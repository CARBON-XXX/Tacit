"""Budget-limited Jev evaluation through the documented TypeSafe-compatible API.

No automatic retries or alternate models. Reads only public benchmark state and
questions. Credentials and the cumulative ledger are outside the repository.
Responses contain Gateway-reported per-request costs and usage for auditing.
"""

import argparse
import hashlib
import json
import os
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import requests
from gateway_budget import BudgetLedger
from typed_common import DATA_REVISION, labels, load_cases, record, summarize

ORIGIN = "https://ai-gateway.vercel.sh"
MODEL = "typesafe-ai/jev"


def evaluate_cases(cases, output, limit):
    private = Path.home() / ".config/tacit"
    key = os.environ.get("AI_GATEWAY_API_KEY") or (private / "gateway.key").read_text().strip()
    ledger = BudgetLedger(private / "jev-budget.json")
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        metadata = session.get(ORIGIN + "/v1/models", timeout=30, allow_redirects=False)
        metadata.raise_for_status()
        model = next(r for r in metadata.json()["data"] if r["id"] == MODEL)
        if (
            model["pricing"] != {"input": "0.000000042", "output": "0"}
            or model["context_window"] != 32000
        ):
            raise RuntimeError("catalog pricing/context changed; re-audit before paid requests")
        existing = {}
        raw_path = output.with_suffix(".jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        if raw_path.exists():
            existing = {
                r["id"]: r for r in [json.loads(line) for line in raw_path.read_text().splitlines()]
            }
        for case in cases[:limit]:
            payload = {"model": MODEL, "state": case["state"], "questions": case["questions"]}
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            identity = hashlib.sha256(encoded).hexdigest()
            if case["id"] in existing:
                if existing[case["id"]]["request_sha256"] != identity:
                    raise RuntimeError("existing response used a different request")
                continue
            if len(encoded) > 32000 or len(case["questions"]) > 16:
                raise ValueError("request exceeds the deliberately conservative pilot input bound")
            # Reserve a full 32K context for state AND each question separately.
            # This overbounds the shared-input billing advertised by the provider.
            reserve = (
                Decimal(model["pricing"]["input"])
                * model["context_window"]
                * (1 + len(case["questions"]))
            )
            ledger.reserve(identity, reserve)
            start = time.perf_counter()
            response = session.post(
                ORIGIN + "/typesafe/v1/systemone", data=encoded, timeout=45, allow_redirects=False
            )
            elapsed = 1000 * (time.perf_counter() - start)
            if response.status_code != 200:
                # Leave the reservation in place because an upstream attempt may bill.
                failure = {
                    "status": response.status_code,
                    "request_sha256": identity,
                    "response": response.text.replace(key, "[REDACTED]"),
                }
                output.with_suffix(".error.json").write_text(json.dumps(failure, indent=2) + "\n")
                raise RuntimeError(
                    "Jev HTTP " + str(response.status_code) + "; stopped without retry"
                )
            answer = response.json()
            gateway = answer.get("provider_metadata", {}).get("gateway", {})
            if "cost" not in gateway:
                raise RuntimeError(
                    "missing reported cost; reservation retained and requests stopped"
                )
            if set(answer.get("answers", {})) != set(case["questions"]):
                raise RuntimeError("incomplete answer schema; stop and inspect response")
            entry = {
                "id": case["id"],
                "request_sha256": identity,
                "latency_ms": elapsed,
                "response": answer,
                "reserved_usd": str(reserve),
            }
            # Save the response before settlement; a crash cannot erase the billing bound.
            with raw_path.open("a") as f:
                f.write(json.dumps(entry, separators=(",", ":")).replace(key, "[REDACTED]") + "\n")
                f.flush()
                os.fsync(f.fileno())
            ledger.settle(identity, gateway["cost"], generation_id=gateway.get("generationId"))
            existing[case["id"]] = entry
            if len(existing) % 25 == 0 or limit <= 3:
                print("completed", len(existing), "budget", ledger.summary(), flush=True)
        records, times = [], []
        for case in cases[:limit]:
            row = existing[case["id"]]
            times.append(row["latency_ms"])
            for name, q in case["questions"].items():
                a = row["response"]["answers"][name]
                p = (
                    [1 - a["noul"], a["noul"]]
                    if q["type"] == "noul"
                    else [a["probabilities"][k] for k in labels(q)]
                )
                records.append(record(case, name, p))
        report = {
            "model": MODEL,
            "catalog_metadata": model,
            "dataset_revision": DATA_REVISION,
            "mode": "Jev generalist; no local training or calibration",
            "endpoint": ORIGIN + "/typesafe/v1/systemone",
            "metrics": summarize(records),
            "latency": {
                "p50_ms": float(np.median(times)),
                "p95_ms": float(np.quantile(times, 0.95)),
                "scope": "serial HTTPS requests through Gateway; includes network; no retries",
            },
            "run_cost_usd": str(
                sum(
                    (
                        Decimal(
                            existing[c["id"]]["response"]["provider_metadata"]["gateway"]["cost"]
                        )
                        for c in cases[:limit]
                    ),
                    Decimal(0),
                )
            ),
            "cumulative_budget": ledger.summary(),
            "records": records,
        }
        output.write_text(json.dumps(report, indent=2) + "\n")
        print("result", report["metrics"]["overall"], "cost", report["run_cost_usd"], flush=True)
    finally:
        session.close()
        ledger.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/typed/jev-gateway-test.json"))
    args = parser.parse_args()
    if not 1 <= args.limit <= 400:
        raise ValueError("pilot limit must be 1..400")
    evaluate_cases(load_cases(split="test"), args.output, args.limit)


if __name__ == "__main__":
    main()
