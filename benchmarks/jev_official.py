"""Pinned Jev evaluation with durable, cumulative spending reservations.

Pricing verified at https://docs.typesafe.ai/models on 2026-09-20.
Charges are estimated from provider-reported input tokens at $0.042/M;
they are not an invoice reconciliation. No retry, redirect, or model fallback.
"""

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import numpy as np
import requests
from gateway_budget import BudgetExceeded, BudgetLedger
from typed_common import DATA_REVISION, labels, load_cases, record, summarize

ORIGIN = "https://api.typesafe.ai"
MODEL = "jev-1.13.0"
INPUT_USD = Decimal("0.000000042")
# Full documented request context, plus a conservative 2x safety margin.
RESERVATION = INPUT_USD * 128000


def token_cost(response):
    if response.get("model") != MODEL:
        raise ValueError("response model differs from pinned version")
    tokens = response.get("usage", {}).get("input_tokens")
    if type(tokens) is not int or not 0 <= tokens <= 64000:
        raise ValueError("missing or out-of-bound billable input token count")
    return INPUT_USD * tokens


def answer_records(case, response):
    if set(response.get("answers", {})) != set(case["questions"]):
        raise ValueError("incomplete or unexpected answer schema")
    output = []
    for name, q in case["questions"].items():
        answer = response["answers"][name]
        if answer.get("type") != q["type"]:
            raise ValueError("response answer type differs from question")
        if q["type"] == "noul":
            p = [1 - answer["noul"], answer["noul"]]
        else:
            if set(answer["probabilities"]) != set(labels(q)):
                raise ValueError("response option labels differ from question")
            p = [answer["probabilities"][k] for k in labels(q)]
        if not np.isfinite(p).all() or min(p) < 0 or max(p) > 1:
            raise ValueError("invalid probability value")
        # The live API rounds probabilities to 2 decimal places. Only accept
        # the maximum mass error explainable by that rounding, not arbitrary
        # malformed vectors. Retain unmodified values in the raw response.
        quantized = all(abs(v * 100 - round(v * 100)) < 1e-8 for v in p)
        tolerance = len(p) * 0.005 + 1e-8 if quantized else 1e-4
        if sum(p) <= 0 or abs(sum(p) - 1) > tolerance:
            raise ValueError("probability mass error exceeds decimal-rounding bound")
        p = (np.asarray(p) / sum(p)).tolist()
        output.append(record(case, name, p))
    return output


def collect_parallel(
    cases,
    output,
    workers,
    private=None,
    session_factory=requests.Session,
    *,
    retry_unresolved=False,
):
    """Bounded waves: reserve all requests before dispatch, settle before next wave.

    Only the main thread touches the ledger/files. An error drains and accounts
    for the current wave, then stops; it never launches another wave or retries.
    A separate explicit recovery run may retry unknown attempts with a NEW
    reservation while retaining every earlier unknown charge.
    """
    if not 1 <= workers <= 8:
        raise ValueError("workers must be 1..8")
    private = private or Path.home() / ".config/tacit"
    key = os.environ.get("TYPESAFE_API_KEY") or (private / "typesafe.key").read_text().strip()
    ledger = BudgetLedger(private / "jev-budget.json", cap="0.50")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix(".jsonl")
    local = threading.local()
    sessions = []

    def fetch(encoded):
        if not hasattr(local, "session"):
            local.session = session_factory()
            local.session.headers.update(
                {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
            )
            sessions.append(local.session)
        start = time.perf_counter()
        response = local.session.post(
            ORIGIN + "/v1/systemone", data=encoded, timeout=45, allow_redirects=False
        )
        return response, 1000 * (time.perf_counter() - start)

    try:
        saved = (
            [json.loads(line) for line in raw_path.read_text().splitlines()]
            if raw_path.exists()
            else []
        )
        existing = {r["id"]: r for r in saved}
        if len(existing) != len(saved):
            raise ValueError("duplicate saved case")
        entries = {r["request_id"]: r for r in ledger.data["entries"]}
        pending, identities = [], set()
        for case in cases:
            payload = {"model": MODEL, "state": case["state"], "questions": case["questions"]}
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            identity = hashlib.sha256(encoded).hexdigest()
            if identity in identities:
                raise ValueError("duplicate request body; deduplicate corpus before evaluation")
            identities.add(identity)
            if case["id"] in existing:
                row = existing[case["id"]]
                attempt = row.get("ledger_request_id", identity)
                if (
                    row["request_sha256"] != identity
                    or attempt not in entries
                    or not (attempt == identity or attempt.startswith(identity + ":retry:"))
                ):
                    raise ValueError("saved response/ledger mismatch")
                cost = token_cost(row["response"])
                if entries[attempt]["status"] == "reserved":
                    ledger.settle(attempt, cost, cost_basis="reported_tokens_times_published_price")
                elif Decimal(entries[attempt]["cost_usd"]) != cost:
                    raise ValueError("settled cost mismatch")
                answer_records(case, row["response"])
            else:
                previous = [
                    e
                    for k, e in entries.items()
                    if k == identity or k.startswith(identity + ":retry:")
                ]
                attempt = identity
                if previous:
                    if not retry_unresolved or any(e["status"] != "reserved" for e in previous):
                        raise ValueError(
                            "previous attempt has no saved response; explicit recovery required"
                        )
                    attempt = identity + ":retry:" + str(len(previous))
                if len(encoded) > 32000 or len(case["questions"]) > 16:
                    raise ValueError("request exceeds conservative input bound")
                pending.append((case, encoded, identity, attempt))
        completed = len(cases) - len(pending)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(pending), workers):
                wave = pending[offset : offset + workers]
                if ledger.committed + RESERVATION * len(wave) > Decimal(ledger.data["cap_usd"]):
                    raise BudgetExceeded("next wave would exceed cumulative budget")
                for _, _, _, attempt in wave:
                    ledger.reserve(attempt, RESERVATION)
                futures = [pool.submit(fetch, encoded) for _, encoded, _, _ in wave]
                failure = None
                for (case, _, identity, attempt), future in zip(wave, futures, strict=True):
                    try:
                        response, elapsed = future.result()
                        if response.status_code != 200:
                            error = {
                                "status": response.status_code,
                                "request_sha256": identity,
                                "response": response.text.replace(key, "[REDACTED]"),
                            }
                            with output.with_suffix(".errors.jsonl").open("a") as stream:
                                stream.write(json.dumps(error) + "\n")
                            raise RuntimeError(
                                "Jev HTTP " + str(response.status_code) + "; stopped without retry"
                            )
                        row = {
                            "id": case["id"],
                            "request_sha256": identity,
                            "ledger_request_id": attempt,
                            "response": response.json(),
                            "latency_ms": elapsed,
                            "reserved_usd": str(RESERVATION),
                            "concurrency": workers,
                        }
                        with raw_path.open("a") as stream:
                            stream.write(
                                json.dumps(row, separators=(",", ":")).replace(key, "[REDACTED]")
                                + "\n"
                            )
                            stream.flush()
                            os.fsync(stream.fileno())
                        ledger.settle(
                            attempt,
                            token_cost(row["response"]),
                            cost_basis="reported_tokens_times_published_price",
                        )
                        answer_records(case, row["response"])
                        completed += 1
                    except Exception as error:
                        with output.with_suffix(".attempt-errors.jsonl").open("a") as stream:
                            stream.write(
                                json.dumps(
                                    {
                                        "id": case["id"],
                                        "request_sha256": identity,
                                        "ledger_request_id": attempt,
                                        "error_type": type(error).__name__,
                                        "error": str(error).replace(key, "[REDACTED]"),
                                    }
                                )
                                + "\n"
                            )
                        failure = failure or error
                if failure is not None:
                    raise failure
                if completed % 40 == 0 or completed == len(cases):
                    print(
                        "completed",
                        completed,
                        "of",
                        len(cases),
                        "budget",
                        ledger.summary(),
                        flush=True,
                    )
    finally:
        for session in sessions:
            session.close()
        ledger.close()


def evaluate_cases(cases, output, private=None, session=None):
    private = private or Path.home() / ".config/tacit"
    key = os.environ.get("TYPESAFE_API_KEY") or (private / "typesafe.key").read_text().strip()
    ledger = BudgetLedger(private / "jev-budget.json", cap="0.50")
    session = session or requests.Session()
    session.headers.update({"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    raw_path = output.with_suffix(".jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    try:
        if raw_path.exists():
            for line in raw_path.read_text().splitlines():
                row = json.loads(line)
                if row["id"] in existing:
                    raise ValueError("duplicate saved response ID")
                existing[row["id"]] = row
        entries = {row["request_id"]: row for row in ledger.data["entries"]}
        for index, case in enumerate(cases):
            payload = {"model": MODEL, "state": case["state"], "questions": case["questions"]}
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            identity = hashlib.sha256(encoded).hexdigest()
            if case["id"] in existing:
                row = existing[case["id"]]
                if row["request_sha256"] != identity:
                    raise ValueError("existing response used a different request")
                attempt = row.get("ledger_request_id", identity)
                if attempt not in entries or not (
                    attempt == identity or attempt.startswith(identity + ":retry:")
                ):
                    raise ValueError("saved response has no cumulative budget entry")
                cost = token_cost(row["response"])
                # Recover a crash between fsync of response and ledger settlement.
                if entries[attempt]["status"] == "reserved":
                    ledger.settle(attempt, cost, cost_basis="reported_tokens_times_published_price")
                elif Decimal(entries[attempt]["cost_usd"]) != cost:
                    raise ValueError("saved response and settled cost disagree")
                answer_records(case, row["response"])
                continue
            # Deliberately small inputs; never silently truncate a benchmark case.
            if len(encoded) > 32000 or len(case["questions"]) > 16:
                raise ValueError("request exceeds conservative evaluation input bound")
            ledger.reserve(identity, RESERVATION)
            start = time.perf_counter()
            response = session.post(
                ORIGIN + "/v1/systemone", data=encoded, timeout=45, allow_redirects=False
            )
            elapsed = 1000 * (time.perf_counter() - start)
            if response.status_code != 200:
                failure = {
                    "status": response.status_code,
                    "request_sha256": identity,
                    "response": response.text.replace(key, "[REDACTED]"),
                    "cumulative_budget": ledger.summary(),
                }
                output.with_suffix(".error.json").write_text(json.dumps(failure, indent=2) + "\n")
                raise RuntimeError(
                    "Jev HTTP " + str(response.status_code) + "; stopped without retry"
                )
            answer = response.json()
            row = {
                "id": case["id"],
                "request_sha256": identity,
                "latency_ms": elapsed,
                "response": answer,
                "reserved_usd": str(RESERVATION),
            }
            # Persist even malformed responses. Unknown charges retain reservation.
            with raw_path.open("a") as f:
                f.write(json.dumps(row, separators=(",", ":")).replace(key, "[REDACTED]") + "\n")
                f.flush()
                os.fsync(f.fileno())
            cost = token_cost(answer)
            ledger.settle(identity, cost, cost_basis="reported_tokens_times_published_price")
            answer_records(case, answer)
            existing[case["id"]] = row
            if (index + 1) % 25 == 0 or len(cases) <= 3:
                print(
                    "completed", index + 1, "of", len(cases), "budget", ledger.summary(), flush=True
                )
        records, times = [], []
        for case in cases:
            row = existing[case["id"]]
            records.extend(answer_records(case, row["response"]))
            times.append(row["latency_ms"])
        request_hashes = {existing[c["id"]]["request_sha256"] for c in cases}
        attempts = [
            e
            for e in ledger.data["entries"]
            if e["request_id"].split(":retry:", 1)[0] in request_hashes
        ]
        report = {
            "model": MODEL,
            "mode": "Jev generalist; no local training or calibration",
            "endpoint": ORIGIN + "/v1/systemone",
            "case_ids_sha256": hashlib.sha256(
                "\n".join(c["id"] for c in cases).encode()
            ).hexdigest(),
            "pricing": {
                "input_usd_per_token": str(INPUT_USD),
                "output_usd_per_token": "0",
                "source": "https://docs.typesafe.ai/models",
                "verified": "2026-09-20",
                "cost_basis": "provider-reported input tokens times published price; not invoice",
            },
            "probability_processing": (
                "API rounds probabilities to two decimals; normalize sum per question. "
                "Keep raw responses. NLL uses common 1e-12 floor for returned zeros. "
                "This evaluates exposed API probabilities, not hidden full precision."
            ),
            "metrics": summarize(records),
            "latency": {
                "p50_ms": float(np.median(times)),
                "p95_ms": float(np.quantile(times, 0.95)),
                "cases": len(cases),
                "concurrency": max(existing[c["id"]].get("concurrency", 1) for c in cases),
                "scope": (
                    "successful HTTPS attempts only; includes network and remote serving; "
                    "failed-attempt reservations reported separately; see concurrency"
                ),
            },
            "run_cost_usd": str(
                sum((token_cost(existing[c["id"]]["response"]) for c in cases), Decimal(0))
            ),
            "input_tokens": sum(
                existing[c["id"]]["response"]["usage"]["input_tokens"] for c in cases
            ),
            "cumulative_budget": ledger.summary(),
            "attempt_accounting": {
                "successful_requests": len(cases),
                "attempts": len(attempts),
                "unresolved_attempts": sum(e["status"] == "reserved" for e in attempts),
                "unresolved_reserved_usd": str(
                    sum(
                        (Decimal(e["reserved_usd"]) for e in attempts if e["status"] == "reserved"),
                        Decimal(0),
                    )
                ),
                "policy": "no automatic retry; explicit recovery retains all older reservations",
            },
            "records": records,
        }
        output.write_text(json.dumps(report, indent=2) + "\n")
        print("result", report["metrics"]["overall"], "cost", report["run_cost_usd"], flush=True)
        return report
    finally:
        session.close()
        ledger.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--suite", choices=["typed", "nli", "expanded"], default="typed")
    parser.add_argument("--concurrency", type=int, default=1, choices=range(1, 9))
    parser.add_argument(
        "--retry-unresolved",
        action="store_true",
        help="explicit recovery; retains prior reservations and reserves again",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.suite == "typed":
        cases = load_cases(split="test")
        revisions = {"LocalLLaMA/typed-decisions": DATA_REVISION}
    elif args.suite == "nli":
        from general_data import MNLI_REVISION, load_corpus

        cases = [c for c in load_corpus(".cache/general-v1", "test") if c["id"].startswith("mnli/")]
        revisions = {"nyu-mll/multi_nli": MNLI_REVISION}
    else:
        from general_data import MNLI_REVISION, load_corpus
        from general_v2_data import REVISIONS

        cases = [
            c
            for c in load_corpus(".cache/general-v2", "test")
            if c["id"].endswith("/structured") or c["id"].startswith(("news/", "emotion/"))
        ]
        revisions = {"nyu-mll/multi_nli": MNLI_REVISION, **REVISIONS}
    if not 1 <= args.limit <= len(cases):
        raise ValueError("limit must be 1.." + str(len(cases)))
    output = args.output or Path(
        "results/" + ("typed" if args.suite == "typed" else "general") + "/jev-official-test.json"
    )
    if args.output is None and args.suite == "expanded":
        output = Path("results/general/jev-expanded-test.json")
    if args.concurrency > 1 or args.retry_unresolved:
        collect_parallel(
            cases[: args.limit], output, args.concurrency, retry_unresolved=args.retry_unresolved
        )
    report = evaluate_cases(cases[: args.limit], output)
    report["dataset_revisions"] = revisions
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
