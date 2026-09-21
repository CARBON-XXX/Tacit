"""Evaluate released Laya or a Tacit run on the same frozen general corpus."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from general_data import load_corpus
from train_general_v2 import measurements
from typed_common import labels, record, summarize


def family(case_id):
    if case_id.startswith("mnli/"):
        return "nli-structured" if case_id.endswith("/structured") else "nli"
    for name in ("news", "emotion"):
        if case_id.startswith(name + "/"):
            return name
    return "typed"


def group_records(records):
    grouped = {}
    for row in records:
        grouped.setdefault(family(row["id"]), []).append(row)
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["laya", "tacit"], default="laya")
    parser.add_argument("--checkpoint", type=Path, default=Path(".cache/laya-general"))
    parser.add_argument("--revision", default="1c5edc17a7acd8701df6fc341c0d179f1c62c982")
    parser.add_argument("--corpus", type=Path, default=Path(".cache/general-v1"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/general/laya-general-test.json")
    )
    parser.add_argument(
        "--families", nargs="+", choices=["nli", "nli-structured", "typed", "news", "emotion"]
    )
    parser.add_argument(
        "--accuracy-only", action="store_true",
        help="omit latency measurements; suitable when GPU training is concurrent",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh output path")
    cases = load_corpus(args.corpus, "test")
    if args.families:
        cases = [c for c in cases if family(c["id"]) in args.families]
    if not cases:
        raise ValueError("no cases in the selected families")
    if args.model == "laya":
        sys.path.insert(0, str(Path(".cache/laya-reference").resolve()))
        import laya

        agent = laya.Agent(str(args.checkpoint.resolve()), device="cuda")
        checkpoint_file = args.checkpoint / "model.safetensors"
        provenance = {
            "model": "convaiinnovations/laya",
            "checkpoint_revision": args.revision,
            "source_revision": subprocess.check_output(
                ["git", "-C", ".cache/laya-reference", "rev-parse", "HEAD"], text=True
            ).strip(),
            "mode": "released general English checkpoint; no local tuning; shipped temperatures",
            "configuration": agent.cfg,
        }
    else:
        from tacit.agent import DecisionAgent

        agent = DecisionAgent(args.checkpoint, device="cuda")
        checkpoint_file = args.checkpoint / "best.pt"
        provenance = {
            "model": str(args.checkpoint),
            "manifest": agent.manifest,
            "temperatures": agent.temperatures,
            "mode": "supervised decision checkpoint; held-out test; fitted calibration",
        }
    if str(agent.device) != "cuda":
        raise RuntimeError("GPU required, no silent fallback")
    for _ in range(3):
        agent.predict(cases[0]["state"], cases[0]["questions"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records, latencies = [], []
    with args.output.with_suffix(".jsonl").open("w") as raw:
        for i, case in enumerate(cases):
            if not args.accuracy_only:
                torch.cuda.synchronize()
                start = time.perf_counter()
            prediction = agent.predict(case["state"], case["questions"])
            if agent.device.type != "cuda":
                raise RuntimeError("model switched to CPU during evaluation")
            entry = {"id": case["id"], "response": prediction}
            if not args.accuracy_only:
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append({"id": case["id"], "latency_ms": elapsed})
                entry["latency_ms"] = elapsed
            raw.write(
                json.dumps(entry, separators=(",", ":"))
                + "\n"
            )
            raw.flush()
            for name, q in case["questions"].items():
                a = prediction["answers"][name]
                p = (
                    [1 - a["noul"], a["noul"]]
                    if q["type"] == "noul"
                    else [a["probabilities"][k] for k in labels(q)]
                )
                records.append(record(case, name, p))
            if (i + 1) % 100 == 0:
                print("evaluated", i + 1, "of", len(cases), flush=True)
    grouped = group_records(records)
    report = {
        **provenance,
        "checkpoint_sha256": hashlib.sha256(checkpoint_file.read_bytes()).hexdigest(),
        "corpus_manifest": json.loads((args.corpus / "manifest.json").read_text()),
        "hardware": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": str(agent.dtype),
        "parameters": sum(p.numel() for p in agent.model.parameters()),
        "metrics": {name: summarize(rows) for name, rows in grouped.items()},
        "tracks": measurements(records),
        "selected_families": sorted(grouped),
        "measurement_mode": "accuracy only; no latency claim" if args.accuracy_only
        else "accuracy and latency; requires an idle GPU for meaningful timing",
        "latency": {
            name: {
                "p50_ms": float(
                    np.median(
                        [r["latency_ms"] for r in latencies if r["id"] in {p["id"] for p in rows}]
                    )
                ),
                "scope": "isolated full SDK per case; CUDA synchronized; includes tokenization",
            }
            for name, rows in grouped.items() if not args.accuracy_only
        },
        "latencies": latencies,
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT", {k: v["overall"] for k, v in report["metrics"].items()}, flush=True)


if __name__ == "__main__":
    main()
