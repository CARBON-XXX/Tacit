"""Run the pinned Laya implementation and checkpoint on exact canonical cases."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from typed_common import DATA_REVISION, labels, load_cases, partition, record, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=".cache/laya-typed")
    parser.add_argument("--split", choices=["test", "validation", "calibration"], default="test")
    parser.add_argument("--output", type=Path, default=Path("results/typed/laya-typed-test.json"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(".cache/laya-reference").resolve()))
    import laya

    cases = (
        load_cases(split="test") if args.split == "test" else partition(load_cases())[args.split]
    )
    agent = laya.Agent(str(Path(args.checkpoint).resolve()), device="cuda")
    if str(agent.device) != "cuda":
        raise RuntimeError("benchmark requires GPU execution, no silent CPU fallback")
    for _ in range(3):
        agent.predict(cases[0]["state"], cases[0]["questions"])
    records, times = [], []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".jsonl")
    with raw_path.open("w") as raw:
        for i, case in enumerate(cases):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = agent.predict(case["state"], case["questions"])
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
            raw.write(
                json.dumps({"id": case["id"], "result": result, "latency_ms": times[-1]}) + "\n"
            )
            raw.flush()
            for name, q in case["questions"].items():
                answer = result["answers"][name]
                p = (
                    [1 - answer["noul"], answer["noul"]]
                    if q["type"] == "noul"
                    else [answer["probabilities"][k] for k in labels(q)]
                )
                records.append(record(case, name, p))
            if i % 25 == 0:
                print(i, len(cases), summarize(records)["overall"], flush=True)
    report = {
        "model": args.checkpoint,
        "mode": "specialist, released checkpoint, no local training",
        "checkpoint_revision": "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
        "source_revision": subprocess.check_output(
            ["git", "-C", ".cache/laya-reference", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dataset_revision": DATA_REVISION,
        "split": args.split,
        "hardware": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "amp_dtype": str(agent.dtype),
        "parameters": sum(p.numel() for p in agent.model.parameters()),
        "metrics": summarize(records),
        "latency": {
            "p50_ms": float(np.median(times)),
            "p95_ms": float(np.quantile(times, 0.95)),
            "cases": len(times),
            "scope": "warm end-to-end SDK, batch=one case, all five questions, GPU synchronization",
        },
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
