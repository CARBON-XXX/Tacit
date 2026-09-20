"""Paired complete-request evaluation against the pinned local Laya baseline."""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from typed_common import DATA_REVISION, labels, load_cases, record, summarize

from tacit.agent import DecisionAgent


def paired_accuracy(candidate, reference, *, samples=10000, seed=2026):
    """Resample cases within workflow; questions from one case stay together."""

    def key(r):
        return r["id"], r["question"]

    a, b = {key(r): r for r in candidate}, {key(r): r for r in reference}
    if len(a) != len(candidate) or len(b) != len(reference) or set(a) != set(b):
        raise ValueError("predictions must match one-to-one by case and question")
    cases = defaultdict(list)
    for k, r in a.items():
        ref = b[k]
        if any(
            r[field] != ref[field]
            for field in ["labels", "gold_index", "target", "workflow", "type"]
        ):
            raise ValueError("reference labels or metadata differ")
        hit_a = int(np.argmax(r["probabilities"]) == r["gold_index"])
        hit_b = int(np.argmax(ref["probabilities"]) == r["gold_index"])
        cases[(r["workflow"], r["id"])].append(hit_a - hit_b)
    workflows = defaultdict(list)
    for (workflow, _), differences in sorted(cases.items()):
        workflows[workflow].append([sum(differences), len(differences)])
    rng = np.random.default_rng(seed)
    numerator, denominator = np.zeros(samples), np.zeros(samples)
    for values in workflows.values():
        v = np.asarray(values)
        selected = v[rng.integers(0, len(v), size=(samples, len(v)))]
        numerator += selected[:, :, 0].sum(1)
        denominator += selected[:, :, 1].sum(1)
    distribution = numerator / denominator
    return {
        "accuracy_difference": float(sum(sum(v) for v in cases.values()) / len(a)),
        "ci95_case_bootstrap": np.quantile(distribution, [0.025, 0.975]).tolist(),
        "resamples": samples,
        "seed": seed,
        "unit": "case, stratified by workflow; five correlated questions per case",
        "scope": "one fixed checkpoint; does not include training-seed uncertainty",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("results/typed/forked-seed0"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/typed/paired-forked-laya.json")
    )
    args = parser.parse_args()
    sys.path.insert(0, str(Path(".cache/laya-reference").resolve()))
    import laya

    cases = load_cases(split="test")
    tacit = DecisionAgent(args.run, device="cuda")
    other = laya.Agent(str(Path(".cache/laya-typed").resolve()), device="cuda")
    if other.device.type != "cuda":
        raise RuntimeError("Laya fell back to CPU")
    agents = {"tacit": tacit, "laya": other}
    # Warm every schema; timings include state tokenization and typed output construction.
    for workflow in sorted({c["workflow"] for c in cases}):
        case = next(c for c in cases if c["workflow"] == workflow)
        for agent in agents.values():
            for _ in range(3):
                agent.predict(case["state"], case["questions"])
    records = {k: [] for k in agents}
    times = {k: [] for k in agents}
    rng = random.Random(2026)
    order = list(range(len(cases)))
    rng.shuffle(order)
    trace = []
    for i, index in enumerate(order):
        case = cases[index]
        models = list(agents)
        rng.shuffle(models)
        entry = {"id": case["id"], "model_order": models, "latency_ms": {}}
        for name in models:
            torch.cuda.synchronize()
            start = time.perf_counter()
            response = agents[name].predict(case["state"], case["questions"])
            torch.cuda.synchronize()
            ms = 1000 * (time.perf_counter() - start)
            times[name].append(ms)
            entry["latency_ms"][name] = ms
            for q, question in case["questions"].items():
                answer = response["answers"][q]
                p = (
                    [1 - answer["noul"], answer["noul"]]
                    if question["type"] == "noul"
                    else [answer["probabilities"][k] for k in labels(question)]
                )
                records[name].append(record(case, q, p))
        trace.append(entry)
        if i % 50 == 0:
            print("paired cases", i, len(cases), flush=True)
    result = {
        "dataset_revision": DATA_REVISION,
        "run": str(args.run),
        "hardware": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "precision": {"tacit": str(tacit.dtype), "laya": str(other.dtype)},
        "protocol": (
            "randomized paired model order and case order, seed 2026; warm schema caches; "
            "complete SDK requests; one case with five questions; GPU synchronization; "
            "no concurrent training"
        ),
        "metrics": {k: summarize(v) for k, v in records.items()},
        "latency": {
            k: {
                "p50_ms": float(np.median(v)),
                "p95_ms": float(np.quantile(v, 0.95)),
                "cases": len(v),
            }
            for k, v in times.items()
        },
        "paired_accuracy": paired_accuracy(records["tacit"], records["laya"]),
        "records": records,
        "trace": trace,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ["records", "trace", "metrics"]}),
        flush=True,
    )
    print(
        "accuracy", {k: v["overall"]["accuracy"] for k, v in result["metrics"].items()}, flush=True
    )


if __name__ == "__main__":
    main()
