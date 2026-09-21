"""Audit immutable data splits and save checksums without inspecting latent factors."""

import hashlib
import json
from pathlib import Path

from typed_common import DATA_REVISION, labels, load_cases, partition, target


def digest(state):
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def audit(splits):
    ids, hashes = {}, {}
    report = {"splits": {}}
    for name, cases in splits.items():
        ids[name] = {c["id"] for c in cases}
        hashes[name] = {digest(c["state"]) for c in cases}
        if len(ids[name]) != len(cases) or len(hashes[name]) != len(cases):
            raise ValueError("duplicate case id or state within " + name)
        for case in cases:
            if set(case["questions"]) != set(case["gold"]):
                raise ValueError("questions and labels do not match")
            for key, q in case["questions"].items():
                if set(labels(q)) != set(case["gold"][key]["probabilities"]):
                    raise ValueError("label spaces do not match")
                p = target(case, key)
                if not ((p >= 0).all() and abs(p.sum() - 1) < 1e-6):
                    raise ValueError("invalid target")
        report["splits"][name] = {
            "cases": len(cases),
            "decisions": sum(len(c["questions"]) for c in cases),
            "unique_ids": len(ids[name]),
            "unique_states": len(hashes[name]),
        }
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if ids[a] & ids[b] or hashes[a] & hashes[b]:
                raise ValueError("split leakage between " + a + " and " + b)
    report["overlapping_ids"] = 0
    report["overlapping_state_hashes"] = 0
    return report


def main():
    splits = {**partition(load_cases()), "test": load_cases(split="test")}
    report = audit(splits)
    report["dataset_revision"] = DATA_REVISION
    report["sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(".cache/typed-data/all").glob("*.parquet"))
    }
    report["input_columns"] = ["state", "questions"]
    report["not_model_input"] = ["id", "workflow", "factors", "gold", "label_agreement"]
    Path("results/typed/data-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
