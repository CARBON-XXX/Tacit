"""Independently audit expanded training inputs without loading model predictions."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def family(case):
    name = case["id"].split("/")[0]
    return name if name in {"mnli", "news", "emotion"} else "typed"


def input_group(case):
    name = family(case)
    state = case["state"]
    if name == "mnli":
        content = state["premise"] if isinstance(state, dict) else state
    elif name in {"news", "emotion"}:
        content = state["text"] if isinstance(state, dict) else state
        content = re.sub(r"\s+", " ", content).strip().casefold()
    else:
        content = case["id"]
    return name + "/" + hashlib.sha256(content.encode()).hexdigest()


def scan(path):
    ids, groups, counts = set(), set(), Counter()
    with Path(path).open() as stream:
        for line in stream:
            case = json.loads(line)
            if case["id"] in ids:
                raise ValueError("duplicate case ID")
            ids.add(case["id"])
            groups.add(input_group(case))
            counts[family(case)] += 1
    return ids, groups, counts


def audit(root, previous, gate):
    root, previous, gate = map(Path, (root, previous, gate))
    manifest = json.loads((root / "manifest.json").read_text())
    gate_manifest = json.loads((gate / "manifest.json").read_text())
    if digest("benchmarks/general_v3_data.py") != manifest["builder_sha256"]:
        raise ValueError("training builder changed")
    if digest(previous / "manifest.json") != manifest["previous_manifest_sha256"]:
        raise ValueError("previous corpus manifest changed")
    if digest(gate / "test.jsonl") != gate_manifest["corpus_sha256"]["test.jsonl"]:
        raise ValueError("fresh gate changed")
    partitions = {}
    for split in ("validation", "calibration", "test", "train", "fresh_gate"):
        path = gate / "test.jsonl" if split == "fresh_gate" else root / (split + ".jsonl")
        if split != "fresh_gate" and digest(path) != manifest["corpus_sha256"][path.name]:
            raise ValueError("expanded corpus hash mismatch")
        if split in {"validation", "calibration", "test"} and digest(path) != digest(
            previous / path.name
        ):
            raise ValueError("held partition changed")
        partitions[split] = scan(path)
        if split != "fresh_gate" and len(partitions[split][0]) != manifest["counts"][split]:
            raise ValueError("expanded corpus count mismatch")
    train = partitions["train"]
    for other in ("validation", "calibration", "test", "fresh_gate"):
        if train[0] & partitions[other][0] or train[1] & partitions[other][1]:
            raise ValueError("training input leakage into " + other)
    return {
        "corpus": str(root),
        "manifest_sha256": digest(root / "manifest.json"),
        "audit_program_sha256": digest(__file__),
        "builder_sha256_verified": manifest["builder_sha256"],
        "partition_counts": {name: len(values[0]) for name, values in partitions.items()},
        "training_families": dict(train[2]),
        "validation_calibration_test_byte_identical_to_v2": True,
        "training_id_and_input_group_overlap_with_all_held_partitions": 0,
        "fresh_gate_manifest_sha256": digest(gate / "manifest.json"),
        "fresh_gate_test_sha256": digest(gate / "test.jsonl"),
        "method": (
            "Independent streaming audit: exact NLI premise groups, whitespace/case-normalized "
            "news/emotion text groups, and case IDs. No model predictions read."
        ),
        "limitations": "Does not establish absence from third-party pretraining data.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path(".cache/general-v3"))
    parser.add_argument("--previous", type=Path, default=Path(".cache/general-v2"))
    parser.add_argument("--gate", type=Path, default=Path(".cache/general-final-gate-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/general-v3-audit.json"))
    args = parser.parse_args()
    report = audit(args.corpus, args.previous, args.gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
