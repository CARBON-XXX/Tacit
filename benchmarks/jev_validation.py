"""Measure Jev on development data to localize gaps without scoring the final gate."""

import argparse
import hashlib
import json
from pathlib import Path

from general_data import load_corpus
from jev_official import collect_parallel, evaluate_cases
from train_general_v2 import measurements


def validation_cases(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / "validation.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["corpus_sha256"][path.name]:
        raise ValueError("validation inputs differ from their pinned corpus manifest")
    return load_corpus(root, "validation"), manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path(".cache/general-v2"))
    parser.add_argument("--output", type=Path, default=Path("results/general/jev-validation.json"))
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--retry-unresolved", action="store_true")
    args = parser.parse_args()
    cases, manifest = validation_cases(args.corpus)
    collect_parallel(cases, args.output, args.concurrency, retry_unresolved=args.retry_unresolved)
    report = evaluate_cases(cases, args.output)
    report["evaluation_split"] = "validation; development only, not a final test"
    report["corpus_manifest"] = manifest
    report["tracks"] = measurements(report["records"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("VALIDATION", {k: v["overall"] for k, v in report["tracks"].items()}, flush=True)


if __name__ == "__main__":
    main()
