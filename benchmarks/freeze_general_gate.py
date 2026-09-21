"""Freeze an unscored NLI/news gate, excluding every previously used input group.

This script reads no predictions and invokes no model. It does not replace the
workflow, emotion, efficiency or calibration requirements of a broad comparison.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from general_data import load_corpus, nli_case, premise_key
from general_v2_data import classification_case, structured_nli, text_key
from publish_comparison import bootstrap_cluster


def select_groups(rows, key, excluded, count, namespace):
    groups = defaultdict(list)
    for row in rows:
        group = key(row)
        if group not in excluded:
            groups[group].append(row)
    order = sorted(groups, key=lambda k: hashlib.sha256((namespace + "/" + k).encode()).digest())
    selected = []
    for group in order:
        selected.extend(sorted(groups[group], key=lambda r: r["source_row"]))
        if len(selected) >= count:
            return selected
    raise ValueError("not enough unseen input groups for the requested gate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, default=Path(".cache/general-v2"))
    parser.add_argument("--output", type=Path, default=Path(".cache/general-final-gate-v1"))
    parser.add_argument("--per-nli-split", type=int, default=1000)
    parser.add_argument("--news-cases", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("gate already exists; never silently resample it")
    if min(args.per_nli_split, args.news_cases) < 1:
        raise ValueError("positive gate sizes required")
    previous = json.loads((args.previous / "manifest.json").read_text())
    used_ids, used_nli, used_news = set(), set(), set()
    for split in ("train", "validation", "calibration", "test"):
        path = args.previous / (split + ".jsonl")
        if hashlib.sha256(path.read_bytes()).hexdigest() != previous["corpus_sha256"][path.name]:
            raise ValueError("previous corpus differs from its manifest")
        for case in load_corpus(args.previous, split):
            used_ids.add(case["id"])
            if case["id"].startswith("mnli/"):
                used_nli.add(bootstrap_cluster(case))
            elif case["id"].startswith("news/"):
                used_news.add(text_key(case["state"]))
    source_hashes, cases, counts = {}, [], {}
    for split in ("validation_matched", "validation_mismatched"):
        path = Path(".cache/multi-nli/data") / (split + "-00000-of-00001.parquet")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != previous["parent"]["source_sha256"][str(path)]:
            raise ValueError("NLI source changed")
        source_hashes[str(path)] = digest
        rows = [
            dict(row, source_row=i)
            for i, row in enumerate(pq.read_table(path).to_pylist())
            if row["label"] in range(3)
        ]
        selected = select_groups(
            rows, premise_key, used_nli, args.per_nli_split, "tacit-final-gate-v1/" + split
        )
        used_nli.update(premise_key(r) for r in selected)
        converted = [nli_case(r, split) for r in selected]
        cases.extend(converted)
        cases.extend(structured_nli(c, rename=True) for c in converted)
        counts[split] = len(selected)
    path = Path(".cache/ag-news/data/test-00000-of-00001.parquet")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != previous["source_sha256"][str(path)]:
        raise ValueError("news source changed")
    source_hashes[str(path)] = digest
    rows = [dict(row, source_row=i) for i, row in enumerate(pq.read_table(path).to_pylist())]
    selected = select_groups(
        rows, lambda r: text_key(r["text"]), used_news, args.news_cases, "tacit-final-gate-v1/news"
    )
    counts["news"] = len(selected)
    cases.extend(classification_case(r, "news", "test") for r in selected)
    ids = {c["id"] for c in cases}
    if len(ids) != len(cases) or ids & used_ids:
        raise ValueError("duplicate or previously used case ID in gate")
    encoded = "".join(
        json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n" for c in cases
    )
    manifest = {
        "status_at_freeze": "unscored; no model outputs consulted",
        "selection": "hash-ordered complete input groups; excludes all previous corpus splits",
        "counts": counts,
        "requests": len(cases),
        "decisions": sum(len(c["questions"]) for c in cases),
        "paired_nli_layouts": True,
        "previous_manifest_sha256": hashlib.sha256(
            (args.previous / "manifest.json").read_bytes()
        ).hexdigest(),
        "dataset_revisions": {
            **previous["parent"]["dataset_revisions"],
            **previous["dataset_revisions"],
        },
        "source_sha256": source_hashes,
        "corpus_sha256": {"test.jsonl": hashlib.sha256(encoded.encode()).hexdigest()},
        "no_previously_used_case_or_input_group": True,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "NLI/news only; does not satisfy workflow, emotion, latency or calibration gates.",
            "Local split separation does not prove absence of pretrained-model contamination.",
            "Both NLI layouts share cases and must be clustered in uncertainty estimates.",
            "Do not tune a checkpoint using these scores before claiming a fresh final test.",
        ],
    }
    args.output.mkdir(parents=True)
    (args.output / "test.jsonl").write_text(encoded)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
