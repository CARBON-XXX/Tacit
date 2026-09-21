"""Expand training data while retaining the exact v2 development/test partitions."""

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from general_data import load_corpus
from general_v2_data import (
    classification_case,
    grouped_partition,
    order_key,
    structured_nli,
    text_key,
)
from publish_comparison import bootstrap_cluster
from publish_typed import sha256


def main():
    previous = Path(".cache/general-v2")
    expanded = Path(".cache/general-nli390k")
    destination = Path(".cache/general-v3")
    if destination.exists():
        raise ValueError("use a fresh corpus directory")
    old_manifest = json.loads((previous / "manifest.json").read_text())
    nli_manifest = json.loads((expanded / "manifest.json").read_text())
    for root, manifest in [(previous, old_manifest), (expanded, nli_manifest)]:
        for filename, digest in manifest["corpus_sha256"].items():
            if sha256(root / filename) != digest:
                raise ValueError("source corpus changed")
    parts = {k: load_corpus(previous, k) for k in ("validation", "calibration", "test")}
    old_training = load_corpus(previous, "train")
    nli = [c for c in load_corpus(expanded, "train") if c["id"].startswith("mnli/")]
    training = [structured_nli(c) if order_key(c["id"])[0] < 128 else c for c in nli]
    training.extend(c for c in old_training if not c["id"].startswith(("mnli/", "news/")))
    news = {}
    for split in ("train", "test"):
        path = Path(".cache/ag-news/data") / (split + "-00000-of-00001.parquet")
        if sha256(path) != old_manifest["source_sha256"][str(path)]:
            raise ValueError("news source changed")
        news[split] = [
            dict(row, source_row=i) for i, row in enumerate(pq.read_table(path).to_pylist())
        ]
    excluded = {text_key(r["text"]) for r in news["test"]}
    selected = grouped_partition(
        news["train"], [("validation", 1000), ("calibration", 1000), ("train", 10**9)], excluded
    )
    for split in ("validation", "calibration"):
        expected = {c["id"]: c for c in parts[split] if c["id"].startswith("news/")}
        actual = {
            c["id"]: c for c in [classification_case(r, "news", "train") for r in selected[split]]
        }
        if actual != expected:
            raise ValueError("news development partition changed")
    training.extend(classification_case(r, "news", "train") for r in selected["train"])
    parts["train"] = training
    ids = {k: {c["id"] for c in v} for k, v in parts.items()}
    groups = {
        k: {
            c["id"].split("/")[0] + "/" + bootstrap_cluster(c)
            for c in v
            if c["id"].startswith(("mnli/", "news/", "emotion/"))
        }
        for k, v in parts.items()
    }
    for i, a in enumerate(parts):
        if len(ids[a]) != len(parts[a]):
            raise ValueError("duplicate case ID")
        for b in list(parts)[i + 1 :]:
            if ids[a] & ids[b] or groups[a] & groups[b]:
                raise ValueError("training/development input group leakage")
    gate = load_corpus(".cache/general-final-gate-v1", "test")
    gate_groups = {c["id"].split("/")[0] + "/" + bootstrap_cluster(c) for c in gate}
    if groups["train"] & gate_groups or ids["train"] & {c["id"] for c in gate}:
        raise ValueError("fresh final gate leaked into training")
    destination.mkdir(parents=True)
    for split in ("validation", "calibration", "test"):
        shutil.copyfile(previous / (split + ".jsonl"), destination / (split + ".jsonl"))
    with (destination / "train.jsonl").open("w") as stream:
        for case in training:
            stream.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "parent": nli_manifest,
        "previous_manifest_sha256": sha256(previous / "manifest.json"),
        "dataset_revisions": old_manifest["dataset_revisions"],
        "source_sha256": old_manifest["source_sha256"],
        "counts": {k: len(v) for k, v in parts.items()},
        "training_counts": {
            "nli": len(nli),
            "news": len(selected["train"]),
            "emotion": sum(c["id"].startswith("emotion/") for c in training),
            "typed": sum(not c["id"].startswith(("mnli/", "news/", "emotion/")) for c in training),
        },
        "development_and_test": (
            "identical bytes to general-v2; no reshuffling or model-based selection"
        ),
        "fresh_gate_training_input_overlap": 0,
        "nli_layouts": old_manifest["nli_layouts"],
        "licenses": old_manifest["licenses"],
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "corpus_sha256": {p.name: sha256(p) for p in destination.glob("*.jsonl")},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "counts": manifest["counts"],
                "training_counts": manifest["training_counts"],
                "fresh_gate_training_input_overlap": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
