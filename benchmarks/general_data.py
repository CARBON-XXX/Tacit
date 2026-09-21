"""Human-labeled NLI decisions with premise-disjoint train/validation/calibration.

The official matched/mismatched validation sets are held-out evaluation here.
No parsing fields, genre, IDs or labels are included in model inputs. MultiNLI
retains its source-specific licenses; the converter is original MIT Tacit code.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from typed_common import DATA_REVISION, load_cases, partition

MNLI_REVISION = "da70db2af9d09693783c3320c4249840212ee221"
NLI_LABELS = ["entailment", "neutral", "contradiction"]
CRITERIA = {
    "entailment": "The hypothesis must be true if the premise is true.",
    "neutral": "The premise does not establish whether the hypothesis is true or false.",
    "contradiction": "The hypothesis must be false if the premise is true.",
}


def premise_key(row):
    return hashlib.sha256(" ".join(row["premise"].split()).encode()).hexdigest()


def key_digest(key):
    return hashlib.sha256(("tacit-general-v1/" + key).encode()).digest()


def split_training(rows, train_cases=24000, held_cases=1000, excluded_premises=()):
    """Keep all hypotheses for one premise together, including across genres."""
    excluded = set(excluded_premises)
    groups = defaultdict(list)
    for row in rows:
        if row["label"] in range(3) and premise_key(row) not in excluded:
            groups[premise_key(row)].append(row)
    ordered = sorted(groups, key=key_digest)
    output = {k: [] for k in ["validation", "calibration", "train"]}
    position = 0
    for name, budget in [
        ("validation", held_cases),
        ("calibration", held_cases),
        ("train", train_cases),
    ]:
        while len(output[name]) < budget:
            if position >= len(ordered):
                raise ValueError("not enough premise groups for requested split")
            output[name].extend(groups[ordered[position]])
            position += 1
    return output


def nli_case(row, split):
    label = NLI_LABELS[row["label"]]
    hypothesis = row["hypothesis"].strip()
    questions = {
        "relation": {
            "type": "choice",
            "instructions": "Classify this hypothesis using the premise: " + hypothesis,
            "criteria": dict(CRITERIA),
        },
        "supported": {
            "type": "noul",
            "instructions": "Does the premise guarantee this statement? " + hypothesis,
            "criteria": {
                "false": "The premise does not guarantee the statement.",
                "true": "The statement follows necessarily from the premise.",
            },
        },
    }
    supported = label == "entailment"
    return {
        "id": "mnli/" + split + "/" + row["pairID"] + "/" + str(row["source_row"]),
        "workflow": "mnli/" + row["genre"],
        "state": row["premise"],
        "questions": questions,
        "gold": {
            "relation": {
                "label": label,
                "probabilities": {k: float(k == label) for k in NLI_LABELS},
            },
            "supported": {
                "label": supported,
                "probabilities": {"false": float(not supported), "true": float(supported)},
            },
        },
    }


def load_corpus(root, split):
    return [json.loads(line) for line in (Path(root) / (split + ".jsonl")).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".cache/multi-nli"))
    parser.add_argument("--output", type=Path, default=Path(".cache/general-v1"))
    parser.add_argument("--train-cases", type=int, default=24000)
    parser.add_argument("--held-cases", type=int, default=1000)
    parser.add_argument("--test-per-split", type=int, default=1000)
    args = parser.parse_args()
    columns = ["premise", "hypothesis", "label", "genre", "promptID", "pairID"]
    source = {}
    hashes = {}
    for split in ["train", "validation_matched", "validation_mismatched"]:
        path = args.root / "data" / (split + "-00000-of-00001.parquet")
        table = pq.read_table(path, columns=columns)
        names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["label"][
            "names"
        ]
        if names != NLI_LABELS:
            raise ValueError("unexpected NLI label ordering")
        source[split] = table.to_pylist()
        # Published pairID values are not unique. Preserve the pinned row offset.
        for index, row in enumerate(source[split]):
            row["source_row"] = index
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    excluded = {premise_key(r) for k, rows in source.items() if k != "train" for r in rows}
    parts = split_training(source["train"], args.train_cases, args.held_cases, excluded)
    prepared = {k: [nli_case(r, "train") for r in rows] for k, rows in parts.items()}
    prepared["test"] = []
    for split in ["validation_matched", "validation_mismatched"]:
        rows = sorted(
            [r for r in source[split] if r["label"] in range(3)],
            key=lambda r: key_digest(split + "/" + r["pairID"] + "/" + str(r["source_row"])),
        )[: args.test_per_split]
        prepared["test"].extend(nli_case(r, split) for r in rows)
    # These samples are replayed only from Tacit's existing training partition.
    typed = partition(load_cases())
    for split in ["train", "validation", "calibration"]:
        prepared[split].extend(typed[split])
    prepared["test"].extend(load_cases(split="test"))
    all_ids = {k: {r["id"] for r in rows} for k, rows in prepared.items()}
    for i, a in enumerate(prepared):
        if len(all_ids[a]) != len(prepared[a]):
            raise ValueError("duplicate example IDs")
        for b in list(prepared)[i + 1 :]:
            if all_ids[a] & all_ids[b]:
                raise ValueError("overlapping split IDs")
    premise_sets = {k: {premise_key(r) for r in rows} for k, rows in parts.items()}
    for i, a in enumerate(parts):
        if premise_sets[a] & excluded:
            raise ValueError("official evaluation premise leaked")
        for b in list(parts)[i + 1 :]:
            if premise_sets[a] & premise_sets[b]:
                raise ValueError("premise split leakage")
    args.output.mkdir(parents=True, exist_ok=True)
    for split, rows in prepared.items():
        with (args.output / (split + ".jsonl")).open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "dataset_revisions": {
            "nyu-mll/multi_nli": MNLI_REVISION,
            "LocalLLaMA/typed-decisions": DATA_REVISION,
        },
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_sha256": hashes,
        "counts": {k: len(v) for k, v in prepared.items()},
        "nli_premises": {k: len(v) for k, v in premise_sets.items()},
        "train_rows_excluded_for_official_premise_overlap": sum(
            premise_key(r) in excluded for r in source["train"]
        ),
        "split_id_overlap": 0,
        "nli_train_validation_calibration_premise_overlap": 0,
        "nli_overlap_with_official_evaluation_premises": 0,
        "input": "premise as state; hypothesis in instructions; relation and guarantee criteria",
        "selection": (
            "hash-based premise groups; evaluation sampled by source row and pair ID before scoring"
        ),
        "corpus_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.jsonl")
        },
        "licenses": "MultiNLI source licenses retained; typed-decisions Apache-2.0; converter MIT",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
