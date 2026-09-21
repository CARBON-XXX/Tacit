"""Expand decision training while keeping prior evaluation groups untouched.

Emotion labels are weak/machine-generated, not independent human ground truth.
Its research-use restriction and AG News' source terms are retained. This is a
research corpus builder, not a grant to redistribute data or relicense weights.
"""

import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from general_data import load_corpus

REVISIONS = {
    "fancyzhx/ag_news": "eb185aade064a813bc0b7f42de02595523103ca4",
    "dair-ai/emotion": "cab853a1dbdf4c42c2b3ef2173804746df8825fe",
}
LABELS = {
    "news": ["World", "Sports", "Business", "Sci/Tech"],
    "emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
}
QUESTIONS = {
    "news": {
        "topic": {
            "type": "choice",
            "instructions": "Which topic best describes this news article?",
            "criteria": {
                "World": "World affairs, government, diplomacy, politics, wars and society.",
                "Sports": "Athletes, teams, sporting events, competitions and match results.",
                "Business": "Companies, finance, markets, economic activity and business deals.",
                "Sci/Tech": "Science, technology, computing, inventions and scientific research.",
            },
        },
    },
    "emotion": {
        "emotion": {
            "type": "choice",
            "instructions": "Which emotion is primarily expressed in this text?",
            "criteria": {
                "sadness": "Sadness, grief, disappointment, loneliness or feeling down.",
                "joy": "Happiness, delight, contentment, optimism or enjoyment.",
                "love": "Love, affection, caring, tenderness or romantic attachment.",
                "anger": "Anger, irritation, frustration, resentment or annoyance.",
                "fear": "Fear, worry, nervousness, anxiety or feeling threatened.",
                "surprise": "Surprise, amazement, astonishment or an unexpected reaction.",
            },
        },
    },
}


def text_key(value):
    return hashlib.sha256(" ".join(value.lower().split()).encode()).hexdigest()


def order_key(value):
    return hashlib.sha256(("tacit-general-v2/" + value).encode()).digest()


def grouped_partition(rows, budgets, excluded=()):
    groups = defaultdict(list)
    for row in rows:
        if text_key(row["text"]) not in excluded:
            groups[text_key(row["text"])].append(row)
    ordered = sorted(groups, key=order_key)
    position, output = 0, {}
    for name, count in budgets:
        output[name] = []
        while len(output[name]) < count and position < len(ordered):
            output[name].extend(groups[ordered[position]])
            position += 1
        if name != "train" and len(output[name]) < count:
            raise ValueError("insufficient rows for development partition")
    return output


def classification_case(row, family, split):
    name = next(iter(QUESTIONS[family]))
    label = LABELS[family][row["label"]]
    return {
        "id": f"{family}/{split}/{row['source_row']}",
        "workflow": family,
        "state": row["text"],
        "questions": copy.deepcopy(QUESTIONS[family]),
        "gold": {
            name: {"label": label, "probabilities": {k: float(k == label) for k in LABELS[family]}}
        },
    }


def structured_nli(case, *, rename=False):
    """A semantically equivalent layout with both statements in the JSON state."""
    result = copy.deepcopy(case)
    prefix = "Classify this hypothesis using the premise: "
    question = case["questions"]["relation"]["instructions"]
    if not isinstance(case["state"], str) or not question.startswith(prefix):
        raise ValueError("expected canonical NLI input")
    result["state"] = {"premise": case["state"], "hypothesis": question[len(prefix) :]}
    result["questions"]["relation"]["instructions"] = (
        "What is the relation of the hypothesis to the premise in the provided record?"
    )
    result["questions"]["supported"]["instructions"] = (
        "Assuming the premise is true, must the hypothesis in the record also be true?"
    )
    if rename:
        result["id"] += "/structured"
    return result


def main():
    base = Path(".cache/general-nli96k")
    destination = Path(".cache/general-v2")
    if destination.exists():
        raise ValueError("use a fresh corpus directory")
    parts = {k: load_corpus(base, k) for k in ["train", "validation", "calibration", "test"]}
    for split, cases in list(parts.items()):
        if split == "train":
            parts[split] = [
                structured_nli(c)
                if c["id"].startswith("mnli/") and order_key(c["id"])[0] < 128
                else c
                for c in cases
            ]
        else:
            parts[split] = cases + [
                structured_nli(c, rename=True) for c in cases if c["id"].startswith("mnli/")
            ]
    sources, notes = {}, {}
    for family, root, folder in [
        ("news", ".cache/ag-news", "data"),
        ("emotion", ".cache/emotion", "split"),
    ]:
        raw = {}
        for split in ["train", "test"] + (["validation"] if family == "emotion" else []):
            path = Path(root) / folder / (split + "-00000-of-00001.parquet")
            table = pq.read_table(path)
            names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["label"][
                "names"
            ]
            if names != LABELS[family]:
                raise ValueError("unexpected classification labels")
            raw[split] = [dict(row, source_row=i) for i, row in enumerate(table.to_pylist())]
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        excluded = {text_key(r["text"]) for r in raw["test"]}
        if family == "news":
            selected = grouped_partition(
                raw["train"],
                [("validation", 1000), ("calibration", 1000), ("train", 32000)],
                excluded,
            )
            source_split = {k: "train" for k in selected}
        else:
            selected = grouped_partition(
                raw["validation"], [("validation", 900), ("calibration", 900)], excluded
            )
            excluded |= {text_key(r["text"]) for r in raw["validation"]}
            selected["train"] = [r for r in raw["train"] if text_key(r["text"]) not in excluded]
            source_split = {k: "validation" if k != "train" else "train" for k in selected}
        selected["test"] = sorted(
            raw["test"], key=lambda r: order_key(f"{family}/test/{r['source_row']}")
        )[:2000]
        source_split["test"] = "test"
        key_sets = {k: {text_key(r["text"]) for r in rows} for k, rows in selected.items()}
        for i, a in enumerate(selected):
            for b in list(selected)[i + 1 :]:
                if key_sets[a] & key_sets[b]:
                    raise ValueError("auxiliary input split leakage")
        for split, rows in selected.items():
            parts[split].extend(classification_case(r, family, source_split[split]) for r in rows)
        notes[family] = {
            "counts": {k: len(v) for k, v in selected.items()},
            "input_overlap": 0,
            "training_excluded_for_official_held_text": sum(
                text_key(r["text"]) in excluded for r in raw["train"]
            ),
            "remaining_official_test_unscored": len(raw["test"]) - len(selected["test"]),
        }
    sets = {k: {r["id"] for r in rows} for k, rows in parts.items()}
    for i, a in enumerate(parts):
        if len(sets[a]) != len(parts[a]):
            raise ValueError("duplicate case ID")
        for b in list(parts)[i + 1 :]:
            if sets[a] & sets[b]:
                raise ValueError("case ID split leakage")
    destination.mkdir(parents=True)
    for split, rows in parts.items():
        (destination / (split + ".jsonl")).write_text(
            "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
        )
    manifest = {
        "parent": json.loads((base / "manifest.json").read_text()),
        "dataset_revisions": REVISIONS,
        "source_sha256": sources,
        "auxiliary": notes,
        "counts": {k: len(v) for k, v in parts.items()},
        "nli_layouts": (
            "train hash-selects one layout; validation/calibration/test include both paired layouts"
        ),
        "licenses": (
            "Original code MIT; Emotion research/education only, machine-generated labels; "
            "AG News source license unspecified; retain source terms. "
            "Research checkpoint, not a commercial license grant."
        ),
        "corpus_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob("*.jsonl")
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in ["parent", "source_sha256"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
