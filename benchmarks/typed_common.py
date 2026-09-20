"""Canonical typed-decision inputs, splits and distribution metrics.

Only state + questions enter models. Factors, labels and case/workflow identifiers
are never appended to input. The public test split is evaluation-only.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATA_REVISION = "ea9306458d6e9563628369a3d1e72e362fb381d2"


def load_cases(root=Path(".cache/typed-data"), split="train"):
    rows = pq.read_table(root / "all" / f"{split}-00000-of-00001.parquet").to_pylist()
    return [
        {
            "id": r["id"],
            "workflow": r["workflow"],
            "state": json.loads(r["state"]),
            "questions": json.loads(r["questions"]),
            "gold": json.loads(r["gold"]),
        }
        for r in rows
    ]


def labels(question):
    if question["type"] == "noul":
        return ["false", "true"]
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return list(question["criteria"])


def target(case, name):
    keys = labels(case["questions"][name])
    p = np.array([case["gold"][name]["probabilities"][k] for k in keys], dtype=np.float64)
    return p / p.sum()


def partition(cases):
    groups = defaultdict(list)
    for case in cases:
        groups[case["workflow"]].append(case)
    splits = {"train": [], "validation": [], "calibration": []}
    for group in groups.values():
        ordered = sorted(
            group, key=lambda r: hashlib.sha256(("tacit-typed-v1/" + r["id"]).encode()).digest()
        )
        # 225 + 37 + 38 cases per workflow; no case split across questions.
        splits["train"].extend(ordered[:225])
        splits["validation"].extend(ordered[225:262])
        splits["calibration"].extend(ordered[262:])
    return splits


def canonical_state(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def metrics(records):
    """Every decision has equal weight; ECE uses max probability, 15 bins."""
    if not records:
        raise ValueError("metrics need at least one decision")
    hits, confidences, nll, soft, brier, kl, tv, mae, within = ([] for _ in range(9))
    confusion = {}
    for r in records:
        p, g = np.array(r["probabilities"], dtype=float), np.array(r["target"], dtype=float)
        if p.ndim != 1 or p.shape != g.shape or len(p) < 2:
            raise ValueError("probability and target shapes must agree")
        if any(not np.isfinite(v).all() or (v < 0).any() or v.sum() <= 0 for v in (p, g)):
            raise ValueError("invalid probability vector")
        p, g = p / p.sum(), g / g.sum()
        pred, gold = int(p.argmax()), r["gold_index"]
        if not 0 <= gold < len(p):
            raise ValueError("gold index is outside the label space")
        key = (r["workflow"], r["question"], tuple(r["labels"]))
        if key not in confusion:
            confusion[key] = np.zeros((len(p), len(p)), dtype=np.int64)
        confusion[key][gold, pred] += 1
        hits.append(pred == gold)
        confidences.append(float(p.max()))
        nll.append(-float((g * np.log(np.maximum(p, 1e-12))).sum()))
        soft.append(float((p * g).sum()))
        brier.append(float(np.square(p - g).sum()))
        kl.append(float((g * np.log(np.maximum(g, 1e-12) / np.maximum(p, 1e-12))).sum()))
        tv.append(float(np.abs(p - g).sum() / 2))
        if r["type"] == "score":
            levels = np.arange(len(p))
            mae.append(abs(float((p * levels).sum() - (g * levels).sum())))
            within.append(abs(pred - gold) <= 1)
    h, c = np.asarray(hits), np.asarray(confidences)
    ece = 0.0
    for i in range(15):
        mask = (c >= i / 15) & (c < (i + 1) / 15 if i < 14 else c <= 1)
        if mask.any():
            ece += float(mask.mean() * abs(h[mask].mean() - c[mask].mean()))
    return {
        "decisions": len(records),
        "accuracy": float(h.mean()),
        "macro_f1_per_question": float(
            np.mean(
                [
                    np.divide(
                        2 * np.diag(m),
                        m.sum(0) + m.sum(1),
                        out=np.zeros(len(m)),
                        where=(m.sum(0) + m.sum(1)) > 0,
                    ).mean()
                    for m in confusion.values()
                ]
            )
        ),
        "soft_accuracy": float(np.mean(soft)),
        "soft_nll": float(np.mean(nll)),
        "brier_sum": float(np.mean(brier)),
        "kl_gold_to_prediction": float(np.mean(kl)),
        "total_variation": float(np.mean(tv)),
        "ece_15_bins": ece,
        "score_expected_mae": float(np.mean(mae)) if mae else None,
        "score_argmax_within_one": float(np.mean(within)) if within else None,
    }


def record(case, name, probabilities):
    q = case["questions"][name]
    keys = labels(q)
    gold_label = (
        str(case["gold"][name]["label"]).lower()
        if q["type"] == "noul"
        else str(case["gold"][name]["label"])
    )
    return {
        "id": case["id"],
        "workflow": case["workflow"],
        "question": name,
        "type": q["type"],
        "labels": keys,
        "gold_index": keys.index(gold_label),
        "target": target(case, name).tolist(),
        "probabilities": list(map(float, probabilities)),
    }


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups["workflow/" + r["workflow"]].append(r)
        groups["type/" + r["type"]].append(r)
        groups["question/" + r["workflow"] + "/" + r["question"]].append(r)
    return {"overall": metrics(records), "groups": {k: metrics(v) for k, v in groups.items()}}
