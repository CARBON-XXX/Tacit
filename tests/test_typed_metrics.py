import copy
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from audit_typed import audit  # noqa: E402
from compare_typed import paired_accuracy  # noqa: E402
from typed_common import metrics, partition  # noqa: E402


def example():
    return {
        "id": "a",
        "workflow": "w",
        "question": "q",
        "type": "noul",
        "labels": ["false", "true"],
        "gold_index": 1,
        "target": [0.0, 1.0],
        "probabilities": [0.0, 1.0],
    }


def test_metrics_exact_endpoints_and_invalid_distributions():
    r = example()
    m = metrics([r])
    assert m["accuracy"] == 1 and m["ece_15_bins"] == 0 and m["brier_sum"] == 0
    # Macro F1 includes both declared classes, including the unobserved class.
    assert m["macro_f1_per_question"] == 0.5
    r["probabilities"] = [1.0, 0.0]
    m = metrics([r])
    assert m["accuracy"] == 0 and m["ece_15_bins"] == 1 and m["brier_sum"] == 2
    r["target"] = [float("nan"), 1.0]
    with pytest.raises(ValueError):
        metrics([r])
    with pytest.raises(ValueError):
        metrics([])


def test_partitions_keep_cases_together_and_are_order_independent():
    rows = [{"id": str(i), "workflow": "w"} for i in range(300)]
    a, b = partition(rows), partition(rows[::-1])
    assert a == b
    assert [len(v) for v in a.values()] == [225, 37, 38]
    assert len({r["id"] for part in a.values() for r in part}) == 300


def test_audit_rejects_same_state_under_different_ids():
    a = {"id": "train", "state": {"x": 1}, "questions": {}, "gold": {}}
    b = copy.deepcopy(a)
    b["id"] = "test"
    with pytest.raises(ValueError, match="leakage"):
        audit({"train": [a], "test": [b]})


def test_paired_statistics_detect_misalignment_and_preserve_case_groups():
    a = [example()]
    assert paired_accuracy(a, a, samples=100)["ci95_case_bootstrap"] == [0.0, 0.0]
    b = copy.deepcopy(a)
    b[0]["probabilities"] = [1.0, 0.0]
    result = paired_accuracy(a, b, samples=100)
    assert result["accuracy_difference"] == 1.0
    assert result["ci95_case_bootstrap"] == [1.0, 1.0]
    b[0]["question"] = "mismatched"
    with pytest.raises(ValueError, match="one-to-one"):
        paired_accuracy(a, b, samples=100)
