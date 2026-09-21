import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("pyarrow")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from general_data import nli_case, premise_key, split_training  # noqa: E402
from publish_comparison import paired_clusters  # noqa: E402
from train_general import batch_loss  # noqa: E402
from typed_common import record  # noqa: E402

from tacit import decision_loss  # noqa: E402
from tacit.semantics import schema_candidates  # noqa: E402


def test_paired_uncertainty_keeps_repeated_premises_together():
    cases = {}
    candidate, reference = [], []
    for i in range(3):
        row = {
            "premise": "premise " + str(i // 2),
            "hypothesis": "hypothesis " + str(i),
            "pairID": str(i),
            "source_row": i,
            "genre": "fixture",
            "label": 0,
        }
        case = nli_case(row, "test")
        cases[case["id"]] = case
        candidate.append(record(case, "relation", [1.0, 0.0, 0.0] if i < 2 else [0.0, 1.0, 0.0]))
        reference.append(record(case, "relation", [0.0, 1.0, 0.0] if i < 2 else [1.0, 0.0, 0.0]))
    result = paired_clusters(candidate, reference, cases, samples=100)
    assert result["clusters"] == 2 and result["decisions"] == 3
    assert result["accuracy_difference"] == pytest.approx(1 / 3)
    assert result["ci95"] == [-1.0, 1.0]
    with pytest.raises(ValueError):
        paired_clusters(candidate + candidate[:1], reference, cases, samples=100)


def test_split_keeps_all_hypotheses_for_same_premise_together():
    rows = [
        {"premise": "premise " + str(i // 3), "label": i % 3, "source_row": i} for i in range(42)
    ]
    excluded = {premise_key(rows[0])}
    parts = split_training(rows, train_cases=12, held_cases=6, excluded_premises=excluded)
    reverse = split_training(rows[::-1], train_cases=12, held_cases=6, excluded_premises=excluded)
    sets = {k: {premise_key(r) for r in v} for k, v in parts.items()}
    assert all(not v & excluded for v in sets.values())
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["calibration"]
    assert {k: {r["source_row"] for r in v} for k, v in parts.items()} == {
        k: {r["source_row"] for r in v} for k, v in reverse.items()
    }


def test_dynamic_hypothesis_is_input_and_labels_are_targets_only():
    row = {
        "premise": "A refund was issued.",
        "hypothesis": "Money was refunded.",
        "pairID": "duplicated",
        "source_row": 7,
        "genre": "fixture",
        "label": 0,
    }
    case = nli_case(row, "train")
    texts, meta = schema_candidates(case["questions"])
    assert all(row["hypothesis"] in text for text in texts)
    assert case["state"] == row["premise"] and case["gold"]["supported"]["label"] is True
    alternate = nli_case({**row, "label": 2}, "train")
    assert alternate["questions"] == case["questions"] and alternate["state"] == case["state"]
    assert nli_case({**row, "source_row": 8}, "train")["id"] != case["id"]
    assert meta[-1][-1] == 5


def test_ragged_decision_loss_excludes_padding_and_weights_decisions_equally():
    logits = torch.tensor([[0.1, 0.2, -torch.inf], [0.3, 0.4, 0.5]], requires_grad=True)
    rows = [
        {"meta": [("a", "noul", [], 0, 2)], "targets": {"a": torch.tensor([0.0, 1.0])}},
        {"meta": [("b", "choice", [], 0, 3)], "targets": {"b": torch.tensor([1.0, 0.0, 0.0])}},
    ]
    expected = (
        decision_loss(logits[:1, :2], rows[0]["targets"]["a"][None], brier_weight=0.1)
        + decision_loss(logits[1:, :3], rows[1]["targets"]["b"][None], brier_weight=0.1)
    ) / 2
    actual = batch_loss(logits, rows)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[0, 2] == 0
