import copy
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from general_data import nli_case  # noqa: E402
from general_v2_data import grouped_partition, structured_nli, text_key  # noqa: E402
from train_general_v2 import memory_batches, selection  # noqa: E402


def test_paired_nli_layout_preserves_target_and_moves_hypothesis_into_state():
    case = nli_case(
        {
            "premise": "A report was approved.",
            "hypothesis": "A report exists.",
            "label": 0,
            "pairID": "example",
            "source_row": 0,
            "genre": "fixture",
        },
        "test",
    )
    original = copy.deepcopy(case)
    result = structured_nli(case, rename=True)
    assert result["gold"] == case["gold"]
    assert result["state"] == {"premise": case["state"], "hypothesis": "A report exists."}
    assert result["id"] == case["id"] + "/structured"
    assert case == original
    assert "A report exists." not in result["questions"]["relation"]["instructions"]


def test_normalized_duplicate_texts_cannot_cross_development_partitions():
    rows = [{"text": f"Example {i}", "label": i % 2, "source_row": i} for i in range(20)]
    rows.append({"text": "  EXAMPLE    1 ", "label": 1, "source_row": 20})
    parts = grouped_partition(
        rows, [("validation", 5), ("calibration", 5), ("train", 100)], {text_key("example 0")}
    )
    keys = {k: {text_key(r["text"]) for r in v} for k, v in parts.items()}
    assert not keys["train"] & keys["validation"]
    assert not keys["train"] & keys["calibration"]
    assert not keys["validation"] & keys["calibration"]
    assert all(text_key("example 0") not in v for v in keys.values())
    assert sum(len(v) for v in parts.values()) == 20


def test_adaptive_batches_bound_padding_cost_and_replay_deterministically():
    rows = [{"length": 50 + i * 53, "id": i} for i in range(45)]
    first = memory_batches(rows, 16, random.Random(12), max_tokens=8192, max_cells=8_388_608)
    second = memory_batches(rows, 16, random.Random(12), max_tokens=8192, max_cells=8_388_608)
    assert [[r["id"] for r in b] for b in first] == [[r["id"] for r in b] for b in second]
    assert sorted(r["id"] for b in first for r in b) == list(range(45))
    for batch in first:
        maximum = max(r["length"] for r in batch)
        assert len(batch) * maximum <= 8192
        assert len(batch) * maximum**2 <= 8_388_608
    with pytest.raises(ValueError):
        memory_batches([{"length": 5000}], 16, random.Random(0))


def test_new_task_improvement_cannot_hide_protected_task_regression():
    names = ["nli/text/choice", "nli/text/noul", "typed", "news", "emotion"]
    baseline = {k: {"overall": {"soft_nll": 0.5}} for k in names}
    candidate = copy.deepcopy(baseline)
    candidate["news"]["overall"]["soft_nll"] = 0.1
    candidate["nli/text/choice"]["overall"]["soft_nll"] = 0.52
    score, failures = selection(candidate, baseline, 0.01)
    assert score < 0.5 and "nli/text/choice" in failures
    candidate["nli/text/choice"]["overall"]["soft_nll"] = 0.49
    assert not selection(candidate, baseline, 0.01)[1]
