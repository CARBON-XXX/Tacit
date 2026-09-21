import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from audit_expansion import input_group, scan  # noqa: E402


def test_nli_group_catches_same_premise_across_layouts_and_hypotheses():
    first = {"id": "mnli/train/a", "state": "Shared premise"}
    second = {
        "id": "mnli/test/b/structured",
        "state": {"premise": "Shared premise", "hypothesis": "Different hypothesis"},
    }
    assert input_group(first) == input_group(second)


def test_auxiliary_group_catches_normalized_duplicate_text_with_different_id():
    first = {"id": "news/train/1", "state": {"text": "  An EXAMPLE\n report"}}
    second = {"id": "news/test/2", "state": {"text": "an example report"}}
    assert input_group(first) == input_group(second)


def test_streaming_scan_counts_typed_family_and_rejects_duplicate_id(tmp_path):
    path = tmp_path / "cases.jsonl"
    line = json.dumps({"id": "tr_customer_service_000001", "state": {}}) + "\n"
    path.write_text(line)
    ids, _, counts = scan(path)
    assert len(ids) == 1 and counts == {"typed": 1}
    path.write_text(line + line)
    with pytest.raises(ValueError, match="duplicate"):
        scan(path)
