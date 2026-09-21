import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("pyarrow")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from evaluate_general import family, group_records  # noqa: E402
from train_general_v2 import track  # noqa: E402


def test_expanded_tasks_do_not_contaminate_workflow_or_canonical_nli_scores():
    ids = [
        "mnli/validation_matched/a/0", "mnli/validation_matched/a/0/structured",
        "news/test/0", "emotion/test/0", "workflow-test-0",
    ]
    rows = [{"id": i, "type": "choice"} for i in ids]
    groups = group_records(rows)
    assert list(groups) == ["nli", "nli-structured", "news", "emotion", "typed"]
    assert all(len(v) == 1 for v in groups.values())
    assert [track(r) for r in rows] == [
        "nli/text/choice", "nli/structured/choice", "news", "emotion", "typed",
    ]
    assert family(ids[0]) == "nli"
    assert track({"id": ids[0], "type": "noul"}) == "nli/text/noul"
