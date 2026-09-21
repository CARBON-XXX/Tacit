import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from freeze_general_gate import select_groups  # noqa: E402


def test_fresh_gate_excludes_seen_groups_and_is_order_independent():
    rows = [{"group": str(i // 2), "source_row": i} for i in range(30)]
    key = lambda row: row["group"]  # noqa: E731
    a = select_groups(rows, key, {"0", "3"}, 5, "fixture")
    b = select_groups(rows[::-1], key, {"0", "3"}, 5, "fixture")
    assert a == b
    assert len(a) == 6
    assert not {r["group"] for r in a} & {"0", "3"}
    assert all(sum(r["group"] == group for r in a) == 2 for group in {r["group"] for r in a})
    with pytest.raises(ValueError, match="not enough unseen"):
        select_groups(rows, key, {str(i) for i in range(15)}, 1, "fixture")
