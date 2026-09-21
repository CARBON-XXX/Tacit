import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from gateway_budget import BudgetExceeded, BudgetLedger  # noqa: E402


def test_budget_survives_restart_and_unknown_request_cost(tmp_path):
    path = tmp_path / "spend.json"
    ledger = BudgetLedger(path)
    ledger.reserve("completed", ".30")
    ledger.settle("completed", ".10")
    ledger.reserve("timed-out", ".35")
    ledger.close()
    ledger = BudgetLedger(path)
    assert ledger.committed == Decimal(".45")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("would-overrun", ".051")
    ledger.reserve("remaining", ".05")
    assert ledger.committed == Decimal(".50")
    ledger.close()


def test_no_duplicate_billing_or_invalid_provider_cost(tmp_path):
    ledger = BudgetLedger(tmp_path / "spend.json")
    ledger.reserve("a", ".01")
    with pytest.raises(ValueError):
        ledger.reserve("a", ".01")
    for cost in ["-1", "NaN", "Infinity"]:
        with pytest.raises(ValueError):
            ledger.settle("a", cost)
    with pytest.raises(BudgetExceeded):
        ledger.settle("a", ".02")
    assert ledger.summary()["unresolved"] == 1
    ledger.settle("a", ".001", generation_id="test")
    assert ledger.summary()["reported_cost_usd"] == "0.001"
    with pytest.raises(ValueError):
        ledger.settle("a", ".001")
    ledger.close()
