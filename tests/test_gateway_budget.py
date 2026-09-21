import copy
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


def test_authorized_cap_change_preserves_charges_unknown_costs_and_history(tmp_path):
    path = tmp_path / "spend.json"
    ledger = BudgetLedger(path)
    ledger.reserve("paid", ".20")
    ledger.settle("paid", ".10")
    ledger.reserve("unknown", ".30")
    entries = copy.deepcopy(ledger.data["entries"])
    ledger.set_cap("5.00", reason="User explicitly raised cumulative cap to USD 5")
    ledger.set_cap("5.00", reason="User repeated the same cumulative limit")
    assert ledger.data["entries"] == entries
    assert ledger.committed == Decimal(".40")
    assert len(ledger.data["cap_changes"]) == 1
    change = ledger.data["cap_changes"][0]
    assert change["previous_cap_usd"] == "0.50"
    assert change["new_cap_usd"] == "5.00"
    assert change["existing_requests"] == 2
    ledger.close()
    ledger = BudgetLedger(path)
    assert ledger.data["cap_usd"] == "5.00"
    assert ledger.summary()["unresolved"] == 1
    ledger.reserve("remaining", "4.60")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("over", ".0001")
    ledger.close()
    with pytest.raises(ValueError, match="silently changed"):
        BudgetLedger(path, cap="9")
    # Failed construction must release the process lock.
    reopened = BudgetLedger(path)
    assert reopened.committed == Decimal("5.00")
    reopened.close()


def test_cap_change_rejects_invalid_values_and_cannot_erase_reservations(tmp_path):
    ledger = BudgetLedger(tmp_path / "spend.json")
    ledger.reserve("unknown", ".30")
    for cap in ["NaN", "Infinity", "-1", "0"]:
        with pytest.raises(ValueError):
            ledger.set_cap(cap, reason="test")
    with pytest.raises(ValueError):
        ledger.set_cap("5", reason="")
    with pytest.raises(BudgetExceeded):
        ledger.set_cap(".20", reason="test")
    assert ledger.data["cap_usd"] == "0.50"
    assert "cap_changes" not in ledger.data
    ledger.close()
