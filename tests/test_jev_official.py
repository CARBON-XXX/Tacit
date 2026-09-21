import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

pytest.importorskip("requests")
pytest.importorskip("pyarrow")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from gateway_budget import BudgetExceeded, BudgetLedger  # noqa: E402
from jev_official import (  # noqa: E402
    MODEL,
    RESERVATION,
    answer_records,
    collect_parallel,
    evaluate_cases,
    token_cost,
)


def case():
    return {
        "id": "case",
        "workflow": "test",
        "state": "public",
        "questions": {"ok": {"type": "noul", "instructions": "Is this public?"}},
        "gold": {"ok": {"label": True, "probabilities": {"false": 0.0, "true": 1.0}}},
    }


def response():
    return {
        "model": MODEL,
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "answers": {"ok": {"type": "noul", "noul": 0.9}},
    }


class FakeSession:
    def __init__(self, data=None, status=200):
        self.headers, self.calls, self.data, self.status = {}, [], data or response(), status

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        data, status = self.data, self.status

        class Response:
            status_code = status
            text = "mock provider failure"

            def json(self):
                return data

        return Response()

    def close(self):
        pass


def test_resume_never_rebills_and_spending_cap_prevents_http(tmp_path):
    (tmp_path / "typesafe.key").write_text("test-only-placeholder")
    output = tmp_path / "result.json"
    s = FakeSession()
    evaluate_cases([case()], output, tmp_path, s)
    assert len(s.calls) == 1
    assert s.calls[0][1]["allow_redirects"] is False
    evaluate_cases([case()], output, tmp_path, s)
    assert len(s.calls) == 1
    ledger = BudgetLedger(tmp_path / "jev-budget.json")
    ledger.reserve("other-provider", Decimal(".50") - ledger.committed)
    ledger.close()
    second = dict(case(), id="second", state="new public input")
    with pytest.raises(BudgetExceeded):
        evaluate_cases([case(), second], output, tmp_path, s)
    assert len(s.calls) == 1


def test_error_retains_reservation_and_never_retries(tmp_path):
    (tmp_path / "typesafe.key").write_text("test-only-placeholder")
    s = FakeSession(status=529)
    with pytest.raises(RuntimeError, match="529"):
        evaluate_cases([case()], tmp_path / "result.json", tmp_path, s)
    ledger = BudgetLedger(tmp_path / "jev-budget.json")
    assert ledger.committed == RESERVATION
    ledger.close()
    assert len(s.calls) == 1
    assert json.loads((tmp_path / "result.error.json").read_text())["status"] == 529


def test_billable_usage_and_probabilities_fail_closed():
    assert token_cost(response()) == Decimal(".0000042")
    for bad in [None, -1, 64001, True, "100"]:
        r = response()
        r["usage"]["input_tokens"] = bad
        with pytest.raises(ValueError):
            token_cost(r)
    r = response()
    r["model"] = "other-version"
    with pytest.raises(ValueError):
        token_cost(r)
    for value in [float("nan"), -0.1, 1.1]:
        r = response()
        r["answers"]["ok"]["noul"] = value
        with pytest.raises(ValueError):
            answer_records(case(), r)


def test_decimal_rounding_normalized_but_invalid_mass_rejected():
    c = case()
    c["questions"]["ok"] = {
        "type": "choice",
        "instructions": "Pick",
        "criteria": {"a": "", "b": "", "c": ""},
    }
    c["gold"]["ok"] = {"label": "b", "probabilities": {"a": 0.0, "b": 1.0, "c": 0.0}}
    r = response()
    r["answers"]["ok"] = {"type": "choice", "probabilities": {"a": 0.04, "b": 0.93, "c": 0.02}}
    p = answer_records(c, r)[0]["probabilities"]
    assert sum(p) == pytest.approx(1.0)
    assert p[1] == pytest.approx(0.93 / 0.99)
    assert r["answers"]["ok"]["probabilities"]["b"] == 0.93
    r["answers"]["ok"]["probabilities"]["b"] = 0.7
    with pytest.raises(ValueError):
        answer_records(c, r)


def test_parallel_waves_reserve_before_sending_and_reuse_responses(tmp_path):
    (tmp_path / "typesafe.key").write_text("test-only-placeholder")
    cases = [dict(case(), id=str(i), state=str(i)) for i in range(5)]
    s = FakeSession()
    output = tmp_path / "parallel.json"
    collect_parallel(cases, output, 2, tmp_path, lambda: s)
    assert len(s.calls) == 5
    collect_parallel(cases, output, 2, tmp_path, lambda: s)
    evaluate_cases(cases, output, tmp_path, s)
    assert len(s.calls) == 5
    ledger = BudgetLedger(tmp_path / "jev-budget.json")
    assert ledger.committed == 5 * token_cost(response())
    ledger.close()


def test_parallel_failure_finishes_current_wave_and_stops(tmp_path):
    (tmp_path / "typesafe.key").write_text("test-only-placeholder")
    cases = [dict(case(), id=str(i), state=str(i)) for i in range(5)]
    s = FakeSession(status=529)
    with pytest.raises(RuntimeError):
        collect_parallel(cases, tmp_path / "parallel.json", 2, tmp_path, lambda: s)
    assert len(s.calls) == 2
    ledger = BudgetLedger(tmp_path / "jev-budget.json")
    assert ledger.committed == 2 * RESERVATION
    assert ledger.summary()["unresolved"] == 2
    ledger.close()


def test_explicit_recovery_retains_old_cost_bound_and_does_not_rebill_success(tmp_path):
    (tmp_path / "typesafe.key").write_text("test-only-placeholder")
    output = tmp_path / "result.json"
    failed = FakeSession(status=529)
    with pytest.raises(RuntimeError):
        collect_parallel([case()], output, 1, tmp_path, lambda: failed)
    success = FakeSession()
    with pytest.raises(ValueError):
        collect_parallel([case()], output, 1, tmp_path, lambda: success)
    assert not success.calls
    collect_parallel([case()], output, 1, tmp_path, lambda: success, retry_unresolved=True)
    evaluate_cases([case()], output, tmp_path, success)
    collect_parallel([case()], output, 1, tmp_path, lambda: success)
    assert len(success.calls) == 1
    ledger = BudgetLedger(tmp_path / "jev-budget.json")
    assert ledger.summary()["unresolved"] == 1
    assert ledger.committed == RESERVATION + token_cost(response())
    ledger.close()
