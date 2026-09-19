"""The decision interface: typed in, typed out, nothing to parse."""

from __future__ import annotations

import pytest
import torch

from tacit import (
    Boolean,
    BooleanAnswer,
    Choice,
    ChoiceAnswer,
    EncoderConfig,
    Score,
    ScoreAnswer,
    Tacit,
    TacitConfig,
    expected_calibration_error,
    fit_temperature,
)

QUESTIONS = {
    "urgent": Boolean("Is this urgent?", when_true="blocked now", when_false="no rush"),
    "team": Choice("Which team?", {"billing": "refunds", "technical": "outages"}),
    "anger": Score("How frustrated?", ["calm", "annoyed", "furious"]),
}


@pytest.fixture(scope="module")
def model() -> Tacit:
    torch.manual_seed(0)
    return Tacit(TacitConfig(encoder=EncoderConfig(d_model=64, n_layers=2))).eval()


def test_answers_are_typed_and_normalised(model):
    answers = model.evaluate("payouts failing for three days", QUESTIONS)

    assert isinstance(answers["urgent"], BooleanAnswer)
    assert isinstance(answers["team"], ChoiceAnswer)
    assert isinstance(answers["anger"], ScoreAnswer)

    for answer in answers.values():
        assert abs(sum(answer.probabilities.values()) - 1.0) < 1e-4
        assert all(0.0 <= p <= 1.0 for p in answer.probabilities.values())
        assert 0.0 <= answer.confidence <= 1.0


def test_answers_stay_inside_the_declared_space(model):
    """The answer cannot be a value the caller did not offer."""
    answers = model.evaluate("the dashboard is down", QUESTIONS)
    assert answers["team"].value in {"billing", "technical"}
    assert answers["anger"].level in {"calm", "annoyed", "furious"}
    assert 0.0 <= answers["anger"].value <= 2.0
    assert set(answers["urgent"].probabilities) == {"true", "false"}


def test_options_are_defined_at_call_time(model):
    """A question can introduce options the model has never seen."""
    fresh = Choice("Route this where?", ["procurement", "legal", "facilities", "security"])
    answer = model.evaluate("we need a vendor contract reviewed", {"route": fresh})["route"]
    assert answer.value in fresh.labels()
    assert len(answer.probabilities) == 4


def test_batched_evaluate_matches_one_at_a_time(model):
    texts = ["refund never arrived", "the API returns 500"]
    batched = model.evaluate(texts, QUESTIONS)
    assert isinstance(batched, list) and len(batched) == 2
    for text, row in zip(texts, batched, strict=True):
        single = model.evaluate(text, QUESTIONS)
        for name in QUESTIONS:
            assert row[name].probabilities.keys() == single[name].probabilities.keys()
            for label, value in row[name].probabilities.items():
                assert abs(value - single[name].probabilities[label]) < 1e-4


def test_session_state_moves_with_observations(model):
    session = model.session()
    session.observe("customer: the payout failed")
    first = session.state_vector().clone()
    session.observe("customer: it works now, thanks")
    assert not torch.allclose(first, session.state_vector(), atol=1e-6)
    assert session.bytes_seen == len("customer: the payout failed") + len(
        "customer: it works now, thanks"
    )


def test_session_reset_clears_history(model):
    session = model.session()
    session.observe("something happened")
    session.reset()
    assert session.bytes_seen == 0
    with pytest.raises(RuntimeError, match="before any observe"):
        session.ask({"urgent": QUESTIONS["urgent"]})


def test_boolean_answer_is_truthy_above_a_half():
    assert bool(BooleanAnswer(probability=0.9, probabilities={"true": 0.9, "false": 0.1}))
    assert not bool(BooleanAnswer(probability=0.2, probabilities={"true": 0.2, "false": 0.8}))


@pytest.mark.parametrize(
    "question, message",
    [
        (lambda: Choice("pick", ["only"]), "at least 2"),
        (lambda: Choice("pick", ["a", "a"]), "duplicate"),
        (lambda: Score("rate", ["single"]), "at least 2"),
        (lambda: Choice("pick", [f"o{i}" for i in range(300)]), "at most 255"),
    ],
)
def test_malformed_questions_are_rejected(question, message):
    with pytest.raises(ValueError, match=message):
        question()


def test_temperature_only_rescales_confidence(model):
    """Calibration must not change which candidate wins."""
    torch.manual_seed(0)
    logits = torch.randn(256, 3) * 3.0
    targets = logits.argmax(-1).clone()
    targets[::4] = (targets[::4] + 1) % 3  # make it imperfect, so a fit is meaningful

    before = logits.argmax(-1)
    temperature = fit_temperature(model.head, logits, targets)
    after = (logits / temperature).argmax(-1)

    assert temperature > 0
    assert torch.equal(before, after)
    model.head.set_temperature(1.0)


def test_temperature_fit_tolerates_masked_padding(model):
    """Ragged question widths are padded; that must not produce NaN."""
    torch.manual_seed(0)
    logits = torch.randn(128, 3)
    logits[:64, 2] = 0.0
    mask = torch.ones(128, 3, dtype=torch.bool)
    mask[:64, 2] = False
    targets = torch.randint(0, 2, (128,))

    temperature = fit_temperature(model.head, logits, targets, candidate_mask=mask)
    assert temperature == temperature  # not NaN
    assert temperature > 0
    model.head.set_temperature(1.0)


def test_calibration_error_is_zero_when_confidence_matches_accuracy():
    probs = torch.full((100,), 0.8)
    correct = torch.zeros(100, dtype=torch.bool)
    correct[:80] = True
    assert expected_calibration_error(probs, correct) < 1e-6


def test_cache_is_dropped_when_switching_to_training(model):
    model.eval()
    model.evaluate("warm the cache", {"urgent": QUESTIONS["urgent"]})
    assert model._cache
    model.train()
    assert not model._cache
    model.eval()
