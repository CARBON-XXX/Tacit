"""Streaming boundaries, decision semantics and calibration invariants."""

import pytest
import torch

from tacit import (
    Boolean,
    ByteEncoder,
    Choice,
    ConformalPolicy,
    EncoderConfig,
    ResonantBlock,
    Score,
    SignalConfig,
    SignalTacit,
    Tacit,
    TacitConfig,
    TemperatureScaler,
    decision_loss,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"collapse_every": 0},
        {"collapse_every": 1},
        {"conv_kernel": 0},
        {"conv_kernel": 1},
        {"phase_selective": False},
        {"mag_selective": True},
        {"selective_readout": False},
    ],
)
def test_resumable_chunks_match_steps(kwargs):
    torch.manual_seed(3)
    block = ResonantBlock(32, **{"collapse_every": 8, **kwargs}).eval()
    x = torch.randn(2, 53, 32)
    state = block.init_state(2, "cpu")
    chunked, pos = [], 0
    with torch.no_grad():
        for size in [1, 3, 8, 2, 19, 20]:
            out, state = block.absorb(x[:, pos : pos + size], state, pos)
            chunked.append(out)
            pos += size
        reference_state = block.init_state(2, "cpu")
        reference = []
        for i in range(53):
            out, reference_state = block.step(x[:, i : i + 1], reference_state, i)
            reference.append(out)
    torch.testing.assert_close(torch.cat(chunked, 1), torch.cat(reference, 1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state[0], reference_state[0], atol=3e-5, rtol=3e-5)
    if state[1] is not None:
        torch.testing.assert_close(state[1], reference_state[1])


@pytest.mark.parametrize("register", [0, 5])
def test_encoder_arbitrary_observations_and_future(register):
    torch.manual_seed(7)
    model = ByteEncoder(EncoderConfig(d_model=32, n_layers=2, register_states=register)).eval()
    tokens = torch.randint(0, 256, (2, 71))
    with torch.no_grad():
        state = model.init_state(2, "cpu")
        for start, end in [(0, 3), (3, 21), (21, 37), (37, 71)]:
            out, state = model.absorb(tokens[:, start:end], state)
            torch.testing.assert_close(out, model(tokens[:, :end])[:, -1], atol=2e-5, rtol=2e-5)
    assert state.pos == 71


def test_batched_heads_gradients_order_and_bounded_cache():
    torch.manual_seed(0)
    model = Tacit(
        TacitConfig(
            encoder=EncoderConfig(d_model=32, n_layers=1),
            question_cache_size=2,
            question_batch_size=3,
        )
    )
    questions = {
        "a": Boolean("active?"),
        "b": Choice("route?", ["x", "y", "z"]),
        "c": Score("level?", ["low", "high"]),
    }
    state = torch.randn(2, 32, requires_grad=True)
    model.eval()
    batch = model.score_many(state, questions, calibrated=False)
    serial = {k: model.score(state, q, calibrated=False) for k, q in questions.items()}
    for k in questions:
        torch.testing.assert_close(batch[k], serial[k], atol=1e-6, rtol=1e-5)
    assert len(model._cache) == 2
    reordered = model.score_many(state, {"b": Choice("route?", ["z", "x", "y"])})
    torch.testing.assert_close(reordered["b"], batch["b"][:, [2, 0, 1]])
    model.train()
    scores = model.score_many(state, questions, calibrated=False)
    sum(v.square().mean() for v in scores.values()).backward()
    assert torch.isfinite(model.encoder.embed.weight.grad).all()
    assert model.encoder.embed.weight.grad.abs().sum() > 0
    model.eval()
    model.score_many(state.detach(), questions)
    model.load_state_dict(model.state_dict())
    assert not model._cache and not model._head_cache
    model.score_many(state.detach(), questions)
    model.to("cpu")
    assert not model._cache and not model._head_cache


def test_losses_accept_soft_targets_and_respect_ordinal_distance():
    logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
    hard = torch.tensor([1])
    soft = torch.tensor([[0.0, 1.0, 0.0]])
    torch.testing.assert_close(decision_loss(logits, hard), decision_loss(logits, soft))
    decision_loss(logits, soft, brier_weight=0.5, ordinal_weight=1).backward()
    assert torch.isfinite(logits.grad).all()
    target = torch.tensor([0])
    near = torch.tensor([[0.0, 3.0, -3.0]])
    far = torch.tensor([[0.0, -3.0, 3.0]])
    assert decision_loss(near, target, ordinal_weight=1) < decision_loss(
        far, target, ordinal_weight=1
    )
    with pytest.raises(ValueError):
        decision_loss(logits, torch.tensor([[0.8, 0.8, 0.8]]))


def test_temperature_improves_validation_nll_and_preserves_decisions():
    raw = torch.tensor([[10.0, -10.0]]).repeat(100, 1)
    y = torch.tensor([0] * 80 + [1] * 20)
    scaler = TemperatureScaler()
    scaler.fit(raw, y)
    assert decision_loss(scaler(raw), y) < decision_loss(raw, y)
    assert torch.equal(raw.argmax(-1), scaler(raw).argmax(-1))


def test_prediction_sets_reject_ambiguous_empty_and_small_calibration():
    probs = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    policy = ConformalPolicy.fit(["yes", "no"], probs, torch.zeros(4, dtype=torch.long), alpha=0.4)
    predictions = policy.predict(torch.tensor([[0.9, 0.1], [0.5, 0.5]]))
    assert predictions[0].automated and predictions[0].value == "yes"
    assert not predictions[1].automated and predictions[1].value is None
    small = ConformalPolicy.fit(["yes", "no"], probs[:1], torch.tensor([0]), alpha=0.01)
    assert small.predict(probs)[0].candidates == ("yes", "no")
    with pytest.raises(ValueError):
        small.predict(torch.tensor([[1.0, 1.0]]))


def test_signal_memory_updates_zero_and_ignores_missing_nan():
    torch.manual_seed(2)
    model = SignalTacit(SignalConfig(("a", "b")), {"go": Boolean("go?")}).eval()
    values = torch.tensor([[[1.0, float("nan")], [0.0, 9.0], [2.0, 0.0]]])
    mask = torch.tensor([[[True, False], [True, False], [False, True]]])
    with torch.no_grad():
        full, state = model(values, mask)
        previous = None
        outputs = []
        for i in range(3):
            out, previous = model(values[:, i : i + 1], mask[:, i : i + 1], previous)
            outputs.append(out["go"])
    torch.testing.assert_close(torch.cat(outputs, 1), full["go"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state.values, torch.zeros(1, 2))
    assert state.known.all() and state.pos == 3
    assert state.nbytes == model.init_state().nbytes
    for pair in state.blocks:
        for tensor in pair:
            if tensor is not None:
                assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    with pytest.raises(ValueError, match="finite"):
        model(values)


def test_signal_training_and_chunk_boundaries():
    torch.manual_seed(5)
    questions = {
        "route": Choice("route", ["a", "b", "c"]),
        "level": Score("level", ["low", "high"]),
    }
    model = SignalTacit(SignalConfig(("x", "y")), questions).eval()
    values = torch.randn(3, 21, 2)
    mask = torch.rand(3, 21, 2) > 0.6
    logits, state = model(values, mask)
    for name in questions:
        decision_loss(logits[name], torch.zeros((3, 21), dtype=torch.long)).backward(
            retain_graph=True
        )
    assert torch.isfinite(model.project.weight.grad).all()
    with torch.no_grad():
        _, carried = model(values[:, :5], mask[:, :5])
        answers, carried = model.evaluate(values[:, 5:], mask[:, 5:], carried)
        expected, _ = model.evaluate(values, mask)
    for actual, reference in zip(answers, expected, strict=True):
        for name in questions:
            assert actual[name].probabilities == pytest.approx(
                reference[name].probabilities, abs=1e-5
            )
    assert carried.nbytes == state.nbytes
