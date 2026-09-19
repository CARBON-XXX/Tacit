"""Properties the collapse operator has to hold, whatever else changes."""

from __future__ import annotations

import torch

from tacit import CollapseSchedule, CyclicRegister, PhaseCollapse


def make_gate(**kwargs) -> PhaseCollapse:
    torch.manual_seed(0)
    return PhaseCollapse(32, 4, 4, n_phases=8, cold_start=False, **kwargs)


def test_codebook_stays_on_the_unit_circle():
    """Training and inference read the same normalised codebook, or they diverge."""
    gate = make_gate()
    with torch.no_grad():
        gate.codebook_real.mul_(3.7)
        gate.codebook_imag.add_(1.2)
    real, imag = gate._unit_codebook()
    assert torch.allclose((real**2 + imag**2).sqrt(), torch.ones_like(real), atol=1e-4)


def test_collapse_preserves_magnitude_when_fully_routed():
    gate = make_gate().eval()
    with torch.no_grad():
        gate.route_proj.bias.fill_(20.0)  # sigmoid saturates, so r == 1
        z_real, z_imag = torch.randn(2, 6, 4, 4), torch.randn(2, 6, 4, 4)
        out_real, out_imag = gate(z_real, z_imag, torch.randn(2, 6, 32))

    before = (z_real**2 + z_imag**2).sqrt()
    after = (out_real**2 + out_imag**2).sqrt()
    assert torch.allclose(before, after, atol=1e-3)


def test_zero_routing_is_the_identity():
    gate = make_gate().eval()
    with torch.no_grad():
        gate.route_proj.bias.fill_(-20.0)
        z_real, z_imag = torch.randn(2, 6, 4, 4), torch.randn(2, 6, 4, 4)
        out_real, out_imag = gate(z_real, z_imag, torch.randn(2, 6, 32))

    assert torch.allclose(out_real, z_real, atol=1e-4)
    assert torch.allclose(out_imag, z_imag, atol=1e-4)


def test_eval_selection_is_deterministic():
    gate = make_gate().eval()
    z_real, z_imag, cond = torch.randn(2, 6, 4, 4), torch.randn(2, 6, 4, 4), torch.randn(2, 6, 32)
    with torch.no_grad():
        first = gate(z_real, z_imag, cond)
        second = gate(z_real, z_imag, cond)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_gradients_reach_the_codebook_through_the_discrete_choice():
    gate = make_gate().train()
    out_real, out_imag = gate(
        torch.randn(2, 6, 4, 4), torch.randn(2, 6, 4, 4), torch.randn(2, 6, 32)
    )
    (out_real.sum() + out_imag.sum()).backward()
    assert gate.codebook_real.grad is not None
    assert gate.codebook_real.grad.abs().sum() > 0
    assert gate.symbol_proj.weight.grad.abs().sum() > 0


def test_schedule_lowers_temperature_to_the_floor():
    torch.manual_seed(0)
    model = torch.nn.Sequential(make_gate(), make_gate())
    schedule = CollapseSchedule(model, tau_init=1.0, tau_min=0.05, steps=100)
    assert len(schedule) == 2
    assert schedule.step(0) == 1.0
    assert abs(schedule.step(100) - 0.05) < 1e-6
    assert abs(schedule.step(500) - 0.05) < 1e-6  # clamped past the end


def test_register_is_the_identity_at_initialisation():
    """Zero-init output means the register can be switched on over a checkpoint."""
    torch.manual_seed(0)
    register = CyclicRegister(32, n_states=10)
    x = torch.randn(2, 8, 32)
    assert torch.allclose(register(x), x, atol=1e-6)


def test_register_counts_exactly_with_hard_shifts():
    """A one-hot shift is an exact cyclic permutation, so counting is exact."""
    torch.manual_seed(0)
    register = CyclicRegister(8, n_states=6)
    state = register.init_state(1, "cpu")

    # Drive the register by hand: advance by one group element each step.
    shift = torch.zeros(1, 6)
    shift[0, 1] = 1.0
    for expected in range(1, 7):
        state = register._advance(state, shift)
        assert int(state.argmax(-1)) == expected % 6
        assert abs(float(state.max()) - 1.0) < 1e-6


def test_register_streaming_matches_the_scan():
    torch.manual_seed(0)
    register = CyclicRegister(16, n_states=5, gate_init=0.5).eval()
    x = torch.randn(1, 7, 16)

    with torch.no_grad():
        parallel = register(x)
        state = register.init_state(1, x.device)
        outputs = []
        for t in range(x.shape[1]):
            out, state = register.step(x[:, t : t + 1], state)
            outputs.append(out)

    assert torch.allclose(parallel, torch.cat(outputs, dim=1), atol=1e-5)
