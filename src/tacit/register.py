r"""Cyclic register — an exact, depth-invariant accumulator.

A soft recurrence is good at compression and bad at exact bookkeeping: per-step
accuracy in the high 60s compounds into nothing over a long chain, and whatever
it does learn stops working past the trained depth. This module is the exact
complement, and it is deliberately narrow.

Counting mod ``N`` is the cyclic group Z/N. So represent the running value as a
distribution over ``N`` states, have each position emit a *shift* distribution,
and apply it by circular convolution — a mixture of cyclic permutations, where
a one-hot shift is an exact permutation. The readout is fixed and positional
(``state[k]`` is P(running value == k)), which forbids hiding the value in a
distributed code and so forces each shift to become the correct permutation.

Because the same learned shift matrices reapply at every position, depth is not
a parameter: a register trained on short chains keeps working on long ones.

The output projection is zero-initialised, so the module is exactly the identity
at load and can be switched on over an existing checkpoint without disturbing it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CyclicRegister"]


def _gate_logit(gate_init: float) -> float:
    g = min(max(float(gate_init), 0.0), 0.999)
    return -8.0 if g <= 0.0 else math.log(g / (1.0 - g))


class CyclicRegister(nn.Module):
    """Track a running element of Z/N and inject it back into the hidden state.

    Args:
        d_model: hidden width.
        n_states: group order ``N``.
        gate_init: initial value of the injection gate, in ``[0, 1)``.
    """

    def __init__(self, d_model: int, *, n_states: int = 10, gate_init: float = 0.0) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_states = int(n_states)

        self.shift_proj = nn.Linear(self.d_model, self.n_states)
        self.state_to_h = nn.Linear(self.n_states, self.d_model)
        nn.init.zeros_(self.state_to_h.weight)
        nn.init.zeros_(self.state_to_h.bias)
        self.gate_logit = nn.Parameter(torch.tensor(_gate_logit(gate_init)))

        # roll_index[i, k] = (i - k) mod N, so a gather implements circular
        # convolution without building N rolled copies of the state.
        i = torch.arange(self.n_states).view(-1, 1)
        k = torch.arange(self.n_states).view(1, -1)
        self.register_buffer("roll_index", (i - k) % self.n_states, persistent=False)

        self._last_state: torch.Tensor | None = None

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_states={self.n_states}"

    def init_state(self, batch: int, device: torch.device | str) -> torch.Tensor:
        """Identity element: all mass on state 0."""
        s = torch.zeros(batch, self.n_states, device=device)
        s[:, 0] = 1.0
        return s

    def _advance(self, state: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """One group multiplication: circular convolution of state with shift."""
        gathered = state[:, self.roll_index]          # [B, N, N] -> gathered[:, i, k] = s[i-k]
        return (gathered * shift.unsqueeze(1)).sum(-1)

    def shifts(self, x: torch.Tensor, *, hard: bool = False) -> torch.Tensor:
        """Per-position shift distribution ``[B, T, N]``."""
        p = F.softmax(self.shift_proj(x), dim=-1)
        if hard:
            idx = p.argmax(-1, keepdim=True)
            p = torch.zeros_like(p).scatter_(-1, idx, 1.0)
        return p

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        hard: bool = False,
    ) -> torch.Tensor:
        """Scan the register over a sequence and inject the running state.

        Args:
            x: ``[B, T, d_model]``.
            mask: ``[B, T]`` of 1 for real positions, 0 to freeze the register
                (padding).
            hard: use a one-hot shift, making each step an exact permutation.

        Returns:
            ``[B, T, d_model]``, equal to ``x`` at initialisation.
        """
        b, t, _ = x.shape
        shift = self.shifts(x, hard=hard)
        if mask is None:
            valid = torch.ones(b, t, device=x.device, dtype=shift.dtype)
        else:
            valid = mask.to(shift.dtype)
            if valid.ndim == 3:
                valid = valid.squeeze(-1)

        state = self.init_state(b, x.device).to(shift.dtype)
        states = []
        for i in range(t):
            advanced = self._advance(state, shift[:, i])
            keep = valid[:, i].unsqueeze(-1)
            state = keep * advanced + (1.0 - keep) * state
            states.append(state)
        running = torch.stack(states, dim=1)                     # [B, T, N]

        self._last_state = running
        gate = torch.sigmoid(self.gate_logit).to(x.dtype)
        return x + gate * self.state_to_h(running).to(x.dtype)

    def step(self, x_t: torch.Tensor, state: torch.Tensor, *, hard: bool = False):
        """Advance one position. ``x_t`` is ``[B, 1, d_model]``."""
        shift = self.shifts(x_t, hard=hard)[:, 0]
        state = self._advance(state, shift)
        gate = torch.sigmoid(self.gate_logit).to(x_t.dtype)
        out = x_t + gate * self.state_to_h(state).unsqueeze(1).to(x_t.dtype)
        return out, state

    def state_log_probs(self) -> torch.Tensor | None:
        """Log distribution of the running value per position, from the last forward.

        Supervise this when you know the intermediate values and want the
        register to commit to exact permutations; never needed at inference.
        """
        return None if self._last_state is None else self._last_state.clamp_min(1e-9).log()
