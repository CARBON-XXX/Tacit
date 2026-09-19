r"""Phase collapse — a non-linear, discrete map on a complex state.

The operator
============
A complex state ``z = m * exp(i*theta)`` is snapped so that its *phase* lands on
a learned codebook of ``K`` points on the unit circle, while its *magnitude* is
left untouched::

    z_out = r * (|z| * code) + (1 - r) * z

``code`` is the selected unit-circle symbol and ``r in (0, 1)`` is a learned
routing confidence: near 0 the state flows through continuously, near 1 it is
fully snapped. Both ``code`` and ``r`` are predicted from the layer input, so
the map is input-conditioned.

Why it matters
==============
The phase map introduces a non-affine operation into the recurrence. Whether
this improves a particular task is an empirical question; nonlinearity alone
does not establish a computational complexity separation.

Training vs inference
=====================
Training selects a symbol with a straight-through Gumbel-Softmax so gradients
reach the codebook; inference takes a plain argmax. Keeping both paths on the
*same* normalised codebook matters: if the two diverge, a model can score well
teacher-forced and then fall apart when run on its own state.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PhaseCollapse", "CollapseSchedule"]


class PhaseCollapse(nn.Module):
    """Snap the phase of a complex state onto a learned unit-circle codebook.

    Args:
        d_in: width of the conditioning input used to predict routing and symbols.
        n_bands: number of frequency bands; the state is ``[..., n_bands, db]``.
        db: complex dimensions per band.
        n_phases: codebook size ``K``. 16 gives a 4-bit symbol alphabet.
        tau_init: initial Gumbel-Softmax temperature.
        tau_min: temperature floor, also the floor of the predicted temperature.
        per_band_gate: predict one routing confidence per band rather than one
            per position.
        cold_start: initialise routing near zero so the model begins fully
            continuous and has to learn where snapping pays off.
        symbol_rank: factorise the symbol-logit projection through a bottleneck
            of this width. 0 keeps the dense projection, which costs
            ``d_in * n_bands * db * K`` parameters and dominates the module.
        fp32_phase: run the angle-sensitive part in fp32 even when the
            surrounding model is bf16. Near the unit circle bf16 resolves angles
            only to about 1/128 rad, which is coarse enough to blur symbols.
    """

    def __init__(
        self,
        d_in: int,
        n_bands: int,
        db: int,
        *,
        n_phases: int = 16,
        tau_init: float = 1.0,
        tau_min: float = 0.05,
        per_band_gate: bool = True,
        cold_start: bool = True,
        symbol_rank: int = 0,
        fp32_phase: bool = True,
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.n_bands = int(n_bands)
        self.db = int(db)
        self.n_phases = int(n_phases)
        self.tau_min = float(tau_min)
        self.per_band_gate = bool(per_band_gate)
        self.fp32_phase = bool(fp32_phase)

        angles = torch.linspace(0.0, 2 * math.pi * (1 - 1 / self.n_phases), self.n_phases)
        init_r = angles.cos().view(1, 1, -1).expand(self.n_bands, self.db, self.n_phases)
        init_i = angles.sin().view(1, 1, -1).expand(self.n_bands, self.db, self.n_phases)
        self.codebook_real = nn.Parameter(init_r.clone())
        self.codebook_imag = nn.Parameter(init_i.clone())

        gate_out = self.n_bands if per_band_gate else 1
        self.route_proj = nn.Linear(self.d_in, gate_out)
        self.temp_proj = nn.Linear(self.d_in, gate_out)
        if cold_start:
            nn.init.zeros_(self.route_proj.weight)
            nn.init.constant_(self.route_proj.bias, -4.0)  # sigmoid(-4) ~ 0.018
            nn.init.zeros_(self.temp_proj.weight)
            nn.init.zeros_(self.temp_proj.bias)  # softplus(0) = ln 2
        else:
            nn.init.xavier_uniform_(self.route_proj.weight)
            nn.init.zeros_(self.route_proj.bias)
            nn.init.xavier_uniform_(self.temp_proj.weight)
            nn.init.zeros_(self.temp_proj.bias)

        out_dim = self.n_bands * self.db * self.n_phases
        self.symbol_rank = int(symbol_rank)
        if self.symbol_rank > 0:
            self.symbol_proj: nn.Module = nn.Sequential(
                nn.Linear(self.d_in, self.symbol_rank, bias=False),
                nn.Linear(self.symbol_rank, out_dim),
            )
            nn.init.xavier_uniform_(self.symbol_proj[0].weight, gain=1.0)
            nn.init.xavier_uniform_(self.symbol_proj[1].weight, gain=0.1)
            nn.init.zeros_(self.symbol_proj[1].bias)
        else:
            self.symbol_proj = nn.Linear(self.d_in, out_dim)
            nn.init.xavier_uniform_(self.symbol_proj.weight, gain=0.1)
            nn.init.zeros_(self.symbol_proj.bias)

        self.register_buffer("tau", torch.tensor(float(tau_init)))
        self.register_buffer("_route_mean", torch.zeros(()), persistent=False)
        self.register_buffer("_symbol_usage", torch.zeros(self.n_phases), persistent=False)
        self._last_symbol_probs: torch.Tensor | None = None

    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, n_bands={self.n_bands}, db={self.db}, "
            f"n_phases={self.n_phases}"
        )

    def set_temperature(self, tau: float) -> None:
        self.tau.fill_(max(float(tau), self.tau_min))

    @property
    def route_mean(self) -> float:
        """Mean routing confidence from the last forward pass."""
        return float(self._route_mean)

    @property
    def symbol_usage(self) -> torch.Tensor:
        """How often each of the ``K`` symbols was selected, last forward pass."""
        return self._symbol_usage.detach().clone()

    @property
    def symbol_perplexity(self) -> float:
        """Effective alphabet size: 1 means the codebook collapsed, K means even use."""
        u = self._symbol_usage.clamp_min(1e-8)
        return float((-(u * u.log()).sum()).exp())

    def usage_penalty(self) -> torch.Tensor:
        """KL(symbol usage || uniform). Add to the loss to keep symbols alive.

        Left out of the module's own output on purpose: whether a forming
        alphabet should be pushed toward uniform depends on the task.
        """
        probs = self._last_symbol_probs
        if probs is None:
            usage = self._symbol_usage.clamp_min(1e-8)
        else:
            usage = probs.mean(dim=tuple(range(probs.ndim - 1))).clamp_min(1e-8)
        return (usage * (usage * self.n_phases).log()).sum()

    def _unit_codebook(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the codebook back onto the unit circle.

        Training and inference must both read this normalised view. Snapping
        against raw parameters in one path and normalised ones in the other
        silently changes the operator between train and eval.
        """
        norm = (self.codebook_real.square() + self.codebook_imag.square() + 1e-8).sqrt()
        return self.codebook_real / norm, self.codebook_imag / norm

    def _select(
        self, logits: torch.Tensor, tau: torch.Tensor, hard: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick a codebook symbol per (band, dim). Returns real and imag parts.

        ``logits`` is ``[..., n_bands, db, K]`` and ``tau`` broadcasts against it.
        """
        code_r, code_i = self._unit_codebook()
        lead = logits.ndim - 3  # batch-like dims in front of (n_bands, db, K)
        shape = (1,) * lead + code_r.shape

        if hard:
            idx = logits.argmax(dim=-1)
            self._last_symbol_probs = None
            with torch.no_grad():
                onehot = F.one_hot(idx, self.n_phases).to(logits.dtype)
                self._symbol_usage.copy_(onehot.mean(dim=tuple(range(onehot.ndim - 1))))
            gather = idx.unsqueeze(-1)
            out_r = code_r.view(shape).expand(logits.shape).gather(-1, gather).squeeze(-1)
            out_i = code_i.view(shape).expand(logits.shape).gather(-1, gather).squeeze(-1)
            return out_r, out_i

        soft = (logits / tau).softmax(dim=-1)
        self._last_symbol_probs = soft
        # Straight-through: the forward symbol is discrete, the gradient is not.
        noise = -torch.empty_like(logits).exponential_().log()
        y = ((logits + noise) / tau).softmax(dim=-1)
        index = y.argmax(dim=-1, keepdim=True)
        weights = torch.zeros_like(y).scatter_(-1, index, 1.0) - y.detach() + y
        with torch.no_grad():
            self._symbol_usage.copy_(weights.mean(dim=tuple(range(weights.ndim - 1))))
        out_r = (weights * code_r.view(shape)).sum(-1)
        out_i = (weights * code_i.view(shape)).sum(-1)
        return out_r, out_i

    def route_and_code(
        self, cond: torch.Tensor, *, hard: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Routing confidence and codebook symbol, without touching a state.

        Splitting this out lets a scan precompute every boundary's symbol in
        parallel and then apply the cheap state-dependent part serially.

        Args:
            cond: ``[B, T, d_in]`` conditioning input.
            hard: force discrete selection. Defaults to ``not self.training``.

        Returns:
            ``(r, code_r, code_i)`` where ``r`` is ``[B, T, n_bands, 1]`` and the
            codes are ``[B, T, n_bands, db]``.
        """
        if hard is None:
            hard = not self.training
        b, t, _ = cond.shape
        host_dtype = cond.dtype

        logits = self.symbol_proj(cond).view(b, t, self.n_bands, self.db, self.n_phases)
        r = torch.sigmoid(self.route_proj(cond))
        r = r.unsqueeze(-1) if self.per_band_gate else r.unsqueeze(-1).unsqueeze(-1)

        tau = F.softplus(self.temp_proj(cond)) + self.tau_min
        tau = tau.unsqueeze(-1).unsqueeze(-1) if self.per_band_gate else tau[..., None, None, None]

        if self.fp32_phase and host_dtype != torch.float32:
            logits, tau = logits.float(), tau.float()

        with torch.no_grad():
            self._route_mean.copy_(r.detach().float().mean())

        code_r, code_i = self._select(logits, tau, hard)
        return r, code_r.to(host_dtype), code_i.to(host_dtype)

    def forward(
        self,
        z_real: torch.Tensor,
        z_imag: torch.Tensor,
        cond: torch.Tensor,
        *,
        hard: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse a complex state.

        Args:
            z_real, z_imag: ``[B, T, n_bands, db]`` state to snap.
            cond: ``[B, T, d_in]`` conditioning input.

        Returns:
            The snapped ``(z_real, z_imag)``, same shape as the input.
        """
        r, code_r, code_i = self.route_and_code(cond, hard=hard)
        mag = (z_real * z_real + z_imag * z_imag + 1e-8).sqrt()
        out_r = r * (mag * code_r) + (1 - r) * z_real
        out_i = r * (mag * code_i) + (1 - r) * z_imag
        return out_r, out_i


class CollapseSchedule:
    """Anneal every collapse gate in a model from ``tau_init`` down to ``tau_min``.

    A high temperature early keeps symbol assignment soft while the codebook is
    still moving; annealing down hands the model a genuinely discrete alphabet
    by the end of training.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        tau_init: float = 1.0,
        tau_min: float = 0.05,
        steps: int = 6000,
    ) -> None:
        self.gates = [m for m in model.modules() if isinstance(m, PhaseCollapse)]
        self.tau_init = float(tau_init)
        self.tau_min = float(tau_min)
        self.steps = max(1, int(steps))

    def __len__(self) -> int:
        return len(self.gates)

    def step(self, global_step: int) -> float:
        frac = min(global_step / self.steps, 1.0)
        tau = self.tau_init + (self.tau_min - self.tau_init) * frac
        for gate in self.gates:
            gate.set_temperature(tau)
        return tau
