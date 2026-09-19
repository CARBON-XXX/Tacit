r"""Resonant block — a complex recurrence punctuated by phase collapse.

The state is a complex wavefield ``z`` of shape ``[B, n_bands, db]`` whose real
and imaginary halves together span ``d_model``. It evolves as::

    z_t = Lambda * z_{t-1} + drive(x_t)          continuous resonance
    z_t = PhaseCollapse(z_t, x_t)                every `collapse_every` steps

``Lambda = rho * exp(i*Omega)`` is near-unitary: ``Omega`` holds per-(band, dim)
carrier frequencies and ``rho`` is bounded below by ``rho_floor``. Between
collapse points the recurrence is linear with a known multiplier, so a whole
chunk is solved in parallel; collapse then acts on the carried state at the
chunk boundary. Cost is ``O(T/K)`` sequential host steps instead of ``O(T)``,
and the state map stays genuinely non-linear.

Two entry points, one recurrence
================================
``forward`` runs the chunked parallel scan over a whole sequence.
``step`` advances a single token against a carried state in constant time and
memory, with no cache that grows with context. They are numerically equivalent;
``tests/test_parity.py`` pins that down, because a core whose decode path
quietly differs from its training path is the failure mode that costs weeks.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from tacit.collapse import PhaseCollapse

__all__ = ["ResonantBlock", "RMSNorm"]


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(self.eps).sqrt()
        return x / rms * self.weight


class ResonantBlock(nn.Module):
    """One layer of resonance plus periodic phase collapse.

    Args:
        d_model: model width. Must be divisible by ``2 * n_bands``.
        n_bands: number of independent resonator banks.
        collapse_every: chunk length ``K``, and the period at which collapse
            acts on the carried state. 0 disables collapse, leaving a linear
            oscillator bank — useful only as an ablation.
        n_phases: codebook size for collapse.
        rho_floor: lower bound on ``|Lambda|``. Also bounds the within-chunk
            prefix reciprocal at ``rho_floor ** -(K-1)``, which is what keeps
            the parallel scan accurate in complex64.
        phase_selective: let the input modulate the per-step phase rotation.
            This is selectivity in the phase domain, distinct from the
            magnitude-decay selectivity a linear SSM applies.
        mag_selective: let the input modulate ``|Lambda|``.
        selective_readout: read the state through a learned complex query,
            ``conj(q) * z``, rather than reading real and imaginary parts
            directly. An output gate can only rescale features; interference
            against a query is what gives content-addressed recall.
        output_gate: SiLU gate on the readout.
        conv_kernel: width of a causal depthwise conv applied before the
            recurrence. A cheap local-mixing prior, not part of the core. 0 off.
        dropout: dropout on the block output.
        n_layers: total depth, used to scale the output projection at init.
        collapse_cold_start: start with routing confidence near zero.
        symbol_rank: bottleneck width for the collapse symbol projection.
    """

    def __init__(
        self,
        d_model: int,
        *,
        n_bands: int = 4,
        collapse_every: int = 16,
        n_phases: int = 16,
        rho_floor: float = 0.9,
        phase_selective: bool = True,
        mag_selective: bool = False,
        selective_readout: bool = True,
        output_gate: bool = True,
        conv_kernel: int = 4,
        dropout: float = 0.0,
        n_layers: int = 1,
        collapse_cold_start: bool = False,
        symbol_rank: int = 0,
    ) -> None:
        super().__init__()
        if d_model % (2 * n_bands):
            raise ValueError(f"d_model={d_model} must be divisible by 2*n_bands={2 * n_bands}")

        self.d_model = int(d_model)
        self.n_bands = int(n_bands)
        self.db = self.d_model // (2 * self.n_bands)
        self.collapse_every = int(collapse_every)
        self.rho_floor = float(rho_floor)
        self.phase_selective = bool(phase_selective)
        self.mag_selective = bool(mag_selective)
        self.selective = self.phase_selective or self.mag_selective
        self.selective_readout = bool(selective_readout)
        self.output_gate = bool(output_gate)
        self.conv_kernel = int(conv_kernel)
        self.chunk = self.collapse_every if self.collapse_every > 0 else 32

        growth = self.rho_floor ** -(self.chunk - 1)
        if growth > 1e3:
            warnings.warn(
                f"rho_floor={self.rho_floor} with chunk={self.chunk} gives a within-chunk "
                f"prefix reciprocal of {growth:.3g}; the complex64 scan may lose precision. "
                f"Raise rho_floor or lower collapse_every.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.norm = RMSNorm(self.d_model)
        self.conv = (
            nn.Conv1d(self.d_model, self.d_model, self.conv_kernel,
                      groups=self.d_model, padding=0, bias=True)
            if self.conv_kernel > 0
            else None
        )

        self.drive_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.normal_(self.drive_proj.weight, std=0.02)
        if self.phase_selective:
            self.dphi_proj = self._sel_proj()
        if self.mag_selective:
            self.dmag_proj = self._sel_proj()
        if self.selective_readout:
            self.query_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        if self.output_gate:
            self.gate_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.normal_(self.out_proj.weight, std=(2.0 * max(n_layers, 1)) ** -0.5 * 0.05)
        self.dropout = nn.Dropout(dropout)

        # Lambda = rho * exp(i * Omega), intrinsic unless a selectivity head runs.
        rho0 = torch.full((self.n_bands, self.db), 0.99)
        u = ((rho0 - self.rho_floor) / (1.0 - self.rho_floor)).clamp(1e-4, 1 - 1e-4)
        self.rho_logit = nn.Parameter(torch.log(u / (1 - u)))
        base = torch.exp(torch.linspace(math.log(1e-2), math.log(math.pi), self.db))
        spread = torch.linspace(0.8, 1.2, self.n_bands).unsqueeze(-1)
        self.omega = nn.Parameter(base.unsqueeze(0) * spread)

        self.collapse = (
            PhaseCollapse(
                self.d_model,
                self.n_bands,
                self.db,
                n_phases=n_phases,
                cold_start=collapse_cold_start,
                symbol_rank=symbol_rank,
            )
            if self.collapse_every > 0
            else None
        )

    def _sel_proj(self) -> nn.Linear:
        proj = nn.Linear(self.d_model, self.n_bands * self.db, bias=False)
        nn.init.normal_(proj.weight, std=0.02)
        return proj

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_bands={self.n_bands}, db={self.db}, "
            f"collapse_every={self.collapse_every}, rho_floor={self.rho_floor}"
        )

    # -- pieces shared by the parallel and the streaming path ---------------

    def _rho(self, dmag: torch.Tensor | None = None) -> torch.Tensor:
        logit = self.rho_logit if dmag is None else self.rho_logit + dmag
        return self.rho_floor + (1.0 - self.rho_floor) * torch.sigmoid(logit)

    def _pre(self, x: torch.Tensor, conv_state: torch.Tensor | None):
        """Normalise then optionally convolve.

        Returns ``(h, normed)``. ``normed`` is what the conv ring buffer holds,
        so a caller can hand the tail of it to a later ``step``. When
        ``conv_state`` is given, only the final position is produced.
        """
        normed = self.norm(x)
        if self.conv is None:
            return normed, normed
        if conv_state is None:
            padded = F.pad(normed.transpose(1, 2), (self.conv_kernel - 1, 0))
            return F.silu(self.conv(padded).transpose(1, 2)), normed
        seq = torch.cat([conv_state, normed], dim=1)
        tip = self.conv(seq.transpose(1, 2))[..., -1:].transpose(1, 2)
        return F.silu(tip), seq

    def _drive(self, h: torch.Tensor) -> torch.Tensor:
        b, t, _ = h.shape
        real, imag = self.drive_proj(h).view(b, t, self.n_bands, 2 * self.db).chunk(2, dim=-1)
        return torch.complex(real.float(), imag.float())

    def _log_lambda(self) -> torch.Tensor:
        return torch.complex(self._rho().log().float(), self.omega.float())

    def _lam_selective(self, h: torch.Tensor) -> torch.Tensor:
        b, t, _ = h.shape
        shape = (b, t, self.n_bands, self.db)
        if self.mag_selective:
            rho = self._rho(self.dmag_proj(h).view(shape))
        else:
            rho = self._rho().view(1, 1, self.n_bands, self.db).expand(shape)
        phase = self.omega.view(1, 1, self.n_bands, self.db).expand(shape)
        if self.phase_selective:
            phase = phase + torch.tanh(self.dphi_proj(h).view(shape))
        return torch.polar(rho.float().contiguous(), phase.float().contiguous())

    def _readout(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Map the complex state back to ``d_model`` real features."""
        b, t = z.shape[0], z.shape[1]
        if self.selective_readout:
            qr, qi = self.query_proj(h).view(b, t, self.n_bands, 2 * self.db).chunk(2, dim=-1)
            feat = torch.conj(torch.complex(qr.float(), qi.float())) * z
        else:
            feat = z
        y = torch.cat([feat.real.reshape(b, t, -1), feat.imag.reshape(b, t, -1)], dim=-1)
        y = y.to(self.out_proj.weight.dtype)
        if self.output_gate:
            y = y * F.silu(self.gate_proj(h))
        return self.dropout(self.out_proj(y))

    def _collapse_state(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Collapse one carried state ``[B, n_bands, db]`` under ``cond`` ``[B, 1, d]``."""
        zr, zi = self.collapse(z.real.unsqueeze(1), z.imag.unsqueeze(1), cond)
        return torch.complex(zr[:, 0], zi[:, 0])

    # -- parallel path -------------------------------------------------------

    def forward(self, x: torch.Tensor, *, return_state: bool = False):
        """Run a whole sequence through the chunked parallel scan.

        Args:
            x: ``[B, T, d_model]``.
            return_state: also return a state that a later ``step`` can resume
                from. Requires ``T`` to be a multiple of ``collapse_every`` so
                that no chunk is padded.

        Returns:
            ``[B, T, d_model]``, or ``(output, state)`` when ``return_state``.
        """
        b, t, _ = x.shape
        nb, db, k = self.n_bands, self.db, self.chunk
        if return_state and t % k:
            raise ValueError(
                f"return_state needs T ({t}) to be a multiple of the chunk length ({k}); "
                f"otherwise the carried state includes padding."
            )

        h, normed = self._pre(x, conv_state=None)
        drive = self._drive(h)
        lam = self._lam_selective(h) if self.selective else None

        pad = (k - t % k) % k
        if pad:
            drive = F.pad(drive, (0, 0, 0, 0, 0, pad))
            if lam is not None:
                lam = F.pad(lam, (0, 0, 0, 0, 0, pad))
                lam[:, t:] = 1.0  # identity on padding
        n_chunk = (t + pad) // k
        drive = drive.view(b, n_chunk, k, nb, db)

        if lam is None:
            # One shared multiplier: its powers are the same in every chunk.
            steps = torch.arange(k, device=x.device, dtype=torch.float32).view(k, 1, 1)
            log_lam = self._log_lambda()
            prefix = torch.exp(steps * log_lam).view(1, 1, k, nb, db)       # Lambda ** j
            inverse = torch.exp(-steps * log_lam).view(1, 1, k, nb, db)     # Lambda ** -j
            carry_gain = torch.exp(float(k) * log_lam)                      # Lambda ** K
            carry_map = torch.exp(log_lam) * prefix                         # Lambda ** (j+1)
            within = prefix * torch.cumsum(drive * inverse, dim=2)
        else:
            lam = lam.view(b, n_chunk, k, nb, db)
            carry_map = torch.cumprod(lam, dim=2)                           # inclusive
            within = carry_map * torch.cumsum(drive / carry_map, dim=2)
            carry_gain = carry_map[:, :, -1]
        chunk_end = within[:, :, -1]

        cond = None
        if self.collapse is not None:
            last = torch.clamp(torch.arange(n_chunk, device=x.device) * k + (k - 1), max=t - 1)
            cond = h[:, last]                                               # [B, n_chunk, d_model]

        carry = torch.zeros(b, nb, db, dtype=drive.dtype, device=x.device)
        chunks = []
        for c in range(n_chunk):
            gain = carry_map[0, 0] if lam is None else carry_map[:, c]
            chunks.append(within[:, c] + gain * carry.unsqueeze(1))
            gain_end = carry_gain if lam is None else carry_gain[:, c]
            carry = gain_end * carry + chunk_end[:, c]
            if self.collapse is not None:
                carry = self._collapse_state(carry, cond[:, c : c + 1])

        z = torch.stack(chunks, dim=1).reshape(b, t + pad, nb, db)
        if pad:
            z = z[:, :t]
        out = x + self._readout(z, h)

        if return_state:
            buf = normed[:, -(self.conv_kernel - 1):] if self.conv is not None else None
            return out, (carry, buf)
        return out

    # -- streaming path ------------------------------------------------------

    def init_state(self, batch: int, device: torch.device | str):
        """Fresh carried state: ``(wavefield, conv ring buffer)``."""
        z = torch.zeros(batch, self.n_bands, self.db, dtype=torch.complex64, device=device)
        buf = (
            torch.zeros(batch, self.conv_kernel - 1, self.d_model, device=device)
            if self.conv is not None
            else None
        )
        return z, buf

    def step(self, x_t: torch.Tensor, state, pos: int):
        """Advance one token in constant time and memory.

        Args:
            x_t: ``[B, 1, d_model]``.
            state: ``(wavefield, conv buffer)`` from ``init_state`` or a prior call.
            pos: absolute 0-indexed position of this token, which decides
                whether a collapse boundary falls here.

        Returns:
            ``(output [B, 1, d_model], next_state)``.
        """
        z, buf = state
        h, seq = self._pre(x_t, conv_state=buf)
        if self.conv is not None:
            buf = seq[:, 1:]
        drive = self._drive(h)[:, 0]
        lam = self._lam_selective(h)[:, 0] if self.selective else self._log_lambda().exp()

        z_new = lam * z + drive
        out = x_t + self._readout(z_new.unsqueeze(1), h)

        if self.collapse is not None and (pos + 1) % self.collapse_every == 0:
            z_new = self._collapse_state(z_new, h)
        return out, (z_new, buf)
