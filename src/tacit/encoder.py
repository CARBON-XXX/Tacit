r"""Byte-level encoder built from resonant blocks.

Input is raw UTF-8 bytes, so there is no tokenizer to ship, version, or
mismatch. The alphabet is 256 byte values plus one padding id.

The encoder exposes the same two entry points as the block it stacks:
``forward`` for a whole sequence in parallel, and ``step`` for one byte at a
time against a carried state. The streaming path is what makes a session
stateful — observations fold into the wavefield and stay there, at constant
cost per byte.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from tacit.register import CyclicRegister
from tacit.resonance import ResonantBlock, RMSNorm

__all__ = ["EncoderConfig", "ByteEncoder", "PAD_ID", "VOCAB_SIZE", "encode_bytes"]

PAD_ID = 256
VOCAB_SIZE = 257


def encode_bytes(text: str, max_len: int | None = None) -> list[int]:
    """UTF-8 encode, optionally keeping only the last ``max_len`` bytes."""
    raw = list(text.encode("utf-8"))
    return raw if max_len is None else raw[-max_len:]


@dataclass
class EncoderConfig:
    """Shape and behaviour of the encoder stack."""

    d_model: int = 256
    n_layers: int = 4
    n_bands: int = 4
    collapse_every: int = 16
    n_phases: int = 16
    rho_floor: float = 0.9
    phase_selective: bool = True
    mag_selective: bool = False
    selective_readout: bool = True
    conv_kernel: int = 4
    dropout: float = 0.0
    collapse_cold_start: bool = False
    symbol_rank: int = 0
    register_states: int = 0  # 0 disables the cyclic register
    register_gate_init: float = 0.0

    def block_kwargs(self) -> dict:
        return {
            "n_bands": self.n_bands,
            "collapse_every": self.collapse_every,
            "n_phases": self.n_phases,
            "rho_floor": self.rho_floor,
            "phase_selective": self.phase_selective,
            "mag_selective": self.mag_selective,
            "selective_readout": self.selective_readout,
            "conv_kernel": self.conv_kernel,
            "dropout": self.dropout,
            "n_layers": self.n_layers,
            "collapse_cold_start": self.collapse_cold_start,
            "symbol_rank": self.symbol_rank,
        }


@dataclass
class EncoderState:
    """Carried state for streaming. Constant size, independent of history."""

    blocks: list = field(default_factory=list)
    register: torch.Tensor | None = None
    pos: int = 0

    def detach(self) -> EncoderState:
        blocks = [(z.detach(), None if buf is None else buf.detach()) for z, buf in self.blocks]
        return EncoderState(
            blocks=blocks,
            register=None if self.register is None else self.register.detach(),
            pos=self.pos,
        )


class ByteEncoder(nn.Module):
    """Stack of resonant blocks over byte embeddings."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        cfg = self.config

        self.embed = nn.Embedding(VOCAB_SIZE, cfg.d_model, padding_idx=PAD_ID)
        nn.init.normal_(self.embed.weight, std=0.02)
        with torch.no_grad():
            self.embed.weight[PAD_ID].zero_()

        self.blocks = nn.ModuleList(
            ResonantBlock(cfg.d_model, **cfg.block_kwargs()) for _ in range(cfg.n_layers)
        )
        self.register = (
            CyclicRegister(cfg.d_model, n_states=cfg.register_states,
                           gate_init=cfg.register_gate_init)
            if cfg.register_states > 0
            else None
        )
        self.norm_out = RMSNorm(cfg.d_model)

    @property
    def d_model(self) -> int:
        return self.config.d_model

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``[B, T]`` byte ids into ``[B, T, d_model]``."""
        h = self.embed(tokens)
        for block in self.blocks:
            h = block(h)
        if self.register is not None:
            h = self.register(h, mask=mask)
        return self.norm_out(h)

    def pool(self, tokens: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode and reduce to one vector per sequence, ``[B, d_model]``.

        Takes the last valid position rather than a mean: the recurrence has
        already folded the whole prefix into that state, so averaging would
        only dilute it.
        """
        h = self.forward(tokens, mask=mask)
        if mask is None:
            return h[:, -1]
        lengths = mask.to(torch.long).sum(dim=1).clamp_min(1) - 1
        return h[torch.arange(h.shape[0], device=h.device), lengths]

    # -- streaming -----------------------------------------------------------

    def init_state(self, batch: int, device: torch.device | str) -> EncoderState:
        return EncoderState(
            blocks=[b.init_state(batch, device) for b in self.blocks],
            register=None if self.register is None else self.register.init_state(batch, device),
            pos=0,
        )

    def step(self, token: torch.Tensor, state: EncoderState) -> tuple[torch.Tensor, EncoderState]:
        """Fold one byte into the state. ``token`` is ``[B]`` or ``[B, 1]``.

        Returns the hidden state at this position, ``[B, 1, d_model]``, and the
        updated encoder state.
        """
        if token.ndim == 1:
            token = token.unsqueeze(1)
        h = self.embed(token)
        blocks = []
        for block, block_state in zip(self.blocks, state.blocks, strict=True):
            h, block_state = block.step(h, block_state, state.pos)
            blocks.append(block_state)
        reg = state.register
        if self.register is not None:
            h, reg = self.register.step(h, reg)
        return self.norm_out(h), EncoderState(blocks=blocks, register=reg, pos=state.pos + 1)

    def absorb(
        self,
        tokens: torch.Tensor,
        state: EncoderState,
    ) -> tuple[torch.Tensor, EncoderState]:
        """Fold a run of bytes into state using resumable parallel chunks.

        ``tokens`` is ``[B, T]``. Returns the hidden state at the final position
        and the updated encoder state. This is the streaming counterpart of
        ``pool``; cost per byte does not depend on how much came before.
        """
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("absorb() needs at least one token")
        h = self.embed(tokens)
        blocks = []
        for block, carried in zip(self.blocks, state.blocks, strict=True):
            h, carried = block.absorb(h, carried, state.pos)
            blocks.append(carried)
        reg = state.register
        if self.register is not None:
            outputs = []
            for i in range(tokens.shape[1]):
                out, reg = self.register.step(h[:, i:i + 1], reg)
                outputs.append(out)
            h = torch.cat(outputs, dim=1)
        return self.norm_out(h[:, -1]), EncoderState(
            blocks=blocks, register=reg, pos=state.pos + tokens.shape[1]
        )
