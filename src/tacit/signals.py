"""A language-free System 1 path for fixed-schema numeric event streams.

Measurements update explicit latest-value registers; a learned recurrent core
captures temporal features. All declared decision heads run together. Field
names and question text are metadata: this model must be trained for its schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from tacit.encoder import EncoderConfig
from tacit.engine import Tacit
from tacit.learning import TemperatureScaler
from tacit.resonance import ResonantBlock, RMSNorm


@dataclass
class SignalConfig:
    features: tuple[str, ...]
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(
            d_model=64, n_layers=2, collapse_every=8, conv_kernel=2
        )
    )
    head_hidden: int = 64

    def __post_init__(self):
        self.features = tuple(self.features)
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("at least one uniquely named feature is required")
        if self.encoder.register_states:
            raise ValueError("SignalTacit uses latest-value registers, not cyclic registers")


@dataclass
class SignalState:
    values: torch.Tensor
    known: torch.Tensor
    blocks: list
    pos: int = 0

    def detach(self):
        return SignalState(
            self.values.detach(),
            self.known.detach(),
            [(z.detach(), None if b is None else b.detach()) for z, b in self.blocks],
            self.pos,
        )

    @property
    def nbytes(self):
        """Persistent tensor bytes, excluding allocator overhead and Python objects."""
        tensors = [self.values, self.known]
        tensors.extend(t for pair in self.blocks for t in pair if t is not None)
        return sum(t.numel() * t.element_size() for t in tensors)


class SignalTacit(nn.Module):
    """Trainable numeric decisions with bounded persistent memory.

    Inputs must use the training feature order and normalization. ``observed``
    distinguishes a new zero from an absent measurement. Unobserved fields keep
    their last value; explicit known bits distinguish missing from zero. No text
    encoder, tokenizer, language-model head, or autoregressive decoding exists.
    Decision names, labels and their order are fixed when constructing the model.
    """

    def __init__(self, config: SignalConfig, questions):
        super().__init__()
        if not questions:
            raise ValueError("at least one decision schema is required")
        self.config = config
        self.questions = dict(questions)
        f, d = len(config.features), config.encoder.d_model
        self.project = nn.Linear(4 * f, d)
        self.blocks = nn.ModuleList(
            [
                ResonantBlock(d, **config.encoder.block_kwargs())
                for _ in range(config.encoder.n_layers)
            ]
        )
        self.norm = RMSNorm(d)
        self.readout = nn.Sequential(
            nn.Linear(d + 2 * f, config.head_hidden),
            nn.SiLU(),
            nn.Linear(config.head_hidden, sum(len(q.labels()) for q in questions.values())),
        )
        self.temperatures = nn.ModuleList([TemperatureScaler() for _ in questions])

    @property
    def device(self):
        return self.project.weight.device

    def init_state(self, batch=1):
        shape = (batch, len(self.config.features))
        return SignalState(
            torch.zeros(shape, device=self.device, dtype=self.project.weight.dtype),
            torch.zeros(shape, device=self.device, dtype=torch.bool),
            [b.init_state(batch, self.device) for b in self.blocks],
        )

    def forward(self, values, observed=None, state=None, *, calibrated=False):
        """Return logits at every event and a resumable state.

        ``values`` and boolean ``observed`` have shape [batch, time, features].
        Omit ``observed`` only when *all* fields are measured at every event.
        Nonfinite values are permitted only in unobserved slots. A batch must
        contain equal-length streams; padding is not an event mask.
        """
        if (
            values.ndim != 3
            or not values.shape[0]
            or not values.shape[1]
            or values.shape[2] != len(self.config.features)
        ):
            raise ValueError("values must be nonempty [batch, time, configured features]")
        values = values.to(device=self.device, dtype=self.project.weight.dtype)
        if observed is None:
            observed = torch.ones_like(values, dtype=torch.bool)
        if observed.shape != values.shape or observed.dtype != torch.bool:
            raise ValueError("observed must be a boolean mask with the same shape as values")
        observed = observed.to(self.device)
        values = torch.where(observed, values, 0)
        if not torch.isfinite(values).all():
            raise ValueError("observed measurements must be finite")
        state = self.init_state(len(values)) if state is None else state
        if state.values.shape != (values.shape[0], values.shape[2]):
            raise ValueError("state batch or feature count does not match input")
        if state.values.device != self.device:
            raise ValueError("state and model must be on the same device")
        t = values.shape[1]
        # Last write wins, computed in parallel. Index zero refers to prior state.
        positions = torch.arange(1, t + 1, device=self.device)[None, :, None]
        index = torch.where(observed, positions, 0).cummax(dim=1).values
        latest = torch.cat([state.values[:, None], values], dim=1).gather(1, index)
        known = state.known[:, None] | (observed.cumsum(dim=1) > 0)
        h = self.project(
            torch.cat([latest, known.to(values.dtype), values, observed.to(values.dtype)], dim=-1)
        )
        blocks = []
        for block, carried in zip(self.blocks, state.blocks, strict=True):
            h, carried = block.absorb(h, carried, state.pos)
            blocks.append(carried)
        raw = self.readout(torch.cat([self.norm(h), latest, known.to(values.dtype)], dim=-1))
        sizes = [len(q.labels()) for q in self.questions.values()]
        logits = {
            name: self.temperatures[i](part) if calibrated else part
            for i, (name, part) in enumerate(zip(self.questions, raw.split(sizes, -1), strict=True))
        }
        return logits, SignalState(
            latest[:, -1].clone(), known[:, -1].clone(), blocks, state.pos + t
        )

    @torch.no_grad()
    def evaluate(self, values, observed=None, state=None):
        """Typed answers for the final event of each batch row, plus next state."""
        logits, state = self(values, observed, state, calibrated=True)
        answers = [{} for _ in range(len(values))]
        for name, question in self.questions.items():
            for i, answer in enumerate(Tacit.answers_from_logits(question, logits[name][:, -1])):
                answers[i][name] = answer
        return answers, state
