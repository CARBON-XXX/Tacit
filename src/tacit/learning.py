"""Decision-only objectives and held-out calibration; no language-model loss."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def decision_loss(logits, targets, *, brier_weight=0.0, ordinal_weight=0.0):
    """Cross entropy plus optional Brier and ranked probability scores.

    Accept integer classes or soft target distributions. RPS penalizes cumulative
    probability errors on an *ordered* scale; leave its weight zero for choices.
    These are supervised proper scoring losses, not reinforcement learning.
    Finite, unpadded logits are required; handle different candidate counts in
    separate calls. All leading dimensions are averaged.
    """
    if brier_weight < 0 or ordinal_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    if logits.ndim < 2 or logits.shape[-1] < 2 or not torch.isfinite(logits).all():
        raise ValueError("finite logits with at least two candidates are required")
    targets = targets.to(logits.device)
    if targets.shape == logits.shape:
        target = targets.float()
        if (
            not torch.isfinite(target).all()
            or (target < 0).any()
            or not torch.allclose(target.sum(-1), torch.ones_like(target[..., 0]), atol=1e-5)
        ):
            raise ValueError("soft targets must be probability distributions")
    else:
        if targets.shape != logits.shape[:-1] or targets.dtype != torch.long:
            raise ValueError("hard targets must be int64 class indices with matching shape")
        target = F.one_hot(targets, logits.shape[-1]).float()
    log_p = logits.float().log_softmax(-1)
    p = log_p.exp()
    loss = -(target * log_p).sum(-1)
    if brier_weight:
        loss = loss + brier_weight * (p - target).square().sum(-1)
    if ordinal_weight:
        loss = loss + ordinal_weight * (
            p.cumsum(-1)[..., :-1] - target.cumsum(-1)[..., :-1]
        ).square().mean(-1)
    return loss.mean()


class TemperatureScaler(nn.Module):
    """A separate temperature per deployed decision schema.

    Fit on a dedicated validation split; use another split for conformal policy
    calibration. A bounded grid includes T=1 so fitting cannot worsen validation
    NLL. Calibration under distribution shift is not guaranteed.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("temperature", torch.ones(()))

    def forward(self, logits):
        return logits / self.temperature

    @torch.no_grad()
    def fit(self, logits, targets):
        if logits.ndim != 2 or len(logits) == 0:
            raise ValueError("fit needs nonempty [examples, candidates] logits")
        decision_loss(logits, targets)  # validate before fitting
        candidates = torch.cat(
            [
                torch.logspace(math.log10(0.05), math.log10(20), 201, device=logits.device),
                torch.ones(1, device=logits.device),
            ]
        )
        losses = torch.stack([decision_loss(logits.detach() / t, targets) for t in candidates])
        best = candidates[losses.argmin()]
        self.temperature.copy_(best.to(self.temperature.device))
        return float(best)


@dataclass(frozen=True)
class PredictionSet:
    candidates: tuple[str, ...]

    @property
    def automated(self) -> bool:
        return len(self.candidates) == 1

    @property
    def value(self) -> str | None:
        return self.candidates[0] if self.automated else None


@dataclass(frozen=True)
class ConformalPolicy:
    """Split-conformal candidate sets, with abstention unless exactly one remains.

    Under exchangeable calibration/test examples, the *marginal set coverage*
    target is 1-alpha. This is not a bound on errors among automated decisions,
    a joint guarantee over many questions, or protection against domain shift.
    Freeze the model and any temperature before fitting this policy.
    """

    labels: tuple[str, ...]
    threshold: float
    alpha: float
    calibration_size: int

    @staticmethod
    def _validate(probs, count):
        if (
            probs.ndim != 2
            or probs.shape[1] != count
            or not torch.isfinite(probs).all()
            or (probs < 0).any()
            or (probs > 1).any()
            or not torch.allclose(probs.sum(-1), torch.ones_like(probs[:, 0]), atol=1e-5)
        ):
            raise ValueError("expected normalized [examples, candidates] probabilities")

    @classmethod
    def fit(cls, labels, probs, targets, *, alpha=0.1):
        labels = tuple(labels)
        if len(labels) < 2 or len(set(labels)) != len(labels) or not 0 < alpha < 1:
            raise ValueError("unique labels and 0 < alpha < 1 are required")
        cls._validate(probs, len(labels))
        if len(probs) == 0 or targets.shape != (len(probs),) or targets.dtype != torch.long:
            raise ValueError("nonempty probabilities and int64 class targets are required")
        targets = targets.to(probs.device)
        if (targets < 0).any() or (targets >= len(labels)).any():
            raise ValueError("target index outside label space")
        scores = 1 - probs[torch.arange(len(probs), device=probs.device), targets]
        rank = math.ceil((len(scores) + 1) * (1 - alpha))
        threshold = 1.0 if rank > len(scores) else float(scores.kthvalue(rank).values)
        return cls(labels, threshold, alpha, len(scores))

    def predict(self, probs):
        self._validate(probs, len(self.labels))
        keep = (1 - probs <= self.threshold).detach().cpu().tolist()
        return [
            PredictionSet(tuple(k for k, yes in zip(self.labels, row, strict=True) if yes))
            for row in keep
        ]
