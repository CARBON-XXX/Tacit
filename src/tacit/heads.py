r"""Decision head — score runtime-defined candidates against a state.

All three question types reduce to the same operation: given a state vector and
a question, rank a set of candidates. The candidates are encoded from text, so
a question can introduce options the model has never seen; meaningful ranking
requires training and evaluation. The answer space is not baked into a fixed
output layer.

The head emits logits over exactly the candidates it was handed. There is no
vocabulary to sample from and no string to parse, which is why a malformed
answer is not a failure mode here.

Calibration
===========
``temperature`` is a single learned scalar applied to every logit. Fit it on
held-out data after training with :func:`fit_temperature`; it does not change
which candidate wins, only how confident the model claims to be. That matters
when a caller is going to branch on ``confidence > 0.9``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DecisionHead", "fit_temperature", "expected_calibration_error"]


class DecisionHead(nn.Module):
    """Score candidate embeddings against a (state, question) pair.

    Args:
        d_model: width of the incoming state, question, and candidate vectors.
        hidden: width of the fusion MLP. Defaults to ``d_model``.
    """

    def __init__(self, d_model: int, *, hidden: int | None = None) -> None:
        super().__init__()
        self.d_model = int(d_model)
        hidden = int(hidden or d_model)

        self.fuse = nn.Sequential(
            nn.Linear(2 * self.d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.d_model),
        )
        self.candidate_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.scale = 1.0 / math.sqrt(self.d_model)

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.exp())

    def set_temperature(self, value: float) -> None:
        with torch.no_grad():
            self.log_temperature.fill_(math.log(max(float(value), 1e-4)))

    def forward(
        self,
        state: torch.Tensor,
        question: torch.Tensor,
        candidates: torch.Tensor,
        *,
        candidate_mask: torch.Tensor | None = None,
        calibrated: bool = True,
    ) -> torch.Tensor:
        """Return logits over candidates.

        Args:
            state: ``[B, d_model]`` summary of everything observed so far.
            question: ``[B, d_model]`` encoding of the prompt.
            candidates: ``[B, n, d_model]`` encoding of each candidate.
            candidate_mask: ``[B, n]``, False where a slot is padding.
            calibrated: divide by the fitted temperature. Turn this off while
                training so the temperature is not learned jointly with the
                rest of the model.

        Returns:
            ``[B, n]`` logits.
        """
        context = self.fuse(torch.cat([state, question], dim=-1))       # [B, D]
        projected = self.candidate_proj(candidates)                     # [B, n, D]
        logits = torch.einsum("bnd,bd->bn", projected, context) * self.scale
        if calibrated:
            logits = logits / self.log_temperature.exp()
        # Mask last: scaling an already-masked -inf makes the gradient wrt the
        # temperature NaN rather than zero.
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, float("-inf"))
        return logits


@torch.no_grad()
def expected_calibration_error(
    probs: torch.Tensor, correct: torch.Tensor, *, n_bins: int = 10
) -> float:
    """Gap between claimed confidence and observed accuracy.

    Args:
        probs: ``[N]`` confidence assigned to the selected candidate.
        correct: ``[N]`` boolean, whether that candidate was right.
        n_bins: number of equal-width confidence bins.

    Returns:
        The sample-weighted mean absolute gap. 0 means perfectly calibrated.
    """
    probs = probs.detach().float().flatten()
    correct = correct.detach().float().flatten()
    if probs.numel() == 0:
        return 0.0
    edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (probs > lo) & (probs <= hi) if i else (probs >= lo) & (probs <= hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        total += n * abs(float(correct[in_bin].mean()) - float(probs[in_bin].mean()))
    return total / probs.numel()


def fit_temperature(
    head: DecisionHead,
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    steps: int = 200,
    lr: float = 0.02,
) -> float:
    """Fit the head's temperature on held-out logits, in place.

    Pass *uncalibrated* logits — the ones produced with ``calibrated=False``.
    Only the temperature moves, so the argmax of every prediction is unchanged;
    what changes is whether a reported 0.9 actually means 0.9.

    Args:
        head: the head whose temperature to set.
        logits: ``[N, n]`` uncalibrated logits from a held-out split. Pad short
            rows with any finite value and mark them in ``candidate_mask``;
            padding with -inf here would make the gradient NaN.
        targets: ``[N]`` index of the correct candidate.
        candidate_mask: ``[N, n]``, False where a slot is padding.
        steps: optimisation steps.
        lr: learning rate for the single scalar.

    Returns:
        The fitted temperature.
    """
    logits = torch.nan_to_num(logits.detach().float(), neginf=0.0, posinf=0.0)
    targets = targets.detach().long()
    mask = None if candidate_mask is None else candidate_mask.to(logits.device)

    log_t = torch.zeros((), requires_grad=True, device=logits.device)
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=steps)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        scaled = logits / log_t.exp()
        if mask is not None:
            scaled = scaled.masked_fill(~mask, float("-inf"))
        loss = F.cross_entropy(scaled, targets)
        loss.backward()
        return loss

    opt.step(closure)
    value = float(log_t.detach().exp())
    head.set_temperature(value)
    return value
