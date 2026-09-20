"""Shared-state transformer with independent candidate branches at every layer.

State tokens attend only to state. Each candidate branch reads that shared state
and its own tokens, never another candidate. Branch positions restart after the
state, so reordering candidate branches permutes logits. This is an experimental
ModernBERT input graph, with no language-model output head or generation loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def fork_masks(owners, positions, local_window):
    """owner 0 = state; positive = candidate branch; -1 = padding."""
    query, key = owners[:, :, None], owners[:, None, :]
    valid = (query >= 0) & (key >= 0)
    full = valid & ((key == 0) | ((query == key) & (query > 0)))
    # Padding cannot affect real tokens, but its own softmax still has a key.
    diagonal = torch.eye(owners.shape[1], dtype=torch.bool, device=owners.device)[None]
    full |= diagonal & (query < 0)
    distance = (positions[:, :, None] - positions[:, None, :]).abs()
    local = full & (distance <= local_window // 2)
    return {"full_attention": full[:, None], "sliding_attention": local[:, None]}


def pack_forks(state_ids, branches, pad_id, *, device="cpu"):
    """Batch equal schemas without sharing any state information across examples."""
    lengths = [len(ids) + sum(map(len, branches)) for ids in state_ids]
    maximum = max(lengths)
    ids = torch.full((len(state_ids), maximum), pad_id, dtype=torch.long, device=device)
    owners = torch.full_like(ids, -1)
    positions = torch.zeros_like(ids)
    markers = torch.empty((len(state_ids), len(branches)), dtype=torch.long, device=device)
    for b, state in enumerate(state_ids):
        n = len(state)
        ids[b, :n] = torch.tensor(state, device=device)
        owners[b, :n] = 0
        positions[b, :n] = torch.arange(n, device=device)
        offset = n
        for i, branch in enumerate(branches):
            end = offset + len(branch)
            ids[b, offset:end] = torch.tensor(branch, device=device)
            owners[b, offset:end] = i + 1
            positions[b, offset:end] = torch.arange(n, n + len(branch), device=device)
            markers[b, i] = offset
            offset = end
    return {"input_ids": ids, "owners": owners, "position_ids": positions, "markers": markers}


class ForkedTacit(nn.Module):
    """Deep state/candidate interaction while retaining shared state computation."""

    def __init__(self, encoder):
        super().__init__()
        if encoder.config.model_type != "modernbert":
            raise ValueError("this experimental attention graph currently requires ModernBERT")
        self.encoder = encoder
        width = encoder.config.hidden_size
        self.readout = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 1)
        )

    @classmethod
    def from_encoder(cls, path):
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(path)
        config.reference_compile = False
        return cls(AutoModel.from_pretrained(path, config=config, attn_implementation="sdpa"))

    def forward(self, input_ids, owners, position_ids, markers):
        masks = fork_masks(owners, position_ids, self.encoder.config.local_attention)
        hidden = self.encoder(
            input_ids=input_ids, position_ids=position_ids, attention_mask=masks
        ).last_hidden_state
        selected = hidden.gather(1, markers[:, :, None].expand(-1, -1, hidden.shape[-1]))
        return self.readout(selected).squeeze(-1).float()
