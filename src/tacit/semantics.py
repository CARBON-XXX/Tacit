"""Optional pretrained semantic input tower with parallel candidate readout.

The transformer is an input encoder only. No vocabulary projection, generation
method, or next-token objective is used. Each state is encoded once; candidate
queries read that shared representation independently. This is a new experimental
stateless text path; it does not inherit the byte core's constant-memory guarantee.
"""

from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn


def schema_candidates(questions):
    texts, metadata = [], []
    for name, q in questions.items():
        kind, criteria = q["type"], q.get("criteria")
        if kind == "noul":
            criteria = criteria or {}
            options = [
                ("false", criteria.get("false", "The statement does not hold.")),
                ("true", criteria.get("true", "The statement holds.")),
            ]
        elif kind == "score":
            options = [(str(i), c) for i, c in enumerate(criteria)]
        elif kind == "choice":
            options = list(criteria.items())
        else:
            raise ValueError("unsupported decision type: " + kind)
        if not 2 <= len(options) <= 255:
            raise ValueError("each decision requires between 2 and 255 candidates")
        start = len(texts)
        for label, description in options:
            texts.append(
                f"Decision type: {kind}. Question: {q['instructions']} "
                f"Candidate: {label}. Criterion: {description or label}"
            )
        metadata.append((name, kind, [k for k, _ in options], start, len(texts)))
    if not texts:
        raise ValueError("at least one decision is required")
    return texts, metadata


class CandidateReadout(nn.Module):
    """Independent candidate queries over shared state tokens.

    No candidate self-attention or candidate-slot embedding: changing the option
    order permutes scores. Two residual cross-attention/MLP layers learn which
    state evidence is relevant to each query and criterion.
    """

    def __init__(self, encoder_width, width=256, layers=2):
        super().__init__()
        self.state_proj = nn.Sequential(
            nn.LayerNorm(encoder_width), nn.Linear(encoder_width, width)
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(encoder_width), nn.Linear(encoder_width, width)
        )
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(width, 4, batch_first=True) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, 4 * width),
                    nn.GELU(),
                    nn.Linear(4 * width, width),
                )
                for _ in range(layers)
            ]
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, state_tokens, state_mask, candidates):
        memory = self.state_proj(state_tokens)
        query = self.query_proj(candidates)[None].expand(len(memory), -1, -1)
        for norm, attention, mlp in zip(self.norms, self.attentions, self.mlps, strict=True):
            context = attention(
                norm(query), memory, memory, key_padding_mask=~state_mask.bool(), need_weights=False
            )[0]
            query = query + context
            query = query + mlp(query)
        return self.scorer(query).squeeze(-1).float()


class SemanticTacit(nn.Module):
    """Pretrained input understanding + a newly trained, generation-free readout.

    ``from_encoder`` initializes only the input tower from pretrained weights.
    The decision readout still requires supervised training. The optional
    transformers dependency is imported only when calling that factory.
    """

    def __init__(self, encoder, width=256, layers=2, query_cache_size=16):
        super().__init__()
        self.encoder = encoder
        self.readout = CandidateReadout(encoder.config.hidden_size, width, layers)
        self.query_cache_size = query_cache_size
        self._queries = OrderedDict()

    @classmethod
    def from_encoder(cls, path, **kwargs):
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(path)
        config.reference_compile = False
        encoder = AutoModel.from_pretrained(path, config=config, attn_implementation="sdpa")
        return cls(encoder, **kwargs)

    def clear_cache(self):
        self._queries.clear()

    def train(self, mode=True):
        self.clear_cache()
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, *args, **kwargs):
        self.clear_cache()
        return super().load_state_dict(*args, **kwargs)

    def encode_candidates(self, tokens, cache_key=None):
        if not self.training and cache_key is not None and cache_key in self._queries:
            self._queries.move_to_end(cache_key)
            return self._queries[cache_key]
        hidden = self.encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        if not self.training and cache_key is not None and self.query_cache_size > 0:
            self._queries[cache_key] = pooled.detach().clone()
            while len(self._queries) > self.query_cache_size:
                self._queries.popitem(last=False)
        return pooled

    def forward(self, states, candidates, *, cache_key=None):
        hidden = self.encoder(**states).last_hidden_state
        queries = self.encode_candidates(candidates, cache_key)
        return self.readout(hidden, states["attention_mask"], queries)
