"""Validate the semantic readout without downloading a pretrained backbone."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tacit.semantics import CandidateReadout, SemanticTacit, schema_candidates


def test_independent_candidate_queries_are_permutation_equivariant():
    torch.manual_seed(0)
    head = CandidateReadout(32, width=16, layers=2).eval()
    states, queries = torch.randn(2, 7, 32), torch.randn(4, 32)
    mask = torch.ones(2, 7, dtype=torch.long)
    first = head(states, mask, queries)
    order = [3, 1, 0, 2]
    second = head(states, mask, queries[order])
    torch.testing.assert_close(second, first[:, order], atol=1e-6, rtol=1e-5)
    extended = head(states, mask, torch.cat([queries, torch.randn(3, 32)]))
    torch.testing.assert_close(extended[:, :4], first, atol=1e-6, rtol=1e-5)
    first.square().mean().backward()
    assert torch.isfinite(head.state_proj[1].weight.grad).all()
    assert head.state_proj[1].weight.grad.abs().sum() > 0


def test_state_padding_is_not_evidence():
    torch.manual_seed(1)
    head = CandidateReadout(32, width=16).eval()
    state, queries = torch.randn(1, 5, 32), torch.randn(2, 32)
    a = head(state, torch.ones(1, 5), queries)
    b = head(
        torch.cat([state, torch.randn(1, 9, 32) * 100], 1),
        torch.tensor([[1] * 5 + [0] * 9]),
        queries,
    )
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.embedding = nn.Embedding(20, 16)
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_encoder_is_shared_and_query_cache_invalidates():
    encoder = TinyEncoder()
    model = SemanticTacit(encoder, width=16, query_cache_size=1).eval()
    states = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)}
    candidates = {"input_ids": torch.tensor([[3, 4], [1, 4]]), "attention_mask": torch.ones(2, 2)}
    with torch.no_grad():
        first = model(states, candidates, cache_key="one")
        assert encoder.calls == 2
        second = model(states, candidates, cache_key="one")
        assert encoder.calls == 3  # one state pass; cached query representations
        torch.testing.assert_close(first, second)
        model(states, candidates, cache_key="two")
        assert list(model._queries) == ["two"]
        model.load_state_dict(model.state_dict())
        assert not model._queries
    model.train()
    model(states, candidates, cache_key="one").square().mean().backward()
    assert not model._queries and encoder.embedding.weight.grad.abs().sum() > 0


def test_schema_is_defined_by_descriptions_not_question_ids():
    q = {
        "private_id": {
            "type": "choice",
            "instructions": "Route the request.",
            "criteria": {"a": "payment issue", "b": "outage"},
        }
    }
    texts, meta = schema_candidates(q)
    renamed, _ = schema_candidates({"different_id": q["private_id"]})
    assert texts == renamed and "private_id" not in " ".join(texts)
    assert meta[0][2] == ["a", "b"]
    with pytest.raises(ValueError):
        schema_candidates({"bad": {"type": "text", "instructions": "generate"}})
