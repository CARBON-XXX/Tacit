import pytest
import torch

from tacit.structured import StructuredTacit, StructuredVectorizer, fields, schema_key


def test_vectorizer_fit_is_frozen_and_missing_differs_from_zero():
    v = StructuredVectorizer(16).fit([{"x": 0, "c": "one"}, {"x": 1, "c": "one"}])
    before = v.transform([{"x": 0, "c": "one"}])
    restored = StructuredVectorizer.from_config(v.config)
    torch.testing.assert_close(before, restored.transform([{"c": "one", "x": 0}]))
    missing = v.transform([{"c": "one"}])
    assert not torch.equal(before, missing)
    v.transform([{"unseen": 100000, "c": "new category"}])
    torch.testing.assert_close(before, v.transform([{"x": 0, "c": "one"}]))
    assert torch.isfinite(v.transform([{"x": 1e20}])).all()
    with pytest.raises(ValueError):
        fields({"x": float("nan")})


def test_structured_schema_and_gradients():
    q = {"route": {"type": "choice", "instructions": "Route.", "criteria": {"a": "A", "b": "B"}}}
    key = schema_key(q)
    assert key == schema_key({"different_name": q["route"]})
    model = StructuredTacit(8, {key: 2})
    model(torch.randn(3, 8), key).square().mean().backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="untrained"):
        model(torch.randn(1, 8), "unseen")


def test_numeric_comparison_and_text_ablation():
    states = [{"a": 1, "b": 2, "text": "one two three four five six"}, {"a": 2, "b": 1}]
    v = StructuredVectorizer(0).fit(states)
    a = v.transform([{"a": 1, "b": 2, "text": "one two three four five six"}])
    b = v.transform([{"a": 1, "b": 2, "text": "changed words that have no effect here"}])
    torch.testing.assert_close(a, b)
    assert v.config["pairs"]
    assert not torch.equal(v.transform(states)[0], v.transform(states)[1])
