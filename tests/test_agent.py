import json

import torch

from tacit.agent import DecisionAgent, answers_from_logits
from tacit.semantics import schema_candidates
from tacit.structured import StructuredTacit, StructuredVectorizer, schema_key


def test_typed_outputs_and_temperature():
    meta = [("yes", "noul", ["false", "true"], 0, 2), ("level", "score", ["0", "1", "2"], 2, 5)]
    result = answers_from_logits(torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0]), meta, {"noul": 2.0})[
        "answers"
    ]
    assert result["yes"]["label"] is True
    assert abs(result["yes"]["noul"] - torch.sigmoid(torch.tensor(1.0)).item()) < 1e-6
    assert abs(result["level"]["score"] - 1.0) < 1e-6


def test_agent_load_and_relabel_request_without_retraining(tmp_path):
    q = {"a": {"type": "choice", "instructions": "Route.", "criteria": {"x": "X", "y": "Y"}}}
    v = StructuredVectorizer(0).fit([{"n": 1}, {"n": 2}])
    model = StructuredTacit(v.width, {schema_key(q): 2}).eval()
    torch.save({"state_dict": model.state_dict()}, tmp_path / "best.pt")
    (tmp_path / "manifest.json").write_text(json.dumps({"arguments": {}}))
    (tmp_path / "result.json").write_text(json.dumps({"temperatures": {}}))
    (tmp_path / "vectorizer.json").write_text(json.dumps(v.config))
    agent = DecisionAgent(tmp_path)
    prediction = agent.predict({"n": 1}, q)
    expected = answers_from_logits(
        model(v.transform([{"n": 1}]), schema_key(q))[0], schema_candidates(q)[1], {}
    )
    assert prediction == expected
    renamed = agent.predict({"n": 1}, {"new_id": q["a"]})
    assert renamed["answers"]["new_id"] == prediction["answers"]["a"]
    agent.clear_cache()
    assert not agent._schemas
