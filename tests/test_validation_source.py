import hashlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from jev_validation import validation_cases  # noqa: E402


def test_reference_validation_loader_pins_inputs_and_never_opens_final_test(tmp_path):
    content = json.dumps({"id": "development-only"}) + "\n"
    (tmp_path / "validation.jsonl").write_text(content)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"corpus_sha256": {"validation.jsonl": hashlib.sha256(content.encode()).hexdigest()}}
        )
    )
    cases, _ = validation_cases(tmp_path)
    assert cases == [{"id": "development-only"}]
    (tmp_path / "validation.jsonl").write_text(content + content)
    with pytest.raises(ValueError, match="differ"):
        validation_cases(tmp_path)
