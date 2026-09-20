"""Experimental inference wrapper for locally trained decision checkpoints."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import torch

from .forked import ForkedTacit, pack_forks
from .semantics import SemanticTacit, schema_candidates
from .structured import StructuredTacit, StructuredVectorizer, schema_key


def answers_from_logits(logits, metadata, temperatures):
    answers = {}
    for name, kind, labels, start, end in metadata:
        p = (logits[start:end].float() / temperatures.get(kind, 1.0)).softmax(-1)
        probabilities = dict(zip(labels, p.tolist(), strict=True))
        item = {"type": kind, "label": labels[int(p.argmax())], "probabilities": probabilities}
        if kind == "noul":
            item["noul"] = probabilities["true"]
            item["label"] = item["label"] == "true"
        elif kind == "score":
            item["label"] = int(item["label"])
            item["score"] = sum(i * probabilities[str(i)] for i in range(len(labels)))
        answers[name] = item
    return {"answers": answers}


class DecisionAgent:
    """Load a training run and answer typed requests without generating tokens.

    The trained benchmark paths remain experimental. ``structured`` accepts only
    trained schemas. Semantic paths accept new schemas mechanically, but their
    unseen-schema accuracy has not been established. Oversize inputs raise rather
    than silently discarding evidence. Run directories contain trusted local
    ``best.pt``, ``manifest.json`` and ``result.json`` artifacts.
    """

    def __init__(self, run, *, device="cpu", encoder=None):
        run = Path(run)
        self.manifest = json.loads((run / "manifest.json").read_text())
        result = json.loads((run / "result.json").read_text())
        self.temperatures = result["temperatures"]
        args = self.manifest["arguments"]
        self.architecture = args.get("architecture", "structured")
        if self.architecture not in {"structured", "late", "forked"}:
            raise ValueError("unknown checkpoint architecture: " + self.architecture)
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self._schemas = OrderedDict()
        saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)["state_dict"]
        if self.architecture == "structured":
            self.vectorizer = StructuredVectorizer.from_config(
                json.loads((run / "vectorizer.json").read_text())
            )
            schemas = {
                k.split(".")[1]: v.shape[0]
                for k, v in saved.items()
                if k.startswith("readout.") and k.endswith(".weight")
            }
            self.model = StructuredTacit(self.vectorizer.width, schemas)
        else:
            from transformers import AutoTokenizer

            encoder = encoder or args["encoder"]
            self.tokenizer = AutoTokenizer.from_pretrained(encoder)
            self.max_length = args["max_length"]
            constructor = ForkedTacit if self.architecture == "forked" else SemanticTacit
            self.model = constructor.from_encoder(encoder)
        self.model.load_state_dict(saved, strict=True)
        self.model.to(self.device).eval()

    def clear_cache(self):
        self._schemas.clear()
        if hasattr(self.model, "clear_cache"):
            self.model.clear_cache()

    def _prepare_schema(self, questions):
        key = json.dumps(questions, ensure_ascii=False)
        if key in self._schemas:
            self._schemas.move_to_end(key)
            return key, self._schemas[key]
        texts, meta = schema_candidates(questions)
        if self.architecture == "structured":
            prepared = (meta, schema_key(questions))
        else:
            tokens = self.tokenizer(
                texts, padding=True, return_tensors="pt", return_token_type_ids=False
            )
            if tokens["input_ids"].shape[1] > 256:
                raise ValueError("candidate description exceeds the 256-token training limit")
            raw = self.tokenizer(texts, add_special_tokens=False, return_token_type_ids=False)[
                "input_ids"
            ]
            branches = [
                [self.tokenizer.mask_token_id, *r, self.tokenizer.sep_token_id] for r in raw
            ]
            prepared = (meta, tokens, branches)
        self._schemas[key] = prepared
        if len(self._schemas) > 16:
            self._schemas.popitem(last=False)
        return key, prepared

    @torch.inference_mode()
    def predict(self, state, questions):
        key, prepared = self._prepare_schema(questions)
        meta = prepared[0]
        if self.architecture == "structured":
            x = self.vectorizer.transform([state]).to(self.device)
            logits = self.model(x, prepared[1])[0]
        else:
            state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            tokens = self.tokenizer(state, return_tensors="pt", return_token_type_ids=False)
            if tokens["input_ids"].shape[1] > self.max_length:
                raise ValueError("state exceeds the configured training token limit")
            with torch.autocast(
                self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"
            ):
                if self.architecture == "forked":
                    packed = pack_forks(
                        tokens["input_ids"].tolist(), prepared[2], self.tokenizer.pad_token_id
                    )
                    logits = self.model(**{k: v.to(self.device) for k, v in packed.items()})[0]
                else:
                    logits = self.model(
                        {k: v.to(self.device) for k, v in tokens.items()},
                        {k: v.to(self.device) for k, v in prepared[1].items()},
                        cache_key=key,
                    )[0]
        return answers_from_logits(logits, meta, self.temperatures)
