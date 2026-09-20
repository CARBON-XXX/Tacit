"""Small fixed-schema decision model over observable JSON fields.

This is an explicit specialist: new question schemas require training. It has no
pretrained language encoder. Optional hashed word features provide a lexical
baseline, not an assertion of semantic understanding. Fit preprocessing on the
training partition only; never pass labels or latent generator factors as state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict

import torch
from torch import nn

from .semantics import schema_candidates


def fields(state):
    """Order-independent object/list aggregates, with missingness preserved."""
    numbers, categories, words = defaultdict(list), Counter(), Counter()

    def visit(value, path):
        if value is None:
            categories[path + "=null"] += 1
        elif isinstance(value, bool):
            categories[path + "=" + str(value).lower()] += 1
        elif isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ValueError("state numbers must be finite")
            numbers[path].append(float(value))
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, path + "/" + str(key).replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            numbers[path + "/#length"].append(float(len(value)))
            for item in value:
                visit(item, path + "/*")
        elif isinstance(value, str):
            tokens = re.findall(r"\w+", value.lower())
            if len(tokens) <= 4 and len(value) <= 80:
                categories[path + "=" + value.lower()] += 1
            for token in tokens:
                words[path + ":" + token] += 1
            for a, b in zip(tokens, tokens[1:], strict=False):
                words[path + ":" + a + " " + b] += 1
        else:
            raise TypeError("state must contain JSON-compatible values")

    visit(state, "")
    aggregates = {}
    for path, values in numbers.items():
        aggregates[path + "/mean"] = sum(values) / len(values)
        # Preserve ranges when a field occurs in multiple list entries.
        aggregates[path + "/min"] = min(values)
        aggregates[path + "/max"] = max(values)
    return aggregates, categories, words


def schema_key(questions):
    # Names are response routing identifiers; descriptions define the trained head.
    texts, _ = schema_candidates(questions)
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()


class StructuredVectorizer:
    def __init__(self, text_features=2048, min_category_count=2):
        self.text_features = text_features
        self.min_category_count = min_category_count
        self.config = None

    def fit(self, states):
        extracted = [fields(s) for s in states]
        paths = sorted({p for n, _, _ in extracted for p in n})
        stats = {}
        for path in paths:
            values = [n[path] for n, _, _ in extracted if path in n]
            transformed = [[v, math.copysign(math.log1p(abs(v)), v)] for v in values]
            t = torch.tensor(transformed, dtype=torch.float64)
            stats[path] = {
                "mean": t.mean(0).tolist(),
                "scale": t.std(0, unbiased=False).clamp_min(1e-4).tolist(),
            }
        counts = Counter()
        for _, c, _ in extracted:
            counts.update(c.keys())
        cats = sorted(k for k, v in counts.items() if v >= self.min_category_count)
        # Generic, bounded pairwise numeric differences expose comparisons without
        # adding task-specific rules or reading generator factors.
        means = [p for p in paths if p.endswith("/mean")]
        pairs = [(a, b) for i, a in enumerate(means) for b in means[i + 1 :]]
        self.config = {
            "stats": stats,
            "categories": cats,
            "pairs": pairs,
            "text_features": self.text_features,
            "min_category_count": self.min_category_count,
        }
        self._category_index = {k: i for i, k in enumerate(cats)}
        return self

    @classmethod
    def from_config(cls, config):
        obj = cls(config["text_features"], config["min_category_count"])
        obj.config = config
        obj._category_index = {k: i for i, k in enumerate(config["categories"])}
        return obj

    @property
    def width(self):
        if self.config is None:
            raise RuntimeError("fit the vectorizer first")
        return (
            3 * len(self.config["stats"])
            + 2 * len(self.config["pairs"])
            + len(self.config["categories"])
            + self.text_features
        )

    def transform(self, states):
        output = torch.zeros(len(states), self.width)
        for row, state in enumerate(states):
            numbers, categories, words = fields(state)
            values = []
            for path, stat in self.config["stats"].items():
                if path not in numbers:
                    values.extend([0.0, 0.0, 0.0])
                else:
                    v = numbers[path]
                    raw = [v, math.copysign(math.log1p(abs(v)), v)]
                    values.extend(
                        [
                            1.0,
                            *[
                                max(-10.0, min(10.0, (x - m) / s))
                                for x, m, s in zip(raw, stat["mean"], stat["scale"], strict=True)
                            ],
                        ]
                    )
            for a, b in self.config["pairs"]:
                if a in numbers and b in numbers:
                    x, y = numbers[a], numbers[b]
                    values.extend([1.0, (x - y) / (abs(x) + abs(y) + 1e-6)])
                else:
                    values.extend([0.0, 0.0])
            offset = len(values)
            output[row, :offset] = torch.tensor(values)
            for key, count in categories.items():
                if key in self._category_index:
                    output[row, offset + self._category_index[key]] = math.log1p(count)
            offset += len(self._category_index)
            if self.text_features:
                for word, count in words.items():
                    digest = int.from_bytes(
                        hashlib.blake2b(word.encode(), digest_size=8).digest(), "little"
                    )
                    index = digest % self.text_features
                    sign = 1 if (digest >> 63) else -1
                    output[row, offset + index] += sign * math.log1p(count)
                text = output[row, offset:]
                text /= text.norm().clamp_min(1)
        return output


class StructuredTacit(nn.Module):
    """Learned JSON features to trained typed heads, without a text decoder."""

    def __init__(self, input_width, schemas, width=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_width, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Dropout(0.1),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.readout = nn.ModuleDict(
            {key: nn.Linear(width, count) for key, count in schemas.items()}
        )

    def forward(self, features, schema):
        if schema not in self.readout:
            raise ValueError("untrained question schema; this model is a specialist")
        return self.readout[schema](self.encoder(features))
