r"""The model and its two calling conventions.

:meth:`Tacit.evaluate` is the stateless one: hand it a state and a set of typed
questions, get back calibrated answers. Nothing carries over between calls.

:meth:`Tacit.session` is the stateful one, and it is the reason this core is
worth wiring up. A session folds each observation into a complex wavefield of
fixed size and keeps it. Later questions are answered against everything seen
so far, at a cost per byte that does not grow with how much came before — there
is no transcript to re-read and no cache to re-scan.

Both paths answer in parallel over the questions you pass, and neither emits
text.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from tacit.encoder import PAD_ID, ByteEncoder, EncoderConfig, EncoderState, encode_bytes
from tacit.heads import DecisionHead
from tacit.questions import (
    Answer,
    Boolean,
    BooleanAnswer,
    ChoiceAnswer,
    Question,
    Score,
    ScoreAnswer,
)

__all__ = ["TacitConfig", "Tacit", "Session"]


@dataclass
class TacitConfig:
    """Model configuration.

    Args:
        encoder: shape of the recurrent stack.
        head_hidden: width of the decision head's fusion MLP.
        max_bytes: longest text fed to the encoder in one go. Longer inputs keep
            their tail. This bounds a single ``evaluate`` call; a session has no
            such limit, because it folds observations into state instead of
            re-reading them.
    """

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_hidden: int | None = None
    max_bytes: int = 4096
    question_batch_size: int = 64
    question_cache_size: int = 128
    prepared_cache_size: int = 8

    def __post_init__(self) -> None:
        if (self.max_bytes < 1 or self.question_batch_size < 1
                or self.question_cache_size < 0 or self.prepared_cache_size < 0):
            raise ValueError("positive byte/batch limits and a nonnegative cache size are required")


class Tacit(nn.Module):
    """A recurrent phase-collapse core with a typed decision interface."""

    def __init__(self, config: TacitConfig | None = None) -> None:
        super().__init__()
        self.config = config or TacitConfig()
        self.encoder = ByteEncoder(self.config.encoder)
        self.head = DecisionHead(self.encoder.d_model, hidden=self.config.head_hidden)
        self._cache: OrderedDict[tuple[str, ...], tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._head_cache: OrderedDict = OrderedDict()

    @property
    def d_model(self) -> int:
        return self.encoder.d_model

    @property
    def device(self) -> torch.device:
        return self.encoder.embed.weight.device

    def train(self, mode: bool = True):  # noqa: D102 - clearing a cache, not new behaviour
        self.clear_cache()
        return super().train(mode)

    def clear_cache(self) -> None:
        """Drop memoised question encodings. Call after updating weights."""
        self._cache.clear()
        self._head_cache.clear()

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, *args, **kwargs):
        self.clear_cache()
        return super().load_state_dict(*args, **kwargs)

    # -- text in, vectors out ------------------------------------------------

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of strings to ``[N, d_model]``.

        Shorter strings are right-padded; the pooled vector is read from each
        sequence's own last real byte.
        """
        if not texts:
            raise ValueError("encode_texts() needs at least one string")
        rows = [encode_bytes(t, self.config.max_bytes) or [PAD_ID] for t in texts]
        width = max(len(r) for r in rows)
        device = self.device
        tokens = torch.full((len(rows), width), PAD_ID, dtype=torch.long, device=device)
        mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
        for i, row in enumerate(rows):
            tokens[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
            mask[i, : len(row)] = True
        return self.encoder.pool(tokens, mask=mask)

    def question_vectors(self, question: Question) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a question and its candidates to ``([d_model], [n, d_model])``.

        Results are memoised while the model is in eval mode, since a service
        usually asks the same handful of questions over and over.
        """
        return self._question_vectors_many([question])[0]

    def _question_vectors_many(self, questions):
        keys = [(q.prompt, *q.candidate_texts()) for q in questions]
        vectors, missing, texts = {}, {}, []
        for key in keys:
            if key in vectors or key in missing:
                continue
            if not self.training and key in self._cache:
                self._cache.move_to_end(key)
                vectors[key] = self._cache[key]
            else:
                missing[key] = (len(texts), len(key))
                texts.extend(key)
        if texts:
            size = self.config.question_batch_size
            encoded = torch.cat([
                self.encode_texts(texts[i:i + size]) for i in range(0, len(texts), size)
            ])
            for key, (start, count) in missing.items():
                pair = encoded[start], encoded[start + 1:start + count]
                vectors[key] = pair
                if not self.training and self.config.question_cache_size:
                    # Clone: a small entry must not retain a whole encoded batch.
                    self._cache[key] = tuple(v.detach().clone() for v in pair)
                    while len(self._cache) > self.config.question_cache_size:
                        self._cache.popitem(last=False)
        return [vectors[key] for key in keys]

    def score_many(self, state_vecs, questions, *, calibrated=True):
        """Score all questions with a shared state encoding and one head call.

        Candidate slots are padded and masked; their order only permutes the
        corresponding scores. Returned logits remain differentiable for training.
        """
        if not questions:
            return {}
        if len(questions) == 1:
            name, question = next(iter(questions.items()))
            return {name: self.score(state_vecs, question, calibrated=calibrated)}
        batch, width = state_vecs.shape
        key = tuple((q.prompt, *q.candidate_texts()) for q in questions.values())
        if not self.training and key in self._head_cache:
            self._head_cache.move_to_end(key)
            prompts, candidates, mask, lengths = self._head_cache[key]
        else:
            vectors = self._question_vectors_many(list(questions.values()))
            lengths = [c.shape[0] for _, c in vectors]
            maximum = max(lengths)
            prompts = torch.stack([p for p, _ in vectors])
            candidates = torch.stack([
                c if len(c) == maximum else torch.nn.functional.pad(c, (0, 0, 0, maximum - len(c)))
                for _, c in vectors
            ])
            mask = torch.arange(maximum, device=state_vecs.device)[None, :] < torch.tensor(
                lengths, device=state_vecs.device
            )[:, None]
            if not self.training and self.config.prepared_cache_size:
                self._head_cache[key] = prompts.detach(), candidates.detach(), mask, lengths
                while len(self._head_cache) > self.config.prepared_cache_size:
                    self._head_cache.popitem(last=False)
        count, maximum = candidates.shape[:2]
        scores = self.head(
            state_vecs[:, None, :].expand(-1, count, -1).reshape(-1, width),
            prompts[None].expand(batch, -1, -1).reshape(-1, width),
            candidates[None].expand(batch, -1, -1, -1).reshape(-1, maximum, width),
            candidate_mask=mask[None].expand(batch, -1, -1).reshape(-1, maximum),
            calibrated=calibrated,
        ).reshape(batch, count, maximum)
        return {name: scores[:, i, :lengths[i]] for i, name in enumerate(questions)}

    # -- scoring -------------------------------------------------------------

    def score(
        self,
        state_vecs: torch.Tensor,
        question: Question,
        *,
        calibrated: bool = True,
    ) -> torch.Tensor:
        """Logits over a question's candidates for each state in the batch.

        Args:
            state_vecs: ``[B, d_model]``.
            question: the question to answer.
            calibrated: apply the fitted temperature. Pass ``False`` when
                producing logits for training or for temperature fitting.

        Returns:
            ``[B, n_candidates]``.
        """
        batch = state_vecs.shape[0]
        prompt_vec, candidate_vecs = self.question_vectors(question)
        return self.head(
            state_vecs,
            prompt_vec.unsqueeze(0).expand(batch, -1),
            candidate_vecs.unsqueeze(0).expand(batch, -1, -1),
            calibrated=calibrated,
        )

    @staticmethod
    def answers_from_logits(question: Question, logits: torch.Tensor) -> list[Answer]:
        """Turn ``[B, n]`` logits into one typed answer per batch row."""
        labels = question.labels()
        probs = logits.float().softmax(dim=-1).detach().cpu()
        out: list[Answer] = []
        for row in probs:
            table = {label: float(p) for label, p in zip(labels, row, strict=True)}
            if isinstance(question, Boolean):
                out.append(BooleanAnswer(probability=table["true"], probabilities=table))
            elif isinstance(question, Score):
                levels = torch.arange(len(labels), dtype=torch.float32)
                out.append(
                    ScoreAnswer(
                        value=float((row * levels).sum()),
                        level=labels[int(row.argmax())],
                        probabilities=table,
                    )
                )
            else:
                out.append(ChoiceAnswer(value=labels[int(row.argmax())], probabilities=table))
        return out

    # -- the stateless path --------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        state: str | list[str],
        questions: dict[str, Question],
    ) -> dict[str, Answer] | list[dict[str, Answer]]:
        """Answer typed questions about a state. No text is generated.

        Args:
            state: one string, or a batch of them.
            questions: named questions, all answered against the same state.

        Returns:
            A name-to-answer mapping, or a list of them when ``state`` is a list.
        """
        batched = isinstance(state, list)
        states = state if batched else [state]
        state_vecs = self.encode_texts(states)
        per_row: list[dict[str, Answer]] = [{} for _ in states]
        all_logits = self.score_many(state_vecs, questions)
        for name, question in questions.items():
            for i, answer in enumerate(
                self.answers_from_logits(question, all_logits[name])
            ):
                per_row[i][name] = answer
        return per_row if batched else per_row[0]

    # -- the stateful path ---------------------------------------------------

    def session(self) -> Session:
        """Open a session that carries state across observations."""
        return Session(self)


class Session:
    """A running judgment context backed by a constant-size recurrent state.

    Each :meth:`observe` folds text into the wavefield and is then forgotten as
    text; only the state remains. :meth:`ask` reads that state. This is what a
    stateless decision endpoint cannot do without being handed the whole
    transcript again on every call.
    """

    def __init__(self, model: Tacit) -> None:
        self.model = model
        self._encoder_state: EncoderState = model.encoder.init_state(1, model.device)
        self._state_vec = torch.zeros(1, model.d_model, device=model.device)
        self._bytes_seen = 0

    @property
    def bytes_seen(self) -> int:
        """Total bytes folded into the state since the last reset."""
        return self._bytes_seen

    def reset(self) -> Session:
        """Forget everything and start over."""
        self._encoder_state = self.model.encoder.init_state(1, self.model.device)
        self._state_vec = torch.zeros(1, self.model.d_model, device=self.model.device)
        self._bytes_seen = 0
        return self

    @torch.no_grad()
    def observe(self, text: str) -> Session:
        """Fold an observation into state with a resumable chunked scan."""
        raw = encode_bytes(text)
        if not raw:
            return self
        tokens = torch.tensor([raw], dtype=torch.long, device=self.model.device)
        self._state_vec, self._encoder_state = self.model.encoder.absorb(
            tokens, self._encoder_state
        )
        self._bytes_seen += len(raw)
        return self

    @torch.no_grad()
    def ask(self, questions: dict[str, Question]) -> dict[str, Answer]:
        """Answer typed questions against everything observed so far."""
        if self._bytes_seen == 0:
            raise RuntimeError("ask() before any observe(); the session has no state yet")
        logits = self.model.score_many(self._state_vec, questions)
        return {
            name: self.model.answers_from_logits(q, logits[name])[0]
            for name, q in questions.items()
        }

    def state_vector(self) -> torch.Tensor:
        """The current pooled state, ``[1, d_model]``. Useful for probing."""
        return self._state_vec.clone()
