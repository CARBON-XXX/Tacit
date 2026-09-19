r"""Typed questions and typed answers.

A question declares its own answer space up front. Because the space is fixed
before the model runs, the answer cannot be malformed: there is no string to
parse, no schema to validate, and no way to get back a value outside the set
you asked about. What comes back is a distribution over that set.

Three shapes cover most judgments software needs:

``Boolean``
    Does this hold? Answered with a probability.
``Choice``
    Which of these? Answered with a distribution over your options.
``Score``
    Where on this scale? Answered with a distribution over ordered levels plus
    the expected level.

Every question also carries ``criteria`` — a short description per candidate.
These are read as text, so the option set is defined at call time rather than
baked into the model, but useful generalization to unseen options requires training and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Question",
    "Boolean",
    "Choice",
    "Score",
    "Answer",
    "BooleanAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "MAX_CANDIDATES",
]

MAX_CANDIDATES = 255


class Question:
    """Base class. Subclasses expose candidate labels and their descriptions."""

    prompt: str

    def labels(self) -> list[str]:
        raise NotImplementedError

    def descriptions(self) -> list[str]:
        raise NotImplementedError

    def candidate_texts(self) -> list[str]:
        """What the encoder actually reads for each candidate."""
        return [
            f"{self.prompt} | {label}: {desc}" if desc else f"{self.prompt} | {label}"
            for label, desc in zip(self.labels(), self.descriptions(), strict=True)
        ]

    def _check(self) -> None:
        n = len(self.labels())
        if n < 2:
            raise ValueError(f"{type(self).__name__} needs at least 2 candidates, got {n}")
        if n > MAX_CANDIDATES:
            raise ValueError(f"{type(self).__name__} allows at most {MAX_CANDIDATES}, got {n}")
        if len(set(self.labels())) != n:
            raise ValueError(f"{type(self).__name__} has duplicate labels")


@dataclass
class Boolean(Question):
    """A yes/no judgment.

    Args:
        prompt: what is being asked.
        when_true: what makes it true. Optional but sharpens the boundary.
        when_false: what makes it false.
    """

    prompt: str
    when_true: str = ""
    when_false: str = ""

    def __post_init__(self) -> None:
        self._check()

    def labels(self) -> list[str]:
        return ["true", "false"]

    def descriptions(self) -> list[str]:
        return [self.when_true, self.when_false]


@dataclass
class Choice(Question):
    """Pick one option from a set defined at call time.

    Args:
        prompt: what is being asked.
        options: either a list of labels, or a mapping from label to a
            description of when that option applies.
    """

    prompt: str
    options: list[str] | dict[str, str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.options, dict):
            self._labels = list(self.options.keys())
            self._descs = [self.options[k] for k in self._labels]
        else:
            self._labels = list(self.options)
            self._descs = [""] * len(self._labels)
        self._check()

    def labels(self) -> list[str]:
        return self._labels

    def descriptions(self) -> list[str]:
        return self._descs


@dataclass
class Score(Question):
    """Place the state on an ordered scale.

    Args:
        prompt: what is being asked.
        levels: ordered level names, lowest first.
    """

    prompt: str
    levels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._check()

    def labels(self) -> list[str]:
        return list(self.levels)

    def descriptions(self) -> list[str]:
        return [""] * len(self.levels)


class Answer:
    """Base class for typed answers."""

    probabilities: dict[str, float]

    @property
    def confidence(self) -> float:
        """Probability mass on the selected candidate."""
        return max(self.probabilities.values())


@dataclass
class BooleanAnswer(Answer):
    probability: float
    probabilities: dict[str, float]

    def __bool__(self) -> bool:
        return self.probability > 0.5

    def __repr__(self) -> str:
        return f"BooleanAnswer(p={self.probability:.3f})"


@dataclass
class ChoiceAnswer(Answer):
    value: str
    probabilities: dict[str, float]

    def __repr__(self) -> str:
        return f"ChoiceAnswer({self.value!r}, confidence={self.confidence:.3f})"


@dataclass
class ScoreAnswer(Answer):
    value: float
    level: str
    probabilities: dict[str, float]

    def __repr__(self) -> str:
        return f"ScoreAnswer({self.value:.2f}, level={self.level!r})"
