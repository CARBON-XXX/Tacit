"""Tacit — a recurrent decision core that answers typed questions instead of writing.

Quick start::

    from tacit import Tacit, Boolean, Choice, Score

    model = Tacit()

    answers = model.evaluate(
        "Help! My payouts have been failing for 3 days.",
        {
            "urgent": Boolean("Is this time-sensitive?"),
            "team": Choice("Who should handle this?", ["billing", "technical", "sales"]),
            "anger": Score("How frustrated is the customer?", ["calm", "annoyed", "furious"]),
        },
    )
    answers["urgent"].probability      # 0.0 - 1.0
    answers["team"].value              # one of the options you passed

For a judgment that depends on history, open a session; observations fold into
a constant-size state rather than being re-read::

    s = model.session()
    s.observe("customer: payouts have been failing")
    s.observe("agent: can you retry now?")
    s.observe("customer: it worked")
    s.ask({"resolved": Boolean("Is the issue resolved?")})
"""

from tacit.collapse import CollapseSchedule, PhaseCollapse
from tacit.encoder import PAD_ID, VOCAB_SIZE, ByteEncoder, EncoderConfig, encode_bytes
from tacit.engine import Session, Tacit, TacitConfig
from tacit.heads import DecisionHead, expected_calibration_error, fit_temperature
from tacit.learning import ConformalPolicy, PredictionSet, TemperatureScaler, decision_loss
from tacit.questions import (
    Answer,
    Boolean,
    BooleanAnswer,
    Choice,
    ChoiceAnswer,
    Question,
    Score,
    ScoreAnswer,
)
from tacit.register import CyclicRegister
from tacit.resonance import ResonantBlock
from tacit.signals import SignalConfig, SignalState, SignalTacit

__version__ = "0.2.0"

__all__ = [
    # interface
    "Tacit",
    "TacitConfig",
    "Session",
    "SignalTacit",
    "SignalConfig",
    "SignalState",
    "decision_loss",
    "TemperatureScaler",
    "ConformalPolicy",
    "PredictionSet",
    "Boolean",
    "Choice",
    "Score",
    "Question",
    "Answer",
    "BooleanAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    # core
    "ResonantBlock",
    "PhaseCollapse",
    "CollapseSchedule",
    "CyclicRegister",
    "ByteEncoder",
    "EncoderConfig",
    "DecisionHead",
    # utilities
    "fit_temperature",
    "expected_calibration_error",
    "encode_bytes",
    "PAD_ID",
    "VOCAB_SIZE",
    "__version__",
]
