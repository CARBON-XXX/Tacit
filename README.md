# Tacit

**A recurrent decision core that answers typed questions instead of writing text — and remembers.**

```python
from tacit import Tacit, Boolean, Choice, Score

model = Tacit()   # untrained; `python examples/triage.py` trains one in ~2 min

answers = model.evaluate(
    "Help! My payouts have been failing for 3 days.",
    {
        "urgent": Boolean("Does this need attention today?"),
        "team":   Choice("Who should handle this?", ["billing", "technical", "sales"]),
        "anger":  Score("How frustrated is the customer?", ["calm", "annoyed", "furious"]),
    },
)

answers["urgent"].probability   # float in [0, 1]
answers["team"].value           # one of the three strings you passed
answers["team"].probabilities   # {"billing": ..., "technical": ..., "sales": ...}
answers["anger"].value          # expected level, 0.0 - 2.0
```

No string comes back, so there is nothing to parse and no way to get a value
outside the set you asked about. The answer space is declared before the model
runs. There are no pretrained weights here — the numbers further down come from
models the example scripts train from scratch, which is also how you can check
them.

## Why this exists

Models that generate text are built for people to read. When your code needs a
judgment it can branch on, generation is the wrong interface — you end up
coercing a text generator into JSON and then parsing your way back out.
[TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
made the case for dropping generation entirely and returning calibrated
probabilities instead. That framing is theirs and it is correct.

Tacit explores a related typed-decision interface on a **recurrent** core.
Its session API carries internal state across observations. This is an
independent experiment, not a Jev reproduction or API-compatible replacement.

```python
session = model.session()
urgent = {"urgent": Boolean("Is the customer still blocked?")}

session.observe("customer: payouts have been failing, we are blocked")
session.ask(urgent)["urgent"].probability  # a model probability, not a guarantee

session.observe("agent: we pushed a fix, can you retry?")
session.observe("customer: it went through, thanks")
session.ask(urgent)["urgent"].probability
```

Each observation folds into a fixed-size complex state and is then forgotten as
text. The session does not retain or re-read the transcript. Its state is a lossy summary. Cost per fixed-size turn does not grow with the
conversation, and no cache grows with context.

## Experiments

- `python examples/triage.py`: train on synthetic support tickets, measure
  accuracy, and fit temperature on a separate calibration split. This is a toy
  task, not evidence of production readiness or general language understanding.
- `python examples/streaming_cost.py`: compare incremental state updates with
  re-encoding a growing transcript using the same Tacit model. This is not a
  Jev or Transformer benchmark. Fixed-size state does not imply lossless memory
  or equal decision quality. Timings depend on hardware and input length.
- `python examples/state_tracking.py`: parameter-budget comparison with phase
  collapse enabled and disabled. No robust advantage has been established;
  run multiple seeds before drawing conclusions.

Probabilities are normalized model scores. Calibration must be measured on
representative held-out data; temperature scaling does not guarantee it.
No complexity-class separation or architectural superiority is claimed.

## How it works

```
bytes ──► ByteEncoder ──────────────────────────────► state vector
             │
             └─ ResonantBlock × N
                  z_t = Λ·z_{t-1} + drive(x_t)        complex resonance
                  z_t = PhaseCollapse(z_t, x_t)       every K steps
                                                              │
questions ──► candidate texts ──► DecisionHead ◄──────────────┘
                                       │
                                       └─► calibrated probabilities
```

- **`ResonantBlock`** — the state is a complex wavefield `z`. Between collapse
  points the recurrence is linear with a known multiplier, so a whole chunk is
  solved in parallel; collapse acts on the carried state at chunk boundaries.
  `forward` and `step` are checked for numerical agreement in evaluation mode, which
  `tests/test_parity.py` enforces. That matters more than it sounds: a core
  whose decode path quietly differs from its training path costs weeks.

- **`PhaseCollapse`** — snaps the phase of the state onto a learned codebook of
  `K` points on the unit circle, leaving magnitude alone, gated by a learned
  routing confidence. Straight-through Gumbel-Softmax while training, argmax at
  inference, both reading the same normalised codebook.

- **`DecisionHead`** — scores candidates that are encoded from text, so options
  are defined at call time. A question can introduce labels the model has never
  seen. Temperature is fitted post-hoc on held-out data, so it changes
  confidence without changing which candidate wins.

- **`CyclicRegister`** — optional exact accumulator over Z/N for the counting
  a soft recurrence is bad at. Zero-initialised output, so it is exactly the
  identity at load.

- **Byte-level I/O** — vocabulary is 256 bytes plus padding. No tokenizer to
  ship, version, or mismatch.

## Install

```bash
git clone https://github.com/CARBON-XXX/Tacit
cd Tacit
pip install -e ".[dev]"
pytest
```

Requires Python 3.10+ and PyTorch 2.1+. Nothing else.

## Status

Alpha, and small. There are no pretrained weights — the examples train their own
in minutes, which is the point: you can verify every number here yourself rather
than take it on trust.

What would make this materially better, roughly in order: a fused streaming
kernel to kill the per-byte launch overhead; a real dataset instead of synthetic
tickets; and a scale at which the collapse ablation actually resolves.

## License

Apache 2.0.
