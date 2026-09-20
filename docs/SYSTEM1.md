# Tacit: a decision core, without language generation

The development target is a small, trainable System 1 component for continuous
classification, routing, ordinal scoring and state-dependent decisions. We are
not pursuing chat, prose generation, chain-of-thought output or next-token loss.
This release is a measured engineering foundation, not evidence that Tacit has
surpassed Jev or Laya in general decision quality.

## Two input paths, one decision objective

`Tacit` accepts UTF-8 text and runtime-defined `Boolean`, `Choice`, and `Score`
questions. It shares the state encoding across questions, batches the decision
head, and independently encodes each candidate. Candidate order only permutes
its scores, up to floating-point error; that structural property does not prove
it understands unseen candidates. Question encodings use a bounded LRU cache
(128 questions plus 8 prepared head batches by default), cleared on weight
loading and device changes. Set either cache limit to zero to disable it.

`SignalTacit` accepts only numeric events and an observation mask. Field names
and output labels are metadata; there is no text processing in this path. The
model learns a **fixed** decision schema. Each event updates explicit latest-value
registers and a recurrent state, and emits all configured decisions together.
Zero and missing are different. An unobserved field retains its previous value.
Explicit registers preserve those measurements; the recurrent state remains a
lossy summary, without a lossless-history guarantee.

```python
import torch
from tacit import Boolean, Choice, Score, SignalConfig, SignalTacit

model = SignalTacit(
    SignalConfig(features=("load", "error_rate", "queue_depth")),
    {
        "alert": Boolean("alert"),
        "route": Choice("route", ["normal", "retry", "escalate"]),
        "severity": Score("severity", ["low", "medium", "high"]),
    },
).eval()  # untrained: do not use these predictions as an operational policy

events = torch.tensor([[[0.8, 0.02, 12.0], [0.0, 0.15, 0.0]]])
observed = torch.tensor([[[True, True, True], [False, True, False]]])
answers, state = model.evaluate(events, observed)
# Last stored measurement is now [0.8, 0.15, 12.0].
# Pass state to the next call. Normalize features exactly as during training.
```

Train with raw logits from `model(events, observed, calibrated=False)`, using
`decision_loss`. It accepts hard classes or soft teacher distributions, with
optional Brier loss and ordinal ranked probability score. These are supervised
proper scoring objectives; this implementation does **not** claim RLCD or new
reinforcement learning. Detach carried state between truncated training chunks.
Streams in a batch must have equal event counts; the observation mask is a field
mask, not padding. Reset state between unrelated streams/windows.

## Streaming implementation

`ResonantBlock.absorb` resumes at any absolute position, handles incomplete
chunks without padding the carried state, and collapses at the same boundaries
as byte-by-byte `step`. Projection, convolution and readout run per observation;
the sequential loop crosses chunk boundaries. This is ordinary PyTorch, not a
fused CUDA/Triton kernel. `step` is retained as a reference implementation.

Persistent inference tensors are copied into compact storage. Otherwise, a small
slice could secretly retain a whole observation allocation. Working memory still
scales with the current observation and batch. Training graphs also grow unless
explicitly detached. Numerical agreement is tested in evaluation mode; stochastic
Gumbel sampling during training need not match across chunk partitions.

## Calibrate, then decide whether to act

1. Train the model on the training split; select it on a separate validation split.
2. Fit one `TemperatureScaler` per decision schema on another split.
3. Freeze both; fit `ConformalPolicy` on a fourth, unused calibration split.
4. Measure set coverage, automation coverage, and accuracy among automated
   decisions on held-out test data. Empty or multi-label sets abstain.

Split-conformal prediction targets **marginal prediction-set coverage** under
exchangeability. It does not guarantee accuracy conditional on automation, joint
coverage across questions, or reliability under domain shift. In our HAR pilot,
overlapping sensor windows and held-out subjects violate the simple exchangeable
setup, so all coverage numbers are explicitly empirical. A threshold cannot repair
missing capability. See [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511).

## What we will compare against

| Reference | What its public evidence suggests | Tacit's testable target |
|---|---|---|
| [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Structured probabilistic decisions from unstructured state | Match quality on the same frozen text tasks before comparing cost/latency |
| [Laya](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) | Useful domain fine-tuning; published issues with high-cardinality options, option order and calibration | Stable candidate order, independent candidate budgets, explicit domain calibration |
| Small local supervised models | Strong, cheap baselines on constrained numeric tasks | Beat them on held-out quality or a measured memory/latency tradeoff |

Laya's report explicitly notes that its Jev figures came from other published
runs with different prompts and sample sizes. We do not turn those rows into a
direct ranking. Tacit's numeric sensor benchmark is also not a Jev/Laya task.

## Evidence gates for the next stage

- **Continuous decisions:** test delayed measurements, revisions, missing fields,
  long distractor histories and schema-valid but wrong outputs. Fixed memory alone
  is not a quality result.
- **Text decisions:** replace the synthetic ticket diagnostic with real labeled
  tasks; separate task and domain holdouts; train on decision labels/distributions.
  Removing generation does not remove the need for semantic training data.
- **Quality:** use at least three training seeds, validation-only model selection,
  a competitive small baseline, and a test set that was not used to choose the
  architecture. Treat this public HAR set as a pilot, not a reusable private gate.
- **Reliability:** report NLL, Brier, ECE, rejection/coverage and conditional errors;
  stress negation, conflicting updates, new domains and option permutations.
- **Efficiency:** compare identical inputs, batch sizes, hardware, precision,
  preprocessing and quality. Measure CPU as well as GPU, and actual resource cost.

Current measurements and reproducible commands: [RESULTS.md](RESULTS.md).
Tacit code remains MIT. Benchmark datasets retain their own licenses.
