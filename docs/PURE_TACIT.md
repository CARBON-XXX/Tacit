# Pure Tacit development rules

Tacit's improvement target is a single neural model that returns typed decisions
and probabilities. Hybrid systems are excluded from this development direction.

- One Tacit checkpoint serves every supported task in a reported run.
- No tree model, score blending, ensemble, external-model fallback or per-task
  checkpoint routing contributes to inference.
- No language-model head, next-token generation or generated-text parsing is used.
- Input encoding, shared state/candidate attention, neural decision heads and
  calibration fitted on a separate calibration partition remain part of Tacit.
- Jev and Laya are evaluation references; their inference outputs are not inputs
  to Tacit. Workflow supervision remains the disclosed synthetic teacher dataset.

The current semantic model is the 149,212,929-parameter `ForkedTacit`: a
ModernBERT-derived input encoder with isolated candidate branches and a learned
decision readout. It is a single neural network, distinct from Tacit's recurrent
streaming experiments. Pretrained encoder provenance remains disclosed; calling
the system pure Tacit does not mean the encoder was pretrained from scratch.

The expanded training run uses 144,886 training cases before workflow
replay: NLI, four structured workflows, news topics and emotion labels. It trains
both premise/instruction and equivalent JSON-state layouts. A single checkpoint
is selected using seven validation tracks. Improvement on new tasks cannot hide
more than 0.01 NLL regression on the original NLI relation, NLI support or workflow
validation tasks. The initial checkpoint remains a candidate. Test scores are
computed only after training, checkpoint selection and separate calibration.

Intermediate validation scores are not test results. A checkpoint that fails
the fixed regression rule is not a successful replacement, even if its mean
validation score improves. All supported tracks must be reported together for
the selected checkpoint; separate checkpoints' best scores cannot be presented
as one model's capability.

The local tree and neural/tree fusion exploration has been removed from the
repository and retained only in an excluded experiment archive. Its 75.65%
workflow result is **not a pure Tacit score** and is not part of the improvement
claim. The published pure-model workflow reference remains 72.15% for the refined
checkpoint, or 71.10% for the mixed checkpoint, as recorded in the
[direct comparison](DIRECT_COMPARISON.md).

## Reproduce the expanded research run

First reproduce the checkpoints in the direct comparison, then:

```bash
python benchmarks/fetch_general.py --auxiliary
python benchmarks/general_data.py --train-cases 96000 --output .cache/general-nli96k
python benchmarks/general_v2_data.py
python benchmarks/train_general_v2.py
```

`train_general_v2.py --resume` restores model, optimizer, scheduler and random
states from an atomic checkpoint. Resume requires matching arguments, source
hashes and corpus provenance. `evaluate_general.py --accuracy-only` allows
accuracy comparisons while other GPU work is running, without claiming latency.

The code remains MIT. Model and data terms are separate: ModernBERT-derived
weights retain Apache 2.0 obligations; MultiNLI retains source-specific terms;
AG News retains its source terms; the emotion dataset has machine-generated
labels and research/education restrictions. The expanded run is a research
artifact, not an unrestricted commercial weight release.
