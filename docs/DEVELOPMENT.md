# Development measurements and next training corpus

The expanded pure Tacit run is still in progress. The measurements below are
development references, not final-test evidence of a stronger Tacit checkpoint.
The [completed test comparisons](EXPANDED_COMPARISON.md) remain the published
comparison until a replacement finishes selection, calibration and evaluation.

## Complete validation references

Both references answered the same 4,052 cases and 6,648 decisions. Predictions
were independently rescored against the pinned validation labels, with complete
case/question coverage and no duplicate decisions. Accuracy uses the exposed
probability argmax. Laya uses its single released general English checkpoint
across every row; its separate workflow specialist is not included here.

| Development track | Decisions | Jev 1.13.0 | Laya general |
|---|---:|---:|---:|
| NLI relation, premise/instruction layout | 1,002 | 81.24% | 63.77% |
| NLI support Boolean, same layout | 1,002 | 87.92% | 78.34% |
| NLI relation, JSON layout | 1,002 | 86.53% | 91.12% |
| NLI support Boolean, JSON layout | 1,002 | 92.61% | 94.21% |
| News topic | 1,000 | 86.10% | 94.30% |
| Emotion | 900 | 58.00% | 58.44% |
| Four workflows | 740 | 75.27% | 38.24% |

The two NLI layouts share examples. These are Tacit's local development
partitions; they are not guaranteed unseen by the released reference models.
For example, Laya's [pinned benchmark description](https://github.com/NandhaKishorM/laya/blob/6a5819129eb220570792e417e49723d697efd76f/README.md)
identifies AG News as part of its training mix. These scores guide development
and cannot substitute for the official test partitions or the fresh final gate.
No reference probabilities are used as Tacit training targets. Local evaluation
ran concurrently with GPU training, so no latency comparison is claimed.

The [snapshot](../results/validation-snapshot/summary.json) contains all exposed
predictions, whitelisted Jev responses, model provenance, NLL/Brier/calibration
metrics and checksums. Jev's rounded probabilities and exact zeros affect NLL
under the shared probability floor.

This validation run used 1,909,625 reported input tokens, costing $0.080204250
for 4,052 successful requests. One interrupted request was explicitly recovered;
its original $0.005376 reservation remains. Across all runs, 12,452 successful
requests cost $0.247844814; the conservative committed total including three
unresolved attempts is $0.266660814, under the cumulative $5.00 cap. The ledger
was retained across interruptions and was never reset.

## Larger training corpus, unchanged holdouts

The next corpus has **524,874 training cases**: 390,000 NLI, 117,990 news,
15,984 emotion and 900 workflow cases before replay. It preserves the exact
validation, calibration and test bytes from the current expanded corpus.
An [independent streaming audit](../results/general-v3-audit.json) verifies input
hashes, case counts, and zero training ID/input-group overlap with those
partitions and the 6,004-case fresh final gate. The gate has been checked for
input separation and remains **unscored**. This audit does not prove absence
from third-party pretraining data.

The larger corpus is prepared; it has **not yet trained a Tacit checkpoint**.
The current run continues on its original 144,886-case corpus and fixed
selection rule. Preparing more data is not an accuracy improvement by itself.

```bash
python benchmarks/general_data.py --train-cases 390000 --output .cache/general-nli390k
python benchmarks/general_v3_data.py
python benchmarks/audit_expansion.py
python benchmarks/jev_validation.py
python benchmarks/evaluate_general.py --model laya --corpus .cache/general-v2 \
  --split validation --accuracy-only --output results/general/laya-validation.json
python benchmarks/publish_validation.py
```

Reproduction requires the pinned source datasets and earlier corpus preparation.
Jev collection reuses durable responses and the cumulative private budget ledger.
Recovery of an unknown request requires the explicit `--retry-unresolved` flag
and retains the earlier reservation. Do not delete the ledger to rerun a test.
Original code is MIT; the dataset and encoder terms described in
[PURE_TACIT.md](PURE_TACIT.md) continue to apply.

## Numerical-input ablation: not adopted

A controlled, single-seed pilot added 50,368 neural numeric-embedding parameters
to the refined ForkedTacit model. Both arms used the same initialization, four
epochs, a frozen encoder, and the same train/validation/calibration partitions.
The control zeroed the numeric features. Selection minimized validation soft
NLL, including epoch zero, before scoring the test set.

The numeric arm selected **epoch zero**: none of its trained checkpoints
improved validation NLL. The zero-feature control selected epoch two. Their
test accuracies were 72.10% and 72.05%; that difference is not evidence of a
numeric-feature improvement. The [full metric summary](../results/numeric-ablation.json)
records the validation history and artifact hashes. The experimental code and
checkpoints are preserved in a local research archive; the default SDK has no
numeric-embedding modification. This negative pilot does not rule out other
numerical representations or training regimes.

The main run has resumed after interruptions. Its recorded optimization time
excludes discarded work since an earlier durable checkpoint and preprocessing;
it must not be presented as total research wall time or total training cost.
