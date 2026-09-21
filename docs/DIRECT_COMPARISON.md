# Tacit, Jev and Laya: direct measurements

Measured on 20 September 2026. Tacit has a promising result on supervised NLI
support judgments, but **overall superiority to Jev and Laya is not established**.
All figures below come from our own runs on identical test requests, with model
variants and supervision identified. Published competitor scores are not substituted
for missing measurements.

**Follow-up:** the [expanded comparison](EXPANDED_COMPARISON.md) finds strong
input-format dependence. Putting both NLI statements in JSON changes Tacit
mixed/Laya general relation accuracy to 39.55%/87.50%. Read both layouts before
interpreting the earlier table below. Its cost figures are historical; the
current cumulative API cap is $5.00 and the expanded report includes later costs.

![Measured comparison](../results/jev-laya-tacit.svg)

## Accuracy and scope

| Task | Tacit | Jev 1.13.0 | Laya |
|---|---:|---:|---:|
| Four workflows, 400 cases / 2,000 decisions | 72.15% refined; 71.10% mixed | 73.45% | 76.80% typed; 36.10% general |
| MultiNLI three-way relation, 2,000 cases | 80.75% mixed | 82.65% | 64.65% general |
| MultiNLI support Boolean, same 2,000 cases | 89.25% mixed | 85.25% | 77.10% general |

Accuracy uses the argmax of the exposed probability vector in canonical label
order, including for score questions. The two NLI tasks are reported separately:
their average is not a three-way NLI accuracy. The Boolean target is true for
entailment and false for neutral/contradiction, so the tasks are correlated.

**Tacit refined** is a 149,212,929-parameter workflow specialist. It adds eight
training epochs from the initial forked checkpoint, retains epoch zero as a
candidate, and selects epoch six by validation soft NLL. Test accuracy rises
from 71.90% to 72.15%; this is a small exploratory improvement.

**Tacit mixed** uses the same parameter count and starts from the refined model.
It trains on 24,000 human-labeled NLI examples plus the original 900 workflow
training cases replayed four times per epoch. Each example can have different
instructions and option counts. The objective is decision cross entropy, Brier
loss and score ordinal loss; there is no language-model head or generation loop.
Two epochs took about 33.7 minutes on GB10. Mean validation NLL across NLI and
workflows selects epoch one, then a separate calibration partition fits temperatures.
Workflow SDK accuracy falls to 71.10%, so this model does not replace the best
workflow specialist. Batch evaluation was 71.15%; three borderline predictions
differ between BF16 batch and SDK execution. NLI predicted labels agree.

**Laya** has separate released general and typed-workflow checkpoints. We keep
them separate and use their shipped inference code and temperatures. We do not
locally fine-tune Laya. **Jev** is the version-pinned official API, with no local
training or calibration. Tacit's task supervision and format familiarity are
advantages specific to this experiment, not evidence of arbitrary-task generality.

## Data, separation and reproducibility

Workflow data: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
revision `ea9306458d6e9563628369a3d1e72e362fb381d2`. Four workflows, five typed
questions per case. Its gold distributions come from a synthetic teacher, so
accuracy measures teacher agreement. Tacit's split is 900 train / 148 validation /
152 calibration / 400 test, with no shared IDs or identical states across splits.

NLI data: [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli),
revision `da70db2af9d09693783c3320c4249840212ee221`. Test is a frozen hash-selected
sample of 1,000 official matched and 1,000 mismatched validation examples; it is
not the full official evaluation or the Laya authors' XNLI benchmark. Training,
validation and calibration are premise-disjoint; all hypotheses for a premise
stay together. We exclude 365 source training rows sharing a premise with either
official evaluation split. NLI development partitions contain 24,000 training,
1,002 validation and 1,002 calibration examples. Official test labels are human
annotations. Public `pairID` values are not unique, so pinned source-row offsets
are part of the reproducible example IDs.

Every model receives the same premise as `state`, with the hypothesis inside
both question instructions. IDs, genre, labels and parses are excluded from
model input. An input audit found **no Laya truncation on these 4,000 NLI decisions**.
On workflows, Laya's shipped 512-token policy truncates state for 34 of 2,000
decisions. Tacit rejects oversize inputs; all test requests completed. Alternative
NLI prompt layouts, multiple training seeds, multilingual use and a fresh untouched
final test remain unmeasured.

| NLI three-way split | Tacit mixed | Jev | Laya general |
|---|---:|---:|---:|
| Matched, 1,000 cases | 81.60% | 82.30% | 64.10% |
| Mismatched, 1,000 cases | 79.90% | 83.00% | 65.20% |

## Paired uncertainty

10,000 bootstrap resamples, seed 2026. Workflow questions remain grouped by case.
NLI hypotheses sharing a premise remain together, yielding 1,797 premise clusters,
stratified by genre. Intervals describe these fixed checkpoints and test samples;
they exclude training-seed uncertainty and multiple-comparison correction.

| Tacit minus reference | Difference | 95% interval |
|---|---:|---:|
| Workflow refined minus Jev | -1.30 pp | [-3.65, +1.00] |
| Workflow refined minus Laya typed | -4.65 pp | [-6.75, -2.60] |
| NLI relation mixed minus Jev | -1.90 pp | [-4.05, +0.20] |
| NLI relation mixed minus Laya general | +16.10 pp | [+13.55, +18.55] |
| NLI support mixed minus Jev | +4.00 pp | [+2.19, +5.78] |
| NLI support mixed minus Laya general | +12.15 pp | [+10.02, +14.30] |

The support-judgment gain is useful evidence for this supervised task. It does not
erase the three-way relation gap or the workflow specialist gap.

## Probability quality

Brier is the sum of squared probability errors, not the class-averaged variant.
ECE has 15 equal-width confidence bins. NLL uses a shared probability floor of
1e-12. Workflow targets are soft teacher distributions; NLI targets are one-hot
human labels, so Brier/NLL magnitudes should be compared within each task.

| Task / model | Brier ↓ | ECE ↓ | NLL ↓ |
|---|---:|---:|---:|
| Workflows / Tacit refined | 0.0871 | 0.1285 | 0.9294 |
| Workflows / Jev | 0.1478 | 0.0367 | 2.2831 |
| Workflows / Laya typed | 0.0615 | 0.2152 | 0.8844 |
| NLI relation / Tacit mixed | 0.2663 | 0.0228 | 0.4722 |
| NLI relation / Jev | 0.2510 | 0.0125 | 0.4490 |
| NLI relation / Laya general | 0.4643 | 0.0722 | 0.7663 |
| NLI support / Tacit mixed | 0.1556 | 0.0193 | 0.2582 |
| NLI support / Jev | 0.1978 | 0.0276 | 0.3126 |
| NLI support / Laya general | 0.3133 | 0.0266 | 0.4788 |

Jev exposes probabilities at roughly two-decimal precision. We preserve raw
responses and normalize only mass errors explainable by that rounding. Returned
zeros increase soft-target NLL substantially; this measures the public API's
exposed output and does not establish its internal full-precision NLL.
Laya's SDK rounds probabilities to four decimals.

Jev's `choice` field differs from exposed-probability argmax on four workflow
questions (no net accuracy change) and one NLI question. Scoring its returned NLI
`choice` instead gives **82.70%**, versus the common argmax protocol's **82.65%**.
Both are recorded in the interface audit; the table uses the common protocol.

## Efficiency and the $0.50 API budget

The refined workflow run randomizes case and local model order, warms schemas,
synchronizes CUDA, and includes input preparation and output construction.
Both local models use BF16 on the same NVIDIA GB10 with no concurrent GPU training.

| Model | Full-request p50 | p95 |
|---|---:|---:|
| Tacit refined | 17.74 ms | 22.72 ms |
| Laya typed | 64.75 ms | 87.59 ms |

Tacit is 3.65x faster in this run and has 2.82x fewer parameters, while its accuracy
is lower. Earlier paired timings were 32.53/94.43 ms: absolute latency varies
between runs. Do not interpret this as a quality-adjusted cost advantage. GPU
purchase, energy and amortized training cost are not measured.

Jev calls use [the official endpoint](https://docs.typesafe.ai/api), pinned to
`jev-1.13.0`. At the [verified price](https://docs.typesafe.ai/models) of $0.042
per million input tokens and free output:

- 400 workflow requests: $0.015885912.
- 2,000 NLI requests: $0.037985388.
- Total successful requests: **$0.053871300**, computed from reported usage.
- Budget committed including the earlier failed Gateway reservation: **$0.061935300**,
  below the cumulative **$0.50** cap. The $0.008064 reservation remains conservative;
  Gateway's usage endpoint reported no increment after that failed call.

These are usage-based charge calculations, not a reconciled invoice. A durable
ledger reserves before sending, survives restart, prevents duplicate paid attempts
and retains unknown charges. Up to eight NLI requests run concurrently in bounded
waves. Errors stop new waves; there are no automatic retries or fallback models.
Jev latency includes network and remote serving and is not a local GPU comparison.
Credentials and the private cumulative ledger stay outside the repository.

## Artifacts and commands

The [comparison snapshot](../results/comparison-snapshot/) includes full-precision
local predictions, raw exposed Jev answers/usage, source revisions, split hashes,
training histories, checkpoint hashes, input audit and paired intervals. The
previous [workflow snapshot](../results/typed-snapshot/) remains available.
Model checkpoint binaries are local artifacts; they are not yet hosted in this repo.

Pinned Laya code: `NandhaKishorM/laya` at `6a5819129eb220570792e417e49723d697efd76f`.
General weights: `convaiinnovations/laya` at
`1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
Typed weights: `convaiinnovations/laya-typed-decisions` at
`f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`.
Tacit's encoder: `answerdotai/ModernBERT-base` at
`8949b909ec900327062f0ebf497f51aef5e6f0c8`.

After the initial workflow training in [COMPETITION.md](COMPETITION.md):

```bash
pip install -e '.[dev,bench,semantic]'
python benchmarks/fetch_general.py
python benchmarks/general_data.py
python benchmarks/train_semantic.py --architecture forked \
  --init-run results/typed/forked-seed0 --epochs 8 --batch-size 4 \
  --encoder-lr 1e-5 --head-lr 1e-4 --output results/typed/forked-refine-seed0
python benchmarks/train_general.py
python benchmarks/evaluate_general.py
python benchmarks/evaluate_general.py --model tacit \
  --checkpoint results/general/mixed-seed0 --output results/general/tacit-sdk-test.json
python benchmarks/compare_typed.py --run results/typed/forked-refine-seed0 \
  --output results/typed/paired-refined-laya.json
python benchmarks/audit_general.py
# Requires TYPESAFE_API_KEY or ~/.config/tacit/typesafe.key; shares the $0.50 ledger.
python benchmarks/jev_official.py --limit 400
python benchmarks/jev_official.py --suite nli --limit 2000 --concurrency 8
python benchmarks/publish_comparison.py
python benchmarks/plot_comparison.py
```

Original Tacit code is MIT. ModernBERT-derived weights retain Apache 2.0 terms;
MultiNLI retains the source-specific licenses and attribution in its dataset card.
Do not relabel external model/data licenses as MIT.
