# Active research objective: surpass Jev and Laya

The objective remains **超过 Jev/Laya**. The sensor pilot and faster byte scan do
not satisfy it. This document tracks work toward a competitive, generation-free
System 1 model, rather than redefining success around a convenient local result.

## Evidence required before claiming the objective

1. **Decision capability:** cover Boolean judgments, classification/routing,
   ordinal scoring, relevance/verification and state changes. Include real
   labeled tasks and independently checkable cases, not only teacher agreement.
2. **Generality:** explicitly separate workflow-specific fine-tuning from unseen
   domains, instructions and candidate schemas. A specialist beating a published
   generalist number on its training domain is insufficient.
3. **Reliability:** measure distributions (NLL, Brier, calibration), errors among
   automated decisions, ambiguity, negation/contradictions and option-order
   sensitivity. Normalized outputs and schema validity do not prove correctness.
4. **Efficiency:** compare complete requests with the same inputs, question counts,
   cache states, hardware/precision where possible, and actual preparation costs.
   Do not compare Tacit pretokenized GPU execution with a competitor's SDK/network
   time and call the ratio an end-to-end win.
5. **Evidence strength:** retain all runs, freeze test data, use multiple training
   seeds and paired comparisons, publish the model/configuration and harness.
   Do not declare a broad win from one selected seed or one favorable task.

The current status is **not achieved**. We will use the available public Jev
measurements with their provenance and limits, as requested; no new Jev calls
have been made. Local Laya checkpoints can be evaluated directly.

## First direct workflow benchmark

Source: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
revision `ea9306458d6e9563628369a3d1e72e362fb381d2`, Apache 2.0. It contains 1,200
training and 400 test cases, five questions/case across four workflows. Models
receive only `state` and `questions`; latent factors, identifiers, labels and
teacher agreement fields are excluded from inputs.

The data author reports Jev 1.13.0 at 0.727 accuracy, 0.148 Brier and 0.144 ECE.
These are the author's API measurements, not our execution. Gold is the average
of three samples from a small teacher endpoint, so this is teacher agreement,
not independent proof of decision correctness. Raw Jev responses and the full
metric implementation are not in the downloaded dataset snapshot.

We reproduce the released specialist Laya checkpoint locally:

- Code: `NandhaKishorM/laya`, revision `6a5819129eb220570792e417e49723d697efd76f`.
- Weights: `convaiinnovations/laya-typed-decisions`, revision
  `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`.
- All 400 cases / 2,000 decisions: **76.8% accuracy**, **0.06148 Brier**,
  **0.21517 ECE**. The first isolated run took 70.41 ms p50; the later randomized
  paired run took 94.43 ms. Use the paired run when comparing with Tacit.
- No local fine-tuning. Its shipped temperature fitting uses training examples
  according to the published notebook; we report it as shipped.

For Tacit, deterministic case-level partitions of the public training set are
900 training / 148 validation / 152 temperature fitting. No case's questions are
split across partitions. Validation selects the checkpoint by soft NLL. Test is
scored only after each run's training/selection/calibration. Repeated public-test
experiments remain exploratory and will require a new untouched final gate.

## Experimental semantic paths

`SemanticTacit` uses an optional pretrained ModernBERT-base input encoder and
independent candidate queries over the final shared token representation. It has
no language-model output head. This first attempt scored 62.7% test accuracy after
eight epochs: clearly below Laya. Its pretokenized forward timing is not directly
comparable with Laya's full SDK time. The result is retained as a failed candidate.

`ForkedTacit` tests deeper interaction: every candidate reads shared state at
every encoder layer, while masks prevent state-to-candidate and cross-candidate
information flow. Candidate positions restart after the state, making option
permutation equivariant in exact arithmetic. Mask visibility, padding isolation,
batch independence and gradients have explicit tests. It remains an experiment;
neither its generalization nor a competitive speed/quality result is assumed.

Both paths are stateless text encoders. They do not inherit the original byte
core's constant-memory stream guarantee. Tacit's numeric and recurrent modules
remain separate capabilities. ModernBERT weights retain Apache 2.0 terms;
Tacit's original implementation remains MIT.

## Recorded workflow results

All local rows use the same 400 test cases and 2,000 decisions. Tacit checkpoints
are trained only on the 900-case training partition. These are **specialists**,
including the public Laya checkpoint fine-tuned on these workflows. The public
Jev number above is a different, author-reported generalist protocol.

| Local model | Parameters | Accuracy | Brier | ECE | Expected score MAE |
|---|---:|---:|---:|---:|---:|
| Tacit late interaction | 151,057,153 | 62.70% | 0.1229 | 0.1041 | 0.4241 |
| Tacit structured + word features | 497,607 | 68.45% | 0.1129 | 0.1212 | 0.3417 |
| Tacit structured, word features off | 235,463 | 62.90% | 0.1403 | 0.1086 | 0.4007 |
| Tacit forked shared state | 149,212,929 | 71.90% | 0.0904 | 0.1352 | 0.3327 |
| Laya typed decisions | 421,293,827 | 76.80% | 0.0615 | 0.2152 | 0.2426 |

Distribution metrics use Tacit's held-out temperature fitting and Laya's shipped
temperatures. Brier is the sum of squared errors against the soft teacher target.
ECE uses 15 bins against hard teacher labels. Lower ECE alone does not establish
a better decision model; Laya wins accuracy, Brier, NLL and score MAE here.
`soft_accuracy` in the local harness means the dot product of the predicted and
target distributions. `macro_f1_per_question` averages all declared classes in
each question, then questions; absent classes contribute zero. The dataset
author's metric implementation is unavailable, so similarly named Jev columns
must not be treated as verified identical formulas.

`StructuredTacit` is a diagnostic small-model path: numeric values and missingness,
categorical values, generic numeric pair comparisons and optional hashed word
features feed shared learned features and fixed trained heads. It has no pretrained
text encoder. List fields are aggregated without preserving event order. Word
features off still retains short categorical strings. Unseen schemas raise an
error; this is not an arbitrary-question semantic model.

The paired run randomizes both case order and model order, warms all four schemas,
and synchronizes the GPU around **complete requests**, including input preparation
and output construction. Both use BF16 on the same GB10, with no concurrent training.

| Model | Request p50 | Request p95 |
|---|---:|---:|
| Tacit forked | 32.53 ms | 85.93 ms |
| Laya typed | 94.43 ms | 198.64 ms |

Tacit's p50 is 2.90x faster in this protocol, while accuracy is 4.90 percentage
points lower. The workflow-stratified paired case bootstrap (10,000 resamples,
seed 2026) gives a 95% interval of **[-7.00, -2.85] percentage points** for Tacit
minus Laya. This quantifies case uncertainty for these fixed checkpoints, not
training-seed uncertainty. It supports a quality gap, not a broad win.

The first forked run selected its last epoch (8): validation soft NLL fell from
1.0846 to 0.9002. A second training stage starts from that checkpoint with a fresh
optimizer, eight additional epochs, encoder LR 1e-5 and head LR 1e-4. This decision
uses the validation trajectory; epoch zero remains eligible if further training
does not improve validation. No result from that stage is claimed before it ends.

Snapshot files preserve full-precision per-decision probabilities, source revisions,
split assignments and local checkpoint hashes. `data-audit.json` checks that all
four partitions have disjoint IDs and SHA-256 state hashes. The public test remains
exploratory across iterations; multiple seeds and unseen domains remain necessary.

## Reproduce the current work

Install the optional dependencies and fetch the exact public revisions. Run from
the repository root:

```bash
pip install -e '.[dev,bench,semantic]'
python benchmarks/fetch_typed.py
python benchmarks/audit_typed.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python benchmarks/laya_baseline.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/train_semantic.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/train_semantic.py \
  --architecture forked --batch-size 4 --output results/typed/forked-seed0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/train_structured.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/compare_typed.py
python benchmarks/summarize_typed.py
python benchmarks/publish_typed.py
```

Do not run GPU latency measurements concurrently with training. Checkpoint files
are local `.pt` artifacts; `manifest.json`, `history.json`, `result.json` and the
Laya response trace retain experiment provenance and predictions. Installations
without semantic extras can still use Tacit's recurrent and numeric modules.

After training, `tacit.agent.DecisionAgent("results/typed/forked-seed0", device="cuda")`
loads the local checkpoint. Its `predict(state, questions)` returns typed answers
and distributions. Oversize input raises rather than silently truncating evidence.
It has no autoregressive generation loop or language output head.

The tracked [snapshot](../results/typed-snapshot/) contains compact predictions
and checksums. Larger working artifacts stay under the ignored `results/typed/`
directory. Validation was performed with PyTorch 2.11.0+cu130 and Transformers
5.4.0; manifests record the exact runtime. Pretrained-derived semantic checkpoint
weights retain Apache 2.0 terms, while original Tacit source is MIT.
