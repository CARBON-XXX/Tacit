# Tacit 0.2 measured results

Measured on NVIDIA GB10, PyTorch 2.11.0+cu130, float32, on 2026-09-19.

Tacit now has a language-free numeric decision path and a faster resumable text core. The real-data pilot beats a small window-statistics MLP, but does not beat the temporal CNN in accuracy or speed. It does not establish superiority to Jev or Laya.

## Real numeric decisions: UCI HAR

[UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) contains video-labeled smartphone sensor activity data. We use 9 preprocessed raw signal channels, 128 samples/window, and six activity classes. No text, tokenizer, pretrained encoder or language generation is involved.

Training: 5,066 windows / 15 subjects; model selection: 801 / 2; temperature fitting: 758 / 2; prediction-set calibration: 727 / 2; official test: 2,947 / 9. Subjects are disjoint across all five sets. Normalization uses only training data. All models train 20 epochs with the same optimizer, batch size 64, and CE + 0.1 Brier loss. The lowest validation-NLL checkpoint is selected separately per run. Seeds: 0, 1, 2.

| Model | Parameters | Test accuracy, mean ± sample SD | Macro F1 | NLL, calibrated | Brier sum, calibrated | ECE, calibrated | GPU p50, median across seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tacit signals | 115,542 | 88.77% ± 1.12 pp | 0.888 | 0.364 | 0.174 | 3.94% | 14.080 ms |
| Window MLP | 22,022 | 82.45% ± 0.69 pp | 0.822 | 0.450 | 0.247 | 2.01% | 0.066 ms |
| Temporal CNN | 44,422 | 89.81% ± 1.17 pp | 0.898 | 0.362 | 0.160 | 6.07% | 0.096 ms |

Latency is a warm batch-one forward pass over a full 128-sample window, including each baseline's window feature calculation, excluding external input normalization, transfers, Python answer formatting and data loading. 100 trials per seed; raw per-run p50/p95 are in the JSON files. Tiny-kernel timings fluctuate on this desktop GPU. These latency differences are material, not an efficiency win for the current Tacit signal implementation.

Tacit uses 2,093 bytes of persistent inference state tensors per stream in this configuration (not total process/GPU memory). State resets between windows: this benchmark does not measure long-history retention. The longest chunk-partition logit difference across the three trained checkpoints is 2.38e-06.

### Confidence and abstention

| Model | Raw ECE | Fitted ECE | Empirical set coverage | Automated fraction | Accuracy among automated decisions |
|---|---:|---:|---:|---:|---:|
| Tacit signals | 6.43% | 3.94% | 87.25% | 97.17% | 89.83% |
| Window MLP | 4.16% | 2.01% | 91.40% | 76.17% | 89.98% |
| Temporal CNN | 5.06% | 6.07% | 83.38% | 88.70% | 94.01% |

Values are means across seeds. The nominal prediction-set target is 90%, but held-out subjects and overlapping windows are not exchangeable. Tacit's empirical coverage falls short. Neither calibration nor abstention is a deployment guarantee. Trial variation over three seeds is not a population confidence interval.

## Same-model runtime improvements

The byte-core comparison uses the same 1,602,265-parameter, randomly initialized Tacit model for both paths. Each update resumes from absolute byte position 7. It checks execution and numerical parity, not learned decision accuracy. Warm, paired, seeded AB/BA order; 30 trials/path/input length.

| Incremental bytes | Byte steps p50 | Chunked p50 | Speedup | Max hidden difference |
|---|---:|---:|---:|---:|
| 32 | 20.31 ms | 2.52 ms | 8.05× | 6.56e-07 |
| 128 | 80.50 ms | 7.14 ms | 11.27× | 5.96e-07 |
| 512 | 324.06 ms | 25.55 ms | 12.68× | 8.05e-07 |

The byte core retains 9,216 bytes of recurrent/convolution storage here, plus the 768-byte pooled vector. This excludes weights, current-observation working memory and bounded question caches.

### Repeated question schemas

State is already encoded; question and prepared-head caches are warm. 200 paired trials/path. These are head-only costs, not full request latencies.

| Questions | Serial p50 | Batched p50 | Max logit difference |
|---|---:|---:|---:|
| 1 | 0.072 ms | 0.072 ms | 0 |
| 3 | 0.210 ms | 0.092 ms | 3.73e-08 |
| 14 | 0.977 ms | 0.191 ms | 7.45e-08 |
| 32 | 2.197 ms | 0.223 ms | 8.94e-08 |

The 255-candidate reversal probe returns finite logits and max aligned difference 0. This is a structural test, with no accuracy claim. The earlier pre-cache runtime run is retained as `results/runtime-before-prepared-cache.json`; final figures use the updated cache implementation and paired timing protocol.

## Reproduce

```bash
pip install -e ".[dev,bench]"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python benchmarks/har.py --seeds 0 1 2
# Run runtime measurements after training finishes:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python benchmarks/runtime.py
python benchmarks/summarize.py
python examples/activity.py
pytest -q
```

`results/har/manifest.json` records the source URL, archive SHA-256, subjects, normalization and labels. Per-run JSON includes learning curves, confusion matrices, selected epochs and calibration. Local `.pt` and prediction `.npz` files are produced by the script and excluded from Git. All nine runs are reported; no successful seed was singled out.

The temporal CNN baseline was added after the Tacit/MLP pilot; Tacit's architecture, training budget and subject splits were not retuned using its test outcomes. These are exploratory public-data experiments, not a private benchmark or a parameter/compute-matched architecture ablation.

Dataset attribution: Reyes-Ortiz, Anguita, Ghio, Oneto and Parra (2013), Human Activity Recognition Using Smartphones, UCI, [DOI 10.24432/C54S4K](https://doi.org/10.24432/C54S4K), CC BY 4.0. Tacit source code is MIT.
