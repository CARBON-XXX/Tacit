"""Render the report and standalone data figure from recorded measurements."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"tacit": "Tacit signals", "window_mlp": "Window MLP", "temporal_cnn": "Temporal CNN"}
COLORS = {"tacit": "#25aaec", "window_mlp": "#8995a6", "temporal_cnn": "#e9a95f"}


def main():
    runs = {
        kind: [
            json.loads((ROOT / f"results/har/{kind}-seed{i}.json").read_text()) for i in range(3)
        ]
        for kind in NAMES
    }
    runtime = json.loads((ROOT / "results/runtime.json").read_text())
    manifest = json.loads((ROOT / "results/har/manifest.json").read_text())

    def values(kind, *keys):
        out = []
        for row in runs[kind]:
            value = row
            for key in keys:
                value = value[key]
            out.append(value)
        return np.asarray(out)

    summary = {}
    for kind in NAMES:
        summary[kind] = {
            "parameters": runs[kind][0]["parameters"],
            "accuracy_mean": float(values(kind, "raw", "accuracy").mean()),
            "accuracy_std": float(values(kind, "raw", "accuracy").std(ddof=1)),
            "calibrated_ece_mean": float(values(kind, "calibrated", "ece_10_bins").mean()),
        }
    (ROOT / "results/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Tacit 0.2 measured results",
        "",
        "Measured on NVIDIA GB10, PyTorch " + manifest["torch"] + ", float32, on 2026-09-19.",
        "",
        "Tacit now has a language-free numeric decision path and a faster resumable text core. "
        "The real-data pilot beats a small window-statistics MLP, but does not beat the temporal "
        "CNN in accuracy or speed. It does not establish superiority to Jev or Laya.",
        "",
        "## Real numeric decisions: UCI HAR",
        "",
        "[UCI HAR](https://archive.ics.uci.edu/dataset/240/"
        "human+activity+recognition+using+smartphones) "
        "contains video-labeled smartphone sensor activity data. We use 9 preprocessed raw signal "
        "channels, 128 samples/window, and six activity classes. No text, tokenizer, pretrained "
        "encoder or language generation is involved.",
        "",
        "Training: 5,066 windows / 15 subjects; model selection: 801 / 2; temperature fitting: "
        "758 / 2; prediction-set calibration: 727 / 2; official test: 2,947 / 9. Subjects are "
        "disjoint across all five sets. Normalization uses only training data. All models train "
        "20 epochs with the same optimizer, batch size 64, and CE + 0.1 Brier loss. The lowest "
        "validation-NLL checkpoint is selected separately per run. Seeds: 0, 1, 2.",
        "",
        "| Model | Parameters | Test accuracy, mean ± sample SD | Macro F1 | NLL, calibrated | "
        "Brier sum, calibrated | ECE, calibrated | GPU p50, median across seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in NAMES:
        s = summary[kind]
        lines.append(
            f"| {NAMES[kind]} | {s['parameters']:,} | "
            f"{100 * s['accuracy_mean']:.2f}% ± {100 * s['accuracy_std']:.2f} pp | "
            f"{values(kind, 'raw', 'macro_f1').mean():.3f} | "
            f"{values(kind, 'calibrated', 'nll').mean():.3f} | "
            f"{values(kind, 'calibrated', 'brier_sum').mean():.3f} | "
            f"{100 * s['calibrated_ece_mean']:.2f}% | "
            f"{np.median(values(kind, 'latency_ms', 'p50')):.3f} ms |"
        )
    lines += [
        "",
        "Latency is a warm batch-one forward pass over a full 128-sample window, "
        "including each baseline's window feature calculation, excluding external input "
        "normalization, transfers, Python answer formatting and data loading. 100 trials "
        "per seed; raw per-run p50/p95 are in the JSON files. Tiny-kernel timings fluctuate "
        "on this desktop GPU. These latency differences are material, not an efficiency win "
        "for the current Tacit signal implementation.",
        "",
        f"Tacit uses {runs['tacit'][0]['persistent_state_tensor_bytes']:,} bytes of persistent "
        "inference state tensors per stream in this configuration (not total process/GPU "
        "memory). State resets between windows: this benchmark does not measure long-history "
        "retention. The longest chunk-partition logit difference across the three trained "
        f"checkpoints is {max(r['chunk_max_logit_error'] for r in runs['tacit']):.3g}.",
        "",
        "### Confidence and abstention",
        "",
        "| Model | Raw ECE | Fitted ECE | Empirical set coverage | Automated fraction | "
        "Accuracy among automated decisions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind in NAMES:
        lines.append(
            f"| {NAMES[kind]} | {100 * values(kind, 'raw', 'ece_10_bins').mean():.2f}% | "
            f"{100 * values(kind, 'calibrated', 'ece_10_bins').mean():.2f}% | "
            f"{100 * values(kind, 'empirical_set_coverage').mean():.2f}% | "
            f"{100 * values(kind, 'automation_coverage').mean():.2f}% | "
            f"{100 * values(kind, 'automated_accuracy').mean():.2f}% |"
        )
    lines += [
        "",
        "Values are means across seeds. The nominal prediction-set target is 90%, "
        "but held-out subjects and overlapping windows are not exchangeable. Tacit's "
        "empirical coverage falls short. Neither calibration nor abstention is a deployment "
        "guarantee. Trial variation over three seeds is not a population confidence interval.",
        "",
        "## Same-model runtime improvements",
        "",
        "The byte-core comparison uses the same 1,602,265-parameter, randomly initialized "
        "Tacit model for both paths. Each update resumes from absolute byte position 7. "
        "It checks execution and numerical parity, not learned decision accuracy. "
        "Warm, paired, seeded AB/BA order; 30 trials/path/input length.",
        "",
        "| Incremental bytes | Byte steps p50 | Chunked p50 | Speedup | Max hidden difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in runtime["absorb"]:
        old, new = row["legacy"]["p50_ms"], row["chunked"]["p50_ms"]
        lines.append(
            f"| {row['bytes']} | {old:.2f} ms | {new:.2f} ms | {old / new:.2f}× | "
            f"{row['hidden_max_error']:.3g} |"
        )
    lines += [
        "",
        "The byte core retains 9,216 bytes of recurrent/convolution storage here, "
        "plus the 768-byte pooled vector. This excludes weights, current-observation "
        "working memory and bounded question caches.",
        "",
        "### Repeated question schemas",
        "",
        "State is already encoded; question and prepared-head caches are warm. "
        "200 paired trials/path. These are head-only costs, not full request latencies.",
        "",
        "| Questions | Serial p50 | Batched p50 | Max logit difference |",
        "|---|---:|---:|---:|",
    ]
    for row in runtime["heads"]:
        lines.append(
            f"| {row['questions']} | {row['serial']['p50_ms']:.3f} ms | "
            f"{row['batched']['p50_ms']:.3f} ms | {row['logit_max_error']:.3g} |"
        )
    lines += [
        "",
        "The 255-candidate reversal probe returns finite logits and max aligned "
        f"difference {runtime['255_candidates']['permutation_max_logit_error']:.3g}. "
        "This is a structural test, with no accuracy claim. The earlier pre-cache runtime "
        "run is retained as `results/runtime-before-prepared-cache.json`; final figures use "
        "the updated cache implementation and paired timing protocol.",
        "",
        "## Reproduce",
        "",
        "```bash",
        'pip install -e ".[dev,bench]"',
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python benchmarks/har.py --seeds 0 1 2",
        "# Run runtime measurements after training finishes:",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python benchmarks/runtime.py",
        "python benchmarks/summarize.py",
        "python examples/activity.py",
        "pytest -q",
        "```",
        "",
        "`results/har/manifest.json` records the source URL, archive SHA-256, subjects, "
        "normalization and labels. Per-run JSON includes learning curves, confusion "
        "matrices, selected epochs and calibration. Local `.pt` and prediction `.npz` "
        "files are produced by the script and excluded from Git. All nine runs are "
        "reported; no successful seed was singled out.",
        "",
        "The temporal CNN baseline was added after the Tacit/MLP pilot; Tacit's "
        "architecture, training budget and subject splits were not retuned using its "
        "test outcomes. These are exploratory public-data experiments, not a private "
        "benchmark or a parameter/compute-matched architecture ablation.",
        "",
        "Dataset attribution: Reyes-Ortiz, Anguita, Ghio, Oneto and Parra (2013), "
        "Human Activity Recognition Using Smartphones, UCI, "
        "[DOI 10.24432/C54S4K](https://doi.org/10.24432/C54S4K), CC BY 4.0. "
        "Tacit source code is MIT.",
    ]
    (ROOT / "docs/RESULTS.md").write_text("\n".join(lines) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f7f9fc",
            "axes.facecolor": "#f7f9fc",
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.19, wspace=0.28, hspace=0.46)
    fig.text(0.08, 0.94, "TACIT  /  SYSTEM 1", fontsize=26, weight="bold", color="#16263d")
    fig.text(
        0.08, 0.895, "v0.2  |  Numeric decisions and recurrent runtime measurements", fontsize=15
    )
    names, x = list(NAMES), np.arange(3)
    ax = axes[0, 0]
    means = [100 * summary[k]["accuracy_mean"] for k in names]
    ax.bar(x, means, color=[COLORS[k] for k in names], width=0.55)
    for i, kind in enumerate(names):
        ax.scatter(
            i + np.array([-0.08, 0, 0.08]),
            100 * values(kind, "raw", "accuracy"),
            s=30,
            color="#15273d",
            zorder=3,
        )
        ax.text(
            i,
            max(100 * values(kind, "raw", "accuracy")) + 2,
            f"{means[i]:.2f}%",
            ha="center",
            weight="bold",
        )
    ax.set(
        xticks=x,
        xticklabels=[NAMES[k] for k in names],
        ylim=(0, 103),
        ylabel="Test accuracy (%)",
        title="Real sensor data: 3 seeds, 2,947 held-out windows",
    )
    ax = axes[0, 1]
    for j, (field, label, color) in enumerate(
        [("raw", "Raw", "#b4c0d1"), ("calibrated", "Temperature fitted", "#25aaec")]
    ):
        bars = ax.bar(
            x + (j - 0.5) * 0.3,
            [100 * values(k, field, "ece_10_bins").mean() for k in names],
            width=0.3,
            label=label,
            color=color,
        )
        ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set(
        xticks=x,
        xticklabels=[NAMES[k] for k in names],
        ylabel="ECE (%) — lower is better",
        title="Fitted confidence: test calibration can improve or worsen",
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.legend(frameon=False, fontsize=10)
    ax = axes[1, 0]
    lengths = [r["bytes"] for r in runtime["absorb"]]
    for field, label, color in [
        ("legacy", "Byte-step reference", "#8995a6"),
        ("chunked", "Resumable chunks", "#25aaec"),
    ]:
        ax.plot(
            lengths, [r[field]["p50_ms"] for r in runtime["absorb"]], "o-", label=label, color=color
        )
    ax.set(
        xlabel="New bytes per observation",
        ylabel="Warm p50 (ms, log scale)",
        yscale="log",
        title="Same 1.60M-parameter byte model, same inputs",
        xticks=lengths,
    )
    ax.legend(frameon=False)
    ax = axes[1, 1]
    counts = [r["questions"] for r in runtime["heads"]]
    for field, label, color in [
        ("serial", "Serial heads", "#8995a6"),
        ("batched", "Prepared batched heads", "#25aaec"),
    ]:
        ax.plot(
            counts, [r[field]["p50_ms"] for r in runtime["heads"]], "o-", label=label, color=color
        )
    ax.set(
        xlabel="Questions per call",
        ylabel="Head-only warm p50 (ms)",
        xticks=counts,
        title="Repeated schemas: shared state and warm caches",
        ylim=(0, None),
    )
    ax.legend(frameon=False)
    fig.text(
        0.08,
        0.055,
        "NVIDIA GB10 · float32 · source-backed HAR pilot + same-model runtime probes\n"
        "Tacit trails the temporal CNN on HAR quality and latency. No direct Jev/Laya result.",
        fontsize=11,
        color="#40546d",
        linespacing=1.7,
    )
    fig.savefig(ROOT / "results/system1-metrics.png", dpi=160)
    svg = ROOT / "results/system1-metrics.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
