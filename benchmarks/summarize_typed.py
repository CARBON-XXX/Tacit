"""Create a compact report and static chart from recorded workflow predictions."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    root = Path("results/typed")
    paired = json.loads((root / "paired-forked-laya.json").read_text())
    rows = []
    for name, run in [
        ("Late interaction", "semantic-seed0"),
        ("Structured + lexical", "structured-seed0"),
        ("Structured, no word features", "structured-no-text-seed0"),
        ("Forked shared state", "forked-seed0"),
    ]:
        p = root / run
        result = json.loads((p / "result.json").read_text())
        manifest = json.loads((p / "manifest.json").read_text())
        rows.append(
            {
                "name": name,
                "run": run,
                "parameters": manifest["parameters"],
                "metrics": result["calibrated"]["overall"],
                "selected_epoch": result["selected_epoch"],
            }
        )
    laya = json.loads((root / "laya-typed-test.json").read_text())
    report = {
        "status": "goal not achieved",
        "goal": "超过 Jev/Laya",
        "runs": rows,
        "laya": {"parameters": laya["parameters"], "metrics": paired["metrics"]["laya"]["overall"]},
        "paired_latency": paired["latency"],
        "paired_accuracy": paired["paired_accuracy"],
        "scope": (
            "four known workflows; 400 held-out cases, 2000 teacher-labeled decisions; "
            "one training seed; specialist pilot"
        ),
    }
    (root / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "tacit-typed-workflow",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    fig.patch.set_facecolor("#f5f7fa")
    names = ["Late interaction", "Structured + words", "Forked shared state", "Laya typed"]
    chosen = [rows[0]["metrics"], rows[1]["metrics"], rows[3]["metrics"], report["laya"]["metrics"]]
    colors = ["#94a3b8", "#94a3b8", "#0891b2", "#8b5cf6"]
    for ax in axes:
        ax.set_facecolor("#f5f7fa")
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    values = [r["accuracy"] * 100 for r in chosen]
    axes[0].barh(names, values, color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_title("Teacher-label accuracy (%)", loc="left", fontweight="bold")
    for i, value in enumerate(values):
        axes[0].text(value + 1, i, f"{value:.2f}", va="center")
    values = [r["brier_sum"] for r in chosen]
    axes[1].barh(names, values, color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 0.17)
    axes[1].set_yticklabels([])
    axes[1].set_title("Brier score (lower is better)", loc="left", fontweight="bold")
    for i, value in enumerate(values):
        axes[1].text(value + 0.002, i, f"{value:.4f}", va="center")
    values = [paired["latency"][k]["p50_ms"] for k in ["tacit", "laya"]]
    axes[2].barh(["Tacit forked", "Laya typed"], values, color=colors[2:])
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 130)
    axes[2].set_title("Complete request p50 (ms)", loc="left", fontweight="bold")
    for i, value in enumerate(values):
        axes[2].text(value + 2, i, f"{value:.2f}", va="center")
    fig.suptitle(
        "TACIT / System 1 research progress",
        x=0.035,
        ha="left",
        y=0.96,
        fontsize=23,
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.87,
        "400 identical cases • 2,000 typed decisions • GB10 • BF16 • five questions per request",
        fontsize=12,
    )
    delta = paired["paired_accuracy"]["accuracy_difference"] * 100
    lower, upper = [x * 100 for x in paired["paired_accuracy"]["ci95_case_bootstrap"]]
    fig.text(
        0.035,
        0.12,
        f"Tacit accuracy gap vs Laya: {delta:+.2f} pp; paired case-bootstrap 95% CI "
        f"[{lower:+.2f}, {upper:+.2f}] pp. Goal not achieved.",
        fontsize=11,
    )
    fig.text(
        0.035,
        0.07,
        "One seed; known-workflow specialists. Teacher agreement is not independent correctness. "
        "Jev public results use a different protocol.",
        fontsize=10,
        color="#475569",
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.76, bottom=0.24, wspace=0.34)
    fig.savefig("results/typed-progress.png", dpi=160)
    svg = Path("results/typed-progress.svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
