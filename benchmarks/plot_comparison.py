"""Create a standalone comparison image from the rescored public snapshot."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    source = Path("results/comparison-snapshot/summary.json")
    data = json.loads(source.read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "tacit-three-model-comparison",
        }
    )
    colors = {"tacit": "#0077a8", "laya": "#c07821", "jev": "#555b66"}
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.patch.set_facecolor("white")

    def bars(ax, title, subtitle, names, values, identities):
        y = np.arange(len(names))
        ax.barh(y, values, height=0.56, color=[colors[k] for k in identities])
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Accuracy (%)")
        ax.grid(axis="x", color="#e7e9ed")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontweight="bold", pad=39, fontsize=15)
        ax.text(0, 1.04, subtitle, transform=ax.transAxes, fontsize=10, color="#555b66")
        for position, value in zip(y, values, strict=True):
            ax.text(value + 1.3, position, f"{value:.2f}", va="center", fontsize=12)

    keys = ["tacit-refined", "tacit-mixed", "jev-typed", "laya-typed", "laya-general"]
    bars(
        axes[0, 0],
        "Four workflows",
        "400 cases / 2,000 decisions; synthetic teacher labels",
        ["Tacit refined", "Tacit mixed", "Jev 1.13.0", "Laya typed", "Laya general"],
        [data["workflow"][k]["accuracy"] * 100 for k in keys],
        ["tacit", "tacit", "jev", "laya", "laya"],
    )
    keys = ["tacit-mixed", "jev-nli", "laya-general"]
    for ax, kind, title in [
        (axes[0, 1], "choice", "NLI: three-way relation"),
        (axes[1, 0], "noul", "NLI: is the statement supported?"),
    ]:
        bars(
            ax,
            title,
            "Same 2,000 MultiNLI cases; human labels",
            ["Tacit mixed", "Jev 1.13.0", "Laya general"],
            [data["nli"][k][kind]["accuracy"] * 100 for k in keys],
            ["tacit", "jev", "laya"],
        )
    ax = axes[1, 1]
    pairs = [
        ("Workflow vs Jev", "workflow/refined-minus-jev-typed"),
        ("Workflow vs Laya typed", "workflow/refined-minus-laya-typed"),
        ("NLI relation vs Jev", "nli/choice/mixed-minus-jev-nli"),
        ("NLI relation vs Laya", "nli/choice/mixed-minus-laya-general"),
        ("NLI support vs Jev", "nli/noul/mixed-minus-jev-nli"),
        ("NLI support vs Laya", "nli/noul/mixed-minus-laya-general"),
    ]
    for i, (_, key) in enumerate(pairs):
        row = data["paired_accuracy"][key]
        point = row["accuracy_difference"] * 100
        lo, hi = [x * 100 for x in row["ci95"]]
        ax.errorbar(
            point,
            i,
            xerr=[[point - lo], [hi - point]],
            fmt="o",
            color=colors["tacit"],
            capsize=4,
            markersize=5,
            linewidth=1.4,
        )
        ax.text(22.5, i, f"{point:+.2f}", va="center", ha="right", fontsize=11)
    ax.set_yticks(range(len(pairs)), [name for name, _ in pairs])
    ax.invert_yaxis()
    ax.set_xlim(-9, 23)
    ax.set_xticks([-5, 0, 5, 10, 15, 20])
    ax.axvline(0, color="#555b66", linewidth=1)
    ax.set_xlabel("Tacit minus reference (percentage points)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e7e9ed")
    ax.set_axisbelow(True)
    ax.set_title("Paired accuracy differences", loc="left", fontweight="bold", pad=39, fontsize=15)
    ax.text(
        0,
        1.04,
        "95% cluster-bootstrap intervals; 10,000 resamples",
        transform=ax.transAxes,
        fontsize=10,
        color="#555b66",
    )
    fig.text(0.045, 0.956, "TACIT / JEV / LAYA", fontsize=26, fontweight="bold", color="#20252c")
    fig.text(
        0.045,
        0.920,
        "Generation-free decisions • fixed test inputs • measured 20 Sep 2026",
        fontsize=13,
        color="#555b66",
    )
    fig.text(
        0.045,
        0.074,
        "Tacit refined: workflow specialist. Tacit mixed: NLI + workflow replay. "
        "One training seed; broader superiority is not established.",
        fontsize=11,
    )
    fig.text(
        0.045,
        0.046,
        "Accuracy uses exposed probability argmax; Jev returns 82.70% via its NLI choice field. "
        "Laya's published XNLI protocol differs from this MultiNLI test.",
        fontsize=10,
        color="#555b66",
    )
    fig.text(
        0.045,
        0.022,
        "Source: results/comparison-snapshot/summary.json • "
        "2,400 Jev requests / estimated API charge USD 0.0538713 / budget USD 0.50",
        fontsize=10,
        color="#555b66",
    )
    fig.subplots_adjust(left=0.155, right=0.97, top=0.80, bottom=0.18, wspace=0.72, hspace=0.85)
    fig.savefig("results/jev-laya-tacit.png", dpi=160)
    svg = Path("results/jev-laya-tacit.svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
