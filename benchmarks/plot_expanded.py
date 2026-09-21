"""Render the expanded frozen-checkpoint measurements as a standalone data image."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    data = json.loads(Path("results/expanded-snapshot/summary.json").read_text())
    models = ["tacit-mixed", "jev", "laya-general"]
    names = {"tacit-mixed": "Tacit mixed", "jev": "Jev 1.13.0", "laya-general": "Laya general"}
    colors = {"tacit-mixed": "#0077a8", "jev": "#555b66", "laya-general": "#c07821"}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "tacit-expanded-frozen-comparison",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.patch.set_facecolor("white")

    def style(ax, title, subtitle):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=15, pad=34)
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=10, color="#555b66")
        ax.grid(axis="x", color="#e7e9ed")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    def bars(ax, title, subtitle, rows):
        for y, (_label, model, track, open_fill) in enumerate(rows):
            value = data["tracks"][model][track]["accuracy"] * 100
            ax.barh(
                y,
                value,
                height=0.57,
                color="white" if open_fill else colors[model],
                edgecolor=colors[model],
                linewidth=1.2,
                hatch="//" if open_fill else None,
            )
            ax.text(value + 1, y, f"{value:.2f}", va="center", fontsize=11)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Accuracy (%)")
        style(ax, title, subtitle)

    for ax, kind, title in [
        (axes[0, 0], "choice", "NLI: three-way relation"),
        (axes[0, 1], "noul", "NLI: support Boolean"),
    ]:
        rows = []
        for model in models:
            rows.extend(
                [
                    (names[model] + " / question", model, "nli/text/" + kind, True),
                    (names[model] + " / JSON", model, "nli/structured/" + kind, False),
                ]
            )
        bars(ax, title, "Same 2,000 cases and labels in both layouts", rows)
    bars(
        axes[1, 0],
        "Additional classification tasks",
        "2,000 cases each; emotion has machine-generated labels",
        [
            (label + " / " + names[model], model, track, False)
            for track, label in [("news", "News"), ("emotion", "Emotion")]
            for model in models
        ],
    )
    ax = axes[1, 1]
    for i, model in enumerate(models):
        result = data["structured_minus_text_layout_accuracy"][model]["choice"]
        point = result["accuracy_difference"] * 100
        lo, hi = [v * 100 for v in result["ci95"]]
        ax.errorbar(
            point,
            i,
            xerr=[[point - lo], [hi - point]],
            fmt="o",
            color=colors[model],
            capsize=4,
            linewidth=1.4,
            markersize=7,
        )
        ax.text(42, i, f"{point:+.2f}", va="center", ha="right", fontsize=12)
    ax.set_yticks(range(len(models)), [names[m] for m in models])
    ax.set_ylim(2.6, -0.6)
    ax.set_xlim(-50, 44)
    ax.set_xticks([-40, -20, 0, 20])
    ax.axvline(0, color="#555b66", linewidth=1)
    ax.set_xlabel("JSON minus question layout (percentage points)")
    style(
        ax,
        "NLI relation: paired layout effect",
        "95% bootstrap intervals; shared premises clustered",
    )
    fig.suptitle(
        "Tacit / Jev / Laya: expanded decision tests",
        x=0.055,
        ha="left",
        fontsize=23,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.055,
        0.928,
        "Frozen mixed Tacit checkpoint. New pure Tacit training is not scored here.",
        fontsize=13,
        color="#555b66",
    )
    fig.text(
        0.055,
        0.037,
        "Question layout: hypothesis in instructions. JSON layout: both statements in state. "
        "NLI rows reuse the same cases.",
        fontsize=11,
        color="#555b66",
    )
    fig.text(
        0.055,
        0.014,
        "Source: results/expanded-snapshot  |  Exposed probability argmax  |  "
        "Supervision differs; no broad superiority claim.",
        fontsize=11,
        color="#555b66",
    )
    fig.subplots_adjust(left=0.18, right=0.97, top=0.84, bottom=0.115, wspace=0.6, hspace=0.63)
    for suffix in ("png", "svg"):
        path = Path("results/expanded-comparison." + suffix)
        fig.savefig(path, dpi=160, metadata={"Date": None} if suffix == "svg" else None)
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


if __name__ == "__main__":
    main()
