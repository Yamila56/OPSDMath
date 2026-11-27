"""Plot the paired seed results on the two frozen-base budget cohorts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


METHODS = [
    ("equations", "Equation SFT", "#79538c"),
    ("privileged", "Uniform privileged OPSD", "#b04f40"),
    ("random", "Random assignment", "#357795"),
    ("behavior", "Behavior-based assignment", "#008052"),
]


def render(source, output):
    result = json.loads(source.read_text(encoding="utf-8"))
    early, late = (result["denominators"][group] for group in ("early", "late"))
    points = {}
    for key, label, color in METHODS:
        for seed in (17, 29):
            row = result["results"][f"{key}-{seed}"]["by_cohort"]
            points[key, seed] = (100 * row["early"] / early, 100 * row["late"] / late, label, color)
    fig, ax = plt.subplots(figsize=(8.6, 4.9), layout="constrained")
    ax.set_title("GSM8K accuracy by frozen-base budget cohort", loc="left", fontweight="bold", pad=10)
    ax.set_xlabel(f"Retention among {early} early-correct questions (%)")
    ax.set_ylabel(f"Acceleration among {late} late-correct questions (%)")
    ax.set(xlim=(61, 102), ylim=(-4, 68))
    inset = ax.inset_axes([.50, .40, .41, .40])
    inset.set(xlim=(94.8, 100.8), ylim=(-.9, 14), title="Detail near frozen-base retention")
    inset.set_xticks([95, 97, 99, 100])
    inset.set_yticks([0, 5, 10])
    for target in (ax, inset):
        target.grid(color="#dee5e9", linewidth=.6)
        target.set_axisbelow(True)
        target.scatter(100, 0, marker="D", color="#273842", s=38, zorder=3)
        for (_, seed), (x, y, _, color) in points.items():
            target.scatter(x, y, marker="o" if seed == 17 else "s", color=color,
                           edgecolors="white", linewidths=.65, s=56, zorder=3)
    keys = [Line2D([], [], marker="o", color=color, linestyle="", label=label) for _, label, color in METHODS]
    keys.append(Line2D([], [], marker="D", color="#273842", linestyle="", label="Frozen base and no-op anchors"))
    ax.legend(handles=keys, loc="center left", bbox_to_anchor=(.015, .51), frameon=False, fontsize=8.5)
    ax.text(.015, .96, "Seed 17 circles · Seed 29 squares", transform=ax.transAxes,
            fontsize=8.2, color="#364854", va="top")
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.source, args.output)
