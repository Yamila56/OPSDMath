"""Plot paired accuracy against the generation cap and mean generated tokens."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    datasets = (("primary", "792 condition questions"), ("confirmation", "288 new questions"))
    methods = {
        "base": ("Frozen base", "#626c7c"),
        "answer": ("Answer-only OPSD", "#076888"),
        "general_brief": ("General notes OPSD", "#6a4f95"),
        "constraint_brief": ("Constraint notes OPSD", "#c45319"),
    }
    budgets = (128, 256, 512)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4), layout="constrained")
    for (name, title), (cap_axis, actual_axis) in zip(datasets, axes, strict=True):
        for method, (label, color) in methods.items():
            variants = [method] if method == "base" else [f"{method}{seed}" for seed in (17, 29)]
            for index, variant in enumerate(variants):
                observations = [report[name][str(b)]["metrics"][variant] for b in budgets]
                accuracies = [r["change_pair"]/r["pairs"] for r in observations]
                actual = [r["mean_output_tokens"] for r in observations]
                style = dict(marker="o", markersize=4, linewidth=1.6, color=color,
                             linestyle="-" if index == 0 else "--")
                cap_axis.plot(budgets, accuracies, label=label if index == 0 else None, **style)
                actual_axis.plot(actual, accuracies, **style)
        cap_axis.set(xscale="log", xlabel="Maximum total generated tokens", ylabel="Effective-condition pairs correct",
                     title=title)
        cap_axis.set_xticks(budgets, labels=[str(b) for b in budgets])
        cap_axis.xaxis.set_minor_locator(NullLocator())
        actual_axis.set(xlabel="Mean total generated tokens per question", title=title)
        for axis in (cap_axis, actual_axis):
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.set_ylim(0, 1)
            axis.grid(axis="y", color="#d9dce1", linewidth=.6)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(loc="lower right", fontsize=8, frameon=False)
    axes[0, 1].text(.98, .05, "Solid: seed 17 · Dashed: seed 29", ha="right",
                    transform=axes[0, 1].transAxes, fontsize=8, color="#454b57")
    fig.suptitle("Teacher context and inference budget", fontsize=15, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
