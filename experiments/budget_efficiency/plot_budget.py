"""Plot saved numerical accuracy and answer coverage at each token cap."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--methods", nargs="+", help="Plot only these method names.")
    args = parser.parse_args()
    styles = {"base": "#4d4d4d", "sft": "#4477aa", "online": "#228833", "fixed": "#aa3377",
              "equation": "#ee7733", "answer": "#66ccee", "shortsft": "#332288", "equationsft": "#ccbb44"}
    methods = defaultdict(list)
    with args.metrics.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["parser"] == "numeric" and (not args.methods or row["method"] in args.methods):
                methods[row["method"]].append(row)
    if args.methods and set(args.methods) != methods.keys():
        parser.error(f"Unknown methods: {sorted(set(args.methods) - methods.keys())}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharex=True)
    for name, rows in methods.items():
        budget = [int(r["budget"]) for r in rows]
        family = name.rstrip("0123456789")
        color = styles[family]
        line = "--" if name.endswith("29") else "-"
        axes[0].plot(budget, [100 * float(r["accuracy"]) for r in rows], line,
                     color=color, marker="o", markersize=3, label=name)
        axes[1].plot(budget, [100 * int(r["parsed"])/int(r["n"]) for r in rows],
                     line, color=color, marker="o", markersize=3)
    for axis, label in zip(axes, ("Numeric accuracy (%)", "Parseable answers (%)"), strict=True):
        axis.set(xscale="log", xticks=[64, 128, 256, 512], xticklabels=["64", "128", "256", "512"],
                 xlabel="Maximum generated tokens", ylabel=label)
        axis.grid(axis="y", alpha=.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, ncols=2, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
