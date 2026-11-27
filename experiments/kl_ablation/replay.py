"""Verify the main experiment from the original prediction archives."""

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from experiments.gradient_audit.replay import verify_eos_audit
from experiments.transfer_evaluation.replay import summarize as summarize_svamp
from opsdmath.analysis.paired_statistics import interval
from opsdmath.evaluation.metrics import LONG, SHORT, cohort, read_correctness
from opsdmath.io import digest

METHODS = ("base", "clipped", "full", "equations")
SEEDS = (17, 29)


def summarize_gsm8k(root):
    folders = {17: root / "seed17/evaluation", 29: root / "evaluation"}
    strata = cohort(folders[17] / "base")
    if strata != cohort(folders[29] / "base"):
        raise ValueError("The two frozen runs disagree on the GSM8K cohorts")
    ids = list(strata)
    summary = {}
    for cap, scorer in ((SHORT, "numeric"), (SHORT, "strict"), (LONG, "numeric")):
        stack = {}
        for method in METHODS:
            rows = []
            for seed in SEEDS:
                folder = (root / "seed17/equation-controls" / f"equations-{seed}"
                          if method == "equations" else folders[seed] / method)
                correctness = read_correctness(folder, cap, scorer)
                if correctness.keys() != strata.keys():
                    raise ValueError(f"Unexpected question IDs in {folder} at {cap} tokens")
                rows.append([correctness[key] for key in ids])
            stack[method] = np.array(rows, dtype=np.int8)
        for name in ("all", "early", "late", "wrong_both"):
            mask = np.array([name == "all" or strata[key] == name for key in ids])
            selected = {m: stack[m][:, mask] for m in METHODS}
            versus = {}
            for control in ("base", "clipped", "equations"):
                difference = selected["full"] - selected[control]
                versus[control] = {"pooled": interval(difference.mean(0)),
                                   "by_seed": [interval(row) for row in difference]}
            summary[f"{cap}/{scorer}/{name}"] = {
                "counts": {method: data.sum(1).tolist() for method, data in selected.items()},
                "n": int(mask.sum()),
                "full_vs": versus,
            }
    return summary


def recompute(root):
    gsm8k = summarize_gsm8k(root)
    svamp = summarize_svamp(root / "seed17/transfer-evaluation")
    return {**gsm8k, "SVAMP": svamp["accuracy"]}, svamp


def verify(repository):
    evidence = repository / "results/kl_ablation"
    archives = {}
    for line in (evidence / "archive-sha256.txt").read_text(encoding="utf-8").splitlines():
        checksum, name = line.split()
        archives[name] = checksum
    with tempfile.TemporaryDirectory(prefix="opsdmath-") as directory:
        root = Path(directory)
        for name, destination in (("seed29", root), ("seed17", root / "seed17"), ("controls", root / "seed17")):
            path = evidence / f"{name}-evidence.tar.gz"
            if digest(path) != archives[path.name]:
                raise ValueError(f"SHA-256 mismatch for {path}")
            with tarfile.open(path) as archive:
                archive.extractall(destination, filter="data")
        main, transfer = recompute(root)
    for actual, path in ((main, evidence / "paired-two-seed.json"),
                         (transfer, repository / "results/transfer_evaluation/summary.json")):
        if actual != json.loads(path.read_text(encoding="utf-8")):
            raise ValueError(f"Replayed statistics differ from {path}")
    return main, transfer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["kl-ablation", "svamp-transfer"], default="kl-ablation")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    paired, transfer = verify(args.repository)
    if args.experiment == "kl-ablation":
        print("Matched archived GSM8K and SVAMP paired statistics.")
        for cap in (SHORT, LONG):
            print(f"GSM8K {cap}-token counts:", paired[f"{cap}/numeric/all"]["counts"])
        for cohort_name in ("early", "late"):
            print(f"GSM8K {cohort_name} counts:", paired[f"{SHORT}/numeric/{cohort_name}"]["counts"])
        print("Clipped EOS decreases:", verify_eos_audit(args.repository / "results/gradient_audit"))
    for budget, entry in transfer["accuracy"].items():
        models = {seed: row["correct"] for seed, row in entry["models"].items()}
        print(f"SVAMP {budget}-token accuracy (out of {entry['n']}):", {"base": entry["base"], **models})
    print("SVAMP early retention:", transfer["early_retention"])
    if args.output:
        report = transfer if args.experiment == "svamp-transfer" else paired
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
