"""Postprocess confirmation predictions by reference operation count and training seed."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from experiments.budget_efficiency.budget_study import BUDGETS, PARSERS, uncertainty
from opsdmath.data.gsm8k import read_rows
from experiments.budget_efficiency.summarize_run import write_csv

COMPARISONS = (("equation", "answer"), ("equation", "online"), ("online", "fixed"), ("online", "sft"))


def stratum(count):
    if count is None or count == 0:
        return "unmarked"
    if count <= 2:
        return "1–2"
    if count <= 4:
        return "3–4"
    return "5+"


def train_metrics(root):
    rows = []
    for seed in (17, 29):
        expected = None
        for name in ("equation", "answer", "fixed"):
            directory = root / f"{name}{seed}"
            events = read_rows(directory / "events.jsonl")
            samples = [row for row in events if row["event"] == "microstep"]
            updates = [row for row in events if row["event"] == "update"]
            ids = [row["id"] for row in samples]
            final = events[-1]
            if len(samples) != 256 or len(set(ids)) != 256 or len(updates) != 64 or final["event"] != "complete":
                raise ValueError(f"Incomplete OPSD group {directory}")
            if expected is None:
                expected = ids
            elif ids != expected:
                raise ValueError(f"Training order changed for {directory}")
            cache = [row for row in events if row["event"] == "fixed_rollouts"]
            if name == "fixed":
                prompts = read_rows(directory / "fixed_rollouts.jsonl")
                if [r["id"] for r in prompts] != ids or len(cache) != 1:
                    raise ValueError(f"Fixed rollouts differ from the training order for {directory}")
            rows.append(dict(method=name, seed=seed, n=len(samples), updates=len(updates),
                teacher_input_tokens=sum(row["teacher_prompt_tokens"] for row in samples),
                student_trajectory_tokens=sum(row["tokens"] for row in samples),
                fixed_generation_seconds=cache[0]["seconds"] if cache else None,
                seconds=final["seconds"], peak_allocated_mib=final["peak_allocated_mib"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train", type=Path)
    args = parser.parse_args()
    groups = {}
    lengths = []
    for path in sorted(args.scores.glob("*.jsonl")):
        records = read_rows(path)
        bins = defaultdict(list)
        for row in records:
            budget, scoring = row["budget"], row["parser"]
            score = groups.setdefault((path.stem, budget, scoring), {})
            if row["id"] in score:
                raise ValueError(f"Duplicate prediction at {path}/{row['id']}")
            score[row["id"]] = int(row["correct"])
            if scoring == "numeric":
                bins[budget, stratum(row["operation_count"])].append(row)
        for (budget, tier), values in bins.items():
            lengths.append(dict(group=path.stem, budget=budget, reference_ops=tier, n=len(values),
                accuracy=mean(r["correct"] for r in values), parseable=mean(r["parsed"] is not None for r in values),
                mean_tokens=mean(r["length"] for r in values)))
    pooled = []
    for left, right in COMPARISONS:
        for budget in BUDGETS:
            for method in PARSERS:
                paired = [groups.get((f"{name}{seed}", budget, method))
                          for seed in (17, 29) for name in (left, right)]
                if any(vector is None for vector in paired):
                    continue
                a, b, c, d = paired
                if not a.keys() == b.keys() == c.keys() == d.keys():
                    raise ValueError("Confirmation IDs differ by seed")
                delta = [(a[k] - b[k] + c[k] - d[k])/2 for k in a]
                low, high = uncertainty(delta)
                pooled.append(dict(left=left, right=right, budget=budget, parser=method, n=len(delta),
                                   delta=mean(delta), low=low, high=high))
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "difficulty.csv", lengths)
    if pooled:
        write_csv(args.output / "two_seed_paired.csv", pooled)
    if args.train:
        write_csv(args.output / "training.csv", train_metrics(args.train))
    (args.output / "method.json").write_text(
        json.dumps({"method": "Bootstrap question IDs with both training seeds kept together."}, indent=2))


if __name__ == "__main__":
    main()
