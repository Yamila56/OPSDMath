"""Run the frozen OPSD protocol one GPU job at a time."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("runs")
BUDGETS = (64, 128, 256, 512)


def call(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def completed(path):
    events = path / "events.jsonl"
    return events.is_file() and json.loads(events.read_text().splitlines()[-1])["event"] == "complete"


def run(path, *args):
    path = ROOT / path
    if completed(path):
        return
    if path.exists():
        path.rename(path.with_name(f"{path.name}-interrupted-{time.time_ns()}"))
    call("run_math.py", *args, "--model", "model", "--output", path)


def evaluate(phase, split, name, adapter=None, batch_size=16):
    flags = ("--adapter", adapter) if adapter else ()
    run(f"{phase}/{name}", "predict", "--data", split, "--save-token-ids", "--cap", 512,
        "--batch-size", batch_size, *flags)
    return f"{name}={ROOT / phase / name / 'base.jsonl'}"


def diagnostic_probe():
    for cap in BUDGETS:
        run(f"dev-direct-{cap}", "predict", "--data", "data/dev.jsonl", "--limit", 16,
            "--cap", cap, "--save-token-ids", "--batch-size", 16)
    def read(cap):
        path = ROOT / f"dev-direct-{cap}" / "base.jsonl"
        return [json.loads(line)["token_ids"] for line in path.read_text().splitlines()]
    longest = read(512)
    for cap in BUDGETS[:-1]:
        if [sequence[:cap] for sequence in longest] != read(cap):
            raise RuntimeError(f"Direct budget differs at {cap}")
    print("Direct and sliced generation match on 16 development rows")


def confirmation_batch():
    """Use a larger confirmation batch only if it preserves development token IDs."""
    paths = {}
    for size in (16, 64):
        prefix = f"dev-confirmation-batch-{size}"
        paths[size] = ROOT / prefix
        run(prefix, "predict", "--data", "data/dev.jsonl", "--limit", 64, "--cap", 512,
            "--save-token-ids", "--batch-size", size, "--adapter", "existing/17/sft")
    def predictions(size):
        return {r["id"]: r["token_ids"] for r in map(json.loads, (paths[size] / "base.jsonl").read_text().splitlines())}
    small, large = predictions(16), predictions(64)
    same = small.keys() == large.keys() and small == large
    def runtime(size):
        return json.loads((paths[size] / "events.jsonl").read_text().splitlines()[-1])["seconds"]
    chosen = 64 if same and runtime(64) < runtime(16) else 16
    data = dict(chosen=chosen, exact_match=same, seconds16=runtime(16), seconds64=runtime(64),
                differing=sum(large.get(key) != ids for key, ids in small.items()))
    (ROOT / "confirmation_batch.json").write_text(json.dumps(data, indent=2) + "\n")
    return chosen


def evaluation(phase):
    confirmation = phase == "confirmation"
    split = "data/next-confirmation/inputs.jsonl" if confirmation else "data/test_inputs.jsonl"
    labels = "data/next-confirmation/labels.jsonl" if confirmation else "data/test_labels.jsonl"
    if not confirmation:
        diagnostic_probe()
    batch_size = confirmation_batch() if confirmation else 16
    estimates = [evaluate(phase, split, "base", batch_size=batch_size)]
    for seed in (17, 29):
        estimates.append(evaluate(phase, split, f"sft{seed}", f"existing/{seed}/sft", batch_size))
        estimates.append(evaluate(phase, split, f"online{seed}", f"existing/{seed}/opsd", batch_size))
        if confirmation:
            for group in ("equation", "answer", "fixed"):
                name = f"{group}{seed}"
                estimates.append(evaluate(phase, split, name, ROOT / "train" / name / "adapter", batch_size))
    scores = ROOT / phase / "score"
    if not (scores / "metrics.csv").exists():
        if scores.exists():
            scores.rename(scores.with_name(f"score-interrupted-{time.time_ns()}"))
        call("experiments/legacy/budget_study.py", "report", "model", labels, scores, *estimates)


def train():
    groups = {
        "equation": ("--teacher-context", "equations"),
        "answer": ("--teacher-context", "answer"),
        "fixed": ("--fixed-rollout",),
    }
    for name, flags in groups.items():
        run(f"train/probe-{name}", "opsd", "--data", "data/train.jsonl", "--probe", "--limit", 4, *flags)
        run(f"train/reload-{name}", "predict", "--data", "data/dev.jsonl", "--limit", 4,
            "--batch-size", 4, "--adapter", ROOT / "train" / f"probe-{name}" / "adapter")
    for seed in (17, 29):
        for name, flags in groups.items():
            run(f"train/{name}{seed}", "opsd", "--data", "data/train.jsonl", "--seed", seed, *flags)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phases", nargs="+", choices=("diagnostic", "train", "confirmation"))
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    os.environ["PYTHONUNBUFFERED"] = "1"
    call("run_math.py", "check-loss")
    for phase in args.phases:
        if phase == "train":
            train()
        else:
            evaluation(phase)
        (ROOT / f"{phase}.complete").touch()


if __name__ == "__main__":
    main()
