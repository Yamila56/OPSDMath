"""Summarize paired prompt interactions and capability-matched budgets on locked splits."""

import argparse
import json
from pathlib import Path

import numpy as np

from opsdmath.evaluation.metrics import interval

from opsdmath.data.gsm8k import number, read_rows
from experiments.budget_efficiency.opsd_calibration import PROMPTS, write
from experiments.budget_efficiency.score_structured_answer import audit

SEEDS = (17, 29)
BUDGETS = (64, 96, 128, 160, 192, 224, 256, 384, 512)
FINE_BUDGETS = tuple(range(64, 257, 8))
GROUPS = ("base", "opsd-17", "opsd-29", "sft-17", "sft-29")


def read_scores(path, ids):
    rows = json.loads(path.read_text(encoding="utf-8"))["numeric"]
    if [r["id"] for r in rows] != ids:
        raise ValueError(f"Evaluated IDs or ordering changed in {path}")
    return np.array([r["correct"] for r in rows], dtype=np.int8)


def collect(root, ids, truth, fine):
    free, fields = {}, {}
    for group in GROUPS:
        folder = root / group
        if not (folder / "complete.json").is_file():
            raise ValueError(f"Unfinished evaluation at {folder}")
        for prompt in PROMPTS:
            for cap in BUDGETS if prompt == "standard" else (64, 96, 128):
                free[group, prompt, cap] = read_scores(folder / f"{prompt}-{cap}.json", ids)
            if fine and prompt == "standard":
                for cap in sorted(set(FINE_BUDGETS) - set(BUDGETS)):
                    free[group, prompt, cap] = read_scores(folder / f"standard-{cap}.json", ids)
            rows = read_rows(folder / f"{prompt}-structured.jsonl")
            if [r["id"] for r in rows] != ids:
                raise ValueError(f"Structured IDs changed for {group}/{prompt}")
            fields[group, prompt] = np.array(
                [audit(row, budget=128, draft_cap=104) == truth[row["id"]] for row in rows], dtype=np.int8)
    return free, fields


def summarize(data, small, large, fine):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    ids = manifest["development_existing"] + manifest["heldout"]
    assert len(ids) == len(set(ids))
    n_dev = len(manifest["development_existing"])
    evaluation = slice(n_dev, None)
    truth = {r["id"]: number(r["answer"]) for r in read_rows(data / "labels.jsonl")}
    if set(ids) != truth.keys() or None in truth.values():
        raise ValueError("The locked split differs from the reference labels")
    scores = {"1.7B": collect(small, ids, truth, fine), "8B": collect(large, ids, truth, fine)}
    pressure = {}
    base_long = {}
    for scale, (free, _) in scores.items():
        base_long[scale] = free["base", "standard", 512].astype(bool)
        long_correct = base_long[scale][:n_dev]
        if not long_correct.any():
            raise ValueError(f"No 512-correct development questions for {scale}")
        pressure[scale] = {cap: float(free["base", "standard", cap][:n_dev][long_correct].mean())
                           for cap in sorted(set(BUDGETS) | (set(FINE_BUDGETS) if fine else set()))}
    target = pressure["1.7B"][128]
    matching = min((cap for cap in pressure["8B"] if cap <= 256),
                   key=lambda cap: (abs(pressure["8B"][cap]-target), cap))
    report = dict(development_n=n_dev, heldout_n=len(ids)-n_dev, development_pressure=pressure,
                  matched_budgets={"1.7B": 128, "8B": matching}, fine_grid=fine, scales={})
    common = (base_long["1.7B"] & base_long["8B"])[evaluation]
    interactions = {}
    method_gaps = {}
    matched_gaps = {}
    for scale, (free, fields) in scores.items():
        result = {}
        for mode, predictions in (
            ("free", {(group, prompt): free[group, prompt, 128] for group in GROUPS for prompt in PROMPTS}),
            ("structured", fields),
        ):
            contrasts = {prompt: np.stack([predictions[f"opsd-{seed}", prompt][evaluation] -
                                          predictions[f"sft-{seed}", prompt][evaluation] for seed in SEEDS])
                         for prompt in PROMPTS}
            delta = contrasts["combined"] - contrasts["standard"]
            interactions[scale, mode] = delta.mean(axis=0)
            method_gaps[scale, mode] = contrasts
            result[mode] = {
                "correct": {prompt: {g: int(predictions[g, prompt][evaluation].sum()) for g in GROUPS} for prompt in PROMPTS},
                "opsd_minus_sft": {p: [interval(x) for x in diffs] for p, diffs in contrasts.items()},
                "combined_minus_standard_interaction_by_seed": [interval(d) for d in delta],
                "combined_minus_standard_interaction": interval(delta.mean(axis=0)),
            }
        result["absolute_budget_correct"] = {cap: {g: int(free[g, "standard", cap][evaluation].sum())
                                                    for g in GROUPS} for cap in BUDGETS}
        matched = report["matched_budgets"][scale]
        result["matched"] = {}
        for key, mask in (("heldout", np.ones(len(common), dtype=bool)), ("common_long_correct", common)):
            preds = {g: free[g, "standard", matched][evaluation][mask] for g in GROUPS}
            base = preds["base"].astype(bool)
            late = ~base & base_long[scale][evaluation][mask]
            result["matched"][key] = {
                "correct": {g: int(arr.sum()) for g, arr in preds.items()}, "n": int(mask.sum()),
                "early_n": int(base.sum()), "late_n": int(late.sum()),
                "early_retained": {g: int(arr[base].sum()) for g, arr in preds.items()},
                "late_accelerated": {g: int(arr[late].sum()) for g, arr in preds.items()},
                "opsd_minus_sft_by_seed": [interval(preds[f"opsd-{s}"]-preds[f"sft-{s}"]) for s in SEEDS],
            }
            matched_gaps[scale, key] = np.stack([preds[f"opsd-{s}"]-preds[f"sft-{s}"] for s in SEEDS])
        report["scales"][scale] = result
    report["cross_scale_prompt_interaction"] = {mode: interval(interactions["8B", mode]-interactions["1.7B", mode])
                                                  for mode in ("free", "structured")}
    report["scale_gap_1p7b_minus_8b"] = {}
    for mode in ("free", "structured"):
        for prompt in PROMPTS:
            delta = method_gaps["1.7B", mode][prompt] - method_gaps["8B", mode][prompt]
            report["scale_gap_1p7b_minus_8b"][f"{mode}/{prompt}"] = {
                "pooled": interval(delta.mean(axis=0)), "by_seed": [interval(d) for d in delta],
            }
    for scope in ("heldout", "common_long_correct"):
        delta = matched_gaps["1.7B", scope] - matched_gaps["8B", scope]
        report["scale_gap_1p7b_minus_8b"][f"matched/{scope}"] = {
            "pooled": interval(delta.mean(axis=0)), "by_seed": [interval(d) for d in delta],
        }
    return report


def score_fine_prefixes(data, output, model):
    from transformers import AutoTokenizer

    from opsdmath.evaluation.budget import score

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    labels = read_rows(data / "labels.jsonl")
    ids = [r["id"] for r in labels]
    truth = {r["id"]: number(r["answer"]) for r in labels}
    if len(ids) != len(truth) or None in truth.values():
        raise ValueError("Invalid or duplicate answer labels")
    for group in GROUPS:
        traces = read_rows(output / group / "standard-tokens.jsonl")
        if [r["id"] for r in traces] != ids:
            raise ValueError(f"Frozen IDs differ for {group}")
        for cap in sorted(set(FINE_BUDGETS) - set(BUDGETS)):
            _, per_item = score(traces, truth, tokenizer, cap)
            write(output / group / f"standard-{cap}.json", per_item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    summary = sub.add_parser("summarize")
    for field in ("data", "small", "large", "output"):
        summary.add_argument(f"--{field}", required=True, type=Path)
    summary.add_argument("--fine-grid", action="store_true")
    fine_prefixes = sub.add_parser("rescore-prefixes")
    for field in ("data", "output", "model"):
        fine_prefixes.add_argument(f"--{field}", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "rescore-prefixes":
        score_fine_prefixes(args.data, args.output, args.model)
    else:
        write(args.output, summarize(args.data, args.small, args.large, args.fine_grid))


if __name__ == "__main__":
    main()
