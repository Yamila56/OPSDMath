"""Audit and compare the outputs under the shared numeric answer field."""
import argparse
import csv
import json
from hashlib import sha256
from pathlib import Path
from statistics import mean

from budget_study import uncertainty
from opsdmath.data.gsm8k import number, read_rows


def audit(row, budget=128, draft_cap=104):
    charged = sum(len(row[field]) for field in ("draft_token_ids", "marker_token_ids", "final_token_ids"))
    if charged != row["generated_tokens"] or charged > budget or len(row["draft_token_ids"]) > draft_cap:
        raise ValueError(f"Token-budget mismatch for {row['id']}")
    marker = "Final answer: "
    line = row["response"]
    if not line.startswith(marker) or "\n" in line:
        raise ValueError(f"Malformed numeric field for {row['id']}")
    parsed = number(line[len(marker):]) if row["final_eos"] else None
    if row["parsed"] != (str(parsed) if parsed is not None else None):
        raise ValueError(f"Incorrect value or EOS handling for {row['id']}")
    return parsed


def table(path, items):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("runs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    label_rows = read_rows(args.labels)
    gold = {r["id"]: number(r["answer"]) for r in label_rows}
    if len(gold) != len(label_rows) or None in gold.values():
        raise ValueError("Duplicate or invalid reference labels")
    args.output.mkdir(parents=True, exist_ok=False)
    metrics, comparisons = [], []
    for seed in (17, 29):
        outcomes = {}
        for family in ("answer", "equationsft"):
            method = f"{family}{seed}"
            raw = args.runs / method / "predictions.jsonl"
            rows = read_rows(raw)
            if {r["id"] for r in rows} != gold.keys() or len(rows) != len(gold):
                raise ValueError(f"Unexpected question IDs in {method}")
            values = {r["id"]: audit(r) for r in rows}
            result = {key: value == gold[key] for key, value in values.items()}
            outcomes[family] = result
            metrics.append(dict(
                method=method, seed=seed, n=len(rows), correct=sum(result.values()),
                valid=sum(value is not None for value in values.values()),
                final_eos=sum(r["final_eos"] for r in rows), draft_hit_cap=sum(r["draft_hit_cap"] for r in rows),
                mean_tokens=mean(r["generated_tokens"] for r in rows), predictions_sha256=sha256(raw.read_bytes()).hexdigest(),
            ))
        diffs = [int(outcomes["answer"][key]) - int(outcomes["equationsft"][key]) for key in gold]
        low, high = uncertainty(diffs)
        comparisons.append(dict(seed=seed, n=len(diffs), delta=mean(diffs), low=low, high=high,
                                opsd_only_correct=diffs.count(1), sft_only_correct=diffs.count(-1)))
    table(args.output / "metrics.csv", metrics)
    table(args.output / "comparisons.csv", comparisons)


if __name__ == "__main__":
    main()
