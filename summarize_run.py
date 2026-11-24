"""Recompute a formal OPSD run and export the metrics, timings and disagreement cases."""

import argparse
import csv
import json
import random
import re
from pathlib import Path

from gsm8k import decorated_number, number, parse_answer, read_rows, score, write_rows


INLINE_FINAL = re.compile(r"\bFinal answer\**[ \t]*:[ \t]*\**[ \t]*([^\r\n]+)", re.IGNORECASE)
UNIT_ENDING = re.compile(
    r"\s+(?:seconds?|minutes?|hours?|days?|weeks?|years?|inches?|feet|foot|yards?|meters?|miles?|mph|"
    r"cents?|dollars?|pounds?|grams?|kilograms?|liters?|gallons?)\.?\**$", re.IGNORECASE,
)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_final_answer(response):
    """Retrospective format check on explicit final lines with inline markers or plain-language units."""
    primary = parse_answer(response)
    answers = INLINE_FINAL.findall(response)
    if primary is not None or len(answers) != 1:
        return primary
    return decorated_number(UNIT_ENDING.sub("", answers[0].strip()))


def summarize(run, data, reused_base=False):
    names = ("base", "sft", "opsd")
    prediction_paths = {name: run / f"eval-{name}" / f"{name}.jsonl" for name in names}
    labels_path = data / "test_labels.jsonl"
    results = score(labels_path, list(prediction_paths.values()), None)
    if results != json.loads((run / "results.json").read_text(encoding="utf-8")):
        raise ValueError("Recomputed scores differ from the cloud run")
    write_csv(run / "results.csv", [{"method": name, **results["metrics"][name]} for name in names])

    probe_phases = [phase for phase in ("probe", "probe-reload") if (run / phase / "events.jsonl").is_file()]
    phases = probe_phases + ["train-opsd", "train-sft"] + ([] if reused_base else ["eval-base"]) + ["eval-sft", "eval-opsd"]
    timings = []
    for phase in phases:
        events = read_rows(run / phase / "events.jsonl")
        summary = events[-1]
        if summary["event"] != "complete":
            raise ValueError(f"Phase {phase} did not complete")
        processing = [row for row in events if row["event"] in ("microstep", "prediction_batch")]
        is_training = any(row["event"] == "microstep" for row in processing)
        timings.append({
            "phase": phase,
            "items": sum(1 if is_training else row["count"] for row in processing),
            "updates": sum(row["event"] == "update" for row in events),
            "completion_tokens": sum(row["tokens"] for row in processing),
            "seconds": round(summary["seconds"], 2),
            "peak_allocated_mib": summary["peak_allocated_mib"],
            "peak_reserved_mib": summary["peak_reserved_mib"],
        })
    write_csv(run / "timings.csv", timings)

    problems = {row["id"]: row["problem"] for row in read_rows(data / "test_inputs.jsonl")}
    truth = {row["id"]: row["answer"] for row in read_rows(labels_path)}
    predictions = {name: {row["id"]: row for row in read_rows(path)} for name, path in prediction_paths.items()}
    format_audit = []
    audit_outcomes = {}
    for name in names:
        values = {key: audit_final_answer(row["response"]) for key, row in predictions[name].items()}
        audit_outcomes[name] = [int(values[key] == number(answer)) for key, answer in truth.items()]
        audit_correct = sum(audit_outcomes[name])
        format_audit.append({
            "method": name, "n": len(truth), "primary_correct": results["metrics"][name]["correct"],
            "audit_correct": audit_correct, "audit_accuracy": audit_correct / len(truth),
            "audit_valid": sum(value is not None for value in values.values()),
        })
    write_csv(run / "format_sensitivity.csv", format_audit)
    audit_deltas = [a - b for a, b in zip(audit_outcomes["opsd"], audit_outcomes["sft"], strict=True)]
    rng = random.Random(17)
    audit_draws = sorted(sum(rng.choices(audit_deltas, k=len(truth))) / len(truth) for _ in range(10000))
    format_comparison = {"delta": sum(audit_deltas) / len(truth), "ci95": [audit_draws[250], audit_draws[9749]]}
    (run / "format_comparison.json").write_text(json.dumps(format_comparison, indent=2) + "\n", encoding="utf-8")
    coverage = {
        (True, True): {"parse_group": "both", "n": 0, "opsd_correct": 0, "sft_correct": 0},
        (True, False): {"parse_group": "opsd_only", "n": 0, "opsd_correct": 0, "sft_correct": 0},
        (False, True): {"parse_group": "sft_only", "n": 0, "opsd_correct": 0, "sft_correct": 0},
        (False, False): {"parse_group": "neither", "n": 0, "opsd_correct": 0, "sft_correct": 0},
    }
    cases = []
    for item, answer in sorted(truth.items()):
        parsed = {name: parse_answer(predictions[name][item]["response"]) for name in names}
        records = {
            name: {**predictions[name][item], "correct": parsed[name] == number(answer)}
            for name in names
        }
        group = coverage[parsed["opsd"] is not None, parsed["sft"] is not None]
        group["n"] += 1
        group["opsd_correct"] += records["opsd"]["correct"]
        group["sft_correct"] += records["sft"]["correct"]
        if records["sft"]["correct"] != records["opsd"]["correct"]:
            cases.append({"id": item, "problem": problems[item], "answer": answer, "predictions": records})
    write_rows(run / "disagreements.jsonl", cases)
    write_csv(run / "parse_coverage.csv", list(coverage.values()))
    print(json.dumps({"timings": timings, **results}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--reused-base", action="store_true", help="Skip timing for the unchanged base predictions.")
    args = parser.parse_args()
    summarize(args.run, args.data, reused_base=args.reused_base)
