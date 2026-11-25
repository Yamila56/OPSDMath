"""Aggregate paired answer-reserve and MATH-500 comparisons."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from budget_study import uncertainty
from condition_pairs import cluster_interval
from gsm8k import read_rows


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_math500(trial, control):
    if trial.keys() != control.keys() or len(trial) != 500:
        raise ValueError("MATH-500 comparison must use the same 500 questions")
    diff = [int(row["correct"])-int(control[key]["correct"]) for key, row in trial.items()]
    strata = {}
    for key, row in trial.items():
        other = control[key]
        status = f"trial_{int(row['parsed'])}_control_{int(other['parsed'])}"
        counts = strata.setdefault(status, dict(n=0, trial_correct=0, control_correct=0))
        counts["n"] += 1
        counts["trial_correct"] += int(row["correct"])
        counts["control_correct"] += int(other["correct"])
    return dict(delta=mean(diff), interval=uncertainty(diff), repaired=diff.count(1), harmed=diff.count(-1),
                by_parsed_status=strata)


def pair_outcomes(rows):
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row["pair"]][row["variant"]] = row
    return {key: (r["base"]["family"], r["base"]["correct"] and r["changed"]["correct"])
            for key, r in pairs.items()}


def reserve_summary(root):
    reports = {str(n): read(root/f"{n}.json") for n in (16, 24, 32, 48)}
    if len({r["labels_sha256"] for r in reports.values()}) != 1:
        raise ValueError("Answer reserves were evaluated against different labels")
    result = {}
    for reserve, current in reports.items():
        deltas = {}
        for name, rows in current["rows"].items():
            trial, control = pair_outcomes(rows), pair_outcomes(reports["24"]["rows"][name])
            if trial.keys() != control.keys():
                raise ValueError(f"Answer reserve changed evaluated pairs for {name}")
            clusters = defaultdict(list)
            for key, (family, hit) in trial.items():
                clusters[family].append(int(hit)-int(control[key][1]))
            deltas[name] = dict(delta=mean(x for c in clusters.values() for x in c),
                                interval=cluster_interval(clusters))
        result[reserve] = dict(metrics=current["metrics"], versus_reserve_24=deltas)
    return result


def math500_summary(root):
    names = ("base", "answer17", "answer29", "constraint_brief17", "constraint_brief29", "fullsft17", "fullsft29")
    runs = {name: read(root/name/"complete.json") for name in names}
    dataset_ids = {r["dataset_sha256"] for r in runs.values()}
    if len(dataset_ids) != 1:
        raise ValueError("MATH-500 runs used different dataset bytes")
    result = {}
    delivery = {name: read(root/name/"delivery.json") for name in names}
    for budget in (128, 256, 512):
        outcomes = {name: {r["id"]: r for r in read_rows(root/name/f"scored-{budget}.jsonl")}
                    for name in names}
        base = outcomes["base"]
        deltas = {name: paired_math500(rows, base) for name, rows in outcomes.items()}
        matched = {}
        for seed in (17, 29):
            for trial, control in (("answer", "fullsft"), ("constraint_brief", "answer"),
                                   ("constraint_brief", "fullsft")):
                left, right = f"{trial}{seed}", f"{control}{seed}"
                matched[f"{left}:{right}"] = paired_math500(outcomes[left], outcomes[right])
        result[str(budget)] = dict(metrics={name: runs[name]["metrics"][str(budget)] for name in names},
                                   explicit_final_audit={name: delivery[name][str(budget)] for name in names},
                                   versus_base=deltas, same_seed_comparisons=matched)
    return dict(dataset_sha256=dataset_ids.pop(), budgets=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--controls-only", action="store_true",
                        help="Summarize the fixed-budget and trainer controls before external evaluation completes.")
    args = parser.parse_args()
    output = dict(answer_reserve=reserve_summary(args.run/"answer-reserve"),
                  upstream_objective=read(args.run/"upstream-objective-parity.json"))
    trainer = {}
    for variant in ("verified", "deterministic"):
        smoke = args.run/f"upstream-trainer-{variant}"
        if (smoke/"DONE").exists():
            trainer[variant] = dict(resumed=[read(smoke/f"step-{step}.json") for step in (1, 2)],
                                    uninterrupted=read(smoke/"uninterrupted.json"))
    if trainer:
        output["upstream_trainer_smoke"] = trainer
    if not args.controls_only:
        output["math500"] = math500_summary(args.run/"math500")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
