"""Generate and score held-out math questions with controlled condition changes."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

from gsm8k import number, read_rows, write_rows


def case(family, a, b, c):
    # Each entry returns a base story, an operative clause, an inert clause,
    # a shared question, and two independently computable answers.
    cases = [
        (f"There are {a} trays with {b} muffins per tray.", f"{c} muffins were discarded.",
         f"{c} muffins have a blue wrapper.", "How many muffins are there to serve?", a*b, a*b-c),
        (f"Omar works {a} hours a day for {b} days.", f"On the final day he works {c} extra hours.",
         f"On the final day he takes {c} photos.", "How many hours does he work?", a*b, a*b+c),
        (f"A tank has {a*b} liters of water. It loses {b} liters each day for {c} days.",
         f"At the end, {a} liters are added.", f"The tank is {a} years old.",
         "How many liters remain?", a*b-b*c, a*b-b*c+a),
        (f"{b} students will split {a*b} stickers equally.", f"Before sharing, {b*c} more stickers are added.",
         f"The stickers come in {c} designs.", "How many stickers will each student get?", a, a+c),
        (f"A shop has {a} scarves at {b} dollars apiece.",
         f"One scarf costs {c} dollars less than the posted price.", f"{c} scarves are blue.",
         "What is the total price in dollars for buying every scarf?", a*b, a*b-c),
        (f"A runner completes {a} laps around a {100*b}-meter track.",
         f"She also runs {100*c} meters before the laps.", f"Her jersey number is {c}.",
         "How many meters does she run in total?", a*b*100, (a*b+c)*100),
        (f"A garden measures {a} meters by {b} meters.", f"A {c}-meter-wide strip along the {a}-meter side is unplanted.",
         f"The garden has {c} signs.", "How many square meters are planted?", a*b, a*(b-c)),
        (f"Mei pays {a*b} dollars for {b} identical books.",
         f"The payment includes a total delivery fee of {b*c} dollars.",
         f"The delivery arrived after {c} days.", "What is the price of each book without delivery?", a, a-c),
        (f"A ticket costs {a} dollars and a snack costs {b} dollars. Lee buys {c} of each.",
         "A coupon makes one snack free.", "Lee pays using a blue card.",
         "How many dollars does Lee pay?", c*(a+b), c*(a+b)-b),
        (f"A printer prints {a} pages per minute. It runs for {b} minutes.",
         f"It pauses for the last {c} minutes of that interval.", f"The printer has {c} buttons.",
         "How many pages does it print?", a*b, a*(b-c)),
        (f"Rita saves {a} dollars per week. She starts with no savings and saves for {b} weeks.",
         f"At the end of week one she receives and saves an extra {c*a} dollars.",
         f"In week one she writes {c} journal entries.",
         "How many dollars has she saved by the end?", a*b, a*(b+c)),
        (f"A bus with {a*b} seats has {a} rows with {b} seats each.",
         f"The rear {c} rows are reserved and unavailable to the public.",
         f"The bus serves {c} routes.", "How many seats are available to the public?", a*b, (a-c)*b),
        (f"A recipe makes {a} servings with {b} grams of spice per serving.",
         f"For the first {c} servings use half the spice.", f"The spice was bought {c} days ago.",
         "How many grams of spice are used?", a*b, a*b-c*(b//2)),
        (f"An orchard has {a} trees with {b} apples each.",
         f"Each of the first {c} trees has one extra apple.", f"Each of the first {c} trees has a yellow tag.",
         "How many apples are on the trees?", a*b, a*b+c),
        (f"A school buys {a} packs of {b} tiles and uses two packs for a project.",
         f"It also uses {c} tiles from another pack.", f"The pack wrappers have {c} colors.",
         "How many unused tiles are left?", (a-2)*b, (a-2)*b-c),
        (f"A delivery van makes {a} trips carrying {b} boxes on each trip.",
         f"On the final {c} trips, one box on each trip is returned to the warehouse.",
         f"On the final {c} trips, the driver wears a hat.", "How many boxes are delivered and kept?", a*b, a*b-c),
    ]
    return cases[family]


def scope_case(family, a, b, c):
    examples = [
        ("boxes", "bottles", "are broken", "intact bottles", "bottles originally packed"),
        ("shelves", "books", "are reserved", "unreserved books", "books originally shelved"),
        ("trays", "rolls", "are set aside", "rolls still available", "rolls originally baked"),
        ("crates", "jars", "are cracked", "uncracked jars", "jars originally loaded"),
        ("cabinets", "files", "are archived", "unarchived files", "files originally sorted"),
        ("lots", "bikes", "are rented out", "bikes still available", "bikes originally parked"),
        ("drawers", "badges", "are marked invalid", "valid badges", "badges originally stored"),
        ("envelopes", "tickets", "are voided", "valid tickets", "tickets originally packed"),
    ]
    container, item, modifier, wanted, total = examples[family]
    story = f"There are {a} {container}, originally with {b} {item} each."
    clause = f"Across all the {container} combined, {c} {item} {modifier}."
    scoped = f"In each of the {container}, {c} {item} {modifier}."
    query = f"How many {wanted} are there?"
    return [
        ("base", f"{story} {query}", a*b),
        ("changed", f"{story} {clause} {query}", a*b-c),
        ("inert", f"{story} {clause} How many {total} were there?", a*b),
        ("relation", f"{story} {scoped} {query}", a*(b-c)),
        ("number", f"There are {a} {container}, originally with {b+1} {item} each. {query}", a*(b+1)),
    ]


def prepare(args):
    rng = random.Random(87231)
    for split, families in (("dev", [*range(4), 16, 17]), ("test", [*range(4, 16), *range(18, 24)])):
        folder = args.output / split
        folder.mkdir(parents=True, exist_ok=False)
        inputs, labels = [], []
        for family in families:
            values = rng.sample([(a, b) for a in range(9, 24) for b in range(8, 19, 2)], args.per_family)
            for index, (a, b) in enumerate(values):
                c = rng.randint(2, 6)
                pair = f"{family:02}-{index:03}"
                if family >= 16:
                    variants = scope_case(family-16, a, b, c)
                    base, changed = variants[0][2], variants[1][2]
                else:
                    story, relevant, inert, query, base, changed = case(family, a, b, c)
                    variants = [(variant, " ".join(part for part in (story, clause, query) if part), value)
                                for variant, clause, value in (("base", "", base), ("changed", relevant, changed),
                                                               ("inert", inert, base))]
                if base == changed:
                    raise ValueError(f"Condition failed to change the answer for {pair}")
                for variant, question, value in variants:
                    key = f"{pair}-{variant}"
                    inputs.append(dict(id=key, question=question))
                    labels.append(dict(id=key, family=family, pair=pair, variant=variant,
                                       answer=str(value), baseline=str(base)))
        rng.shuffle(inputs)
        if len({r["question"] for r in inputs}) != len(inputs):
            raise ValueError("Duplicate questions")
        write_rows(folder / "inputs.jsonl", inputs)
        write_rows(folder / "labels.jsonl", labels)


def cluster_interval(diffs):
    rng = random.Random(229)
    families = list(diffs)
    samples = sorted(mean(value for family in rng.choices(families, k=len(families))
                          for value in diffs[family]) for _ in range(5000))
    return [samples[125], samples[4874]]


def structural_groups(diffs):
    groups = {family: values for family, values in diffs.items() if family < 16}
    shared_scope = [value for family, values in diffs.items() if family >= 16 for value in values]
    if shared_scope:
        groups["shared_scope_template"] = shared_scope
    return groups


def score(args):
    from score_structured_answer import audit

    labels = read_rows(args.labels)
    keys = {r["id"] for r in labels}
    related_truth = {(r["pair"], r["variant"]): number(r["answer"]) for r in labels}
    outcomes, details, metrics = {}, {}, {}
    for spec in args.predictions:
        method, raw_path = spec.split("=", 1)
        predictions = read_rows(Path(raw_path))
        if len(predictions) != len(keys) or {r["id"] for r in predictions} != keys:
            raise ValueError(f"Unexpected prediction IDs for {method}")
        values = {r["id"]: audit(r, budget=args.budget, draft_cap=args.budget - args.answer_reserve)
                  for r in predictions}
        paired, copied, spurious, correct, valid = {}, {}, {}, [], []
        completed = {}
        capped = {r["id"]: r["draft_hit_cap"] for r in predictions}
        rows = []
        for row in labels:
            value = values[row["id"]]
            ok = value == number(row["answer"])
            correct.append(ok)
            valid.append(value is not None)
            completed.setdefault(row["pair"], {})[row["variant"]] = value is not None
            paired.setdefault(row["pair"], {"family": row["family"]})[row["variant"]] = ok
            if row["variant"] == "changed":
                copied[row["pair"]] = value == number(row["baseline"])
            if row["variant"] == "inert":
                spurious[row["pair"]] = value is not None and value != number(row["baseline"])
            rows.append(dict(row, parsed=str(value) if value is not None else None, correct=ok,
                             draft_hit_cap=capped[row["id"]]))
        outcomes[method] = paired
        details[method] = rows
        scoped = [r for r in paired.values() if "relation" in r]
        scoped_labels = [r for r in labels if r["variant"] == "relation"]
        base_correct = sum(r["base"] for r in paired.values())
        metrics[method] = dict(n=len(correct), correct=sum(correct), valid=sum(valid), pairs=len(paired),
                               mean_output_tokens=mean(r["generated_tokens"] for r in predictions),
                               draft_cap_hits=sum(capped.values()),
                               incorrect_at_draft_cap=sum(r["draft_hit_cap"] and not r["correct"] for r in rows),
                               base=sum(r["base"] for r in paired.values()),
                               changed=sum(r["changed"] for r in paired.values()),
                               inert=sum(r["inert"] for r in paired.values()),
                               change_pair=sum(r["base"] and r["changed"] for r in paired.values()),
                               inert_pair=sum(r["base"] and r["inert"] for r in paired.values()),
                               change_pair_completed=sum(r["base"] and r["changed"] for r in completed.values()),
                               inert_pair_completed=sum(r["base"] and r["inert"] for r in completed.values()),
                               unchanged_answer_on_changed=sum(copied.values()), spurious_change_on_inert=sum(spurious.values()),
                               unchanged_answer_given_base=sum(hit and paired[k]["base"] for k, hit in copied.items()),
                               spurious_change_given_base=sum(hit and paired[k]["base"] for k, hit in spurious.items()),
                               changed_given_base=sum(r["base"] and r["changed"] for r in paired.values())/base_correct
                               if base_correct else None,
                               scoped_n=len(scoped), relation=sum(r["relation"] for r in scoped),
                               relation_draft_cap_hits=sum(capped[r["id"]] for r in scoped_labels),
                               relation_valid=sum(values[r["id"]] is not None for r in scoped_labels),
                               relation_used_global_count=sum(
                                   values[r["id"]] == related_truth[r["pair"], "changed"] for r in scoped_labels),
                               numeric=sum(r["number"] for r in scoped),
                               all_variants=sum(all(v for k, v in r.items() if k != "family") for r in paired.values()))
    comparisons = {}
    for left, right in (pair.split(":", 1) for pair in args.compare):
        results = {}
        for metric in ("changed", "inert"):
            diffs = {}
            for pair, a in outcomes[left].items():
                b = outcomes[right][pair]
                diffs.setdefault(a["family"], []).append(
                    int(a["base"] and a[metric]) - int(b["base"] and b[metric]))
            structural = structural_groups(diffs)
            shared = [(a[metric], outcomes[right][pair][metric]) for pair, a in outcomes[left].items()
                      if a["base"] and outcomes[right][pair]["base"]]
            results[metric] = dict(delta=mean(v for d in diffs.values() for v in d), interval=cluster_interval(diffs),
                                   structural_interval=cluster_interval(structural),
                                   structural_clusters=len(structural),
                                   common_base_n=len(shared), left_correct=sum(a for a, _ in shared),
                                   right_correct=sum(b for _, b in shared))
        comparisons[f"{left}:{right}"] = results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(metrics=metrics, comparisons=comparisons, rows=details,
                                           labels_sha256=hashlib.sha256(args.labels.read_bytes()).hexdigest()),
                                     indent=2),
                           encoding="utf-8")


def compare_budgets(args):
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (args.lower, args.higher)]
    if reports[0]["labels_sha256"] != reports[1]["labels_sha256"]:
        raise ValueError("Budget comparisons require the same frozen labels")
    tables = [report["rows"] for report in reports]
    if tables[0].keys() != tables[1].keys():
        raise ValueError("Budget comparisons require the same model groups")

    def paired_results(rows, variant):
        questions = {(row["pair"], row["variant"]): row for row in rows}
        return {
            row["pair"]: (row["family"], row["correct"] and questions[row["pair"], variant]["correct"])
            for row in rows if row["variant"] == "base"
        }

    output = {}
    for variant in ("changed", "inert"):
        lower = {name: paired_results(rows, variant) for name, rows in tables[0].items()}
        higher = {name: paired_results(rows, variant) for name, rows in tables[1].items()}
        results = {}
        for name in lower:
            group_diffs, interactions = {}, {}
            for key, (family, low) in lower[name].items():
                high_family, high = higher[name][key]
                if family != high_family:
                    raise ValueError(f"Inconsistent family for {key}")
                delta = int(high) - int(low)
                base_delta = int(higher["base"][key][1]) - int(lower["base"][key][1])
                group_diffs.setdefault(family, []).append(delta)
                interactions.setdefault(family, []).append(delta-base_delta)
            results[name] = dict(n=len(lower[name]), lower_correct=sum(r[1] for r in lower[name].values()),
                                 higher_correct=sum(r[1] for r in higher[name].values()),
                                 gain=mean(v for group in group_diffs.values() for v in group),
                                 gain_interval=cluster_interval(group_diffs),
                                 extra_gain_vs_base=mean(v for group in interactions.values() for v in group),
                                 extra_gain_vs_base_interval=cluster_interval(interactions),
                                 extra_gain_vs_base_structural_interval=cluster_interval(structural_groups(interactions)))
        output[variant] = results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("output", type=Path)
    build.add_argument("--per-family", type=int, default=16)
    report = commands.add_parser("score")
    report.add_argument("labels", type=Path)
    report.add_argument("output", type=Path)
    report.add_argument("predictions", nargs="+")
    report.add_argument("--compare", nargs="*", default=[])
    report.add_argument("--budget", type=int, default=128)
    report.add_argument("--answer-reserve", type=int, default=24)
    budgets = commands.add_parser("compare-budgets")
    budgets.add_argument("lower", type=Path)
    budgets.add_argument("higher", type=Path)
    budgets.add_argument("output", type=Path)
    args = parser.parse_args()
    actions = {"prepare": prepare, "score": score, "compare-budgets": compare_budgets}
    actions[args.command](args)


if __name__ == "__main__":
    main()
