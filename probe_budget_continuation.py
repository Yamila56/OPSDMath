"""Freeze a GSM8K development pool and compare teachers on identical token prefixes."""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from gsm8k import NUMBER, dataset, number, parse_answer, parse_strict_answer, read_rows, write_rows
from short_sft_baseline import EQUATIONS

SHORT, LONG = 128, 512
CUTS = (32, 64, 96)
CONTEXTS = ("none", "answer", "masked_equations")
COHORTS = ("early", "late", "early_wrong_eos", "other_wrong")
SEED = 672384
POOL_SIZE = 384
PER_COHORT = 24


def masked_equations(row):
    """Remove every numeric literal equal to the final answer, including intermediate uses."""
    gold = number(row["answer"])
    pattern = re.compile(r"[0-9][0-9]*(?:,[0-9]{3})*(?:\.[0-9]+)?")
    notes = []
    for expression in EQUATIONS.findall(row["solution"]):
        masked = pattern.sub(lambda match: "?" if number(match[0]) in (gold, abs(gold)) else match[0], expression)
        # A final expression with a hidden result can still be evaluated, which is intentional.
        if masked != "=" and any(char.isdigit() for char in masked):
            notes.append(masked)
    return "\n".join(notes)


def prepare(args):
    train, _ = dataset("train", args.source)
    test, _ = dataset("test", args.source)
    excluded = set(test) | {r["id"] for path in args.exclude for r in read_rows(path)}
    candidates = sorted(train.keys() - excluded)
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    pool = [{**train[key], "masked_equations": masked_equations(train[key])}
            for key in candidates[:POOL_SIZE]]
    # The masked-reference intervention is defined only on items with a usable reference.
    pool = [row for row in pool if row["masked_equations"] and
            number(row["answer"]) not in {number(match[0]) for match in NUMBER.finditer(row["masked_equations"])}]
    args.output.parent.mkdir(exist_ok=True, parents=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    write_rows(args.output, pool)
    print(dict(pool=len(pool), excluded=len(excluded), sha256=sha(args.output)))


def sha(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def generate(model, tokenizer, inputs, cap, batch_size):
    from run_math import batch_suffixes

    for offset in range(0, len(inputs), batch_size):
        yield from batch_suffixes(model, tokenizer, inputs[offset:offset + batch_size],
                                  cap, False, model.device)


def classify(row, tokens, tokenizer, eos_ids):
    gold = number(row["answer"])
    short = tokenizer.decode(tokens[:SHORT], skip_special_tokens=True)
    long = tokenizer.decode(tokens, skip_special_tokens=True)
    if parse_answer(short) == gold:
        return "early"
    if parse_answer(long) == gold:
        return "late"
    if len(tokens) <= SHORT and tokens[-1] in eos_ids:
        return "early_wrong_eos"
    return "other_wrong"


def run(args):
    import torch

    from opsd_calibration import load_model
    from run_math import prompt

    args.output.mkdir(parents=True, exist_ok=False)
    rows = read_rows(args.pool)
    model, tokenizer = load_model(args.model)
    eos_id = model.generation_config.eos_token_id
    eos_ids = {eos_id} if isinstance(eos_id, int) else set(eos_id)
    student_inputs = [prompt(tokenizer, row, output_budget=SHORT) for row in rows]
    first = min(args.batch_size, len(rows))
    direct_short = list(generate(model, tokenizer, student_inputs[:first], SHORT, args.batch_size))
    grouped = defaultdict(list)
    traces = []
    for i, (tokens, hit_cap) in enumerate(generate(model, tokenizer, student_inputs, LONG, args.batch_size)):
        if i < first and tokens[:SHORT] != direct_short[i][0]:
            raise ValueError("Direct short generation differs from the long trajectory prefix")
        category = classify(rows[i], tokens, tokenizer, eos_ids)
        traces.append(dict(id=rows[i]["id"], tokens=tokens, cohort=category, hit_cap=hit_cap))
        grouped[category].append((rows[i], tokens))
    write_rows(args.output / "base-tokens.jsonl", traces)
    selected = [(category, row, tokens) for category in COHORTS for row, tokens in grouped[category][:PER_COHORT]]
    write_rows(args.output / "selected.jsonl", [dict(id=row["id"], cohort=category) for category, row, _ in selected])
    write_summary(args.output / "protocol.json", dict(pool_sha256=sha(args.pool), model=str(args.model),
                  model_index_sha256=sha(args.model / "model.safetensors.index.json"),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(), batch=args.batch_size,
                  quotas=PER_COHORT, counts={category: len(grouped[category]) for category in COHORTS},
                  cuts=CUTS, eligible=sum(len(tokens) > cut or
                      (len(tokens) == cut and tokens[-1] not in eos_ids)
                      for _, _, tokens in selected for cut in CUTS)))
    print(f"Selected {len(selected)} items from {len(rows)} base trajectories", flush=True)
    results = []
    for cut in CUTS:
        eligible = [(category, row, tokens[:cut]) for category, row, tokens in selected
                    if len(tokens) >= cut and tokens[cut - 1] not in eos_ids]
        for context in CONTEXTS:
            for remaining_hint in (False, True):
                inputs = []
                for _, row, prefix in eligible:
                    example = dict(row)
                    if remaining_hint:
                        example["problem"] += f"\n\nAn answer prefix of {cut} tokens has already been written. Continue it within the remaining {SHORT-cut} tokens."
                    teacher_context = context
                    if context == "masked_equations":
                        example["general_brief"] = example["masked_equations"]
                        teacher_context = "general_brief"
                    teacher = prompt(tokenizer, example, privileged=context != "none",
                                     context=teacher_context, output_budget=SHORT)
                    inputs.append(torch.cat((teacher, torch.tensor([prefix])), dim=1))
                continuations = generate(model, tokenizer, inputs, SHORT - cut, args.batch_size)
                for (category, row, prefix), (suffix, hit_cap) in zip(eligible, continuations, strict=True):
                    combined = tokenizer.decode(prefix + suffix, skip_special_tokens=True)
                    gold = number(row["answer"])
                    results.append(dict(id=row["id"], cohort=category, cut=cut, context=context,
                                        remaining_hint=remaining_hint, suffix=suffix, text=combined,
                                        numeric=parse_answer(combined) == gold,
                                        strict=parse_strict_answer(combined) == gold, hit_cap=hit_cap))
                write_rows(args.output / "continuations.jsonl", results)
                print(f"cut={cut} context={context} hint={remaining_hint} n={len(eligible)}", flush=True)
    summarize(results, args.output / "summary.json")


def write_summary(path, values):
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize(results, output):
    groups = defaultdict(Counter)
    for row in results:
        key = f"{row['cohort']}/{row['cut']}/{row['context']}/{row['remaining_hint']}"
        groups[key].update(n=1, numeric=row["numeric"], strict=row["strict"], hit_cap=row["hit_cap"],
                           continuation_tokens=len(row["suffix"]))
    late = {(r["id"], r["context"], r["remaining_hint"]): r for r in results
            if r["cohort"] == "late" and r["cut"] == 64}
    comparisons = {}
    for context, hint in (("masked_equations", False), ("none", True), ("answer", False)):
        differences = [int(r["numeric"]) - int(late[key, context, hint]["numeric"])
                       for (key, source, remaining), r in late.items()
                       if source == "masked_equations" and remaining]
        comparisons[f"{context}/{hint}"] = dict(n=len(differences), improved=differences.count(1),
                                               regressed=differences.count(-1))
    write_summary(output, dict(groups=dict(groups), late_64_numeric_paired=comparisons,
                              audit_required="Judge whether the derivation supports each numeric success before selecting a teacher."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", required=True, type=Path)
    prep.add_argument("--exclude", required=True, type=Path, nargs="+")
    prep.add_argument("--output", required=True, type=Path)
    probe = commands.add_parser("run")
    probe.add_argument("--model", required=True, type=Path)
    probe.add_argument("--pool", required=True, type=Path)
    probe.add_argument("--output", required=True, type=Path)
    probe.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
