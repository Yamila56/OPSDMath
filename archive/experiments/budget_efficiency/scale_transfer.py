"""Historical GSM8K and structured-answer scale-transfer protocol."""

import argparse
import json
import time
from pathlib import Path
from statistics import mean

import numpy as np

from opsdmath.evaluation.budget import score, validate_rows
from opsdmath.evaluation.answer_parser import number
from opsdmath.io import digest, read_rows, write_rows

BUDGETS = (128, 256, 512)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def saved_rows(path, inputs):
    if not path.exists():
        return None
    rows = read_rows(path)
    if [row["id"] for row in rows] != [row["id"] for row in inputs]:
        raise ValueError(f"Saved rows in {path} do not match the evaluation input order")
    return rows


def save_rows(path, rows):
    temporary = path.with_suffix(".jsonl.tmp")
    write_rows(temporary, rows)
    temporary.replace(path)


def natural_rollout(model, tokenizer, rows, cap, budget_hint, batch_size):
    from opsdmath.training.prompts import prompt
    from opsdmath.training.rollout import batch_suffixes

    records = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        prompts = [prompt(tokenizer, row, output_budget=cap if budget_hint else None) for row in batch]
        suffixes = batch_suffixes(model, tokenizer, prompts, cap, False, model.device)
        if not budget_hint and cap == 512 and start == 0:
            probe = batch_suffixes(model, tokenizer, prompts, 128, False, model.device)
            if any(ids[:128] != shorter for (ids, _), (shorter, _) in zip(suffixes, probe, strict=True)):
                raise RuntimeError("Direct short rollout differs from the cached 512-token prefix")
        records.extend(dict(id=row["id"], token_ids=ids, hit_cap=capped)
                       for row, (ids, capped) in zip(batch, suffixes, strict=True))
        print(json.dumps(dict(protocol="natural", hint=budget_hint, cap=cap,
                              processed=len(records), total=len(rows))), flush=True)
    return records


def evaluate(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from archive.experiments.budget_efficiency.structured import generate

    gsm, labels = validate_rows(args.gsm8k/"inputs.jsonl", args.gsm8k/"labels.jsonl", 512)
    condition, _ = validate_rows(args.conditions/"inputs.jsonl", args.conditions/"labels.jsonl", 288)
    truth = {r["id"]: number(r["answer"]) for r in labels}
    if None in truth.values():
        raise ValueError("Invalid GSM8K reference number")
    if args.batch_size < 1:
        raise ValueError("Positive batch size required")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if (args.output/"complete.json").exists():
        raise ValueError(f"Evaluation already completed in {args.output}")
    identity = dict(
        gsm8k_sha256=digest(args.gsm8k/"labels.jsonl"), condition_sha256=digest(args.conditions/"labels.jsonl"),
        model_sha256=digest(args.model/"model.safetensors.index.json"),
        gsm8k_inputs_sha256=digest(args.gsm8k/"inputs.jsonl"),
        condition_inputs_sha256=digest(args.conditions/"inputs.jsonl"), batch_size=args.batch_size,
        adapter_sha256=digest(args.adapter/"adapter_model.safetensors") if args.adapter else None,
        adapter_config_sha256=digest(args.adapter/"adapter_config.json") if args.adapter else None,
    )
    identity_path = args.output/"evaluation-inputs.json"
    if identity_path.exists():
        saved_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if saved_identity != identity:
            raise ValueError(f"Evaluation inputs changed since the previous run in {args.output}")
    else:
        write(identity_path, identity)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True,
    )
    base.generation_config.pad_token_id = tokenizer.pad_token_id
    model = (PeftModel.from_pretrained(base, args.adapter) if args.adapter else base).eval()
    natural = {}
    for setting in ("standard", "hint"):
        started = time.perf_counter()
        natural[setting] = {}
        for cap in BUDGETS:
            hinted = setting == "hint"
            rollout_cap = cap if hinted else max(BUDGETS)
            if hinted or cap == min(BUDGETS):
                path = args.output/f"tokens-{setting}-{rollout_cap}.jsonl"
                responses = saved_rows(path, gsm)
                if responses is None:
                    responses = natural_rollout(model, tokenizer, gsm, rollout_cap, hinted, args.batch_size)
                    save_rows(path, responses)
                elif any(len(row["token_ids"]) > rollout_cap or
                         (row["hit_cap"] and len(row["token_ids"]) != rollout_cap) for row in responses):
                    raise ValueError(f"Saved token rows exceed the budget or have an inconsistent cap in {path}")
            natural[setting][str(cap)], per_parser = score(responses, truth, tokenizer, cap)
            write(args.output/f"gsm8k-{setting}-{cap}.json", per_parser)
        path = args.output/f"condition-{setting}.jsonl"
        if saved_rows(path, condition) is None:
            predictions = []
            for start_index in range(0, len(condition), args.batch_size):
                predictions.extend(generate(model, tokenizer, condition[start_index:start_index+args.batch_size],
                                            budget_hint=setting == "hint"))
                print(json.dumps(dict(protocol="structured", hint=setting == "hint", cap=128,
                                      processed=len(predictions), total=len(condition))), flush=True)
            save_rows(path, predictions)
        print(json.dumps(dict(setting=setting, seconds=time.perf_counter()-started)), flush=True)
    write(args.output/"complete.json", dict(
        **identity,
        natural=natural, seconds=time.perf_counter()-start, peak_mib=torch.cuda.max_memory_allocated()/2**20))


def match_questions(*score_sets):
    groups = [{row["id"]: row for row in scores} for scores in score_sets]
    if not groups[0] or any(group.keys() != groups[0].keys() or len(group) != len(scores)
                            for group, scores in zip(groups, score_sets, strict=True)):
        raise ValueError("Paired outputs must use identical, non-repeated question IDs")
    return groups


def paired_interval(diff, categories):
    frequencies = np.array([diff.count(value) for value in categories])/len(diff)
    draws = np.random.default_rng(55).multinomial(len(diff), frequencies, size=10000)
    return np.quantile((draws @ categories)/len(diff), [.025, .975]).tolist()


def paired(a, b):
    a, b = match_questions(a, b)
    diff = [int(a[i]["correct"])-int(b[i]["correct"]) for i in a]
    common = [d for i, d in zip(a, diff, strict=True) if a[i]["parsed"] and b[i]["parsed"]]
    return dict(delta=mean(diff), interval=paired_interval(diff, np.arange(-1, 2)),
                repaired=diff.count(1), harmed=diff.count(-1),
                common_parsed=len(common), common_net=sum(common))


def prompt_interaction(standard_left, standard_right, hint_left, hint_right):
    a, b, c, d = match_questions(standard_left, standard_right, hint_left, hint_right)
    diff = [int(a[i]["correct"])-int(b[i]["correct"])-int(c[i]["correct"])+int(d[i]["correct"]) for i in a]
    return dict(delta=mean(diff), interval=paired_interval(diff, np.arange(-2, 3)))


def summarize(args):
    directory = args.run/"evaluation"
    groups = {}
    for name in args.groups:
        groups[name] = json.loads((directory/name/"complete.json").read_text(encoding="utf-8"))
    for key in ("gsm8k_sha256", "condition_sha256", "model_sha256"):
        if len({g[key] for g in groups.values()}) != 1:
            raise ValueError(f"Evaluation groups used different {key}")
    comparisons = {}
    short_scores = {}
    controls = []
    for seed in (17, 29):
        controls.extend([(f"answer{seed}", "base"), (f"fullopsd{seed}", "base"),
                         (f"answer{seed}", f"equationsft{seed}"), (f"answer{seed}", f"fullsft{seed}"),
                         (f"answer{seed}", f"fullopsd{seed}"), (f"fullopsd{seed}", f"fullsft{seed}")])
    for setting in ("standard", "hint"):
        comparisons[setting] = {}
        for cap in BUDGETS:
            scores = {name: json.loads((directory/name/f"gsm8k-{setting}-{cap}.json").read_text(encoding="utf-8"))
                      for name in groups}
            if cap == 128:
                short_scores[setting] = scores
            comparisons[setting][str(cap)] = {
                f"{a}:{b}": {parser: paired(scores[a][parser], scores[b][parser]) for parser in ("numeric", "strict")}
                for a, b in controls if a in groups and b in groups
            }
    interactions = {}
    for seed in (17, 29):
        left, right = f"answer{seed}", f"equationsft{seed}"
        if left in groups and right in groups:
            interactions[f"{left}:{right}"] = prompt_interaction(
                *(short_scores[setting][group]["numeric"]
                  for setting in ("standard", "hint") for group in (left, right)))
    condition = json.loads(args.condition.read_text(encoding="utf-8"))
    condition = {key: condition[key] for key in ("metrics", "comparisons", "labels_sha256")}
    if condition["labels_sha256"] != next(iter(groups.values()))["condition_sha256"]:
        raise ValueError("Structured condition scoring used different labels")
    timing = {}
    for name in groups:
        if name == "base":
            continue
        events = read_rows(args.run/"train"/name/"events.jsonl")
        timing[name] = dict(total_seconds=events[-1]["seconds"],
                            updates=sum(r["event"] == "update" for r in events),
                            peak_mib=events[-1]["peak_allocated_mib"],
                            rollout_tokens=sum(r["tokens"] for r in events if r["event"] == "rollout_batch"))
    write(args.output, dict(gsm8k=groups, gsm8k_comparisons=comparisons,
                            standard_minus_hint_interactions_at_128=interactions,
                            conditions=condition, training=timing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inference = commands.add_parser("evaluate")
    for name in ("model", "gsm8k", "conditions", "output"):
        inference.add_argument(f"--{name}", type=Path, required=True)
    inference.add_argument("--adapter", type=Path)
    inference.add_argument("--batch-size", type=int, default=64)
    inference.add_argument("--resume", action="store_true",
                           help="Reuse complete stages only for the original checkpoint and evaluation inputs")
    report = commands.add_parser("summarize")
    for name in ("run", "condition", "output"):
        report.add_argument(f"--{name}", type=Path, required=True)
    report.add_argument("--groups", nargs="+", required=True)
    args = parser.parse_args()
    (evaluate if args.command == "evaluate" else summarize)(args)


if __name__ == "__main__":
    main()
