"""Replay successful base trajectories under existing adapters and report policy drift."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from opsdmath.data.gsm8k import read_rows
from experiments.retention.retention_study import cohort


def outcome_inputs(path):
    return {row["id"]: row["correct"] for row in
            json.loads((path / "score-128.json").read_text(encoding="utf-8"))["numeric"]}


def native_token_types(tokenizer, tokens, eos_ids):
    text = ""
    labels = []
    for token in tokens:
        if token in eos_ids:
            labels.append("eos")
            continue
        piece = tokenizer.decode([token], skip_special_tokens=True)
        current_line = (text + piece).split("\n")[-1].lstrip().lower()
        if current_line.startswith("final answer"):
            labels.append("answer")
        elif any(c.isdigit() for c in piece):
            labels.append("non_answer_numeric")
        else:
            labels.append("other")
        text += piece
    return labels


def first_difference(before, after):
    for i, (left, right) in enumerate(zip(before, after)):
        if left != right:
            return i
    return None if len(before) == len(after) else min(len(before), len(after))


def run(args):
    import torch
    from peft import PeftModel

    from experiments.budget_efficiency.opsd_calibration import load_model
    from opsdmath.training.trainer import completion_logits, prompt

    args.output.mkdir(parents=True, exist_ok=False)
    baseline = args.evaluations / "base"
    frozen = cohort(baseline)
    rows = [r for r in read_rows(args.inputs) if frozen[r["id"]] == "early"]
    paths = [("base", None), *[tuple(spec.split("=", 1)) for spec in args.adapter]]
    traces = {row["id"]: row["token_ids"][:128] for row in read_rows(baseline / "tokens.jsonl")}
    start_scores = {}
    results = []
    for name, adapter in paths:
        base, tokenizer = load_model(args.model)
        model = PeftModel.from_pretrained(base, adapter).eval() if adapter else base
        eos = model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos)
        evaluation = args.evaluations / name
        correct = outcome_inputs(evaluation)
        generated = {r["id"]: r["token_ids"][:128] for r in read_rows(evaluation / "tokens.jsonl")}
        with torch.inference_mode():
            for row in rows:
                key = row["id"]
                tokens = traces[key]
                prefix = prompt(tokenizer, row, output_budget=128).to(model.device)
                suffix = torch.tensor([tokens], device=model.device)
                log_prob = completion_logits(model, prefix, suffix)[0].float().log_softmax(-1)
                recorded = log_prob.gather(-1, suffix.T).flatten().tolist()
                eos_prob = log_prob[:, eos_ids].exp().sum(dim=1).tolist()
                if name == "base":
                    start_scores[key] = (recorded, eos_prob)
                    continue
                before, eos_before = start_scores[key]
                labels = native_token_types(tokenizer, tokens, eos_ids)
                result = defaultdict(list)
                for category, left, right in zip(labels, before, recorded, strict=True):
                    result[category].append(right - left)
                results.append(dict(id=key, adapter=name, retained=correct[key],
                                    first_difference=first_difference(tokens, generated[key]),
                                    mean_eos_probability_change=mean(b-a for a, b in zip(eos_before, eos_prob)),
                                    logp_delta={k: mean(v) for k, v in result.items()}))
        del model, base
        torch.cuda.empty_cache()
        print(name, Counter(r["retained"] for r in results if r["adapter"] == name), flush=True)
    from opsdmath.data.gsm8k import write_rows

    write_rows(args.output / "per-item.jsonl", results)
    groups = defaultdict(list)
    for row in results:
        groups[row["adapter"], row["retained"]].append(row)
    summary = {}
    for (adapter, retained), members in groups.items():
        categories = sorted({key for r in members for key in r["logp_delta"]})
        summary[f"{adapter}/{retained}"] = dict(
            n=len(members),
            mean_eos_probability_change=mean(r["mean_eos_probability_change"] for r in members),
            logp_delta={key: mean(r["logp_delta"][key] for r in members if key in r["logp_delta"])
                        for key in categories},
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "inputs", "evaluations", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--adapter", required=True, action="append", metavar="NAME=PATH")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
