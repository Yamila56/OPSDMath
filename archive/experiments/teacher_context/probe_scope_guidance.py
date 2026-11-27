"""Probe selection of an intermediate operation under a fixed derivation prefix."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from opsdmath.data.gsm8k import read_rows, write_rows
from opsdmath.training.prompts import BRIEF_CONTEXTS, prompt


def requested_operation(row, variant):
    a, b, c = map(int, re.findall(r"\d+", row["problem"])[:3])
    answer = {"changed": 1, "inert": 0, "relation": 2}[variant]
    foil = a*b-c if variant != "changed" else a*(b-c)
    return f"Originally there are {a} × {b} items.", ["0", str(c), f"{a} × {c}"], answer, foil


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("model", "notes", "labels", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    labels = {r["id"]: r for r in read_rows(args.labels)}
    notes = [r for r in read_rows(args.notes)
             if labels[r["id"]]["family"] >= 16 and labels[r["id"]]["variant"] in {"changed", "inert", "relation"}]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    labels_for_choice = ("A", "B", "C")
    letter_ids = [tokenizer.encode(" " + letter, add_special_tokens=False) for letter in labels_for_choice]
    if any(len(ids) != 1 for ids in letter_ids):
        raise ValueError("Choice labels must be single tokens")
    choice_indices = [ids[0] for ids in letter_ids]
    trials = []
    for row in notes:
        variant = labels[row["id"]]["variant"]
        step, options, correct, foil = requested_operation(row, variant)
        for shift in range(3):
            order = [(index+shift) % 3 for index in range(3)]
            display = "\n".join(f"{letter}. {options[index]}"
                                for letter, index in zip(labels_for_choice, order, strict=True))
            suffix = f"{step}\nWhich quantity should be subtracted to answer the question?\n{display}\nChoice:"
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False, return_tensors="pt").to(model.device)
            contexts = [(None, row, "full"), *[(mode, row, mode) for mode in BRIEF_CONTEXTS],
                        ("correct_answer", dict(row, answer=labels[row["id"]]["answer"]), "answer"),
                        ("foil_answer", dict(row, answer=foil), "answer")]
            for mode, example, context in contexts:
                ids = prompt(tokenizer, example, privileged=mode is not None, context=context).to(model.device)
                output = model(torch.cat((ids, suffix_ids), dim=1), use_cache=False, logits_to_keep=1)
                scores = output.logits[0, 0, choice_indices].float().tolist()
                gold = order.index(correct)
                margin = scores[gold] - max(value for i, value in enumerate(scores) if i != gold)
                trials.append(dict(id=row["id"], variant=variant, rotation=shift, mode=mode or "no_notes",
                                   gold=labels_for_choice[gold], scores=scores, margin=margin))
    summary = defaultdict(dict)
    for variant in ("changed", "inert", "relation"):
        for mode in ("no_notes", *BRIEF_CONTEXTS, "correct_answer", "foil_answer"):
            subset = [r for r in trials if r["variant"] == variant and r["mode"] == mode]
            summary[variant][mode] = dict(items=len({r["id"] for r in subset}), rotations=len(subset),
                                          correct=sum(r["margin"] > 0 for r in subset),
                                          mean_margin=mean(r["margin"] for r in subset))
    args.output.mkdir(parents=True, exist_ok=False)
    write_rows(args.output / "trials.jsonl", trials)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
