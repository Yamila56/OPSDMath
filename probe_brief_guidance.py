"""Compare teachers on the same student prefixes with two explicit numeric candidates."""
import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k import read_rows, write_rows
from run_math import BRIEF_CONTEXTS, prompt
from structured_answer_budget import MARKER


@torch.inference_mode()
def candidates(model, prefix, completions):
    result = []
    for tokens in completions:
        suffix = torch.tensor([tokens], device=model.device)
        tensor = torch.cat((prefix, suffix), dim=1)
        logits = model(tensor, logits_to_keep=len(tokens)+1, use_cache=False).logits[0, :-1].float()
        log_prob = logits.log_softmax(-1)
        realized = log_prob.gather(-1, suffix.T).flatten()
        result.append((float(realized.sum()), log_prob[0]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("model", "notes", "labels", "student", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    labels = {r["id"]: r for r in read_rows(args.labels)}
    student = {r["id"]: r for r in read_rows(args.student)}
    examples = [r for r in read_rows(args.notes) if labels[r["id"]]["variant"] in {"changed", "inert"}]
    paired_notes = {(labels[r["id"]]["pair"], labels[r["id"]]["variant"]): r for r in examples}
    changed_gold = {r["pair"]: r["answer"] for r in labels.values() if r["variant"] == "changed"}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    eos = tokenizer.eos_token_id
    all_eos = model.generation_config.eos_token_id
    all_eos = {all_eos} if isinstance(all_eos, int) else set(all_eos)
    marker = tokenizer.encode(MARKER, add_special_tokens=False)
    results = []
    for row in examples:
        key = row["id"]
        label = labels[key]
        draft = student[key]["draft_token_ids"]
        if draft and draft[-1] in all_eos:
            draft = draft[:-1]
        competing = label["baseline"] if label["variant"] == "changed" else changed_gold[label["pair"]]
        suffixes = [tokenizer.encode(value, add_special_tokens=False) + [eos]
                    for value in (label["answer"], competing)]
        log_probs = {}
        result = dict(id=key, family=label["family"], variant=label["variant"], competing_answer=competing)
        opposite = "inert" if label["variant"] == "changed" else "changed"
        swap = dict(row, constraint_brief=paired_notes[label["pair"], opposite]["constraint_brief"])
        contexts = [(None, row, "full"), *[(mode, row, mode) for mode in BRIEF_CONTEXTS],
                    ("swapped_constraint", swap, "constraint_brief"),
                    ("correct_answer", dict(row, answer=label["answer"]), "answer"),
                    ("foil_answer", dict(row, answer=competing), "answer")]
        for mode, example, context in contexts:
            ids = prompt(tokenizer, example, privileged=mode is not None, context=context).to(model.device)
            ids = torch.cat((ids, torch.tensor([draft + marker], device=model.device)), dim=1)
            (gold, distribution), (stale, _) = candidates(model, ids, suffixes)
            log_probs[mode] = distribution
            base = log_probs[None]
            kl = (distribution.exp() * (distribution - base)).sum()
            result[mode or "no_notes"] = dict(gold_logp=gold, stale_logp=stale, margin=gold-stale,
                                               kl_to_unprivileged_next_token=float(kl))
        results.append(result)
    args.output.mkdir(parents=True, exist_ok=False)
    write_rows(args.output / "paired_trace.jsonl", results)
    summary = {}
    for variant in ("changed", "inert"):
        subset = [r for r in results if r["variant"] == variant]
        summary[variant] = {
            mode: dict(n=len(subset), gold_preferred=sum(r[mode]["margin"] > 0 for r in subset),
                       mean_margin=mean(r[mode]["margin"] for r in subset),
                       mean_kl=mean(r[mode]["kl_to_unprivileged_next_token"] for r in subset))
            for mode in ("no_notes", *BRIEF_CONTEXTS, "swapped_constraint", "correct_answer", "foil_answer")
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
