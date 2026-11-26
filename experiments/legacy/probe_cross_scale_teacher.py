"""Replay identical student prefixes in both frozen self-teachers."""

import argparse
import hashlib
from pathlib import Path

from opsdmath.data.gsm8k import number, parse_strict_answer, read_rows, write_rows
from opsd_calibration import load_model, write
from summarize_run import audit_final_answer

SCALES = ("1p7b", "8b")
CONTEXTS = ("none", "answer", "masked_equations")
PREFIX_TOKENS = 60
TOTAL = 120


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    import torch

    from opsdmath.training.trainer import batch_suffixes, prompt

    models = dict(zip(SCALES, (args.small_model, args.large_model), strict=True))
    if len({digest(model / "tokenizer.json") for model in models.values()}) != 1:
        raise ValueError("Identical tokenizer vocabulary and tokenization are required")
    rows = read_rows(args.cases)
    sources = {scale: {r["id"]: r["token_ids"][:PREFIX_TOKENS] for r in
                       read_rows(args.evaluations / scale / "base" / "standard-tokens.jsonl")}
               for scale in SCALES}
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", dict(
        cases_sha256=digest(args.cases), n=len(rows), prefix=PREFIX_TOKENS, total=TOTAL,
        prefix_sha256={s: digest(args.evaluations / s / "base" / "standard-tokens.jsonl") for s in SCALES},
        model_tokenizer_sha256=digest(args.small_model / "tokenizer.json"),
        models={s: str(p) for s, p in models.items()}, batch=24, greedy=True, thinking=False,
        status="retrospective common-late diagnostic",
    ))
    for scale, model_path in models.items():
        model, tokenizer = load_model(model_path)
        eos = model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos)
        results = []
        for owner, trajectories in sources.items():
            for condition in CONTEXTS:
                prompts = []
                for row in rows:
                    prefix = trajectories[row["id"]]
                    if len(prefix) != PREFIX_TOKENS or set(prefix) & eos:
                        raise ValueError(f"Incomplete fixed prefix for {row['id']}")
                    example = dict(row)
                    example["problem"] += (
                        f"\n\nA response prefix of {PREFIX_TOKENS} tokens has already been written. "
                        f"Continue it within the remaining {TOTAL-PREFIX_TOKENS} tokens."
                    )
                    example["general_brief"] = row["masked_equations"]
                    mode = "general_brief" if condition == "masked_equations" else condition
                    start = prompt(tokenizer, example, privileged=condition != "none",
                                   context=mode, output_budget=TOTAL, key_steps=False)
                    prompts.append(torch.cat((start, torch.tensor([prefix])), dim=1))
                for i in range(0, len(rows), 24):
                    generated = batch_suffixes(model, tokenizer, prompts[i:i+24], TOTAL-PREFIX_TOKENS,
                                               False, model.device)
                    for row, (tokens, hit_cap) in zip(rows[i:i+24], generated, strict=True):
                        text = tokenizer.decode(trajectories[row["id"]] + tokens, skip_special_tokens=True)
                        gold = number(row["answer"])
                        results.append(dict(id=row["id"], prefix_source=owner, condition=condition,
                                            text=text, suffix_token_ids=tokens, hit_cap=hit_cap,
                                            correct=audit_final_answer(text) == gold,
                                            strict=parse_strict_answer(text) == gold))
                print(scale, owner, condition, len(rows), flush=True)
        write_rows(args.output / f"{scale}.jsonl", results)
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("small-model", "large-model", "evaluations", "cases", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
