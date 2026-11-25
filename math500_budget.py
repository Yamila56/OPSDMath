"""Evaluate GSM8K-trained checkpoints on MATH-500 with a shared boxed-answer prompt."""
import argparse
import hashlib
from importlib.metadata import version
import json
import time
from collections import Counter
from pathlib import Path

import torch
from math_verify import LatexExtractionConfig, parse, verify
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k import read_rows, write_rows
from run_math import batch_suffixes

INSTRUCTION = r"Solve the problem briefly. Put your final answer in \boxed{...}."


def provenance(model, adapter, batch_size, *, model_revision=None, retrospective=False):
    def hashes(directory, names):
        entries = {}
        for name in names:
            path = directory/name
            if path.is_file():
                with path.open("rb") as stream:
                    entries[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        return entries

    return dict(
        model_revision=model_revision,
        model_files=hashes(model, ("config.json", "generation_config.json", "model.safetensors.index.json")),
        tokenizer_files=hashes(model, ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")),
        adapter_files=hashes(adapter, ("adapter_config.json", "adapter_model.safetensors")) if adapter else None,
        prompt_sha256=hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        decoding=dict(cap=512, batch_size=batch_size, dtype="bfloat16", attention="sdpa",
                      do_sample=False, enable_thinking=False),
        software={p: version(p) for p in ("transformers", "torch", "peft", "math-verify")},
        retrospective=retrospective,
    )


def last_box(text):
    spans = []
    start = 0
    while (marker := text.find(r"\boxed{", start)) >= 0:
        start = marker+len(r"\boxed{")
        depth = 1
        for offset, char in enumerate(text[start:], start=start):
            if text[offset-1] == "\\":
                continue
            depth += (char == "{") - (char == "}")
            if depth == 0:
                spans.append(text[marker:offset+1])
                start = offset+1
                break
    return spans[-1] if spans else None


def parse_box(box):
    if box is None:
        return []
    return parse("$"+box+"$", extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
                 fallback_mode="no_fallback", parsing_timeout=2)


def score(dataset, predictions, tokenizer, budget):
    metrics, scored = Counter(), []
    for row, prediction in zip(dataset, predictions, strict=True):
        tokens = prediction["token_ids"][:budget]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        boxed = last_box(text)
        answer = parse_box(boxed)
        correct = bool(row["gold"] and answer and verify(row["gold"], answer, timeout_seconds=2))
        metrics.update(dict(n=1, correct=correct, closed_box=boxed is not None,
                            parsed=bool(answer), gold_parsed=bool(row["gold"]),
                            generated_tokens=len(tokens),
                            at_limit=len(prediction["token_ids"]) > budget or
                            (len(prediction["token_ids"]) == budget and prediction["hit_cap"])))
        scored.append(dict(id=prediction["id"], correct=correct, closed_box=boxed,
                           parsed=bool(answer), gold_parsed=bool(row["gold"]), actual_tokens=len(tokens)))
    return dict(metrics, mean_generated_tokens=metrics["generated_tokens"]/metrics["n"]), scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "model", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    data = read_rows(args.data)
    if len(data) != 500 or len({row["unique_id"] for row in data}) != 500:
        parser.error("Expected all 500 distinct official test questions")
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    predictions = []
    start = time.perf_counter()
    for offset in range(0, len(data), args.batch_size):
        batch = data[offset:offset+args.batch_size]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{r['problem']}\n\n{INSTRUCTION}"}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
        ) for r in batch]
        generated = batch_suffixes(model, tokenizer, prompts, 512, False, model.device)
        if offset == 0:
            short = batch_suffixes(model, tokenizer, prompts, 128, False, model.device)
            if [ids for ids, _ in short] != [ids[:128] for ids, _ in generated]:
                raise ValueError("128-token outputs differ from the long-generation prefix")
        predictions.extend(dict(id=row["unique_id"], token_ids=ids, hit_cap=capped)
                           for row, (ids, capped) in zip(batch, generated, strict=True))
        print(json.dumps(dict(processed=len(predictions))), flush=True)
    generation_seconds = time.perf_counter()-start
    write_rows(args.output/"tokens.jsonl", predictions)
    for row in data:
        row["gold"] = parse_box(r"\boxed{" + row["answer"] + "}")
    results = {}
    for budget in (128, 256, 512):
        results[str(budget)], scored = score(data, predictions, tokenizer, budget)
        write_rows(args.output/f"scored-{budget}.jsonl", scored)
    (args.output/"complete.json").write_text(json.dumps(dict(
        dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        prefix_probe=args.batch_size, generation_seconds=generation_seconds, metrics=results,
        provenance=provenance(args.model, args.adapter, args.batch_size, model_revision=args.model_revision),
    ), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
