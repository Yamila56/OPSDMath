"""Cache matched-length teacher notes generated from questions alone."""
import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k import read_rows, write_rows
from run_math import batch_suffixes

GUIDANCE = {
    "general_brief": "Identify the target, a general mathematical principle, and a generic check on the solution.",
    "constraint_brief": ("Identify the target and its units, which facts affect it and at what scope, "
                         "and one condition that can be missed or is irrelevant."),
}
HEADINGS = ("Target", "Notes", "Check")
FIELDS = re.compile(r"^\s*Target:\s*(.+?)\s*\n\s*Notes:\s*(.+?)\s*\n\s*Check:\s*(.+?)\s*$",
                    re.I | re.S)
PARTIAL_NUMBER = re.compile(r"\d$")


def match_tokens(tokenizer, notes):
    lengths = {mode: tokenizer.encode(note, add_special_tokens=False) for mode, note in notes.items()}
    limit = min(map(len, lengths.values()))
    while limit:
        matched = {mode: tokenizer.decode(ids[:limit], clean_up_tokenization_spaces=False)
                   for mode, ids in lengths.items()}
        intact = all(not (PARTIAL_NUMBER.search(note) and notes[mode][len(note):len(note)+1].isdigit())
                     for mode, note in matched.items())
        if intact and all(len(tokenizer.encode(note, add_special_tokens=False)) == limit for note in matched.values()):
            return matched, limit
        limit -= 1
    raise ValueError("At least one generated note field is empty")


def match_notes(tokenizer, notes):
    structured = {mode: FIELDS.fullmatch(note) for mode, note in notes.items()}
    if not all(structured.values()):
        return match_tokens(tokenizer, notes)
    fields = {mode: [] for mode in notes}
    for index, name in enumerate(HEADINGS, 1):
        snippets, _ = match_tokens(tokenizer, {mode: parsed[index] for mode, parsed in structured.items()})
        for mode, snippet in snippets.items():
            fields[mode].append(f"{name}: {snippet}")
    return match_tokens(tokenizer, {mode: "\n".join(parts) for mode, parts in fields.items()})


def note_prompt(tokenizer, problem, instruction):
    text = (f"Problem: {problem}\n\n{instruction}\n"
            "Write exactly three concise lines beginning Target:, Notes:, and Check:. "
            "Put the core information first on each line. Prefer short phrases. "
            "Refer to quantities by name and units. Use words, without digits, equations, calculated results, or a final answer.")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("model", "data", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    original = read_rows(args.data)
    rows = [dict(id=r["id"], problem=r["problem"] if "problem" in r else r["question"]) for r in original]
    if not rows or len(rows) != len({r["id"] for r in rows}):
        parser.error("Expected nonempty questions with unique IDs")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    stats = Counter()
    raw = []
    for index in range(0, len(rows), args.batch_size):
        batch = rows[index:index + args.batch_size]
        generations = {}
        for mode, instruction in GUIDANCE.items():
            prompts = [note_prompt(tokenizer, row["problem"], instruction) for row in batch]
            generations[mode] = batch_suffixes(model, tokenizer, prompts, 128, False, model.device)
        for offset, row in enumerate(batch):
            notes = {}
            for mode, results in generations.items():
                tokens, capped = results[offset]
                full = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                notes[mode] = full
                stats[f"{mode}_generated_tokens"] += len(tokens)
                stats[f"{mode}_cap_hits"] += capped
            matched, limit = match_notes(tokenizer, notes)
            row.update(matched)
            raw.append(dict(id=row["id"], original=notes, matched_tokens=limit,
                            original_lengths={mode: len(tokenizer.encode(note, add_special_tokens=False))
                                              for mode, note in notes.items()}))
            stats["matched_tokens"] += limit
        print(json.dumps(dict(processed=min(index+len(batch), len(rows)), n=len(rows))), flush=True)
    write_rows(args.output / "notes.jsonl", rows)
    write_rows(args.output / "generation-audit.jsonl", raw)
    (args.output / "complete.json").write_text(json.dumps(dict(
        n=len(rows), stats=stats, seconds=time.perf_counter()-start,
        source_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
    ), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
