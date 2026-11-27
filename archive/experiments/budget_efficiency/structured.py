"""Compare numeric answer delivery with a shared, token-budgeted output field."""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from opsdmath.evaluation.answer_parser import number
from opsdmath.io import read_rows, write_rows
from opsdmath.training.prompts import prompt
from opsdmath.training.rollout import batch_suffixes

BUDGET = 128
DRAFT_CAP = 104
MARKER = "\nFinal answer: "
NUMBER_PREFIX = re.compile(r"[+-]?(?:[0-9]+(?:[./][0-9]*)?)?\Z")
COMPLETE_NUMBER = re.compile(r"[+-]?[0-9]+(?:[./][0-9]+)?\Z")
PIECE = re.compile(r"[0-9+./-]+\Z")


def grammar_state(text):
    if not text:
        return ""
    if text in {"+", "-"}:
        return "-"
    if text.endswith("."):
        return "0."
    if text.endswith("/"):
        return "0/"
    if "." in text:
        return "0.1"
    if "/" in text:
        return "0/1"
    return "1"


class NumericField(LogitsProcessor):
    def __init__(self, tokenizer, eos, prompt_width, device):
        self.tokenizer = tokenizer
        self.eos = set(eos)
        self.prompt_width = prompt_width
        token_ids = set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids)
        pieces = {idx: token for idx in token_ids if PIECE.fullmatch(token := tokenizer.convert_ids_to_tokens(idx))}
        self.allowed = {}
        for state in ("", "-", "1", "0.", "0/", "0.1", "0/1"):
            options = [idx for idx, value in pieces.items() if NUMBER_PREFIX.fullmatch(state + value)]
            if COMPLETE_NUMBER.fullmatch(state):
                options.extend(self.eos)
            if not options:
                raise RuntimeError(f"Numeric grammar has no valid continuation for {state!r}")
            self.allowed[state] = torch.tensor(sorted(options), device=device)

    def __call__(self, input_ids, scores):
        viable = torch.full_like(scores, -torch.inf)
        suffixes = input_ids[:, self.prompt_width:].tolist()
        for i, raw in enumerate(suffixes):
            if any(token in self.eos for token in raw):
                viable[i, min(self.eos)] = 0
                continue
            text = self.tokenizer.decode(raw, clean_up_tokenization_spaces=False)
            if not NUMBER_PREFIX.fullmatch(text):
                raise RuntimeError(f"Tokenizer violated the numeric grammar: {text!r}")
            allowed = self.allowed[grammar_state(text)]
            viable[i, allowed] = scores[i, allowed]
        return viable


@torch.inference_mode()
def generate(model, tokenizer, batch, *, budget=BUDGET, draft_cap=DRAFT_CAP, teacher_context=None,
             budget_hint=False, key_steps=None, thinking=False):
    device = model.device
    prompts = [prompt(tokenizer, row, privileged=teacher_context is not None,
                      context=teacher_context or "full", output_budget=budget if budget_hint else None,
                      key_steps=key_steps, thinking=thinking) for row in batch]
    drafts = batch_suffixes(model, tokenizer, prompts, draft_cap, False, device)
    eos_config = model.generation_config.eos_token_id
    eos = {eos_config} if isinstance(eos_config, int) else set(eos_config)
    marker = tokenizer.encode(MARKER, add_special_tokens=False)
    if tokenizer.decode(marker, clean_up_tokenization_spaces=False) != MARKER:
        raise RuntimeError("Final answer marker did not round trip")
    prefix = []
    for ids, (draft, _) in zip(prompts, drafts, strict=True):
        context = draft[:-1] if draft and draft[-1] in eos else draft
        prefix.append(ids[0].tolist() + context + marker)
    # The common reserve charges the marker and the draft's EOS when present.
    # Drafts that end early leave unused allowance.
    final_cap = budget - draft_cap - len(marker)
    tokenizer.padding_side = "left"
    inputs = tokenizer.pad({"input_ids": prefix}, return_tensors="pt").to(device)
    width = inputs.input_ids.shape[1]
    processor = NumericField(tokenizer, eos, width, device)
    sequences = model.generate(
        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
        logits_processor=LogitsProcessorList([processor]), max_new_tokens=final_cap,
        do_sample=False, temperature=None, top_k=None, top_p=None, use_cache=True,
    )[:, width:].tolist()
    result = []
    for row, (draft, draft_hit_cap), raw in zip(batch, drafts, sequences, strict=True):
        end = next((i for i, idx in enumerate(raw) if idx in eos), None)
        tokens = raw if end is None else raw[:end+1]
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        parsed = number(text) if end is not None and COMPLETE_NUMBER.fullmatch(text) else None
        charged = len(draft) + len(marker) + len(tokens)
        if charged > budget:
            raise RuntimeError("Joint token budget exceeded")
        result.append(dict(id=row["id"], response=MARKER.lstrip("\n") + text,
                           parsed=str(parsed) if parsed is not None else None, final_eos=end is not None,
                           draft=tokenizer.decode(draft, skip_special_tokens=True), draft_hit_cap=draft_hit_cap,
                           draft_token_ids=draft, marker_token_ids=marker, final_token_ids=tokens,
                           generated_tokens=charged))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("model", "data", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--teacher-context", choices=["general_brief", "constraint_brief"])
    parser.add_argument("--budget", type=int, choices=[128, 256, 512], default=BUDGET)
    parser.add_argument("--answer-reserve", type=int, default=BUDGET-DRAFT_CAP,
                        help="Reserve for the answer marker, numeric answer, and EOS.")
    parser.add_argument("--budget-hint", action="store_true",
                        help="Tell the model the total output budget before generation.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = read_rows(args.data)[:args.limit]
    if not rows or args.batch_size < 1 or any({"answer", "solution"} & row.keys() for row in rows):
        parser.error("Pass a nonempty question-only split and a positive batch size")
    if args.teacher_context and (args.adapter or any(args.teacher_context not in row for row in rows)):
        parser.error("Teacher probes require the base model and cached briefs on every row")
    draft_cap = args.budget - args.answer_reserve
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    marker_size = len(tokenizer.encode(MARKER, add_special_tokens=False))
    if not marker_size+1 < args.answer_reserve < args.budget:
        parser.error("Reserve must fit the marker, a numeric token and EOS, with room for a draft")
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), draft_cap=draft_cap, marker=MARKER)
    (args.output / "config.json").write_text(json.dumps(config, default=str, indent=2) + "\n", encoding="utf-8")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
        local_files_only=True,
    )
    base.generation_config.pad_token_id = tokenizer.pad_token_id
    model = (PeftModel.from_pretrained(base, args.adapter) if args.adapter else base).eval()
    predictions = []
    for index in range(0, len(rows), args.batch_size):
        predictions.extend(generate(model, tokenizer, rows[index:index + args.batch_size],
                                    budget=args.budget, draft_cap=draft_cap, teacher_context=args.teacher_context,
                                    budget_hint=args.budget_hint))
        print(json.dumps(dict(event="batch", processed=len(predictions), total=len(rows))), flush=True)
    write_rows(args.output / "predictions.jsonl", predictions)
    (args.output / "complete.json").write_text(json.dumps(dict(
        n=len(predictions), seconds=time.perf_counter()-start,
        peak_mib=torch.cuda.max_memory_allocated()/2**20,
    )), encoding="utf-8")


if __name__ == "__main__":
    main()
