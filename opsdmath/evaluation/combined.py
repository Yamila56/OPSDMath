"""Inference protocol used by the two-seed KL ablation."""

import json
from pathlib import Path

from opsdmath.evaluation.answer_parser import number
from opsdmath.io import read_rows, write_rows
from opsdmath.evaluation.budget import digest, score, validate_rows

SHORT = 128
LONG = 512

def load_model(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True).eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def evaluate(args):
    from peft import PeftModel

    from opsdmath.training.prompts import prompt
    from opsdmath.training.rollout import batch_suffixes

    n = len(read_rows(args.data / "inputs.jsonl"))
    rows, labels = validate_rows(args.data / "inputs.jsonl", args.data / "labels.jsonl", n)
    truth = {row["id"]: number(row["answer"]) for row in labels}
    protocol = dict(batch_size=args.batch_size, short=SHORT, long=LONG,
                    model=digest(args.model / "model.safetensors.index.json"),
                    inputs=digest(args.data / "inputs.jsonl"), labels=digest(args.data / "labels.jsonl"),
                    prompt="combined-128", thinking=False)
    args.output.mkdir(parents=True, exist_ok=True)
    for spec in args.group:
        name, _, adapter = spec.partition("=")
        output = args.output / name
        identity = dict(protocol, adapter=digest(Path(adapter) / "adapter_model.safetensors") if adapter else None)
        output.mkdir(exist_ok=True)
        manifest = output / "protocol.json"
        if manifest.exists() and identity != json.loads(manifest.read_text(encoding="utf-8")):
            raise ValueError(f"Evaluation inputs changed for {name}")
        if (output / "complete.json").exists():
            continue
        write(manifest, identity)
        base, tokenizer = load_model(args.model)
        model = PeftModel.from_pretrained(base, adapter).eval() if adapter else base
        traces = []
        for start in range(0, n, args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = [prompt(tokenizer, r, output_budget=SHORT) for r in batch]
            generated = batch_suffixes(model, tokenizer, prompts, LONG, False, model.device)
            if start == 0:
                direct = batch_suffixes(model, tokenizer, prompts, SHORT, False, model.device)
                if any(a[:SHORT] != b for (a, _), (b, _) in zip(generated, direct, strict=True)):
                    raise ValueError("Direct short decoding differs from the long trajectory prefix")
            traces.extend(dict(id=row["id"], token_ids=ids, hit_cap=hit)
                          for row, (ids, hit) in zip(batch, generated, strict=True))
        temporary = output / "tokens.tmp"
        write_rows(temporary, traces)
        temporary.replace(output / "tokens.jsonl")
        metrics = {}
        for cap in (SHORT, LONG):
            metrics[str(cap)], per_item = score(traces, truth, tokenizer, cap)
            write(output / f"score-{cap}.json", per_item)
        write(output / "complete.json", metrics)
        print(name, metrics["128"]["numeric"], flush=True)
        del model, base
