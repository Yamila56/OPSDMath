"""Inference protocol used by the two-seed KL ablation."""

import json
from pathlib import Path

from opsdmath.evaluation.answer_parser import number
from opsdmath.io import digest, read_rows, write_rows
from opsdmath.evaluation.budget import score, validate_rows
from opsdmath.evaluation.metrics import LONG, SHORT
from opsdmath.training.modeling import load_eval_model

PROMPT = "combined-128"
THINKING = False


def validate_recipe(config):
    expected = {"short": SHORT, "long": LONG, "prompt": PROMPT, "thinking": THINKING}
    for key, value in expected.items():
        if type(config.get(key)) is not type(value) or config[key] != value:
            raise ValueError(f"Evaluation recipe {key} must be {value!r}")
    for key in ("batch_size", "questions"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Evaluation recipe {key} must be a positive integer")

def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def evaluate(args, *, expected_questions=None):
    from peft import PeftModel

    from opsdmath.training.prompts import prompt
    from opsdmath.training.rollout import batch_suffixes

    n = expected_questions if expected_questions is not None else len(read_rows(args.data / "inputs.jsonl"))
    rows, labels = validate_rows(args.data / "inputs.jsonl", args.data / "labels.jsonl", n)
    truth = {row["id"]: number(row["answer"]) for row in labels}
    protocol = dict(batch_size=args.batch_size, short=SHORT, long=LONG,
                    model=digest(args.model / "model.safetensors.index.json"),
                    inputs=digest(args.data / "inputs.jsonl"), labels=digest(args.data / "labels.jsonl"),
                    prompt=PROMPT, thinking=THINKING)
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
        base, tokenizer = load_eval_model(args.model)
        model = PeftModel.from_pretrained(base, adapter).eval() if adapter else base
        traces = []
        for start in range(0, n, args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = [prompt(tokenizer, r, output_budget=SHORT, thinking=THINKING) for r in batch]
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
        print(name, metrics[str(SHORT)]["numeric"], flush=True)
        del model, base
