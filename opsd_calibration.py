"""Freeze a GSM8K development split, probe teacher signals, and evaluate budget prompts."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from gsm8k import REVISION, dataset, number, read_rows, write_rows


PROMPTS = {"standard": (False, False), "budget": (True, False),
           "key-steps": (False, True), "combined": (True, True)}
TEACHERS = {"off-answer": (False, "answer"), "on-answer": (True, "answer"),
            "on-full": (True, "full"), "on-none": (True, "none"), "off-none": (False, "none")}
CLIPS = (0.05, 0.001, 1e-7, None)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"manifest.json").exists():
        raise ValueError(f"Development split already exists at {args.output}")
    source = args.output / "source"
    source.mkdir(exist_ok=True)
    train, _ = dataset("train", source)
    test, _ = dataset("test", source)
    available = sorted(train.keys() - test.keys())
    previous_train = read_rows(args.train)
    if [row["id"] for row in previous_train] != available[128:384]:
        raise ValueError("The historical training split does not match the pinned GSM8K source")
    rows = [train[key] for key in available[384:448]]
    assert len(rows) == 64
    fields = {"reference": ("id", "problem", "answer", "solution"),
              "inputs": ("id", "problem"), "labels": ("id", "answer")}
    for name, keys in fields.items():
        path = args.output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps({k: row[k] for k in keys}, ensure_ascii=False) + "\n")
    write(args.output/"manifest.json", dict(source_revision=REVISION, count=len(rows),
          rule="sorted unique GSM8K train IDs [384:448] after removing test overlap",
          sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.jsonl")}))


def load_model(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True).eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def probe(args):
    import torch
    import torch.nn.functional as F

    from run_math import ROLLOUT_TEMPERATURE, batch_suffixes, completion_logits, prompt
    from summarize_run import audit_final_answer

    rows = read_rows(args.data)
    if len({row["id"] for row in rows}) != len(rows) or not rows:
        raise ValueError("Expected distinct calibration questions")
    args.output.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model(args.model)
    device = model.device
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else eos
    trajectories = batch_suffixes(model, tokenizer, [prompt(tokenizer, r) for r in rows], 256, False, device)
    write_rows(args.output/"trajectories.jsonl", [dict(id=row["id"], token_ids=ids, hit_cap=hit)
               for row, (ids, hit) in zip(rows, trajectories, strict=True)])

    records = []
    for row, (ids, hit) in zip(rows, trajectories, strict=True):
        suffix = torch.tensor([ids], device=device)
        question = prompt(tokenizer, row).to(device)
        texts = tokenizer.batch_decode([[i] for i in ids], skip_special_tokens=True)
        math_like = torch.tensor([bool(re.search(r"[0-9+*/=]", t)) for t in texts], device=device)
        truth = number(row["answer"])
        short = audit_final_answer(tokenizer.decode(ids[:128], skip_special_tokens=True))
        long = audit_final_answer(tokenizer.decode(ids, skip_special_tokens=True))
        with torch.no_grad():
            student_raw = completion_logits(model, question, suffix).float()
        logits = student_raw.requires_grad_()
        for key, (thinking, context) in TEACHERS.items():
            teacher_prompt = prompt(tokenizer, row, privileged=True, context=context, thinking=thinking).to(device)
            with torch.no_grad():
                if key == "off-none":
                    torch.testing.assert_close(teacher_prompt, question, rtol=0, atol=0)
                teacher = (completion_logits(model, teacher_prompt, suffix).float() / ROLLOUT_TEMPERATURE).log_softmax(-1)
            student = (logits / ROLLOUT_TEMPERATURE).log_softmax(-1)
            points = F.kl_div(student, teacher, reduction="none", log_target=True)
            raw = points.detach()[0].sum(-1)
            eos_delta = (teacher[..., eos_ids].exp() - student.detach()[..., eos_ids].exp()).sum(-1)
            reference = None
            for i, clip in enumerate(CLIPS):
                loss = (points if clip is None else points.clamp(max=clip)).sum(-1).mean()
                gradient, = torch.autograd.grad(loss, logits, retain_graph=(i+1 < len(CLIPS)))
                norm = torch.linalg.vector_norm(gradient)
                if reference is None:
                    reference = gradient
                cosine = F.cosine_similarity(reference.flatten(), gradient.flatten(), dim=0) if norm > 0 else None
                clipped = points.detach() > clip if clip is not None else None
                records.append(dict(id=row["id"], teacher=key, clip=clip, tokens=len(ids), hit_cap=hit,
                    short_correct=short == truth, long_correct=long == truth,
                    short_parsed=short is not None, long_parsed=long is not None,
                    unclipped_kl=float(raw.mean()), objective=float(loss.detach()),
                    clipped_vocab_fraction=float(clipped.float().mean()) if clipped is not None else 0.0,
                    clipped_position_fraction=float(clipped.any(-1).float().mean()) if clipped is not None else 0.0,
                    math_like_kl=float(raw[math_like].mean()) if math_like.any() else None,
                    other_position_kl=float(raw[~math_like].mean()) if (~math_like).any() else None,
                    logit_gradient_norm=float(norm), cosine_to_005=float(cosine) if cosine is not None else None,
                    eos_delta=float(eos_delta.mean()), last_eos_delta=float(eos_delta[0, -1])))
        del logits, student_raw, points, gradient, reference
        if len(records) % (8*len(TEACHERS)*len(CLIPS)) == 0:
            print(f"Probed {len(records)//(len(TEACHERS)*len(CLIPS))}/{len(rows)} trajectories", flush=True)
    write_rows(args.output/"token-probe.jsonl", records)


def evaluate(args):
    from peft import PeftModel

    from eval_scale_transfer import score, validate_rows
    from run_math import batch_suffixes, prompt
    from score_structured_answer import audit
    from structured_answer_budget import generate

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    rows, labels = validate_rows(args.data/"inputs.jsonl", args.data/"labels.jsonl", args.questions)
    truth = {row["id"]: number(row["answer"]) for row in labels}
    if None in truth.values():
        raise ValueError("Invalid reference answer")
    args.output.mkdir(parents=True, exist_ok=True)
    model_signature = args.model/"model.safetensors.index.json"
    if not model_signature.is_file():
        model_signature = args.model/"model.safetensors"
    protocol = dict(dataset_sha256={key: digest(args.data/f"{key}.jsonl") for key in ("inputs", "labels")},
                    model_sha256=digest(model_signature), prompts=args.prompts,
                    questions=args.questions, batch_size=args.batch_size, cap=args.cap,
                    structured=args.structured, thinking=args.thinking)
    if args.prefix_budgets:
        protocol["prefix_budgets"] = args.prefix_budgets
    for specification in args.group:
        name, _, adapter = specification.partition("=")
        if Path(name).name != name or not name or name in (".", ".."):
            raise ValueError(f"Invalid evaluation group: {name}")
        output = args.output/name
        identity = dict(protocol, adapter_sha256=digest(Path(adapter)/"adapter_model.safetensors") if adapter else None)
        manifest = output/"evaluation-inputs.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Existing results use a different evaluation for {name}")
        if (output/"complete.json").exists():
            if not manifest.exists():
                raise ValueError(f"Existing evaluation has no input manifest for {name}")
            continue
        output.mkdir(exist_ok=True)
        write(manifest, identity)
        base, tokenizer = load_model(args.model)
        model = PeftModel.from_pretrained(base, adapter).eval() if adapter else base
        metrics = {}
        for mode in args.prompts:
            budget, key_steps = PROMPTS[mode]
            cap = 128 if mode != "standard" else args.cap
            traces, fields = [], []
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start+args.batch_size]
                prompts = [prompt(tokenizer, r, output_budget=128 if budget else None,
                                  key_steps=key_steps, thinking=args.thinking) for r in batch]
                generated = batch_suffixes(model, tokenizer, prompts, cap, False, model.device)
                traces.extend(dict(id=r["id"], token_ids=ids, hit_cap=hit)
                              for r, (ids, hit) in zip(batch, generated, strict=True))
                if args.structured:
                    fields.extend(generate(model, tokenizer, batch, budget_hint=budget,
                                           key_steps=key_steps, thinking=args.thinking))
            write_rows(output/f"{mode}-tokens.jsonl", traces)
            metrics[mode] = {}
            for budget in args.prefix_budgets or (128, 256, 512):
                if budget <= cap:
                    metrics[mode][str(budget)], per_item = score(traces, truth, tokenizer, budget)
                    write(output/f"{mode}-{budget}.json", per_item)
            if fields:
                parsed = [audit(row, budget=128, draft_cap=104) for row in fields]
                metrics[mode]["structured"] = dict(n=len(fields), parsed=sum(v is not None for v in parsed),
                     correct=sum(v == truth[row["id"]] for row, v in zip(fields, parsed, strict=True)),
                     mean_tokens=sum(row["generated_tokens"] for row in fields)/len(fields))
                write_rows(output/f"{mode}-structured.jsonl", fields)
            print(f"{name}: {mode} complete", flush=True)
        write(output/"complete.json", metrics)
        del model, base


def delivery(args):
    """Decompose outcomes on fixed-prompt prefixes using strata defined by the frozen model."""
    def read(group, cap):
        folder = args.evaluation/group
        path = folder/f"standard-{cap}.json"
        if not path.exists():
            path = folder/f"gsm8k-standard-{cap}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))["numeric"]
        indexed = {row["id"]: row for row in rows}
        if not indexed or len(indexed) != len(rows):
            raise ValueError(f"Missing or repeated question IDs in {path}")
        return indexed

    base_short, base_long = (read(args.base, cap) for cap in (args.short, args.long))
    if base_short.keys() != base_long.keys():
        raise ValueError("Frozen short and long prefixes use different questions")
    cohorts = {key: [] for key in ("frozen_early", "frozen_late", "frozen_wrong_at_both")}
    for key, row in base_short.items():
        if row["correct"]:
            cohorts["frozen_early"].append(key)
        elif base_long[key]["correct"]:
            cohorts["frozen_late"].append(key)
        else:
            cohorts["frozen_wrong_at_both"].append(key)
    groups = {}
    for group in dict.fromkeys((args.base, *args.group)):
        short, long = (read(group, cap) for cap in (args.short, args.long))
        if short.keys() != base_short.keys() or long.keys() != base_short.keys():
            raise ValueError(f"Mismatched prefix question IDs: {group}")
        groups[group] = dict(
            short_correct_by_frozen_cohort={name: sum(short[k]["correct"] for k in keys) for name, keys in cohorts.items()},
            long_correct=sum(row["correct"] for row in long.values()),
            late_previously_unparsed=sum(long[k]["correct"] and not row["parsed"] for k, row in short.items()),
            late_previously_wrong=sum(long[k]["correct"] and row["parsed"] and not row["correct"] for k, row in short.items()),
            early_correct_later_wrong=sum(row["correct"] and not long[k]["correct"] for k, row in short.items()),
        )
    write(args.output, dict(short=args.short, long=args.long, frozen_group=args.base,
                            cohorts={name: len(keys) for name, keys in cohorts.items()}, groups=groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--train", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for name in ("probe", "evaluate"):
        command = commands.add_parser(name)
        for field in ("model", "data", "output"):
            command.add_argument(f"--{field}", type=Path, required=True)
    command.add_argument("--group", nargs="+", required=True)
    command.add_argument("--questions", type=int, required=True)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--cap", type=int, choices=(128, 256, 512), default=128)
    command.add_argument("--prefix-budgets", nargs="+", type=int)
    command.add_argument("--prompts", nargs="+", choices=PROMPTS, default=list(PROMPTS))
    command.add_argument("--structured", action="store_true")
    command.add_argument("--thinking", action="store_true")
    decompose = commands.add_parser("delivery")
    decompose.add_argument("--evaluation", type=Path, required=True)
    decompose.add_argument("--output", type=Path, required=True)
    decompose.add_argument("--base", default="base")
    decompose.add_argument("--short", type=int, choices=(128, 256), default=128)
    decompose.add_argument("--long", type=int, choices=(256, 512), default=512)
    decompose.add_argument("--group", nargs="+", required=True)
    args = parser.parse_args()
    if args.mode == "delivery" and args.short >= args.long:
        parser.error("Delivery analysis requires a shorter prefix and a longer prefix")
    {"prepare": prepare, "probe": probe, "evaluate": evaluate, "delivery": delivery}[args.mode](args)


if __name__ == "__main__":
    main()
