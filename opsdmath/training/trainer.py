"""Single-device Qwen3 short-answer SFT and on-policy self-distillation."""

import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

import torch

from opsdmath.evaluation.answer_parser import audit_final_answer, number
from opsdmath.io import read_rows, write_rows

from opsdmath.evaluation.prediction import predict
from opsdmath.objectives.kl import check_loss
from opsdmath.training.arguments import trainer_parser, training_rows
from opsdmath.training.checkpoint import save_state, training_identity, validate_resume
from opsdmath.training.modeling import load_model, load_tokenizer, lora_config
from opsdmath.training.losses import sft_loss, opsd_loss
from opsdmath.training.prompts import BRIEF_CONTEXTS, prompt, teacher_context
from opsdmath.training.rollout import batch_suffixes, completion_logits, fixed_teacher_logits
from opsdmath.training.runtime import log, memory, synchronize


def check_prompts(tokenizer, row, output, context):
    collator_path = Path(__file__).resolve().parents[2] / "upstream" / "data_collator.py"
    spec = importlib.util.spec_from_file_location("official_opsd_collator", collator_path)
    collator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collator_module)
    official = collator_module.SelfDistillationDataCollator(tokenizer, reason_first=False)([row])
    student = prompt(tokenizer, row)
    teacher = prompt(tokenizer, row, privileged=True, context=context)
    student_text = tokenizer.decode(student[0])
    teacher_text = tokenizer.decode(teacher[0])
    if "Reference Solution Begin" in student_text or teacher_context(row, context) not in teacher_text:
        raise RuntimeError("Reference context is missing from teacher or present in student")
    log({"event": "prompt_audit", "student": student_text, "teacher": teacher_text,
         "official_collator_fields": sorted(official)}, output)


def train(args, model, tokenizer, rows, device):
    from peft import PeftModel, get_peft_model

    state = torch.load(args.resume_from, weights_only=True, map_location="cpu") if args.resume_from else None
    random.Random(args.data_seed if args.data_seed is not None else args.seed).shuffle(rows)
    identity = training_identity(args, rows)
    if state:
        validate_resume(state["identity"], identity, args.resume_from)
    model = (PeftModel.from_pretrained(model, args.resume_from.parent / state["adapter"], is_trainable=True)
             if state else get_peft_model(model, lora_config(args.rank)))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    steps = (len(rows) + args.accumulation - 1) // args.accumulation
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0, 1 - step / steps))
    def student_prompt(row):
        return prompt(tokenizer, row, output_budget=args.output_budget, thinking=args.student_thinking)

    first = 0
    if args.rollout_seed is not None:
        torch.manual_seed(args.rollout_seed)
    if state:
        first = state["next_index"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        log(dict(event="resume", microsteps=first, update=state["update"]), args.output)
        del state
    fixed_rollouts = []
    if args.fixed_rollout:
        started = time.perf_counter()
        for index in range(0, len(rows), args.accumulation):
            group = rows[index:index + args.accumulation]
            samples = batch_suffixes(model, tokenizer, [student_prompt(row) for row in group],
                                     args.cap, True, device)
            fixed_rollouts.extend({"id": row["id"], "tokens": ids, "hit_cap": hit}
                                  for row, (ids, hit) in zip(group, samples, strict=True))
        cache = args.output / "fixed_rollouts.jsonl"
        write_rows(cache, fixed_rollouts)
        fixed_rollouts = read_rows(cache)
        log({"event": "fixed_rollouts", "seconds": time.perf_counter() - started,
             "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(), "count": len(fixed_rollouts)}, args.output)
    fixed_reference = None
    sampled_group = []
    for index in range(first, len(rows)):
        row = rows[index]
        started = time.perf_counter()
        ids = student_prompt(row).to(device)
        if args.mode == "opsd":
            context = args.teacher_by_id[row["id"]] if args.teacher_by_id is not None else args.teacher_context
            teacher_ids = prompt(tokenizer, row, privileged=True, context=context, output_budget=args.output_budget,
                                 thinking=args.teacher_thinking).to(device)
            group_index = index % args.accumulation
            if not fixed_rollouts and group_index == 0:
                group_prompts = [student_prompt(item) for item in rows[index:index + args.accumulation]]
                rollout_batch_size = args.rollout_batch_size or len(group_prompts)
                sampled_group = []
                for offset in range(0, len(group_prompts), rollout_batch_size):
                    sampled_group.extend(batch_suffixes(
                        model, tokenizer, group_prompts[offset:offset + rollout_batch_size], args.cap, True, device,
                    ))
                log({
                    "event": "rollout_batch", "items": len(sampled_group),
                    "tokens": sum(len(tokens) for tokens, _ in sampled_group),
                    "rollout_batch_size": min(rollout_batch_size, len(group_prompts)),
                    "seconds": time.perf_counter() - started,
                }, args.output)
            # All trajectories in this accumulation group use the policy before its next optimizer update.
            if fixed_rollouts:
                cached = fixed_rollouts[index]
                if cached["id"] != row["id"]:
                    raise ValueError("Fixed rollout ID mismatch")
                suffix_ids = cached["tokens"]
                hit_cap = cached["hit_cap"]
            else:
                suffix_ids, hit_cap = sampled_group[group_index]
            suffix = torch.tensor([suffix_ids], device=device)
            synchronize(device)
            generated_at = time.perf_counter()
            teacher = fixed_teacher_logits(model, teacher_ids, suffix)
            if args.probe and index == 0:
                fixed_reference = (teacher_ids, suffix, teacher[0, :2, :32].clone())
                del teacher
                teacher = fixed_teacher_logits(model, teacher_ids, suffix)
            synchronize(device)
            teacher_at = time.perf_counter()
        else:
            suffix = tokenizer(row["solution"], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            eos = torch.tensor([[tokenizer.eos_token_id]], device=device)
            if suffix.shape[1] + 1 > args.sft_reference_cap:
                log({"event": "reference_truncated", "id": row["id"], "tokens": suffix.shape[1]}, args.output)
                suffix = suffix[:, :args.sft_reference_cap]
            else:
                suffix = torch.cat((suffix, eos), dim=1)
            generated_at = teacher_at = time.perf_counter()
        model.train()
        logits = completion_logits(model, ids, suffix)
        diagnostic = {}
        if args.mode == "opsd":
            loss = opsd_loss(logits, teacher, args, context, diagnostic)
            if args.log_diagnostics:
                parsed = audit_final_answer(tokenizer.decode(suffix_ids, skip_special_tokens=True))
                diagnostic.update(hit_cap=hit_cap, parsed=parsed is not None,
                                  correct=parsed is not None and parsed == number(row["answer"]))
            del teacher
        else:
            loss = sft_loss(logits, suffix)
        if not loss.isfinite():
            raise RuntimeError(f"Nonfinite loss on {row['id']}")
        group_start = index // args.accumulation * args.accumulation
        (loss / min(args.accumulation, len(rows) - group_start)).backward()
        del logits
        update = (index + 1) % args.accumulation == 0 or index + 1 == len(rows)
        if update:
            before = [parameter.detach().abs().max().item() for parameter in parameters] if args.probe else []
            grad = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if args.probe:
                after = [parameter.detach().abs().max().item() for parameter in parameters]
                if before == after:
                    raise RuntimeError("Adapter parameters did not update")
                if fixed_reference is not None:
                    teacher_ids, reference_suffix, before_teacher = fixed_reference
                    after_teacher = fixed_teacher_logits(model, teacher_ids, reference_suffix)[0, :2, :32]
                    torch.testing.assert_close(before_teacher, after_teacher, rtol=0, atol=0)
            log({"event": "update", "microsteps": index + 1, "grad_norm": float(grad),
                 "learning_rate": optimizer.param_groups[0]["lr"]}, args.output)
        synchronize(device)
        done = time.perf_counter()
        log({
            "event": "microstep", "id": row["id"], "loss": loss.item(), "tokens": suffix.shape[1],
            "teacher_prompt_tokens": teacher_ids.shape[1] if args.mode == "opsd" else None,
            **({"teacher_context": context} if args.teacher_by_id is not None else {}),
            "rollout_seconds": generated_at - started, "teacher_seconds": teacher_at - generated_at,
            "student_update_seconds": done - teacher_at, "seconds": done - started, **memory(device), **diagnostic,
        }, args.output)
        step = (index // args.accumulation) + 1
        if update and step in args.save_steps:
            adapter = f"checkpoint-{step:03d}"
            model.save_pretrained(args.output / adapter)
            save_state(args.output, adapter, identity, index+1, step, optimizer, scheduler, device)
            log(dict(event="checkpoint", update=step, path=adapter), args.output)
    model.save_pretrained(args.output / "adapter")


def main(argv=None):
    parser = trainer_parser()
    args = parser.parse_args(argv)
    check_loss()
    if args.mode == "check-loss":
        print("Forward KL, clipping, zero KL and gradient checks passed.")
        return
    rows = training_rows(parser, args)
    checker = args.mode != "predict"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    torch.manual_seed(args.init_seed if args.init_seed is not None else args.seed)
    device = torch.device(args.device)
    start = time.perf_counter()
    tokenizer = load_tokenizer(args.model)
    if args.probe and checker and args.teacher_context not in (*BRIEF_CONTEXTS, "none"):
        check_prompts(tokenizer, rows[0], args.output, args.teacher_context)
    model = load_model(args.model, tokenizer, device)
    synchronize(device)
    log({"event": "loaded", "seconds": time.perf_counter() - start, **memory(device)}, args.output)
    if checker:
        train(args, model, tokenizer, rows, device)
    else:
        predict(args, model, tokenizer, rows, device)
    log({"event": "complete", "seconds": time.perf_counter() - start, **memory(device)}, args.output)


if __name__ == "__main__":
    main()
