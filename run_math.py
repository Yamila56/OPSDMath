"""Single-device Qwen3 short-answer SFT and on-policy self-distillation."""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k import number, read_rows, write_rows
from summarize_run import audit_final_answer

INSTRUCTION = "Solve briefly. End with Final answer: <number>."
ROLLOUT_TEMPERATURE = 1.1
DEFAULT_KL_CLIP = 0.05
EQUATIONS = re.compile(r"<<([^<>]+)>>")
BRIEF_CONTEXTS = ("general_brief", "constraint_brief")


def teacher_context(row, mode):
    if mode in BRIEF_CONTEXTS:
        return row[mode]
    if mode == "full":
        return row["solution"]
    final = f"Final answer: {row['answer']}"
    if mode == "answer":
        return final
    return "\n".join([*EQUATIONS.findall(row["solution"]), final])


def prompt(tokenizer, row, privileged=False, context="full", output_budget=None, *,
           key_steps=None, thinking=False):
    problem = row["problem"] if "problem" in row else row["question"]
    message = f"Problem: {problem}\n\n"
    if privileged and context in BRIEF_CONTEXTS:
        message += f"Problem notes:\n{teacher_context(row, context)}\n\n"
    elif privileged and context != "none":
        message += (
            "Here is a reference solution to this problem:\n"
            f"=== Reference Solution Begin ===\n{teacher_context(row, context)}\n=== Reference Solution End ===\n\n"
            "After reading the reference, derive the answer in your own words.\n"
        )
    if key_steps is None:
        key_steps = output_budget is not None
    guidance = []
    if output_budget is not None:
        guidance.append(f"The complete response must fit within {output_budget} tokens.")
    if key_steps:
        guidance.append("Prioritize the key steps.")
    if guidance:
        message += " ".join(guidance) + "\n"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message + INSTRUCTION}],
        tokenize=True, add_generation_prompt=True, enable_thinking=thinking, return_tensors="pt",
    )


def rollout(model, ids, cap, sample, attention_mask=None):
    generation = {"temperature": ROLLOUT_TEMPERATURE, "top_p": 0.95, "top_k": 20} if sample else {
        "temperature": None, "top_p": None, "top_k": None,
    }
    model.eval()
    with torch.no_grad():
        return model.generate(
            ids, attention_mask=attention_mask if attention_mask is not None else torch.ones_like(ids),
            max_new_tokens=cap, do_sample=sample,
            use_cache=True, **generation,
        )[:, ids.shape[1]:]


def batch_suffixes(model, tokenizer, prompts, cap, sample, device):
    """Generate with left padding and remove post-EOS padding from each trajectory."""
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    tokenizer.padding_side = "left"
    inputs = tokenizer.pad({"input_ids": [ids[0].tolist() for ids in prompts]}, return_tensors="pt").to(device)
    suffixes = rollout(model, inputs.input_ids, cap, sample, attention_mask=inputs.attention_mask)
    tokens = []
    for suffix in suffixes.tolist():
        eos = next((index for index, token in enumerate(suffix) if token in eos_ids), None)
        tokens.append((suffix if eos is None else suffix[:eos + 1], eos is None and len(suffix) == cap))
    return tokens


def completion_logits(model, ids, suffix):
    merged = torch.cat((ids, suffix), dim=1)
    logits = model(merged, use_cache=False, logits_to_keep=suffix.shape[1] + 1).logits
    return logits[:, :-1]


def fixed_teacher_logits(model, ids, suffix):
    model.eval()
    with model.disable_adapter(), torch.no_grad():
        return completion_logits(model, ids, suffix)


def distillation_loss(student_logits, teacher_logits, diagnostics=None, *, kl_clip=DEFAULT_KL_CLIP):
    if torch.equal(student_logits, teacher_logits):
        # At identical logits the KL and its gradient are exactly zero. Computing
        # log-softmax backward here leaves roundoff that Adam can magnify.
        if diagnostics is not None:
            diagnostics.update(unclipped_kl=0.0, clipped_kl=0.0,
                               clipped_vocabulary_fraction=0.0, clipped_position_fraction=0.0,
                               exact_logits_match=True)
        return student_logits.reshape(-1)[0].float() * 0.0
    student = F.log_softmax(student_logits.float() / ROLLOUT_TEMPERATURE, dim=-1)
    teacher = F.log_softmax(teacher_logits.float() / ROLLOUT_TEMPERATURE, dim=-1)
    # The upstream clip applies to each vocabulary contribution before summation.
    contributions = F.kl_div(student, teacher, reduction="none", log_target=True)
    loss = (contributions if kl_clip is None else contributions.clamp(max=kl_clip)).sum(-1).mean()
    if diagnostics is not None:
        with torch.no_grad():
            clipped = contributions > kl_clip if kl_clip is not None else torch.zeros_like(contributions, dtype=torch.bool)
            diagnostics.update(unclipped_kl=float(contributions.sum(-1).mean()),
                               clipped_kl=float(loss),
                               clipped_vocabulary_fraction=float(clipped.float().mean()),
                               clipped_position_fraction=float(clipped.any(dim=-1).float().mean()),
                               exact_logits_match=False)
    return loss


def check_loss():
    teacher = torch.tensor([[[0.1, -0.2, 0.3], [2.0, 0.0, -1.0]]])
    student = torch.tensor([[[0.3, 0.1, -0.2], [-1.0, 2.0, 0.0]]], requires_grad=True)
    p = (teacher / ROLLOUT_TEMPERATURE).softmax(-1)
    q = (student / ROLLOUT_TEMPERATURE).softmax(-1)
    for threshold in (DEFAULT_KL_CLIP, 0.001, 1e-7, None):
        terms = p * (p.log() - q.log())
        expected = (terms if threshold is None else terms.clamp(max=threshold)).sum() / 2
        actual = distillation_loss(student, teacher, kl_clip=threshold)
        torch.testing.assert_close(actual, expected)
        gradient, = torch.autograd.grad(actual, student)
        assert gradient.isfinite().all() and gradient.abs().sum() > 0
        match = teacher.clone().requires_grad_()
        zero = distillation_loss(match, teacher, kl_clip=threshold)
        torch.testing.assert_close(zero, torch.tensor(0.0), rtol=0, atol=0)
        exact_gradient, = torch.autograd.grad(zero, match)
        torch.testing.assert_close(exact_gradient, torch.zeros_like(match), rtol=0, atol=0)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def log(event, output):
    row = json.dumps(event, ensure_ascii=False)
    with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(row + "\n")
    print(row, flush=True)


def memory(device):
    return {
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
    } if device.type == "cuda" else {}


def check_prompts(tokenizer, row, output, context):
    collator_path = Path(__file__).parent / "upstream" / "data_collator.py"
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


def lora_config(rank):
    return LoraConfig(
        r=rank, lora_alpha=2 * rank, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def train(args, model, tokenizer, rows, device):
    state = torch.load(args.resume_from, weights_only=True, map_location="cpu") if args.resume_from else None
    model = (PeftModel.from_pretrained(model, args.resume_from.parent / state["adapter"], is_trainable=True)
             if state else get_peft_model(model, lora_config(args.rank)))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    random.Random(args.data_seed if args.data_seed is not None else args.seed).shuffle(rows)
    steps = (len(rows) + args.accumulation - 1) // args.accumulation
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0, 1 - step / steps))
    def student_prompt(row):
        return prompt(tokenizer, row, output_budget=args.output_budget, thinking=args.student_thinking)

    first = 0
    identity = {key: str(getattr(args, key)) for key in (
        "mode", "model", "seed", "init_seed", "data_seed", "rollout_seed", "rank", "accumulation", "cap",
        "sft_reference_cap", "teacher_context", "teacher_thinking", "student_thinking", "learning_rate",
        "max_grad_norm", "kl_clip",
    )}
    identity["ordered_rows_sha256"] = hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if args.output_budget is not None:
        identity["output_budget"] = args.output_budget
    if args.rollout_batch_size is not None:
        identity["rollout_batch_size"] = args.rollout_batch_size
    if args.teacher_by_id is not None:
        identity["teacher_by_id"] = args.teacher_by_id
    if args.unclipped_anchor:
        identity["unclipped_anchor"] = True
    if args.rollout_seed is not None:
        torch.manual_seed(args.rollout_seed)
    if state:
        if state["identity"] != identity:
            raise ValueError("Resume requires the original model, data order and training configuration")
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
            clip = None if args.unclipped_anchor and context == "none" else args.kl_clip
            loss = distillation_loss(logits, teacher, diagnostics=diagnostic if args.probe or args.log_diagnostics else None,
                                     kl_clip=clip)
            if args.log_diagnostics:
                parsed = audit_final_answer(tokenizer.decode(suffix_ids, skip_special_tokens=True))
                diagnostic.update(hit_cap=hit_cap, parsed=parsed is not None,
                                  correct=parsed is not None and parsed == number(row["answer"]))
            del teacher
        else:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), suffix.flatten())
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
            temporary = args.output / "train-state.tmp"
            torch.save(dict(adapter=adapter, identity=identity, next_index=index+1, update=step,
                            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                            cpu_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None), temporary)
            temporary.replace(args.output / "train-state.pt")
            log(dict(event="checkpoint", update=step, path=adapter), args.output)
    model.save_pretrained(args.output / "adapter")


def predict(args, model, tokenizer, rows, device):
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    predictions = []
    for index in range(0, len(rows), args.batch_size):
        batch = rows[index:index + args.batch_size]
        start = time.perf_counter()
        suffixes = batch_suffixes(
            model, tokenizer, [prompt(tokenizer, row, thinking=args.student_thinking) for row in batch],
            args.cap, False, device,
        )
        token_counts = []
        for row, (suffix, hit_cap) in zip(batch, suffixes, strict=True):
            token_counts.append(len(suffix))
            prediction = {
                "id": row["id"], "response": tokenizer.decode(suffix, skip_special_tokens=True),
                "generated_tokens": len(suffix), "hit_cap": hit_cap,
            }
            if args.save_token_ids:
                prediction["token_ids"] = suffix
            predictions.append(prediction)
        log({"event": "prediction_batch", "count": len(batch), "tokens": sum(token_counts),
             "seconds": time.perf_counter() - start, **memory(device)}, args.output)
    write_rows(args.output / f"{args.prediction_name}.jsonl", predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["opsd", "sft", "predict", "check-loss"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=17)
    for seed in ("init", "data", "rollout"):
        parser.add_argument(f"--{seed}-seed", type=int)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--kl-clip", type=lambda value: None if value == "none" else float(value), default=DEFAULT_KL_CLIP)
    parser.add_argument("--student-thinking", action="store_true")
    parser.add_argument("--teacher-thinking", action="store_true")
    parser.add_argument("--save-steps", nargs="+", type=int, default=[])
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--log-diagnostics", action="store_true")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--rollout-batch-size", type=int,
                        help="Maximum simultaneous generations for online OPSD within one optimizer step.")
    parser.add_argument("--batch-size", type=int, default=1, help="Left-padded prediction batch size.")
    parser.add_argument("--cap", type=int, default=256)
    parser.add_argument("--output-budget", type=int, help="Advertise a common output budget in both training prompts.")
    parser.add_argument("--sft-reference-cap", type=int, default=320)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--teacher-context", choices=["full", "equations", "answer", "none", *BRIEF_CONTEXTS], default="full")
    parser.add_argument("--teacher-routing", type=Path, help="An exact training-ID map to answer or none.")
    parser.add_argument("--unclipped-anchor", action="store_true",
                        help="Use the true KL for unprivileged teacher rows and retain clipping for privileged rows.")
    parser.add_argument("--fixed-rollout", action="store_true")
    parser.add_argument("--save-token-ids", action="store_true")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--prediction-name", default="base", choices=["base", "sft", "opsd"])
    args = parser.parse_args()
    check_loss()
    if args.mode == "check-loss":
        print("Forward KL, clipping, zero KL and gradient checks passed.")
        return
    if not (args.model and args.data and args.output) or min(
        args.cap, args.sft_reference_cap, args.accumulation, args.rank, args.batch_size,
    ) < 1:
        parser.error("Provide --model, --data and --output and use positive generation and adapter sizes.")
    checker = args.mode != "predict"
    rows = read_rows(args.data)[:args.limit]
    if args.output_budget is not None and (not checker or args.output_budget <= 0):
        parser.error("--output-budget requires a positive training prompt budget.")
    if args.unclipped_anchor and args.mode != "opsd":
        parser.error("--unclipped-anchor is available for OPSD only.")
    if args.rollout_batch_size is not None and (args.rollout_batch_size < 1 or
                                                args.mode != "opsd" or args.fixed_rollout):
        parser.error("--rollout-batch-size requires a positive size and online OPSD.")
    args.teacher_by_id = None
    if args.teacher_routing:
        if args.mode != "opsd" or args.probe or args.teacher_context != "answer":
            parser.error("--teacher-routing requires answer-only OPSD without --probe.")
        args.teacher_by_id = json.loads(args.teacher_routing.read_text(encoding="utf-8"))
        ids = {row["id"] for row in rows}
        if (not isinstance(args.teacher_by_id, dict) or len(ids) != len(rows)
                or args.teacher_by_id.keys() != ids or any(v not in ("answer", "none") for v in args.teacher_by_id.values())):
            parser.error("Every distinct training ID needs an answer or none routing entry.")
    if any(not math.isfinite(x) or x <= 0 for x in (args.learning_rate, args.max_grad_norm)
           + (() if args.kl_clip is None else (args.kl_clip,))):
        parser.error("Learning rate, gradient norm and a numeric KL clip must be finite and positive.")
    if args.resume_from and (args.fixed_rollout or not checker):
        parser.error("Resume is available for online OPSD and SFT training.")
    if set(args.save_steps) - set(range(1, (len(rows)+args.accumulation-1)//args.accumulation+1)):
        parser.error("Checkpoint steps must be within the training run.")
    brief_training = args.mode == "opsd" and args.teacher_context in BRIEF_CONTEXTS
    required = {"id", "problem"}
    if brief_training:
        required.add(args.teacher_context)
    elif args.mode == "sft" or args.teacher_context != "none":
        required.add("solution")
    if args.mode == "opsd" and (args.log_diagnostics or args.teacher_context in ("answer", "equations")):
        required.add("answer")
    if not rows or (checker and any(not required <= row.keys() for row in rows)):
        parser.error("Pass a nonempty dataset with reference solutions for training.")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    torch.manual_seed(args.init_seed if args.init_seed is not None else args.seed)
    device = torch.device(args.device)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.probe and checker and not brief_training and args.teacher_context != "none":
        check_prompts(tokenizer, rows[0], args.output, args.teacher_context)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map={"": str(device)}, attn_implementation="sdpa", local_files_only=True,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    synchronize(device)
    log({"event": "loaded", "seconds": time.perf_counter() - start, **memory(device)}, args.output)
    if checker:
        train(args, model, tokenizer, rows, device)
    else:
        predict(args, model, tokenizer, rows, device)
    log({"event": "complete", "seconds": time.perf_counter() - start, **memory(device)}, args.output)


if __name__ == "__main__":
    main()
