"""Single-device Qwen3 short-answer SFT and on-policy self-distillation."""

import argparse
import hashlib
import importlib.util
import json
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k import read_rows, write_rows

INSTRUCTION = "Solve briefly. End with Final answer: <number>."
ROLLOUT_TEMPERATURE = 1.1
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


def prompt(tokenizer, row, privileged=False, context="full"):
    problem = row["problem"] if "problem" in row else row["question"]
    message = f"Problem: {problem}\n\n"
    if privileged and context in BRIEF_CONTEXTS:
        message += f"Problem notes:\n{teacher_context(row, context)}\n\n"
    elif privileged:
        message += (
            "Here is a reference solution to this problem:\n"
            f"=== Reference Solution Begin ===\n{teacher_context(row, context)}\n=== Reference Solution End ===\n\n"
            "After reading the reference, derive the answer in your own words.\n"
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message + INSTRUCTION}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
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


def distillation_loss(student_logits, teacher_logits):
    student = F.log_softmax(student_logits.float() / ROLLOUT_TEMPERATURE, dim=-1)
    teacher = F.log_softmax(teacher_logits.float() / ROLLOUT_TEMPERATURE, dim=-1)
    # The upstream clip applies to each vocabulary contribution before summation.
    contributions = F.kl_div(student, teacher, reduction="none", log_target=True).clamp(max=0.05)
    return contributions.sum(dim=-1).mean()


def check_loss():
    teacher = torch.tensor([[[0.1, -0.2, 0.3], [2.0, 0.0, -1.0]]])
    student = torch.tensor([[[0.3, 0.1, -0.2], [-1.0, 2.0, 0.0]]], requires_grad=True)
    p = (teacher / ROLLOUT_TEMPERATURE).softmax(-1)
    q = (student / ROLLOUT_TEMPERATURE).softmax(-1)
    expected = (p * (p.log() - q.log())).clamp(max=0.05).sum() / 2
    actual = distillation_loss(student, teacher)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert student.grad.isfinite().all() and student.grad.abs().sum() > 0
    torch.testing.assert_close(distillation_loss(teacher, teacher), torch.tensor(0.0))


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


def train(args, model, tokenizer, rows, device):
    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-6)
    random.Random(args.seed).shuffle(rows)
    steps = (len(rows) + args.accumulation - 1) // args.accumulation
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0, 1 - step / steps))
    fixed_rollouts = []
    if args.fixed_rollout:
        started = time.perf_counter()
        for index in range(0, len(rows), args.accumulation):
            group = rows[index:index + args.accumulation]
            samples = batch_suffixes(model, tokenizer, [prompt(tokenizer, row) for row in group],
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
    for index, row in enumerate(rows):
        started = time.perf_counter()
        ids = prompt(tokenizer, row).to(device)
        if args.mode == "opsd":
            teacher_ids = prompt(tokenizer, row, privileged=True, context=args.teacher_context).to(device)
            group_index = index % args.accumulation
            if not fixed_rollouts and group_index == 0:
                group_prompts = [prompt(tokenizer, item) for item in rows[index:index + args.accumulation]]
                sampled_group = batch_suffixes(model, tokenizer, group_prompts, args.cap, True, device)
                log({
                    "event": "rollout_batch", "items": len(sampled_group),
                    "tokens": sum(len(tokens) for tokens, _ in sampled_group),
                    "seconds": time.perf_counter() - started,
                }, args.output)
            # All trajectories in this accumulation group use the policy before its next optimizer update.
            if fixed_rollouts:
                cached = fixed_rollouts[index]
                if cached["id"] != row["id"]:
                    raise ValueError("Fixed rollout ID mismatch")
                suffix_ids = cached["tokens"]
            else:
                suffix_ids = sampled_group[group_index][0]
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
        if args.mode == "opsd":
            loss = distillation_loss(logits, teacher)
            del teacher
        else:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), suffix.flatten())
        if not loss.isfinite():
            raise RuntimeError(f"Nonfinite loss on {row['id']}")
        group_start = index // args.accumulation * args.accumulation
        (loss / min(args.accumulation, len(rows) - group_start)).backward()
        del logits
        if (index + 1) % args.accumulation == 0 or index + 1 == len(rows):
            before = [parameter.detach().abs().max().item() for parameter in parameters] if args.probe else []
            grad = torch.nn.utils.clip_grad_norm_(parameters, 0.1, error_if_nonfinite=True)
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
            log({"event": "update", "microsteps": index + 1, "grad_norm": float(grad)}, args.output)
        synchronize(device)
        done = time.perf_counter()
        log({
            "event": "microstep", "id": row["id"], "loss": loss.item(), "tokens": suffix.shape[1],
            "teacher_prompt_tokens": teacher_ids.shape[1] if args.mode == "opsd" else None,
            "rollout_seconds": generated_at - started, "teacher_seconds": teacher_at - generated_at,
            "student_update_seconds": done - teacher_at, "seconds": done - started, **memory(device),
        }, args.output)
    model.save_pretrained(args.output / "adapter")


def predict(args, model, tokenizer, rows, device):
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    predictions = []
    for index in range(0, len(rows), args.batch_size):
        batch = rows[index:index + args.batch_size]
        start = time.perf_counter()
        suffixes = batch_suffixes(
            model, tokenizer, [prompt(tokenizer, row) for row in batch], args.cap, False, device,
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
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1, help="Left-padded prediction batch size.")
    parser.add_argument("--cap", type=int, default=256)
    parser.add_argument("--sft-reference-cap", type=int, default=320)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--teacher-context", choices=["full", "equations", "answer", *BRIEF_CONTEXTS], default="full")
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
    brief_training = args.mode == "opsd" and args.teacher_context in BRIEF_CONTEXTS
    required = {"id", "problem", args.teacher_context if brief_training else "solution"}
    if not rows or (checker and any(not required <= row.keys() for row in rows)):
        parser.error("Pass a nonempty dataset with reference solutions for training.")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.probe and checker and not brief_training:
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
