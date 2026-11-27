"""Compare native OPSD with upstream Trainer.compute_loss on a fixed update."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from accelerate import Accelerator
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from opsdmath.data.gsm8k import read_rows
from opsdmath.objectives.kl import ROLLOUT_TEMPERATURE, distillation_loss
from opsdmath.training.prompts import prompt
from opsdmath.training.rollout import completion_logits, fixed_teacher_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "adapter", "training", "rollouts", "upstream", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))
    from opsd_trainer import OPSDTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True).eval()
    params = [p for p in model.parameters() if p.requires_grad]
    starting = [p.detach().clone() for p in params]
    reference = OPSDTrainer.generalized_jsd_loss
    state = SimpleNamespace(use_thinking_machines_loss=False, use_ema_teacher=False, fixed_teacher=True,
                            accelerator=Accelerator(), beta=0, top_k_loss=None, jsd_token_clip=.05,
                            temperature=ROLLOUT_TEMPERATURE, generalized_jsd_loss=reference)
    train = {row["id"]: row for row in read_rows(args.training)}
    inputs = []
    for entry in read_rows(args.rollouts)[:4]:
        row = train[entry["id"]]
        inputs.append((prompt(tokenizer, row).to(model.device),
                       prompt(tokenizer, row, privileged=True, context="answer").to(model.device),
                       torch.tensor([entry["tokens"]], device=model.device)))
    if len(inputs) != 4:
        raise ValueError("Expected four fixed rollouts for one accumulation group")

    def trainer_loss(student, teacher, suffix, *, keep_tail=False, omit_mask=False):
        student_ids, teacher_ids = [torch.cat((p, suffix), dim=1) for p in (student, teacher)]
        labels = torch.cat((torch.full_like(student, -100), suffix), dim=1)
        batch = dict(
            student_input_ids=student_ids, teacher_input_ids=teacher_ids, labels=labels,
            student_attention_mask=torch.ones_like(student_ids), teacher_attention_mask=torch.ones_like(teacher_ids),
            student_prompt_length=student.size(1), teacher_prompt_length=teacher.size(1),
        )
        original = model.forward

        def audit_forward(**kwargs):
            if omit_mask:
                kwargs.pop("attention_mask", None)
            if not keep_tail:
                return original(**kwargs)
            tail = suffix.size(1)+1
            out = original(**kwargs, logits_to_keep=tail)
            out.logits = torch.nn.functional.pad(out.logits, (0, 0, kwargs["input_ids"].size(1)-tail, 0))
            return out

        with patch.object(model, "forward", audit_forward):
            return OPSDTrainer.compute_loss(state, model, batch)

    def update(loss_fn):
        optimizer = torch.optim.AdamW(params, lr=5e-6)
        losses = []
        for student, teacher, suffix in inputs:
            loss = loss_fn(student, teacher, suffix)
            if not loss.isfinite():
                raise RuntimeError("Nonfinite fixed-trajectory loss")
            (loss/len(inputs)).backward()
            losses.append(float(loss.detach()))
        before_clip = [p.grad.detach().clone() for p in params]
        norm = torch.nn.utils.clip_grad_norm_(params, .1, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updated = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p, old in zip(params, starting, strict=True):
                p.copy_(old)
        return losses, before_clip, updated, float(norm)

    def native(student, teacher, suffix):
        teacher_logits = fixed_teacher_logits(model, teacher, suffix)
        return distillation_loss(completion_logits(model, student, suffix), teacher_logits)

    native_losses, native_grads, native_params, native_norm = update(native)
    cases = {}
    def fp32_loss(student_logits, teacher_logits, **kwargs):
        return reference(student_logits.float(), teacher_logits.float(), **kwargs)

    for mode, score, options in (
        ("upstream_bf16", reference, {}),
        ("upstream_fp32_objective", fp32_loss, {}),
        ("upstream_fp32_keep_tail", fp32_loss, {"keep_tail": True}),
        ("upstream_fp32_no_mask", fp32_loss, {"omit_mask": True}),
        ("upstream_fp32_keep_tail_no_mask", fp32_loss, {"keep_tail": True, "omit_mask": True}),
    ):
        state.generalized_jsd_loss = score
        loss, grads, weights, norm = update(lambda *inputs: trainer_loss(*inputs, **options))
        cases[mode] = dict(
            losses=loss, gradient_norm=norm,
            max_loss_diff=max(abs(a-b) for a, b in zip(native_losses, loss, strict=True)),
            max_gradient_diff=max((a-b).abs().max().item() for a, b in zip(native_grads, grads, strict=True)),
            max_updated_weight_diff=max((a-b).abs().max().item() for a, b in zip(native_params, weights, strict=True)),
        )
    if all(torch.equal(a, b) for a, b in zip(starting, native_params, strict=True)):
        raise RuntimeError("Native optimizer did not change the starting adapter")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        trajectories=len(inputs), gradient_tensors=len(params), native_losses=native_losses,
        native_gradient_norm=native_norm, cases=cases), indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
