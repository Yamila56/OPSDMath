"""Compare the pinned upstream loss helper with native loss and LoRA gradients."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from opsdmath.data.gsm8k import read_rows
from opsdmath.objectives.kl import ROLLOUT_TEMPERATURE, distillation_loss
from opsdmath.training.prompts import prompt
from opsdmath.training.rollout import completion_logits, fixed_teacher_logits


def upstream_loss(path):
    contents = path.read_bytes()
    parsed = ast.parse(contents, filename=str(path))
    trainer = next(item for item in parsed.body if isinstance(item, ast.ClassDef) and item.name == "OPSDTrainer")
    formula = next(method for method in trainer.body
                   if isinstance(method, ast.FunctionDef) and method.name == "generalized_jsd_loss")
    # Load the objective alone to keep the parity check independent of the trainer's runtime.
    formula.decorator_list = []
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[formula], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[formula.name], hashlib.sha256(contents).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "adapter", "training", "rollouts", "upstream", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args()
    reference, digest = upstream_loss(args.upstream)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True).eval()
    params = [p for p in model.parameters() if p.requires_grad]
    data = {row["id"]: row for row in read_rows(args.training)}
    output = []
    for row in read_rows(args.rollouts)[:args.examples]:
        example = data[row["id"]]
        suffix = torch.tensor([row["tokens"]], device=model.device)
        student_ids = prompt(tokenizer, example).to(model.device)
        teacher_ids = prompt(tokenizer, example, privileged=True, context="answer").to(model.device)
        teacher = fixed_teacher_logits(model, teacher_ids, suffix).float()
        student = completion_logits(model, student_ids, suffix)
        native = distillation_loss(student, teacher)
        upstream = reference(student.float(), teacher, labels=suffix, beta=0,
                             temperature=ROLLOUT_TEMPERATURE, token_clip=.05)
        torch.testing.assert_close(native, upstream, rtol=1e-5, atol=1e-6)
        native_gradients = torch.autograd.grad(native, params, retain_graph=True)
        upstream_gradients = torch.autograd.grad(upstream, params)
        largest = 0.0
        for actual, expected in zip(native_gradients, upstream_gradients, strict=True):
            torch.testing.assert_close(actual, expected, rtol=.02, atol=3e-6)
            largest = max(largest, float((actual-expected).abs().max()))
        padded = torch.cat((suffix, torch.full_like(suffix[:, :2], -100)), dim=1)
        padding = torch.zeros_like(teacher[:, :2])
        masked_upstream = reference(torch.cat((student.detach().float(), padding), dim=1),
                                    torch.cat((teacher, padding), dim=1), labels=padded,
                                    beta=0, temperature=ROLLOUT_TEMPERATURE, token_clip=.05)
        torch.testing.assert_close(native.detach(), masked_upstream, rtol=1e-5, atol=1e-6)
        output.append(dict(id=row["id"], tokens=suffix.shape[1], native=float(native.detach()),
                           upstream=float(upstream.detach()), max_gradient_difference=largest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(upstream_sha256=digest, cases=output, gradient_tensors=len(params),
                                           clipping=.05, beta=0, temperature=ROLLOUT_TEMPERATURE),
                                      indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
