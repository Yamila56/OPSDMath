"""Verify a zero-signal update, accumulated gradients, and a real KL recovery gradient."""
import argparse
import json
from pathlib import Path

import torch
from peft import get_peft_model

from gsm8k import read_rows
from opsd_calibration import load_model
from run_math import (check_loss, completion_logits, distillation_loss, fixed_teacher_logits,
                      lora_config, prompt, rollout)


def norm(tensors):
    return float(torch.stack([tensor.detach().float().norm() for tensor in tensors]).norm())


@torch.no_grad()
def step(optimizer, named):
    tensors = [parameter for _, parameter in named]
    previous = [parameter.detach().clone() for parameter in tensors]
    before = float(torch.nn.utils.clip_grad_norm_(tensors, .1, error_if_nonfinite=True))
    after = norm([p.grad for p in tensors])
    optimizer.step()
    results = dict(grad_norm_before=before, grad_norm_after=after, learning_rate=optimizer.param_groups[0]["lr"])
    for kind in ("lora_A", "lora_B"):
        selected = [(p, old) for (name, p), old in zip(named, previous, strict=True) if kind in name]
        results[kind] = dict(
            delta_l2=norm([p - old for p, old in selected]),
            first_moment_l2=norm([optimizer.state[p]["exp_avg"] for p, _ in selected]),
            second_moment_l2=norm([optimizer.state[p]["exp_avg_sq"] for p, _ in selected]),
        )
    optimizer.zero_grad(set_to_none=True)
    return results


def controlled_checks():
    teacher = torch.zeros(1, 1, 2)
    student = torch.tensor([[[6., -6.]]], requires_grad=True)
    true_kl = float(distillation_loss(student, teacher, kl_clip=None).detach())
    clipped_kl = float(distillation_loss(student, teacher, kl_clip=.05).detach())
    assert true_kl > 0 > clipped_kl
    parameter = torch.nn.Parameter(torch.tensor([.5]))
    optimizer = torch.optim.AdamW([parameter], lr=5e-6, weight_decay=0.)
    parameter.grad = torch.tensor([.2])
    optimizer.step()
    before = float(parameter.detach())
    parameter.grad.zero_()
    optimizer.step()
    after = float(parameter.detach())
    assert after != before
    return dict(two_token_full_kl=true_kl, two_token_clipped_kl=clipped_kl,
                momentum_only_delta=after - before, first_moment=float(optimizer.state[parameter]["exp_avg"]),
                second_moment=float(optimizer.state[parameter]["exp_avg_sq"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "train", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    check_loss()
    controls = controlled_checks()
    rows = read_rows(args.train)[:2]
    base, tokenizer = load_model(args.model)
    torch.manual_seed(args.seed)
    model = get_peft_model(base, lora_config(64))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([p for _, p in named], lr=5e-6)
    ids = [prompt(tokenizer, row, output_budget=128).to(model.device) for row in rows]
    suffixes = [rollout(model, sample, 64, False) for sample in ids]
    frozen = [fixed_teacher_logits(model, sample, suffix)
              for sample, suffix in zip(ids, suffixes, strict=True)]

    def students(index):
        model.train()
        return completion_logits(model, ids[index], suffixes[index])

    pure = []
    for index in range(len(rows)):
        observed = students(index)
        if not torch.equal(observed, frozen[index]):
            raise ValueError("The initial or pure-anchor student differs from the frozen teacher")
        diagnostics = {}
        distillation_loss(observed, frozen[index], diagnostics, kl_clip=None).backward()
        evidence = step(opt, named)
        assert diagnostics["exact_logits_match"] and evidence["grad_norm_before"] == 0
        assert evidence["lora_B"]["delta_l2"] == 0
        pure.append(evidence)
    assert all(torch.equal(a, rollout(model, b, 64, False)) for a, b in zip(suffixes, ids, strict=True))

    privileged_ids = prompt(tokenizer, rows[0], privileged=True, context="answer",
                            output_budget=128).to(model.device)
    privileged = fixed_teacher_logits(model, privileged_ids, suffixes[0])
    (distillation_loss(students(0), privileged, kl_clip=.05) / 2).backward()
    previous_grad = [p.grad.clone() for _, p in named]
    assert norm(previous_grad) > 0
    (distillation_loss(students(1), frozen[1], kl_clip=None) / 2).backward()
    assert all(torch.equal(old, p.grad) for old, (_, p) in zip(previous_grad, named, strict=True))
    del previous_grad
    mixed = step(opt, named)

    # This perturbation is separate from training and has a known seed.
    parameter = next(p for name, p in reversed(named) if "lora_B" in name)
    perturbation = torch.Generator(device=model.device).manual_seed(args.seed + 10000)
    with torch.no_grad():
        parameter.add_(.02 * torch.randn(parameter.shape, generator=perturbation, device=model.device))
    recovery_opt = torch.optim.AdamW([p for _, p in named], lr=5e-6)
    observed = students(1)
    fixed_logits = observed.detach().float().requires_grad_()
    objective = {}
    for label, clip in (("full", None), ("clipped", .05)):
        loss = distillation_loss(fixed_logits, frozen[1].float(), kl_clip=clip)
        gradient, = torch.autograd.grad(loss, fixed_logits)
        objective[label] = dict(loss=float(loss.detach()), grad_norm=float(gradient.norm()),
                               toward_teacher=float((gradient * (fixed_logits.detach() - frozen[1])).sum()))
    assert objective["full"]["loss"] > 0 and objective["full"]["toward_teacher"] > 0
    anchor = distillation_loss(observed, frozen[1], kl_clip=None)
    anchor.backward()
    recovery = step(recovery_opt, named)
    with torch.no_grad():
        after = float(distillation_loss(students(1), frozen[1], kl_clip=None))
    result = dict(seed=args.seed, training_ids=[r["id"] for r in rows], controlled_checks=controls,
                  exact_initial_logits=True, identical_greedy_suffixes_after_pure=True, pure_updates=pure,
                  mixed_preserved_all_gradients=True, mixed_update=mixed, perturbed_logits=objective,
                  perturbed_anchor_update=recovery, full_kl_after_recovery=after)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not after < objective["full"]["loss"]:
        raise RuntimeError("One recovery update did not reduce the KL on the fixed trajectory")
    print(f"Verified the exact zero, mixed accumulation, and recovery update at seed {args.seed}.", flush=True)


if __name__ == "__main__":
    main()
