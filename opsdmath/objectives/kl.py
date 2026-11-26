"""Exact project forward-KL objectives and CPU checks."""

import torch
import torch.nn.functional as F


ROLLOUT_TEMPERATURE = 1.1


DEFAULT_KL_CLIP = 0.05


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
