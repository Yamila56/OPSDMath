import pytest
import torch
import torch.nn.functional as F

from opsdmath.analysis.gradients import gradients
from opsdmath.objectives.kl import DEFAULT_KL_CLIP, ROLLOUT_TEMPERATURE, check_loss, distillation_loss


def test_forward_clipped_and_exact_zero():
    check_loss()


def test_analytic_clipped_and_forward_gradients():
    student = torch.tensor([[0., 1., 2., 3.]], requires_grad=True)
    teacher = torch.tensor([[3., 1., 1., 0.]])
    _, analytic, _ = gradients(student, teacher, ROLLOUT_TEMPERATURE, DEFAULT_KL_CLIP)
    for name, cap in (("clipped", DEFAULT_KL_CLIP), ("forward", None)):
        actual, = torch.autograd.grad(distillation_loss(student, teacher, kl_clip=cap), student)
        torch.testing.assert_close(actual, analytic[name])


@pytest.mark.parametrize("clip", [None, 1e-7, 0.001, 0.05, 2.0])
def test_random_batched_kl_and_gradient(clip):
    rng = torch.Generator().manual_seed(17)
    student = torch.randn(2, 3, 11, generator=rng, requires_grad=True)
    teacher = torch.randn(2, 3, 11, generator=rng)
    p_log = (student.detach() / ROLLOUT_TEMPERATURE).log_softmax(-1)
    q_log = (teacher / ROLLOUT_TEMPERATURE).log_softmax(-1)
    terms = q_log.exp() * (q_log - p_log)
    expected = (terms if clip is None else terms.clamp(max=clip)).sum(-1).mean()
    actual = distillation_loss(student, teacher, kl_clip=clip)
    torch.testing.assert_close(actual, expected)
    _, analytic, _ = gradients(student, teacher, ROLLOUT_TEMPERATURE, clip if clip is not None else float("inf"))
    actual_gradient, = torch.autograd.grad(actual, student)
    torch.testing.assert_close(actual_gradient, analytic["clipped"] / 6)


def test_gradient_at_exact_clip_and_degenerate_vocabulary():
    student = torch.tensor([[0., 1., 2.]], requires_grad=True)
    teacher = torch.tensor([[2., 1., 0.]])
    p_log = (student.detach() / ROLLOUT_TEMPERATURE).log_softmax(-1)
    q_log = (teacher / ROLLOUT_TEMPERATURE).log_softmax(-1)
    threshold = F.kl_div(p_log, q_log, log_target=True, reduction="none").max().item()
    _, analytic, mass = gradients(student, teacher, ROLLOUT_TEMPERATURE, threshold)
    actual, = torch.autograd.grad(distillation_loss(student, teacher, kl_clip=threshold), student)
    assert mass.item() == 0
    torch.testing.assert_close(actual, analytic["clipped"])

    only_token = torch.tensor([[1.], [-1.]], requires_grad=True)
    zero = distillation_loss(only_token, torch.zeros_like(only_token))
    gradient, = torch.autograd.grad(zero, only_token)
    assert zero.item() == 0
    assert torch.count_nonzero(gradient).item() == 0
