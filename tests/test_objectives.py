import torch

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
