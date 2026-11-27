"""Objectives applied to completion-only logits by the common training loop."""

import torch.nn.functional as F

from opsdmath.objectives.kl import distillation_loss


def opsd_loss(logits, teacher, args, context, diagnostic):
    clip = None if args.unclipped_anchor and context == "none" else args.kl_clip
    return distillation_loss(logits, teacher, kl_clip=clip,
                             diagnostics=diagnostic if args.probe or args.log_diagnostics else None)


def sft_loss(logits, suffix):
    return F.cross_entropy(logits.flatten(0, 1).float(), suffix.flatten())
