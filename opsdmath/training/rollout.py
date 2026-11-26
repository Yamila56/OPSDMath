"""Sampling and completion-position logits."""

import torch

from opsdmath.objectives.kl import ROLLOUT_TEMPERATURE


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
