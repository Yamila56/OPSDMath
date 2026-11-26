"""Checkpoint identity and atomic publication of optimizer and RNG state."""

import hashlib
import json

import torch


def training_identity(args, rows):
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
    return identity


def save_state(output, adapter, identity, next_index, step, optimizer, scheduler, device):
    temporary = output / "train-state.tmp"
    torch.save(dict(adapter=adapter, identity=identity, next_index=next_index, update=step,
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None), temporary)
    temporary.replace(output / "train-state.pt")
