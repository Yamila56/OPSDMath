"""Command-line options and validation for training and prediction."""

import argparse
import json
import math
from pathlib import Path

from opsdmath.io import read_rows
from opsdmath.objectives.kl import DEFAULT_KL_CLIP
from opsdmath.training.prompts import BRIEF_CONTEXTS


def trainer_parser():
    parser = argparse.ArgumentParser(description="Single-device Qwen3 SFT and on-policy self-distillation.")
    parser.add_argument("mode", choices=["opsd", "sft", "predict", "check-loss"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=17)
    for seed in ("init", "data", "rollout"):
        parser.add_argument(f"--{seed}-seed", type=int)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--kl-clip", type=lambda value: None if value == "none" else float(value), default=DEFAULT_KL_CLIP)
    parser.add_argument("--student-thinking", action="store_true")
    parser.add_argument("--teacher-thinking", action="store_true")
    parser.add_argument("--save-steps", nargs="+", type=int, default=[])
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--log-diagnostics", action="store_true")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--rollout-batch-size", type=int,
                        help="Maximum simultaneous generations for online OPSD within one optimizer step.")
    parser.add_argument("--batch-size", type=int, default=1, help="Left-padded prediction batch size.")
    parser.add_argument("--cap", type=int, default=256)
    parser.add_argument("--output-budget", type=int, help="Advertise a common output budget in both training prompts.")
    parser.add_argument("--sft-reference-cap", type=int, default=320)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--teacher-context", choices=["full", "equations", "answer", "none", *BRIEF_CONTEXTS], default="full")
    parser.add_argument("--teacher-routing", type=Path, help="An exact training-ID map to answer or none.")
    parser.add_argument("--unclipped-anchor", action="store_true",
                        help="Use the true KL for unprivileged teacher rows and retain clipping for privileged rows.")
    parser.add_argument("--fixed-rollout", action="store_true")
    parser.add_argument("--save-token-ids", action="store_true")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--prediction-name", default="base", choices=["base", "sft", "opsd"])
    return parser


def training_rows(parser, args):
    if not (args.model and args.data and args.output) or min(
        args.cap, args.sft_reference_cap, args.accumulation, args.rank, args.batch_size,
    ) < 1:
        parser.error("Provide --model, --data and --output and use positive generation and adapter sizes.")
    checker = args.mode != "predict"
    rows = read_rows(args.data)[:args.limit]
    if args.output_budget is not None and (not checker or args.output_budget <= 0):
        parser.error("--output-budget requires a positive training prompt budget.")
    if args.unclipped_anchor and args.mode != "opsd":
        parser.error("--unclipped-anchor is available for OPSD only.")
    if args.rollout_batch_size is not None and (args.rollout_batch_size < 1 or
                                                args.mode != "opsd" or args.fixed_rollout):
        parser.error("--rollout-batch-size requires a positive size and online OPSD.")
    args.teacher_by_id = None
    if args.teacher_routing:
        if args.mode != "opsd" or args.probe or args.teacher_context != "answer":
            parser.error("--teacher-routing requires answer-only OPSD without --probe.")
        args.teacher_by_id = json.loads(args.teacher_routing.read_text(encoding="utf-8"))
        ids = {row["id"] for row in rows}
        if (not isinstance(args.teacher_by_id, dict) or len(ids) != len(rows)
                or args.teacher_by_id.keys() != ids or any(v not in ("answer", "none") for v in args.teacher_by_id.values())):
            parser.error("Every distinct training ID needs an answer or none routing entry.")
    if any(not math.isfinite(x) or x <= 0 for x in (args.learning_rate, args.max_grad_norm)
           + (() if args.kl_clip is None else (args.kl_clip,))):
        parser.error("Learning rate, gradient norm and a numeric KL clip must be finite and positive.")
    if args.resume_from and (args.fixed_rollout or not checker):
        parser.error("Resume is available for online OPSD and SFT training.")
    if set(args.save_steps) - set(range(1, (len(rows)+args.accumulation-1)//args.accumulation+1)):
        parser.error("Checkpoint steps must be within the training run.")
    brief_training = args.mode == "opsd" and args.teacher_context in BRIEF_CONTEXTS
    required = {"id", "problem"}
    if brief_training:
        required.add(args.teacher_context)
    elif args.mode == "sft" or args.teacher_context != "none":
        required.add("solution")
    if args.mode == "opsd" and (args.log_diagnostics or args.teacher_context in ("answer", "equations")):
        required.add("answer")
    if not rows or (checker and any(not required <= row.keys() for row in rows)):
        parser.error("Pass a nonempty dataset with reference solutions for training.")
    return rows


def reference_arguments(config, seed, paths, row_count):
    """Translate a matched recipe and check its reported fixed protocol."""
    from opsdmath.objectives.kl import ROLLOUT_TEMPERATURE
    from opsdmath.training.modeling import lora_alpha
    from opsdmath.training.rollout import ROLLOUT_TOP_K, ROLLOUT_TOP_P

    if config["mode"] not in ("sft", "opsd"):
        raise ValueError("A reference training recipe must select SFT or OPSD")
    command = [config["mode"], "--model", str(paths.model), "--data", str(paths.data),
               "--output", str(paths.output), "--seed", str(paths.seed)]
    options = config["arguments"] | seed
    if {"model", "data", "output", "seed"} & options.keys():
        raise ValueError("Reference recipes take input paths and the seed from the command line")
    for name, value in options.items():
        option = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                command.append(option)
        elif value is not None or name == "kl_clip":
            command.extend((option, str(value) if value is not None else "none"))

    args = trainer_parser().parse_args(command)
    for name, value in options.items():
        if not hasattr(args, name) or getattr(args, name) != value:
            raise ValueError(f"Recipe argument {name} does not match the effective trainer option")
    if args.accumulation < 1 or args.limit is not None:
        raise ValueError("Reference training requires a positive accumulation size and the full split")
    protocol = {
        "updates": (row_count + args.accumulation - 1) // args.accumulation,
        "optimizer": "AdamW",
        "scheduler": "linear decay to zero",
        "lora_alpha": lora_alpha(args.rank),
    }
    if args.mode == "opsd":
        protocol.update(rollout_temperature=ROLLOUT_TEMPERATURE, rollout_top_p=ROLLOUT_TOP_P,
                        rollout_top_k=ROLLOUT_TOP_K)
    for field, actual in protocol.items():
        if config.get(field) != actual:
            raise ValueError(f"Recorded {field} is {config.get(field)!r}; trainer uses {actual!r}")
    return command
