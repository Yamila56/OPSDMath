"""Run the archived training and evaluation recipes with local input paths."""

import argparse
import hashlib
import json
from pathlib import Path


def verified_file(path, expected):
    """Reject changed inputs before a reference run starts."""
    from opsdmath.io import digest

    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def load_config(path):
    return json.loads(path.read_text(encoding="utf-8"))


def train():
    parser = argparse.ArgumentParser(description="Run a matched training recipe.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    for name in ("model", "data", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = cfg["seeds"].get(str(args.seed))
    if seed is None:
        parser.error(f"Available training seeds: {', '.join(cfg['seeds'])}")

    from opsdmath.io import read_rows

    rows = read_rows(args.data)
    if len(rows) != cfg["training_examples"]:
        parser.error(f"Expected {cfg['training_examples']} training rows")
    ids_hash = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
    if ids_hash != cfg["ordered_ids_sha256"]:
        parser.error("The order or membership of the training split differs from the reference")
    verified_file(args.model / "model.safetensors.index.json", cfg["model_index_sha256"])
    from opsdmath.training.arguments import reference_arguments
    from opsdmath.training.trainer import main

    try:
        command = reference_arguments(cfg, seed, args, len(rows))
    except ValueError as error:
        parser.error(str(error))
    main(command)


def evaluate():
    parser = argparse.ArgumentParser(description="Run the matched 128/512-token inference protocol.")
    for name in ("config", "model", "data", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--group", nargs="+", required=True, help="Use NAME or NAME=ADAPTER_DIR.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    from opsdmath.evaluation.inference import evaluate as run, validate_recipe

    try:
        validate_recipe(cfg)
    except ValueError as error:
        parser.error(str(error))
    for spec in args.group:
        name = spec.partition("=")[0]
        if not name or Path(name).name != name or name in (".", ".."):
            parser.error(f"Invalid group name: {name!r}")
    if len({spec.partition("=")[0] for spec in args.group}) != len(args.group):
        parser.error("Group names must be unique")
    verified_file(args.model / "model.safetensors.index.json", cfg["model_index_sha256"])
    for name in ("inputs", "labels"):
        verified_file(args.data / f"{name}.jsonl", cfg[f"{name}_sha256"])
    args.batch_size = cfg["batch_size"]

    run(args, expected_questions=cfg["questions"])



def reproduce():
    from experiments.kl_ablation.replay import main

    main()
