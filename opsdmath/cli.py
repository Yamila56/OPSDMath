"""Run the archived training and evaluation recipes with local input paths."""

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path


def verified_file(path, expected):
    """Reject changed inputs before a reference run starts."""
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
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

    from opsdmath.evaluation.combined import evaluate as run

    run(args)


def reproduce():
    parser = argparse.ArgumentParser(description="Recompute and verify the published KL ablation.")
    parser.add_argument("--experiment", choices=["kl-ablation"], default="kl-ablation")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from opsdmath.analysis.kl_replay import recompute

    pilot = args.repository / "results/scale-transfer/privileged-gradient-audit/kl-pilot"
    checksums = {}
    for line in (pilot / "archive-sha256.txt").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split()
        checksums[filename] = digest
    with tempfile.TemporaryDirectory(prefix="opsdmath-") as directory:
        root = Path(directory)
        for filename, target in (("seed29", root), ("seed17", root / "seed17"), ("controls", root / "seed17")):
            path = pilot / f"{filename}-evidence.tar.gz"
            verified_file(path, checksums[path.name])
            with tarfile.open(path) as archive:
                archive.extractall(target, filter="data")
        recompute(root)
        report = root / "paired-two-seed.json"
        data = load_config(report)
        if data != load_config(pilot / report.name):
            raise ValueError("Replayed statistics differ from the published statistics")
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(report.read_text(encoding="utf-8"))
    print("Matched the archived two-seed KL ablation and SVAMP statistics.")
    print("128-token counts:", data["128/numeric/all"]["counts"])
    print("512-token counts:", data["512/numeric/all"]["counts"])
