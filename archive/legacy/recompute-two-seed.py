"""Recompute the report from already extracted evidence using the former positional API."""

import argparse
import json
from pathlib import Path

from experiments.kl_ablation.replay import recompute


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report, _ = recompute(args.directory)
    (args.directory / "paired-two-seed.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
