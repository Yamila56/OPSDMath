"""Prepare the pinned GSM8K split for the reference recipes."""

import argparse
from pathlib import Path

from opsdmath.data.gsm8k import prepare


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
