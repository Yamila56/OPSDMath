#!/usr/bin/env bash
# One optimizer update and adapter reload on a local Qwen3-1.7B checkpoint.
set -euo pipefail

usage() {
    printf 'Usage: %s MODEL_DIRECTORY DATA_DIRECTORY OUTPUT_DIRECTORY\n' "$0"
    printf 'Run from the checkout root. Set PYTHON to the CUDA-enabled OPSDMath Python.\n'
}
if [[ "${1:-}" == --help ]]; then
    usage
    exit 0
fi
if (( $# != 3 )); then
    usage >&2
    exit 2
fi

python="${PYTHON:-python}"
model="$1"
data="$2"
output="$3"
if [[ ! -f "$model/config.json" || ! -f "$data/train.jsonl" || ! -f "$data/dev.jsonl" ]]; then
    printf 'Provide local Qwen3-1.7B weights and the prepared GSM8K data directory.\n' >&2
    exit 2
fi
if [[ ! -f upstream/data_collator.py ]]; then
    printf 'Check out the pinned upstream OPSD revision in ./upstream for the prompt parity probe.\n' >&2
    exit 2
fi
"$python" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "The GPU smoke test requires CUDA-enabled PyTorch")'
"$python" -m opsdmath.training.trainer check-loss
"$python" -m opsdmath.training.trainer opsd --model "$model" --data "$data/train.jsonl" \
    --output "$output/opsd-update" --probe --limit 4 --cap 64
"$python" -m opsdmath.training.trainer predict --model "$model" --data "$data/dev.jsonl" \
    --adapter "$output/opsd-update/adapter" --prediction-name opsd \
    --output "$output/opsd-reload" --limit 4
