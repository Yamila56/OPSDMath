#!/usr/bin/env bash
# Train matching SFT and OPSD controls on the original 256-question split.
set -euo pipefail

model=${1:?model directory}
training=${2:?GSM8K training JSONL}
output=${3:?output directory}
shift 3
py=${OPSD_PYTHON:-python}
mkdir -p "$output"
if [[ ! -f "$output/equations/train.jsonl" ]]; then
    "$py" -m experiments.sft_baselines.short_sft_baseline prepare "$training" "$output/equations" --target equations
fi

for group in "$@"; do
    if [[ ! "$group" =~ ^(answer|fullopsd|fullsft|equationsft)(17|29)$ ]]; then
        echo "Unsupported scale-transfer group: $group" >&2
        exit 2
    fi
    method=${BASH_REMATCH[1]}
    seed=${BASH_REMATCH[2]}
    dir="$output/$group"
    if [[ -f "$dir/DONE" && -s "$dir/adapter/adapter_model.safetensors" ]]; then
        continue
    fi
    if [[ -e "$dir" ]]; then
        echo "Incomplete or missing adapter in $dir. Recover it or move this directory aside before retraining." >&2
        exit 2
    fi
    run=(--model "$model" --seed "$seed" --cap 256 --accumulation 4 --output "$dir")
    case "$method" in
        answer) "$py" run_math.py opsd "${run[@]}" --data "$training" --teacher-context answer ;;
        fullopsd) "$py" run_math.py opsd "${run[@]}" --data "$training" --teacher-context full ;;
        fullsft) "$py" run_math.py sft "${run[@]}" --data "$training" ;;
        equationsft) "$py" run_math.py sft "${run[@]}" --data "$output/equations/train.jsonl" ;;
    esac
    test -s "$dir/adapter/adapter_model.safetensors"
    "$py" - "$dir/events.jsonl" <<'PY'
import json
import sys
from pathlib import Path

events = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
if sum(row["event"] == "update" for row in events) != 64 or events[-1]["event"] != "complete":
    raise SystemExit("Expected 64 completed updates")
PY
    touch "$dir/DONE"
done
