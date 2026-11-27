#!/usr/bin/env bash
set -euo pipefail
root="${OPSD_HOME:-$HOME/.local/share/opsd-math}"
python="$root/venv/bin/python"
model="$root/Qwen3-1.7B"
output="${1:?Pass a new output directory}"

if [[ "${2:-}" == --wait ]]; then
    deadline=$((SECONDS + 5400))
    until [[ -s "$model/model-00001-of-00002.safetensors" && -s "$model/model-00002-of-00002.safetensors" ]] &&
        "$python" -c 'import torch, transformers, peft' >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
            echo "Model or environment did not finish preparing within 90 minutes" >&2
            exit 1
        fi
        sleep 15
    done
fi

"$python" -m opsdmath.training.trainer check-loss
"$python" -m opsdmath.training.trainer opsd --model "$model" --data data/train.jsonl \
    --output "$output/opsd-update" --probe --limit 4 --cap 64
"$python" -m opsdmath.training.trainer predict --model "$model" --data data/dev.jsonl \
    --adapter "$output/opsd-update/adapter" --prediction-name opsd \
    --output "$output/opsd-reload" --limit 4
