#!/usr/bin/env bash
set -euo pipefail

output="${1:?Pass a new output directory}"
python=.venv/bin/python
model="$(realpath model)"
mkdir -p "$output"

while [[ ! -f env.ready || ! -s "$model/model-00001-of-00002.safetensors" || ! -s "$model/model-00002-of-00002.safetensors" ]]; do
    sleep 5
done

sha256sum gsm8k.py run_math.py upstream/data_collator.py data/{train,dev,test_inputs,test_labels}.jsonl > "$output/input_sha256.txt"
cp data/manifest.json "$output/data_manifest.json"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$output/gpu.csv"
"$python" -m pip freeze > "$output/environment.txt"
"$python" run_math.py check-loss

"$python" run_math.py opsd --model "$model" --data data/train.jsonl \
    --output "$output/probe" --probe --limit 4
"$python" run_math.py predict --model "$model" --data data/dev.jsonl \
    --output "$output/probe-reload" --adapter "$output/probe/adapter" \
    --prediction-name opsd --batch-size 4 --limit 4

for group in opsd sft; do
    "$python" run_math.py "$group" --model "$model" --data data/train.jsonl \
        --output "$output/train-$group"
done

for group in base sft opsd; do
    adapter=()
    if [[ "$group" != base ]]; then
        adapter=(--adapter "$output/train-$group/adapter")
    fi
    "$python" run_math.py predict --model "$model" --data data/test_inputs.jsonl \
        --output "$output/eval-$group" --prediction-name "$group" --batch-size 16 "${adapter[@]}"
done

"$python" gsm8k.py score data/test_labels.jsonl \
    "$output/eval-base/base.jsonl" "$output/eval-sft/sft.jsonl" "$output/eval-opsd/opsd.jsonl" \
    > "$output/results.json"
touch "$output/complete"
