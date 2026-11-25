#!/usr/bin/env bash
# Build paired scorecards for the specified completed 8B groups.
set -euo pipefail

run=${1:?run directory}
labels=${2:?condition labels JSONL}
suffix=${3:?report name}
shift 3
py=${OPSD_PYTHON:-python}
declare -A present
inputs=()
contrasts=()
for method in "$@"; do
    present[$method]=1
    for prompt in standard hint; do
        inputs+=("$prompt-$method=$run/evaluation/$method/condition-$prompt.jsonl")
    done
    contrasts+=("hint-$method:standard-$method")
done
for seed in 17 29; do
    for pair in "answer$seed:base" "fullopsd$seed:base" "answer$seed:equationsft$seed" \
                "answer$seed:fullsft$seed" "answer$seed:fullopsd$seed" "fullopsd$seed:fullsft$seed"; do
        left=${pair%:*}
        right=${pair#*:}
        if [[ -v present[$left] && -v present[$right] ]]; then
            for prompt in standard hint; do
                contrasts+=("$prompt-$left:$prompt-$right")
            done
        fi
    done
done
"$py" condition_pairs.py score "$labels" "$run/$suffix-conditions.json" "${inputs[@]}" --compare "${contrasts[@]}"
"$py" eval_scale_transfer.py summarize --run "$run" --condition "$run/$suffix-conditions.json" \
    --output "$run/$suffix.json" --groups "$@"
