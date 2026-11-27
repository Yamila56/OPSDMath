#!/usr/bin/env bash
# Run a fixed two-seed comparison, with the same budget prompt in every arm.
set -euo pipefail

py=$(realpath -s "${1:?environment Python}")
model=$(realpath "${2:?frozen model directory}")
root=$(realpath "${3:?run directory containing data}")
project=$(cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$project/../../..${PYTHONPATH:+:$PYTHONPATH}"
cd "$root"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
anchor=()
if [[ "${OPSD_UNCLIPPED_ANCHOR:-0}" == 1 ]]; then
    anchor=(--unclipped-anchor)
fi

if [[ -n "${WAIT_FOR_CALIBRATION:-}" ]]; then
    while [[ ! -f "$WAIT_FOR_CALIBRATION/PROMPT-DEV-DONE" ]]; do
        pid=$(cat "$WAIT_FOR_CALIBRATION/prompts.pid")
        if ! ps -p "$pid" -o stat= | grep -q '^[^Z]'; then
            echo "Original prompt diagnostics exited without a completion marker; check its log." >&2
            exit 8
        fi
        sleep 30
    done
fi

study="$project/retention_study.py"
trainer="$project/../../../opsdmath/training/trainer.py"
"$py" - data <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data = Path(sys.argv[1])
inputs = json.loads((data / "data-manifest.json").read_text())
for name, digest in inputs["sha256"].items():
    if hashlib.sha256((data / name).read_bytes()).hexdigest() != digest:
        raise ValueError(f"Mismatched data checksum: {name}")
PY

"$py" "$study" evaluate --model "$model" --data data/training-cohort --output training-base --group base
if [[ ! -s routing/cohort-counts.json ]]; then
    "$py" "$study" route --base training-base/base --train data/train.jsonl --output routing
fi
cat routing/cohort-counts.json
mkdir -p training
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > gpu.csv
sha256sum "$trainer" "$study" "$project/../../../experiments/kl_ablation/short_sft_baseline.py" \
    "$project"/../../../opsdmath/{training,objectives,evaluation}/*.py > code-sha256.txt
groups=(base)
for seed in 17 29; do
    for method in equations privileged unprivileged random behavior; do
        name="$method-$seed"
        group="training/$name"
        groups+=("$name=$group/adapter")
        if [[ -s "$group/adapter/adapter_model.safetensors" && -f "$group/DONE" ]]; then
            continue
        fi
        [[ ! -e "$group" ]] || mv -- "$group" "$group.interrupted-$(date -u +%Y%m%dT%H%M%SZ)"
        run=(--model "$model" --seed "$seed" --init-seed "$seed" --data-seed "$seed"
             --cap 128 --output-budget 128 --output "$group")
        if [[ "$method" != equations && "${OPSD_SPLIT_ROLLOUT_RNG:-1}" == 1 ]]; then
            run+=(--rollout-seed "$((10000 + seed))")
        fi
        case "$method" in
            equations)
                "$py" -m opsdmath.training.trainer sft "${run[@]}" --data data/equations/train.jsonl ;;
            privileged|unprivileged|random|behavior)
                context=answer
                routing=()
                case "$method" in
                    unprivileged) context=none ;;
                    random) routing=(--teacher-routing "routing/random-$seed.json") ;;
                    behavior) routing=(--teacher-routing routing/behavior.json) ;;
                esac
                "$py" -m opsdmath.training.trainer opsd "${run[@]}" --data data/train.jsonl \
                    --kl-clip 0.05 --teacher-context "$context" --log-diagnostics "${anchor[@]}" "${routing[@]}" ;;
        esac
        test -s "$group/adapter/adapter_model.safetensors"
        "$py" - "$group/events.jsonl" <<'PY'
import json
import sys
from pathlib import Path

events = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
if sum(e["event"] == "update" for e in events) != 64 or events[-1]["event"] != "complete":
    raise RuntimeError("Incomplete training run")
rollout = sum(e["tokens"] for e in events if e["event"] == "rollout_batch")
print(f"Finished {Path(sys.argv[1]).parent.name}. Online rollout tokens: {rollout}", flush=True)
PY
        touch "$group/DONE"
    done
done
touch TRAIN-DONE
"$py" "$study" evaluate --model "$model" --data data/test --output test --group "${groups[@]}"
"$py" "$study" summarize --evaluation test --output results.json
touch GSM8K-DONE
outputs=(routing training-base test training)
if [[ -d transfer-data ]]; then
    "$py" "$study" evaluate --model "$model" --data transfer-data --output transfer --group "${groups[@]}"
    "$py" "$study" summarize --evaluation transfer --output transfer-results.json
    outputs+=(transfer)
fi
find "${outputs[@]}" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.safetensors' \) \
    -print0 | sort -z | xargs -0 sha256sum > artifacts-sha256.txt
touch STUDY-DONE
