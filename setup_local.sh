#!/usr/bin/env bash
set -euo pipefail
root="${OPSD_HOME:-$HOME/.local/share/opsd-math}"
uv_bin="${UV_BIN:-$HOME/.local/share/physvlm-reachability/tools/uv-x86_64-unknown-linux-gnu/uv}"

if [[ "${1:-}" == weights ]]; then
    dest="$root/Qwen3-1.7B"
    mkdir -p "$dest"
    url=https://huggingface.co/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
    for item in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors.index.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
        if [[ ! -s "$dest/$item" ]]; then
            curl -fsSL --retry 2 --connect-timeout 15 "$url/$item" -o "$dest/$item.part"
            mv "$dest/$item.part" "$dest/$item"
        fi
    done
    du -sh "$dest"
else
    if [[ ! -x "$root/venv/bin/python" ]]; then
        "$uv_bin" venv --python 3.11 "$root/venv"
    fi
    "$uv_bin" pip install --python "$root/venv/bin/python" torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
    "$uv_bin" pip install --python "$root/venv/bin/python" transformers==4.57.1 peft==0.17.1 accelerate==1.11.0 safetensors==0.5.3
fi
