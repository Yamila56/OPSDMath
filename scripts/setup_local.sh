#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s [cpu|cuda] [venv-directory]\n' "$0"
    printf 'Set PYTHON to a Python 3.12+ executable. CUDA mode uses the PyTorch 2.8 CUDA 12.6 wheel.\n'
}

if [[ "${1:-}" == --help ]]; then
    usage
    exit 0
fi
if (( $# > 2 )); then
    usage >&2
    exit 2
fi

repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
mode="${1:-cpu}"
venv="${2:-$repo/.venv}"
case "$mode" in
    cpu) extras=test; wheel_index=cpu ;;
    cuda) extras=training; wheel_index=cu126 ;;
    *) usage >&2; exit 2 ;;
esac

python="${PYTHON:-python3}"
"$python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "OPSDMath requires Python 3.12 or later")'
if [[ ! -f "$venv/pyvenv.cfg" ]]; then
    "$python" -m venv "$venv"
fi
if [[ -x "$venv/Scripts/python.exe" ]]; then
    runtime="$venv/Scripts/python.exe"
else
    runtime="$venv/bin/python"
fi
"$runtime" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "Choose a new virtual environment for Python 3.12+")'
# Pin the build flavor even when an environment already has torch 2.8.0.
"$runtime" -m pip install "torch==2.8.0+$wheel_index" --index-url "https://download.pytorch.org/whl/$wheel_index"
"$runtime" -m pip install -e "$repo[$extras]"
printf 'OPSDMath %s environment ready. Python: %s\n' "$mode" "$runtime"
