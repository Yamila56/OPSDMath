#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
opsdmath-reproduce --repository "$repo" "$@"
