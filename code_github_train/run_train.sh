#!/usr/bin/env bash
set -euo pipefail

# Use the currently active environment. No server-specific Conda installation is required.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CUDA_ID:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
fi
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/train.py" "$@"
