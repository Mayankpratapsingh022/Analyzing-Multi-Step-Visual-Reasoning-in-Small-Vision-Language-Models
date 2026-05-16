#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export VCR_DIR="${VCR_DIR:-/workspace/data/vcr}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$VCR_DIR"

echo "[cloud-bootstrap] project root: $PROJECT_ROOT"
echo "[cloud-bootstrap] HF_HOME: $HF_HOME"
echo "[cloud-bootstrap] VCR_DIR: $VCR_DIR"

cd "$PROJECT_ROOT"
mkdir -p logs results

python - <<'PY'
import sys
try:
    import torch
except ImportError:
    print("torch is not installed yet")
    raise SystemExit(0)

print(f"python: {sys.version.split()[0]}")
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"vram_gb: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}")
PY

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  python -m pip install flash-attn --no-build-isolation
else
  echo "[cloud-bootstrap] skipping flash-attn. Set INSTALL_FLASH_ATTN=1 to install it."
fi

python cloud_gpu_vcr/scripts/check_cloud_state.py

echo "[cloud-bootstrap] complete"
