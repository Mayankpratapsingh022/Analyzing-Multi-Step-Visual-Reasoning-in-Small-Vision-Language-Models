#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export WANDB_PROJECT="${WANDB_PROJECT:-small-vlm-reasoning-vcr}"

cd "$PROJECT_ROOT"
echo "[run-vcr-reference] config=cloud_gpu_vcr/configs/vcr_reference.yml"
python run_experiment.py --config cloud_gpu_vcr/configs/vcr_reference.yml
