#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-qwen2-vl-2b}"
SAMPLES="${2:-100}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export VCR_DIR="${VCR_DIR:-/workspace/data/vcr}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export WANDB_PROJECT="${WANDB_PROJECT:-small-vlm-reasoning-vcr}"
export USE_WANDB="${USE_WANDB:-1}"

WANDB_ARGS=()
if [[ "$USE_WANDB" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_group "vcr-single")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
  fi
fi

SAMPLE_ARGS=()
if [[ "$SAMPLES" != "full" ]]; then
  SAMPLE_ARGS+=(--max_samples "$SAMPLES")
fi

cd "$PROJECT_ROOT"
echo "[run-single] model=$MODEL samples=$SAMPLES vcr_dir=$VCR_DIR"

python -m src.baselines.evaluate \
  --model "$MODEL" \
  --dataset vcr \
  --vcr_dir "$VCR_DIR" \
  "${SAMPLE_ARGS[@]}" \
  --seed "${SEED:-42}" \
  --device cuda \
  --batch_size "${BATCH_SIZE:-1}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-8}" \
  --prompt_strategy "${PROMPT_STRATEGY:-zero_shot_direct}" \
  --csv_path "${CSV_PATH:-results/vcr_single.csv}" \
  "${WANDB_ARGS[@]}"
