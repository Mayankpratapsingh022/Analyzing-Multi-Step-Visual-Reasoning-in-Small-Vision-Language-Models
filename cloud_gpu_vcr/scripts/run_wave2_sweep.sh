#!/usr/bin/env bash
# Wave-2 VCR sweep: full val split for the 4 GPU models.
# gpt-4o is intentionally skipped to control API cost (already covered at n=300).
#
# Usage:
#   bash cloud_gpu_vcr/scripts/run_wave2_sweep.sh
#
# Overrides via env vars:
#   CSV_PATH=results/foo.csv MODELS="qwen2-vl-2b qwen2-vl-7b" bash ...

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CSV_PATH="${CSV_PATH:-results/vcr_full_val.csv}"
export WANDB_PROJECT="${WANDB_PROJECT:-small-vlm-reasoning-vcr}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR" results

MODELS_DEFAULT="qwen2-vl-2b qwen2-vl-7b llava-next-7b llava-1.5-13b"
MODELS="${MODELS:-$MODELS_DEFAULT}"

ts="$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG="$LOG_DIR/wave2_${ts}.log"

echo "[wave2] CSV: $CSV_PATH" | tee -a "$SWEEP_LOG"
echo "[wave2] models: $MODELS" | tee -a "$SWEEP_LOG"
echo "[wave2] samples: full val (~26534)" | tee -a "$SWEEP_LOG"
echo "[wave2] log: $SWEEP_LOG" | tee -a "$SWEEP_LOG"

run_with_script() {
  local model="$1"
  echo "" | tee -a "$SWEEP_LOG"
  echo "===== $(date) :: $model =====" | tee -a "$SWEEP_LOG"
  CSV_PATH="$CSV_PATH" \
    bash cloud_gpu_vcr/scripts/run_single_model.sh "$model" full \
    2>&1 | tee -a "$SWEEP_LOG"
  local rc="${PIPESTATUS[0]}"
  echo "[wave2] $model exit code: $rc" | tee -a "$SWEEP_LOG"
}

run_llava_13b_8bit() {
  echo "" | tee -a "$SWEEP_LOG"
  echo "===== $(date) :: llava-1.5-13b (8bit, full val) =====" | tee -a "$SWEEP_LOG"
  python -m src.baselines.evaluate \
    --model llava-1.5-13b \
    --dataset vcr \
    --vcr_dir /workspace/data/vcr \
    --seed 42 \
    --device cuda \
    --batch_size 1 \
    --max_new_tokens 8 \
    --prompt_strategy zero_shot_direct \
    --quantization 8bit \
    --csv_path "$CSV_PATH" \
    --wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_group vcr-full-val \
    2>&1 | tee -a "$SWEEP_LOG"
  local rc="${PIPESTATUS[0]}"
  echo "[wave2] llava-1.5-13b exit code: $rc" | tee -a "$SWEEP_LOG"
}

for m in $MODELS; do
  if [[ "$m" == "llava-1.5-13b" ]]; then
    run_llava_13b_8bit
  else
    run_with_script "$m"
  fi
done

echo "" | tee -a "$SWEEP_LOG"
echo "[wave2] DONE at $(date)" | tee -a "$SWEEP_LOG"
echo "[wave2] CSV: $CSV_PATH" | tee -a "$SWEEP_LOG"
echo "[wave2] log: $SWEEP_LOG" | tee -a "$SWEEP_LOG"
