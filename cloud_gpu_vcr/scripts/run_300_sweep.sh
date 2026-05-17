#!/usr/bin/env bash
# Run the 300-example VCR sweep across the remaining models (qwen2-vl-2b has
# typically already been run as a smoke test/standalone before this).
#
# Each model run frees VRAM before the next one starts. Failures don't abort
# the sweep — the next model still launches.
#
# Usage:
#   bash cloud_gpu_vcr/scripts/run_300_sweep.sh
#
# Override the CSV path or model list via env vars if needed:
#   CSV_PATH=results/foo.csv MODELS="qwen2-vl-7b llava-next-7b" bash ...

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CSV_PATH="${CSV_PATH:-results/vcr_300_comparable.csv}"
SAMPLES="${SAMPLES:-300}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR" results

MODELS_DEFAULT="qwen2-vl-7b llava-next-7b llava-1.5-13b gpt-4o"
MODELS="${MODELS:-$MODELS_DEFAULT}"

ts="$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG="$LOG_DIR/sweep_${ts}.log"
echo "[sweep] CSV: $CSV_PATH" | tee -a "$SWEEP_LOG"
echo "[sweep] models: $MODELS" | tee -a "$SWEEP_LOG"
echo "[sweep] samples: $SAMPLES" | tee -a "$SWEEP_LOG"
echo "[sweep] log: $SWEEP_LOG" | tee -a "$SWEEP_LOG"

run_with_script() {
  local model="$1"
  echo "" | tee -a "$SWEEP_LOG"
  echo "===== $(date) :: $model =====" | tee -a "$SWEEP_LOG"
  CSV_PATH="$CSV_PATH" bash cloud_gpu_vcr/scripts/run_single_model.sh "$model" "$SAMPLES" \
    2>&1 | tee -a "$SWEEP_LOG"
  local rc="${PIPESTATUS[0]}"
  echo "[sweep] $model exit code: $rc" | tee -a "$SWEEP_LOG"
}

run_llava_13b_8bit() {
  echo "" | tee -a "$SWEEP_LOG"
  echo "===== $(date) :: llava-1.5-13b (8bit) =====" | tee -a "$SWEEP_LOG"
  python -m src.baselines.evaluate \
    --model llava-1.5-13b \
    --dataset vcr \
    --vcr_dir /workspace/data/vcr \
    --max_samples "$SAMPLES" \
    --seed 42 \
    --device cuda \
    --batch_size 1 \
    --max_new_tokens 8 \
    --prompt_strategy zero_shot_direct \
    --quantization 8bit \
    --csv_path "$CSV_PATH" \
    --wandb \
    --wandb_project small-vlm-reasoning-vcr \
    --wandb_group vcr-300-comparable \
    2>&1 | tee -a "$SWEEP_LOG"
  local rc="${PIPESTATUS[0]}"
  echo "[sweep] llava-1.5-13b exit code: $rc" | tee -a "$SWEEP_LOG"
}

for m in $MODELS; do
  if [[ "$m" == "llava-1.5-13b" ]]; then
    run_llava_13b_8bit
  else
    run_with_script "$m"
  fi
done

echo "" | tee -a "$SWEEP_LOG"
echo "[sweep] DONE at $(date)" | tee -a "$SWEEP_LOG"
echo "[sweep] CSV: $CSV_PATH" | tee -a "$SWEEP_LOG"
echo "[sweep] log: $SWEEP_LOG" | tee -a "$SWEEP_LOG"
