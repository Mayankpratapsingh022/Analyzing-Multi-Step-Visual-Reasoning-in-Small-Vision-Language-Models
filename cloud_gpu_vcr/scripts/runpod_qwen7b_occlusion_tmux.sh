#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Mayankpratapsingh022/Analyzing-Multi-Step-Visual-Reasoning-in-Small-Vision-Language-Models.git}"
BRANCH="${BRANCH:-feature/mmmu-mathvista-baselines}"
PROJECT_DIR="${PROJECT_DIR:-/workspace/Analyzing-Multi-Step-Visual-Reasoning-in-Small-Vision-Language-Models}"
SESSION_NAME="${SESSION_NAME:-qwen7b_occ}"

HF_DATASET_ID="${HF_DATASET_ID:-JaydeepR/vcr-mirror}"
HF_DETAILS_PATH="${HF_DETAILS_PATH:-wave2_latest/results/qwen2-vl-7b_vcr_20260517_151410_details.json}"
HF_ID="${HF_ID:-Qwen/Qwen2-VL-7B-Instruct}"
MODEL="${MODEL:-qwen2-vl-7b}"
NUM_EXAMPLES="${NUM_EXAMPLES:-500}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bf16}"
OCCLUSION_GRID="${OCCLUSION_GRID:-5}"

VCR_DIR="${VCR_DIR:-/workspace/data/vcr}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/workspace/.cache/matplotlib}"
HF_RAW_DIR="${HF_RAW_DIR:-/workspace/data/vcr_raw}"
OUT_DIR="${OUT_DIR:-results/occlusion_overlap_qwen7b_n500_h100_hf7b_provenance}"
LOG_FILE="${LOG_FILE:-logs/occlusion_overlap_qwen7b_n500_h100_hf7b_provenance.log}"
BOOTSTRAP_LOG_FILE="${BOOTSTRAP_LOG_FILE:-logs/runpod_qwen7b_occ_bootstrap.log}"
ENV_FILE="${ENV_FILE:-/workspace/.qwen7b_occ_env}"
RUNNER_FILE="${RUNNER_FILE:-/workspace/run_qwen7b_occ_inner.sh}"

log() {
  printf '[runpod-qwen7b-occ] %s\n' "$*"
}

shell_quote() {
  printf '%q' "$1"
}

install_system_packages() {
  local missing=()
  command -v git >/dev/null 2>&1 || missing+=(git)
  command -v tmux >/dev/null 2>&1 || missing+=(tmux)
  if (( ${#missing[@]} == 0 )); then
    return
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    printf 'Missing required command(s): %s\n' "${missing[*]}" >&2
    printf 'Install them manually, then rerun this script.\n' >&2
    exit 1
  fi
  log "installing system packages: ${missing[*]}"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"
}

prompt_hf_token_if_needed() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    return
  fi
  printf 'Enter Hugging Face token (input hidden; leave blank only if public access works): '
  read -r -s HF_TOKEN || true
  printf '\n'
  export HF_TOKEN
}

sync_repo() {
  mkdir -p "$(dirname "$PROJECT_DIR")"
  if [[ ! -d "$PROJECT_DIR/.git" ]]; then
    log "cloning repo into $PROJECT_DIR"
    git clone "$REPO_URL" "$PROJECT_DIR"
  fi

  cd "$PROJECT_DIR"
  log "checking out $BRANCH"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull origin "$BRANCH"
}

write_env_file() {
  mkdir -p "$VCR_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR" "$HF_RAW_DIR"
  cat > "$ENV_FILE" <<EOF
export HF_TOKEN=$(shell_quote "${HF_TOKEN:-}")
export VCR_DIR=$(shell_quote "$VCR_DIR")
export HF_HOME=$(shell_quote "$HF_HOME")
export HF_DATASETS_CACHE=$(shell_quote "$HF_DATASETS_CACHE")
export TRANSFORMERS_CACHE=$(shell_quote "$TRANSFORMERS_CACHE")
export MPLCONFIGDIR=$(shell_quote "$MPLCONFIGDIR")
export HF_HUB_ENABLE_HF_TRANSFER=0
export USE_WANDB=0
export HF_DATASET_ID=$(shell_quote "$HF_DATASET_ID")
export HF_DETAILS_PATH=$(shell_quote "$HF_DETAILS_PATH")
export HF_ID=$(shell_quote "$HF_ID")
export MODEL=$(shell_quote "$MODEL")
export NUM_EXAMPLES=$(shell_quote "$NUM_EXAMPLES")
export SEED=$(shell_quote "$SEED")
export DTYPE=$(shell_quote "$DTYPE")
export OCCLUSION_GRID=$(shell_quote "$OCCLUSION_GRID")
export OUT_DIR=$(shell_quote "$OUT_DIR")
export LOG_FILE=$(shell_quote "$LOG_FILE")
export BOOTSTRAP_LOG_FILE=$(shell_quote "$BOOTSTRAP_LOG_FILE")
export HF_RAW_DIR=$(shell_quote "$HF_RAW_DIR")
EOF
  chmod 600 "$ENV_FILE"
  log "wrote env file: $ENV_FILE"
}

write_runner_file() {
  cat > "$RUNNER_FILE" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

source /workspace/.qwen7b_occ_env

cd /workspace/Analyzing-Multi-Step-Visual-Reasoning-in-Small-Vision-Language-Models
mkdir -p logs results "$VCR_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR" "$HF_RAW_DIR"

echo "[qwen7b-occ] repo: $(pwd)"
echo "[qwen7b-occ] gpu:"
nvidia-smi || true

echo "[qwen7b-occ] bootstrap start"
bash cloud_gpu_vcr/scripts/bootstrap_cloud.sh 2>&1 | tee "$BOOTSTRAP_LOG_FILE"

echo "[qwen7b-occ] cloud state"
python cloud_gpu_vcr/scripts/check_cloud_state.py || true

echo "[qwen7b-occ] occlusion run start"
python cloud_gpu_vcr/scripts/quantify_occlusion_overlap.py \
  --prepare-from-hf \
  --hf-dataset-id "$HF_DATASET_ID" \
  --hf-raw-dir "$HF_RAW_DIR" \
  --hf-details-path "$HF_DETAILS_PATH" \
  --num-examples "$NUM_EXAMPLES" \
  --model "$MODEL" \
  --hf-id "$HF_ID" \
  --out-dir "$OUT_DIR" \
  --dtype "$DTYPE" \
  --occlusion-grid "$OCCLUSION_GRID" \
  --seed "$SEED" \
  2>&1 | tee "$LOG_FILE"

echo "[qwen7b-occ] done"
echo "[qwen7b-occ] summary: $OUT_DIR/summary.json"
EOF
  chmod +x "$RUNNER_FILE"
  log "wrote runner file: $RUNNER_FILE"
}

launch_tmux() {
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    printf 'tmux session already exists: %s\n' "$SESSION_NAME" >&2
    printf 'Attach with: tmux attach -t %s\n' "$SESSION_NAME" >&2
    exit 1
  fi
  log "starting detached tmux session: $SESSION_NAME"
  tmux new-session -d -s "$SESSION_NAME" "bash '$RUNNER_FILE'"
}

main() {
  install_system_packages
  prompt_hf_token_if_needed
  sync_repo
  write_env_file
  write_runner_file
  launch_tmux

  cat <<EOF

Started Qwen2-VL-7B occlusion run in tmux session: $SESSION_NAME

Attach:
  tmux attach -t $SESSION_NAME

Detach after attaching:
  Ctrl-b then d

Watch log:
  cd $PROJECT_DIR
  tail -f $LOG_FILE

Final summary:
  cat $PROJECT_DIR/$OUT_DIR/summary.json

EOF
}

main "$@"
