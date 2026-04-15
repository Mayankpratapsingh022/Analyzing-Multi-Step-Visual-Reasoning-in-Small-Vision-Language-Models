#!/usr/bin/env bash
# =============================================================================
# setup_runpod.sh — Bootstrap this project on a fresh RunPod GPU pod.
#
# Recommended RunPod template:
#   Image : runpod/pytorch:2.3.1-py3.10-cuda12.1.1-devel-ubuntu22.04
#   Volume: attach a Network Volume at /workspace (persists across restarts)
#
# Usage (run once after pod starts):
#   bash scripts/setup_runpod.sh [--skip-flash-attn] [--smoke-test]
#
# Options:
#   --skip-flash-attn   Skip flash-attention compilation (~10-15 min).
#                       Use this for quick iteration; re-run without flag
#                       before serious benchmarking.
#   --smoke-test        Run the model smoke test after setup completes.
# =============================================================================

set -euo pipefail

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_FLASH_ATTN=0
RUN_SMOKE_TEST=0
for arg in "$@"; do
    case "$arg" in
        --skip-flash-attn) SKIP_FLASH_ATTN=1 ;;
        --smoke-test)      RUN_SMOKE_TEST=1 ;;
        *) echo "Unknown argument: $arg" && exit 1 ;;
    esac
done

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn] ${NC} $*"; }
section() { echo -e "\n${GREEN}══════════════════════════════════════════${NC}"; \
            echo -e "${GREEN} $*${NC}"; \
            echo -e "${GREEN}══════════════════════════════════════════${NC}"; }

# ── 1. Verify CUDA ────────────────────────────────────────────────────────────
section "1/6  System check"
python - <<'EOF'
import torch, sys
print(f"  Python      : {sys.version.split()[0]}")
print(f"  PyTorch     : {torch.__version__}")
print(f"  CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU         : {torch.cuda.get_device_name()}")
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"  VRAM        : {vram:.1f} GB")
    print(f"  CUDA version: {torch.version.cuda}")
else:
    print("  WARNING: No GPU detected — inference will be very slow.")
EOF

# ── 2. Point HuggingFace cache at /workspace (survives pod restarts) ──────────
section "2/6  HuggingFace cache → /workspace"
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
info "HF_HOME=$HF_HOME"

# Persist cache paths across future shell sessions
PROFILE_LINE='export HF_HOME=/workspace/.cache/huggingface'
grep -qxF "$PROFILE_LINE" ~/.bashrc 2>/dev/null || {
    {
        echo ''
        echo '# HuggingFace cache on persistent volume'
        echo 'export HF_HOME=/workspace/.cache/huggingface'
        echo 'export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers'
        echo 'export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets'
    } >> ~/.bashrc
    info "Cache paths added to ~/.bashrc"
}

# ── 3. Load env vars from .env (API keys, HF token) ──────────────────────────
section "3/6  Environment variables"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ENV_FILE="$PROJECT_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
    info "Loading $ENV_FILE"
    set -o allexport
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +o allexport
else
    warn ".env not found — copy .env.example to .env and fill in your tokens."
    warn "  cp $PROJECT_ROOT/.env.example $PROJECT_ROOT/.env"
fi

# HuggingFace login (required for some gated models, e.g. LLaVA, Phi-3)
if [[ -n "${HF_TOKEN:-}" ]]; then
    info "Logging in to HuggingFace Hub..."
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"
    info "HuggingFace login successful."
else
    warn "HF_TOKEN not set. Gated models (LLaVA-NeXT, Phi-3) may fail to download."
    warn "Set HF_TOKEN in your .env file."
fi

# ── 4. Install Python dependencies ───────────────────────────────────────────
section "4/6  Installing Python dependencies"
cd "$PROJECT_ROOT"

if [[ "$SKIP_FLASH_ATTN" -eq 1 ]]; then
    warn "--skip-flash-attn: installing everything except flash-attn"
    grep -v '^flash-attn' requirements.txt > /tmp/requirements_no_flash.txt
    pip install --upgrade -r /tmp/requirements_no_flash.txt
else
    info "Installing all deps including flash-attn (this will take ~10-15 min the first time)"
    pip install --upgrade -r requirements.txt
fi

# Install the project itself in editable mode so `src.*` imports resolve
pip install -e . --no-deps --quiet 2>/dev/null || true

info "Dependencies installed."

# ── 5. Verify imports ─────────────────────────────────────────────────────────
section "5/6  Verifying imports"
python - <<'EOF'
checks = [
    ("torch",               "torch"),
    ("transformers",        "transformers"),
    ("datasets",            "datasets"),
    ("accelerate",          "accelerate"),
    ("bitsandbytes",        "bitsandbytes"),
    ("PIL",                 "Pillow"),
    ("src.data_loader",     "project src"),
    ("src.models",          "project models"),
]
all_ok = True
for mod, label in checks:
    try:
        __import__(mod)
        print(f"  {'OK':<6} {label}")
    except ImportError as e:
        print(f"  {'FAIL':<6} {label}  ({e})")
        all_ok = False
if not all_ok:
    raise SystemExit("One or more imports failed. Re-run setup or check errors above.")
EOF

# ── 6. Optional smoke test ────────────────────────────────────────────────────
if [[ "$RUN_SMOKE_TEST" -eq 1 ]]; then
    section "6/6  Smoke test (all open-weight small models)"
    cd "$PROJECT_ROOT"
    python scripts/smoke_test.py --skip-api
else
    section "6/6  Setup complete"
    info "Skip smoke test. Run it manually when ready:"
    info "  python scripts/smoke_test.py --skip-api"
fi

echo ""
info "Next steps:"
info "  # Quick baseline (100 examples each)"
info "  python run_experiment.py --config configs/baseline_mmmu.yml"
info "  python run_experiment.py --config configs/baseline_mathvista.yml"
info ""
info "  # Single model"
info "  python -m src.baselines.evaluate --model qwen2-vl-2b --dataset mmmu --subject Math --max_samples 50"
