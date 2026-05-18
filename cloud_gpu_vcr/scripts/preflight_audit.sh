#!/usr/bin/env bash
# Pre-termination audit: enumerate everything on the pod that could be lost.
# Run before backup_to_hf.sh so you can see if anything needs special handling.
#
# Usage:
#   bash cloud_gpu_vcr/scripts/preflight_audit.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

section() { echo; echo "===== $* ====="; }

section "1. project root"
ls -la

section "2. results/ (counts + sizes)"
ls -la results/ 2>/dev/null
echo "file count: $(find results -type f 2>/dev/null | wc -l)"
du -sh results 2>/dev/null

section "3. results/checkpoints/"
if [[ -d results/checkpoints ]]; then
  ls -la results/checkpoints/
  echo "checkpoint count: $(find results/checkpoints -type f | wc -l)"
else
  echo "no checkpoints/"
fi

section "4. logs/"
ls -la logs/ 2>/dev/null
du -sh logs 2>/dev/null

section "5. data/ (project-local; should be empty or symlink-only)"
ls -la data/ 2>/dev/null
du -sh data/ 2>/dev/null

section "6. git status (uncommitted local changes)"
git status --short
echo
git log --oneline -5

section "7. wandb/ local runs (synced to cloud already, but check)"
ls -la wandb/ 2>/dev/null | head -15
echo "unsynced .wandb files (if any):"
find wandb -name "*.wandb" 2>/dev/null | head -10

section "8. /workspace top-level"
ls -la /workspace/

section "9. HF model cache (FYI, not backed up — public models re-downloadable)"
du -sh /workspace/.cache/huggingface/ 2>/dev/null
ls /workspace/.cache/huggingface/hub/ 2>/dev/null | head

section "10. /root home dir"
ls -la /root/ 2>/dev/null | head -20

section "11. /tmp scripts/notebooks"
ls /tmp/*.py /tmp/*.ipynb /tmp/*.sh /tmp/*.json 2>/dev/null

section "12. /workspace/data/vcr (the dataset itself)"
ls /workspace/data/vcr/ 2>/dev/null
echo "val.jsonl:"
ls -la /workspace/data/vcr/val.jsonl 2>/dev/null
echo "jpg count:"
find /workspace/data/vcr/vcr1images -type f -name '*.jpg' 2>/dev/null | wc -l

section "13. /workspace/data/vcr_raw (symlinks; should be redundant)"
ls -la /workspace/data/vcr_raw/ 2>/dev/null
ls -la /workspace/data/vcr_raw/vcr1annots/ 2>/dev/null
ls -la /workspace/data/vcr_raw/vcr1images/ 2>/dev/null

section "14. any other files under /workspace not in the repo"
find /workspace -maxdepth 2 -not -path '*/\.*' 2>/dev/null | head -30

section "done"
echo "If anything above looks important and isn't in the backup script,"
echo "tell me and I'll patch backup_to_hf.sh before you run it."
