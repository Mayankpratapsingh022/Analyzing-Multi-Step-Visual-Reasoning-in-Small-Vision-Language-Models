#!/usr/bin/env bash
# Pre-termination backup: push everything that can't be regenerated to
# JaydeepR/vcr-mirror on Hugging Face. Idempotent — safe to re-run.
#
# Usage:
#   bash cloud_gpu_vcr/scripts/backup_to_hf.sh
#
# Override repo via env if needed:
#   REPO=user/other-mirror bash cloud_gpu_vcr/scripts/backup_to_hf.sh

set -u

REPO="${REPO:-JaydeepR/vcr-mirror}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

ts="$(date +%Y%m%d_%H%M%S)"
echo "[backup] target repo: $REPO"
echo "[backup] timestamp:   $ts"
echo "[backup] project:     $PROJECT_ROOT"
echo

upload() {
  local src="$1" dst="$2" label="$3"
  if [[ ! -e "$src" ]]; then
    echo "[backup] SKIP $label (missing: $src)"
    return 0
  fi
  echo "[backup] >>> uploading $label: $src -> $REPO/$dst"
  hf upload "$REPO" "$src" "$dst" --repo-type dataset
  local rc=$?
  if [[ "$rc" == 0 ]]; then
    echo "[backup] OK   $label"
  else
    echo "[backup] FAIL $label (exit $rc)"
  fi
  echo
}

# results: CSVs, per-example details JSONs, plots, checkpoints
upload "results" "wave2_results_${ts}" "results dir"

# logs: sweep + per-run + fetch + bootstrap
upload "logs" "wave2_logs_${ts}" "logs dir"

# also pin a stable "latest" copy so future-you doesn't have to find the ts
upload "results" "wave2_latest/results" "results latest"
upload "logs" "wave2_latest/logs" "logs latest"

echo "[backup] all uploads attempted."
echo "[backup] verify in browser: https://huggingface.co/datasets/$REPO/tree/main"
echo "[backup] or list via API:"
echo "  python -c \"from huggingface_hub import HfApi; [print(f.path) for f in HfApi().list_repo_tree('$REPO', repo_type='dataset', path_in_repo='wave2_latest', recursive=True)]\""
