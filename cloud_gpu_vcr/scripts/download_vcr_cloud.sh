#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export VCR_DIR="${VCR_DIR:-/workspace/data/vcr}"

mkdir -p "$VCR_DIR"

echo "[vcr-download] target: $VCR_DIR"
cd "$PROJECT_ROOT"
bash scripts/download_vcr.sh "$VCR_DIR"
python cloud_gpu_vcr/scripts/validate_vcr.py --vcr-dir "$VCR_DIR" --samples 5
