#!/usr/bin/env bash
# Download VCR v1.0 dataset.
#
# The VCR dataset requires accepting terms at https://visualcommonsense.com/
# After accepting, you receive download links for the images and annotations.
#
# Usage:
#   bash scripts/download_vcr.sh <path-to-data-dir>
#
# Example:
#   bash scripts/download_vcr.sh data/vcr

set -euo pipefail

DATA_DIR="${1:-data/vcr}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  VCR v1.0 Dataset Download"
echo "  Target: $DATA_DIR"
echo "============================================"

# --- Annotations ---
# These are publicly available on the VCR website
ANNOT_URL="https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1annots.zip"
if [ ! -f "$DATA_DIR/val.jsonl" ]; then
    echo "[1/3] Downloading annotations..."
    wget -q --show-progress -O "$DATA_DIR/vcr1annots.zip" "$ANNOT_URL"
    echo "      Extracting annotations..."
    unzip -qo "$DATA_DIR/vcr1annots.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/vcr1annots.zip"
    echo "      Done."
else
    echo "[1/3] Annotations already exist, skipping."
fi

# --- Images ---
# ~25GB -- requires the download link from VCR website after accepting terms
IMAGE_URL="https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1images.zip"
if [ ! -d "$DATA_DIR/vcr1images" ]; then
    echo "[2/3] Downloading images (~25GB, this will take a while)..."
    wget -q --show-progress -O "$DATA_DIR/vcr1images.zip" "$IMAGE_URL"
    echo "      Extracting images..."
    unzip -qo "$DATA_DIR/vcr1images.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/vcr1images.zip"
    echo "      Done."
else
    echo "[2/3] Images directory already exists, skipping."
fi

# --- Validation ---
echo "[3/3] Validating dataset..."
EXPECTED_SPLITS=("train.jsonl" "val.jsonl" "test.jsonl")
for split in "${EXPECTED_SPLITS[@]}"; do
    if [ -f "$DATA_DIR/$split" ]; then
        COUNT=$(wc -l < "$DATA_DIR/$split")
        echo "      $split: $COUNT examples"
    else
        echo "      WARNING: $split not found!"
    fi
done

if [ -d "$DATA_DIR/vcr1images" ]; then
    IMG_COUNT=$(find "$DATA_DIR/vcr1images" -name "*.jpg" | wc -l)
    echo "      Images: $IMG_COUNT .jpg files"
else
    echo "      WARNING: vcr1images/ directory not found!"
fi

echo ""
echo "Done. Dataset stored at: $DATA_DIR"
