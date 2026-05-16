#!/usr/bin/env python3
"""Validate the VCR dataset layout and loader on the cloud machine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import VCRDataset


def count_lines(path: Path) -> int:
    with path.open() as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcr-dir", default="/workspace/data/vcr")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    vcr_dir = Path(args.vcr_dir)
    print(f"[validate-vcr] vcr_dir={vcr_dir}")

    required = ["train.jsonl", "val.jsonl", "test.jsonl"]
    missing = [name for name in required if not (vcr_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing annotation files: {missing}")

    for name in required:
        path = vcr_dir / name
        print(f"[validate-vcr] {name}: {count_lines(path)} lines")

    image_dir = vcr_dir / "vcr1images"
    if not image_dir.exists():
        raise SystemExit(f"Missing image directory: {image_dir}")

    image_count = sum(1 for _ in image_dir.rglob("*.jpg"))
    meta_count = sum(1 for _ in image_dir.rglob("*.metadata.json"))
    print(f"[validate-vcr] images: {image_count}")
    print(f"[validate-vcr] metadata files: {meta_count}")
    if image_count == 0:
        raise SystemExit("No VCR images found.")

    ds = VCRDataset(str(vcr_dir), split="val", max_samples=args.samples, seed=42)
    print(f"[validate-vcr] sampled val size: {len(ds)}")

    box_examples = 0
    for idx in range(len(ds)):
        ex = ds[idx]
        boxes = ex["metadata"].get("boxes", [])
        if boxes:
            box_examples += 1
        print(f"[validate-vcr] sample {idx}")
        print(f"  image_size={ex['image'].size}")
        print(f"  annot_id={ex['metadata'].get('annot_id')}")
        print(f"  question={ex['question'][:160]}")
        print(f"  answer_choices={json.dumps(ex['answer_choices'][:2])[:220]}")
        print(f"  rationale_choices={json.dumps(ex['rationale_choices'][:2])[:220]}")
        print(f"  question_refs={ex['metadata'].get('question_object_refs')}")
        print(f"  boxes={len(boxes)}")

    if box_examples == 0:
        raise SystemExit("Loader worked, but no boxes were found in sampled examples.")

    print("[validate-vcr] OK")


if __name__ == "__main__":
    main()
