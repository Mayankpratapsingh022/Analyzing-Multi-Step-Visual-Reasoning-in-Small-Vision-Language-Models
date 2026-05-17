"""
Fetch only the image files needed for the 300-example VCR val sweep.

Why this exists: the full vcr1images set is ~30 GB and ~200k files. When pulling
the JaydeepR/vcr-mirror dataset from HF, default workers + small files trip the
3000 req/5min rate limit on the Team plan. The 300-example sweep only touches
300 random val examples (seed=42), which is ~600 files total — well under the
rate limit.

This script:
  1. Loads val.jsonl from the local mirror.
  2. Reproduces the loader's sampling: random.Random(42).sample(annots, 300).
  3. Checks which sampled images and metadata files are already on disk.
  4. Downloads only the missing files from JaydeepR/vcr-mirror.

The seed and sample count must match src.data_loader.VCRDataset usage in the
300-example sweep.

Usage:
  python cloud_gpu_vcr/scripts/fetch_300_val.py

Override defaults via env vars or CLI flags if needed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "JaydeepR/vcr-mirror"
HF_PATH_PREFIX = "vcr1images/vcr1images"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vcr-raw",
        default="/workspace/data/vcr_raw",
        help="Where the HF mirror was downloaded to (must contain vcr1annots/val.jsonl).",
    )
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="val", choices=["val", "train", "test"])
    args = parser.parse_args()

    vcr_raw = Path(args.vcr_raw)
    val_path = vcr_raw / "vcr1annots" / f"{args.split}.jsonl"
    img_root = vcr_raw / "vcr1images" / "vcr1images"

    if not val_path.exists():
        raise SystemExit(f"missing annotations file: {val_path}")

    with open(val_path) as f:
        annots = [json.loads(line) for line in f]
    print(f"[fetch] {args.split}.jsonl total: {len(annots)}")

    sampled = random.Random(args.seed).sample(annots, args.samples)
    print(f"[fetch] sampled: {len(sampled)} (seed={args.seed})")

    needed: list[str] = []
    for a in sampled:
        needed.append(a["img_fn"])
        needed.append(a["metadata_fn"])

    missing = [rel for rel in needed if not (img_root / rel).exists()]
    present = len(needed) - len(missing)
    print(f"[fetch] files needed: {len(needed)}")
    print(f"[fetch] already present: {present}")
    print(f"[fetch] missing: {len(missing)}")

    if not missing:
        print("[fetch] nothing to download.")
        return

    img_root.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for i, rel in enumerate(missing, 1):
        repo_path = f"{HF_PATH_PREFIX}/{rel}"
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_path,
                repo_type="dataset",
                local_dir=str(vcr_raw),
            )
        except Exception as exc:
            failures.append((rel, str(exc)))
        if i % 50 == 0 or i == len(missing):
            print(f"[fetch] {i}/{len(missing)} downloaded ({len(failures)} failures)")

    if failures:
        print(f"[fetch] {len(failures)} failures:")
        for rel, err in failures[:10]:
            print(f"  {rel} :: {err}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
        raise SystemExit(1)

    print("[fetch] done.")


if __name__ == "__main__":
    main()
