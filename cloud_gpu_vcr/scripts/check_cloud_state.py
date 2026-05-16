#!/usr/bin/env python3
"""Print cloud runner state before launching long VCR jobs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    print("[cloud-state] environment")
    for key in [
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "VCR_DIR",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_MODE",
    ]:
        print(f"  {key}={os.environ.get(key, '')}")

    print("[cloud-state] paths")
    for path in [
        os.environ.get("HF_HOME", "/workspace/.cache/huggingface"),
        os.environ.get("VCR_DIR", "/workspace/data/vcr"),
        "results",
        "logs",
    ]:
        p = Path(path)
        print(f"  {p}: {'exists' if p.exists() else 'missing'}")

    print("[cloud-state] python imports")
    for mod in ["torch", "transformers", "datasets", "wandb", "src.data_loader", "src.models"]:
        try:
            imported = __import__(mod)
            version = getattr(imported, "__version__", "")
            print(f"  OK {mod} {version}")
        except Exception as exc:
            print(f"  FAIL {mod}: {exc}")

    try:
        import torch

        print("[cloud-state] cuda")
        print(f"  available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  gpu={torch.cuda.get_device_name(0)}")
            print(f"  vram_gb={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}")
    except Exception as exc:
        print(f"[cloud-state] cuda check failed: {exc}")


if __name__ == "__main__":
    main()
