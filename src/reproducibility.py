"""Reproducibility helpers for inference experiments."""

from __future__ import annotations

import os
import platform
import random
import subprocess

import numpy as np
import torch


def set_reproducibility(seed: int, deterministic: bool = True) -> None:
    """Set common random seeds and deterministic flags.

    This project runs inference only, but deterministic sampling, dataset
    selection, and CUDA kernel choices still matter for reproducible evals.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def collect_environment_metadata() -> dict:
    """Collect runtime metadata for result JSON and W&B configs."""
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        metadata["gpu_count"] = torch.cuda.device_count()
        metadata["gpu_vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if commit.returncode == 0:
            metadata["git_commit"] = commit.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode == 0:
            metadata["git_dirty"] = bool(status.stdout.strip())
    except OSError:
        metadata["git_commit"] = ""
        metadata["git_dirty"] = None

    return metadata
