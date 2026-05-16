"""
Checkpoint manager for evaluation runs.

Saves progress periodically (default every 5 min) and on interruption (Ctrl+C),
so runs can be resumed from where they left off.

Checkpoint files live in results/checkpoints/ with deterministic names:
    {model}_{dataset}_{split}_{seed}.checkpoint.json

Status flow:
    in_progress -> completed   (normal finish)
    in_progress -> interrupted  (Ctrl+C or crash, detected on next save)
"""

import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

log = logging.getLogger("vlm-eval")

CHECKPOINT_DIR = Path("results/checkpoints")


def checkpoint_path(
    model_name: str,
    dataset: str,
    split: str,
    seed: int,
    run_key: str | None = None,
) -> Path:
    """Deterministic checkpoint filename for a run config."""
    suffix = f"_{run_key}" if run_key else ""
    return CHECKPOINT_DIR / f"{model_name}_{dataset}_{split}_{seed}{suffix}.checkpoint.json"


def load_checkpoint(path: Path) -> Optional[dict]:
    """Load an existing checkpoint. Returns None if not found."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_checkpoint(path: Path, data: dict):
    """Atomically save checkpoint (write tmp then rename to avoid corruption)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    data["last_saved"] = datetime.now().isoformat()
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.rename(path)


def list_checkpoints() -> list[dict]:
    """List all checkpoints with summary info (for checker.py)."""
    if not CHECKPOINT_DIR.exists():
        return []

    results = []
    for p in sorted(CHECKPOINT_DIR.glob("*.checkpoint.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            results.append({
                "file": str(p),
                "filename": p.name,
                "model_name": data.get("model_name", "?"),
                "dataset": data.get("dataset", "?"),
                "split": data.get("split", "?"),
                "seed": data.get("seed", "?"),
                "quantization": data.get("quantization", "none"),
                "subset_pct": data.get("subset_pct"),
                "max_samples": data.get("max_samples"),
                "prompt_strategy": data.get("prompt_strategy"),
                "max_new_tokens": data.get("max_new_tokens"),
                "run_key": data.get("run_key"),
                "status": data.get("status", "?"),
                "completed": data.get("completed_examples", 0),
                "total": data.get("total_examples", 0),
                "last_saved": data.get("last_saved", "?"),
                "run_id": data.get("run_id", "?"),
                # partial metrics
                "metrics": data.get("running_metrics", {}),
            })
        except (json.JSONDecodeError, KeyError):
            results.append({"file": str(p), "filename": p.name, "status": "corrupt"})

    return results


class CheckpointManager:
    """Manages periodic saving and graceful interruption for an eval run.

    Usage:
        mgr = CheckpointManager(model, dataset, split, seed, total, ...)
        start_idx = mgr.resume_index()         # 0 if fresh, N if resuming
        prior_results = mgr.resumed_examples()  # [] if fresh

        for i in range(start_idx, total):
            ... do inference ...
            mgr.record(i, example_result_dict)  # auto-saves every 5 min

        mgr.mark_completed(final_results_dict)
    """

    def __init__(
        self,
        model_name: str,
        dataset: str,
        split: str,
        seed: int,
        total_examples: int,
        run_id: str = "",
        run_key: str | None = None,
        quantization: str = "none",
        subset_pct: Optional[float] = None,
        max_samples: Optional[int] = None,
        prompt_strategy: str = "zero_shot_direct",
        max_new_tokens: int = 8,
        save_interval_sec: float = 300,  # 5 minutes
    ):
        self.path = checkpoint_path(model_name, dataset, split, seed, run_key=run_key)
        self.save_interval_sec = save_interval_sec
        self._last_save_time = time.time()
        self._interrupted = False

        self._data = {
            "run_id": run_id,
            "run_key": run_key,
            "model_name": model_name,
            "dataset": dataset,
            "split": split,
            "seed": seed,
            "quantization": quantization,
            "subset_pct": subset_pct,
            "max_samples": max_samples,
            "prompt_strategy": prompt_strategy,
            "max_new_tokens": max_new_tokens,
            "total_examples": total_examples,
            "completed_examples": 0,
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
            "last_saved": None,
            "per_example": [],
            "running_metrics": {},
        }

        # Check for existing checkpoint
        self._existing = load_checkpoint(self.path)

        # Install SIGINT handler for graceful Ctrl+C
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """On Ctrl+C: save checkpoint immediately, then re-raise."""
        if self._interrupted:
            # Second Ctrl+C: force exit
            log.warning("Force exit (second Ctrl+C)")
            signal.signal(signal.SIGINT, self._original_sigint)
            raise KeyboardInterrupt
        self._interrupted = True
        log.warning("")
        log.warning("Interrupted! Saving checkpoint before exit...")
        self._data["status"] = "interrupted"
        save_checkpoint(self.path, self._data)
        log.warning(f"Checkpoint saved: {self.path}")
        log.warning(
            f"Progress: {self._data['completed_examples']}/{self._data['total_examples']} examples"
        )
        log.warning("Run the same command again to resume, or use checker.py for details.")
        # Restore original handler and re-raise
        signal.signal(signal.SIGINT, self._original_sigint)
        raise KeyboardInterrupt

    def can_resume(self) -> bool:
        """Is there an existing checkpoint we can resume from?"""
        if self._existing is None:
            return False
        return (
            self._existing.get("status") in ("in_progress", "interrupted")
            and self._existing.get("completed_examples", 0) > 0
        )

    def is_completed(self) -> bool:
        """Was this run already completed?"""
        return self._existing is not None and self._existing.get("status") == "completed"

    def resume_index(self) -> int:
        """Return the index to start from. 0 if fresh, N if resuming."""
        if self.can_resume():
            n = self._existing["completed_examples"]
            log.info(f"Resuming from checkpoint: {n}/{self._data['total_examples']} examples done")
            # Carry over prior results
            self._data["per_example"] = self._existing.get("per_example", [])[:n]
            self._data["completed_examples"] = n
            self._data["running_metrics"] = self._existing.get("running_metrics", {})
            self._data["resumed_from"] = self._existing.get("last_saved", "")
            return n
        return 0

    def resumed_examples(self) -> list[dict]:
        """Return per-example results from the resumed checkpoint."""
        if self.can_resume():
            return self._existing.get("per_example", [])
        return []

    def record(self, idx: int, example_result: dict, running_metrics: dict = None):
        """Record a completed example. Auto-saves if interval has elapsed."""
        self._data["per_example"].append(example_result)
        self._data["completed_examples"] = idx + 1
        if running_metrics:
            self._data["running_metrics"] = running_metrics

        now = time.time()
        if now - self._last_save_time >= self.save_interval_sec:
            self._periodic_save()

    def _periodic_save(self):
        """Save checkpoint to disk."""
        save_checkpoint(self.path, self._data)
        self._last_save_time = time.time()
        done = self._data["completed_examples"]
        total = self._data["total_examples"]
        log.info(f"[Checkpoint saved] {done}/{total} examples -> {self.path}")

    def mark_completed(self, final_results: dict):
        """Mark run as completed and save final checkpoint."""
        self._data["status"] = "completed"
        self._data["completed_at"] = datetime.now().isoformat()
        self._data["final_metrics"] = {
            k: v for k, v in final_results.items()
            if k != "per_example" and not isinstance(v, (list, dict))
        }
        save_checkpoint(self.path, self._data)
        log.info(f"[Checkpoint] Run completed -> {self.path}")

        # Restore original signal handler
        signal.signal(signal.SIGINT, self._original_sigint)

    def force_restart(self):
        """Delete existing checkpoint to start fresh."""
        if self.path.exists():
            self.path.unlink()
            log.info(f"Deleted checkpoint: {self.path}")
        self._existing = None
        self._data["per_example"] = []
        self._data["completed_examples"] = 0
