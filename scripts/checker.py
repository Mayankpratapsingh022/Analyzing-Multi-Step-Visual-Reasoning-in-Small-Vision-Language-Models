#!/usr/bin/env python3
"""
Checkpoint checker: inspect evaluation progress, decide to resume or restart.

Shows all checkpoints with their status, progress, and partial metrics.
Generates the exact command to resume or restart each run.

Usage:
    python scripts/checker.py                    # Show all checkpoints
    python scripts/checker.py --status            # Quick status table
    python scripts/checker.py --detail <file>     # Detailed view of one checkpoint
    python scripts/checker.py --delete <file>     # Delete a checkpoint (start fresh)
    python scripts/checker.py --delete-completed  # Clean up completed checkpoints
    python scripts/checker.py --resume-cmd        # Print resume commands for all interrupted runs
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checkpoint import list_checkpoints, CHECKPOINT_DIR


def format_pct(done: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{done/total*100:.1f}%"


def progress_bar(done: int, total: int, width: int = 20) -> str:
    if total == 0:
        return "[" + "." * width + "]"
    pct = done / total
    filled = int(width * pct)
    return "[" + "=" * filled + ">" * (1 if filled < width else 0) + "." * (width - filled - 1) + "]"


STATUS_COLORS = {
    "completed": "\033[32m",    # green
    "interrupted": "\033[33m",  # yellow
    "in_progress": "\033[36m",  # cyan
    "corrupt": "\033[31m",      # red
}
RESET = "\033[0m"


def colored_status(status: str) -> str:
    color = STATUS_COLORS.get(status, "")
    return f"{color}{status}{RESET}"


def show_status_table(checkpoints: list[dict]):
    """Compact overview of all checkpoints."""
    if not checkpoints:
        print("No checkpoints found.")
        print(f"(checkpoint dir: {CHECKPOINT_DIR})")
        return

    print(f"\n{'='*95}")
    print(f"  CHECKPOINT STATUS ({len(checkpoints)} runs found)")
    print(f"{'='*95}\n")

    print(
        f"  {'Model':<22} {'Dataset':<10} {'Status':<14} "
        f"{'Progress':<28} {'Metrics':<20} {'Last Saved'}"
    )
    print(f"  {'-'*110}")

    for c in checkpoints:
        if c.get("status") == "corrupt":
            print(f"  {c['filename']:<22} {'CORRUPT':<10}")
            continue

        done = c["completed"]
        total = c["total"]
        status = colored_status(c["status"])
        pbar = f"{progress_bar(done, total)} {done}/{total} ({format_pct(done, total)})"

        # Show key metric
        metrics = c.get("metrics", {})
        if "q_ar_accuracy" in metrics:
            metric_str = f"Q->AR: {metrics['q_ar_accuracy']:.3f}"
        elif "q_a_accuracy" in metrics:
            metric_str = f"Q->A: {metrics['q_a_accuracy']:.3f}"
        elif "accuracy" in metrics:
            metric_str = f"Acc: {metrics['accuracy']:.3f}"
        else:
            metric_str = "-"

        last_saved = c.get("last_saved", "?")
        if isinstance(last_saved, str) and len(last_saved) > 16:
            last_saved = last_saved[:16]

        print(
            f"  {c['model_name']:<22} {c['dataset']:<10} {status:<24} "
            f"{pbar:<28} {metric_str:<20} {last_saved}"
        )

    print()


def show_detail(checkpoint_file: str):
    """Detailed view of a single checkpoint."""
    path = Path(checkpoint_file)
    if not path.exists():
        # Try in checkpoint dir
        path = CHECKPOINT_DIR / checkpoint_file
    if not path.exists():
        print(f"Checkpoint not found: {checkpoint_file}")
        return

    with open(path) as f:
        data = json.load(f)

    done = data.get("completed_examples", 0)
    total = data.get("total_examples", 0)
    status = data.get("status", "?")

    print(f"\n{'='*70}")
    print(f"  CHECKPOINT DETAIL")
    print(f"{'='*70}")
    print(f"  File         : {path}")
    print(f"  Run ID       : {data.get('run_id', '?')}")
    print(f"  Model        : {data.get('model_name', '?')}")
    print(f"  Dataset      : {data.get('dataset', '?')}")
    print(f"  Split        : {data.get('split', '?')}")
    print(f"  Seed         : {data.get('seed', '?')}")
    print(f"  Quantization : {data.get('quantization', 'none')}")
    print(f"  Subset %     : {data.get('subset_pct', 'full')}")
    print(f"")
    print(f"  Status       : {colored_status(status)}")
    print(f"  Progress     : {done}/{total} ({format_pct(done, total)})")
    print(f"  Remaining    : {total - done} examples")
    print(f"")
    print(f"  Started at   : {data.get('started_at', '?')}")
    print(f"  Last saved   : {data.get('last_saved', '?')}")
    if data.get("completed_at"):
        print(f"  Completed at : {data['completed_at']}")
    if data.get("resumed_from"):
        print(f"  Resumed from : {data['resumed_from']}")

    # Metrics
    metrics = data.get("running_metrics", {})
    if data.get("final_metrics"):
        metrics = data["final_metrics"]
        print(f"\n  Final Metrics:")
    elif metrics:
        print(f"\n  Running Metrics (partial, at {done}/{total}):")

    if metrics:
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"    {k:<30}: {v:.4f}")
            else:
                print(f"    {k:<30}: {v}")

    # Resume / restart commands
    model = data.get("model_name", "MODEL")
    dataset = data.get("dataset", "DATASET")
    seed = data.get("seed", 42)
    subset = data.get("subset_pct")

    base_cmd = f"python -m src.baselines.evaluate --model {model} --dataset {dataset} --seed {seed}"
    if subset:
        base_cmd += f" --subset_pct {subset}"
    if dataset == "vcr":
        base_cmd += " --vcr_dir data/vcr"

    print(f"\n  Commands:")
    if status in ("interrupted", "in_progress"):
        print(f"    Resume : {base_cmd}")
        print(f"    Restart: {base_cmd} --restart")
    elif status == "completed":
        print(f"    Re-run : {base_cmd} --restart")

    print(f"{'='*70}\n")


def show_resume_commands(checkpoints: list[dict]):
    """Print resume commands for all interrupted/in-progress runs."""
    resumable = [c for c in checkpoints if c.get("status") in ("interrupted", "in_progress")]

    if not resumable:
        print("No interrupted runs to resume.")
        return

    print(f"\nResumable runs ({len(resumable)}):\n")
    for c in resumable:
        model = c["model_name"]
        dataset = c["dataset"]
        seed = c.get("seed", 42)
        subset = c.get("subset_pct")
        done = c["completed"]
        total = c["total"]

        cmd = f"python -m src.baselines.evaluate --model {model} --dataset {dataset} --seed {seed}"
        if subset:
            cmd += f" --subset_pct {subset}"
        if dataset == "vcr":
            cmd += " --vcr_dir data/vcr"

        print(f"  # {model} on {dataset} -- {done}/{total} done ({format_pct(done, total)})")
        print(f"  {cmd}")
        print()


def delete_checkpoint(checkpoint_file: str):
    """Delete a checkpoint file."""
    path = Path(checkpoint_file)
    if not path.exists():
        path = CHECKPOINT_DIR / checkpoint_file
    if not path.exists():
        print(f"Checkpoint not found: {checkpoint_file}")
        return

    with open(path) as f:
        data = json.load(f)

    model = data.get("model_name", "?")
    done = data.get("completed_examples", 0)
    total = data.get("total_examples", 0)
    status = data.get("status", "?")

    print(f"Deleting: {path.name}")
    print(f"  Model: {model} | Status: {status} | Progress: {done}/{total}")

    confirm = input("  Are you sure? [y/N] ").strip().lower()
    if confirm == "y":
        path.unlink()
        print("  Deleted.")
    else:
        print("  Cancelled.")


def delete_completed():
    """Delete all completed checkpoints."""
    checkpoints = list_checkpoints()
    completed = [c for c in checkpoints if c.get("status") == "completed"]

    if not completed:
        print("No completed checkpoints to clean up.")
        return

    print(f"Found {len(completed)} completed checkpoints:")
    for c in completed:
        print(f"  {c['model_name']} on {c['dataset']} -- {c['filename']}")

    confirm = input(f"\nDelete all {len(completed)}? [y/N] ").strip().lower()
    if confirm == "y":
        for c in completed:
            Path(c["file"]).unlink()
            print(f"  Deleted: {c['filename']}")
        print("Done.")
    else:
        print("Cancelled.")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect evaluation checkpoints -- see progress, resume, or restart runs"
    )
    parser.add_argument("--status", action="store_true",
                        help="Show compact status table (default action)")
    parser.add_argument("--detail", type=str, metavar="FILE",
                        help="Show detailed info for one checkpoint")
    parser.add_argument("--resume-cmd", action="store_true",
                        help="Print resume commands for interrupted runs")
    parser.add_argument("--delete", type=str, metavar="FILE",
                        help="Delete a checkpoint file")
    parser.add_argument("--delete-completed", action="store_true",
                        help="Delete all completed checkpoints")
    args = parser.parse_args()

    if args.detail:
        show_detail(args.detail)
    elif args.resume_cmd:
        show_resume_commands(list_checkpoints())
    elif args.delete:
        delete_checkpoint(args.delete)
    elif args.delete_completed:
        delete_completed()
    else:
        show_status_table(list_checkpoints())


if __name__ == "__main__":
    main()
