#!/usr/bin/env python3
"""
Main experiment runner. Parses a YAML config and runs evaluations.

Usage:
    python run_experiment.py --config configs/baseline_small.yml
    python run_experiment.py --config configs/baseline_large.yml
"""

import argparse
import multiprocessing as mp
import sys

import yaml

from src.baselines.evaluate import run_single_evaluation, append_to_csv, run_sweep, _eval_worker


def main():
    parser = argparse.ArgumentParser(description="Run experiments from YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.config}")
    print(f"Experiment: {config.get('experiment_name', 'unnamed')}")
    print(f"Dataset: {config.get('dataset', 'vcr')}")

    if args.dry_run:
        print("\nFull config:")
        yaml.dump(config, sys.stdout, default_flow_style=False)
        return

    # Support sweep mode
    if "sweep" in config:
        run_sweep(
            model_group=config["sweep"],
            dataset_name=config.get("dataset", "mmmu"),
            vcr_dir=config.get("vcr_dir", "data/vcr"),
            subset_pct=config.get("subset_pct"),
            max_samples=config.get("max_samples"),
            subject=config.get("subject"),
            quantization=config.get("quantization"),
            seed=config.get("seed", 42),
            device=config.get("device", "cuda"),
            output_dir=config.get("output_dir", "results"),
            csv_path=config.get("csv_path", "results/baselines.csv"),
        )
        return

    # Run individual models from list
    models = config.get("models", [])
    if not models:
        print("No models specified in config.")
        return

    for model_cfg in models:
        model_name = model_cfg if isinstance(model_cfg, str) else model_cfg["name"]
        quant = model_cfg.get("quantization") if isinstance(model_cfg, dict) else None

        device = config.get("device", "cuda")
        if model_name in ("gpt-4o", "claude"):
            device = "cpu"

        eval_kwargs = dict(
            model_name=model_name,
            dataset_name=config.get("dataset", "mmmu"),
            vcr_dir=config.get("vcr_dir", "data/vcr"),
            subset_pct=config.get("subset_pct"),
            max_samples=config.get("max_samples"),
            subject=config.get("subject"),
            quantization=quant or config.get("quantization"),
            seed=config.get("seed", 42),
            device=device,
            output_dir=config.get("output_dir", "results"),
        )
        csv_path = config.get("csv_path", "results/baselines.csv")

        # Isolate each model in its own spawned process so a CUDA
        # device-side assertion cannot cascade to subsequent models.
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(target=_eval_worker, args=(eval_kwargs, csv_path, q))
        proc.start()
        proc.join()

        if not q.empty():
            item = q.get()
            if item[0] == "ok":
                print(f"OK: {model_name}")
            else:
                print(f"FAILED: {model_name} -- {item[1]}")
        else:
            print(f"FAILED: {model_name} -- worker process exited with code {proc.exitcode}")


if __name__ == "__main__":
    main()
