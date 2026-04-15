#!/usr/bin/env python3
"""
Baseline evaluation script with comprehensive logging.

All runs produce:
  - Console output: progress bars, running accuracy, ETA
  - Log file (logs/): full prompts, raw model responses, parsed answers
  - Results CSV: summary metrics per model
  - Detail JSON: per-example results with prompts and outputs

Usage:
    # Single model on VCR (5% subset)
    python -m src.baselines.evaluate --model qwen2-vl-2b --dataset vcr --vcr_dir data/vcr --subset_pct 5

    # Full VCR val
    python -m src.baselines.evaluate --model qwen2-vl-2b --dataset vcr --vcr_dir data/vcr

    # MMMU
    python -m src.baselines.evaluate --model phi3-vision --dataset mmmu --subset_pct 10

    # All small models sweep
    python -m src.baselines.evaluate --sweep small --dataset vcr --vcr_dir data/vcr --subset_pct 5

    # From YAML config
    python -m src.baselines.evaluate --config configs/baseline_small.yml
"""

import argparse
import csv
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import yaml

from src.data_loader import get_dataset
from src.logger import setup_logger, format_time
from src.models import get_evaluator, list_evaluators


log = logging.getLogger("vlm-eval")

SMALL_MODELS = [
    "moondream2",
    "qwen2-vl-2b",
    "internvl2-2b",
    "phi3-vision",
    "qwen2-vl-7b",
    "llava-next-7b",
    "internvl2-8b",
]

LARGE_MODELS = [
    "llava-1.5-13b",
    "internvl2-26b",
    "llava-1.6-34b",
    "gpt-4o",
    "claude",
]

ALL_MODELS = SMALL_MODELS + LARGE_MODELS


def get_vram_usage() -> float:
    """Current GPU memory allocated in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


def get_vram_total() -> float:
    """Total GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return 0.0


def log_system_info():
    """Log system and GPU info at the start of a run."""
    log.info(f"System Info:")
    log.info(f"  PyTorch     : {torch.__version__}")
    log.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"  GPU         : {torch.cuda.get_device_name()}")
        log.info(f"  VRAM total  : {get_vram_total():.1f} GB")
        log.info(f"  CUDA version: {torch.version.cuda}")
    log.info("")


def run_single_evaluation(
    model_name: str,
    dataset_name: str,
    vcr_dir: str = "data/vcr",
    subset_pct: Optional[float] = None,
    max_samples: Optional[int] = None,
    subject: Optional[str] = None,
    quantization: Optional[str] = None,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "results",
    restart: bool = False,
    batch_size: int = 0,
) -> dict:
    """Evaluate a single model on a single dataset. Returns results dict.

    If a checkpoint exists for this model+dataset+seed, resumes from it.
    Pass restart=True to ignore the checkpoint and start fresh.
    batch_size=0 auto-detects from VRAM, 1=sequential.

    Args:
        max_samples: Exact number of examples to evaluate. Takes priority over
                     subset_pct when both are set.
        subject: MMMU only — subject config name to load (e.g. "Math").
                 Ignored for other datasets.
    """
    run_id = f"{model_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Build a human-readable sample description for the log header
    if max_samples is not None:
        sample_desc = f"{max_samples} samples"
    elif subset_pct is not None:
        sample_desc = f"{subset_pct}%"
    else:
        sample_desc = "full"

    log.info("")
    log.info(f"{'#'*70}")
    log.info(f"  NEW EVALUATION RUN")
    log.info(f"{'#'*70}")
    log.info(f"  Run ID       : {run_id}")
    log.info(f"  Model        : {model_name}")
    log.info(f"  Dataset      : {dataset_name}" + (f" ({subject})" if subject else ""))
    log.info(f"  Samples      : {sample_desc}")
    log.info(f"  Quantization : {quantization or 'none'}")
    log.info(f"  Seed         : {seed}")
    log.info(f"  Device       : {device}")
    log.info(f"  Output dir   : {output_dir}")
    log.info(f"{'#'*70}")
    log.info("")

    # ---- Load model ----
    evaluator = get_evaluator(model_name)
    log.info(f"Loading model: {model_name} ...")
    vram_before = get_vram_usage()
    t0 = time.time()
    evaluator.load_model(device=device, quantization=quantization)
    load_time = time.time() - t0
    vram_after = get_vram_usage()

    log.info(f"Model loaded successfully")
    log.info(f"  Load time    : {format_time(load_time)}")
    log.info(f"  VRAM before  : {vram_before:.2f} GB")
    log.info(f"  VRAM after   : {vram_after:.2f} GB")
    log.info(f"  VRAM used    : {vram_after - vram_before:.2f} GB")
    log.info("")

    # ---- Load dataset ----
    log.info(f"Loading dataset: {dataset_name} ...")
    ds_t0 = time.time()

    if dataset_name == "vcr":
        dataset = get_dataset(
            "vcr", vcr_dir=vcr_dir, split="val",
            subset_pct=subset_pct, seed=seed,
        )
    elif dataset_name in ("mmmu", "mathvista"):
        ds_kwargs = {"subset_pct": subset_pct, "max_samples": max_samples, "seed": seed}
        if dataset_name == "mmmu":
            ds_kwargs["split"] = "validation"
            if subject:
                ds_kwargs["subject"] = subject
        else:
            ds_kwargs["split"] = "testmini"
        dataset = get_dataset(dataset_name, **ds_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ds_load_time = time.time() - ds_t0
    log.info(f"Dataset loaded: {len(dataset)} examples in {format_time(ds_load_time)}")
    log.info("")

    # ---- Run evaluation ----
    if dataset_name == "vcr":
        results = evaluator.evaluate_vcr(
            dataset, split="val", subset_pct=subset_pct, seed=seed,
            run_id=run_id, restart=restart, batch_size=batch_size,
        )
    else:
        results = evaluator.evaluate_mcq(
            dataset, dataset_name=dataset_name, seed=seed,
            run_id=run_id, restart=restart, batch_size=batch_size,
        )

    # ---- Add metadata ----
    results["run_id"] = run_id
    results["quantization"] = quantization or "none"
    results["device"] = device
    results["load_time_sec"] = load_time
    results["vram_gb"] = get_vram_usage()
    results["vram_model_gb"] = round(vram_after - vram_before, 2)
    results["timestamp"] = datetime.now().isoformat()

    # ---- Save per-example results ----
    os.makedirs(output_dir, exist_ok=True)
    detail_path = Path(output_dir) / f"{run_id}_details.json"
    with open(detail_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Per-example results saved to: {detail_path}")

    # ---- Free GPU memory ----
    del evaluator.model
    if hasattr(evaluator, "processor") and evaluator.processor:
        del evaluator.processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info(f"GPU memory freed. VRAM now: {get_vram_usage():.2f} GB")

    return results


def append_to_csv(results: dict, csv_path: str = "results/baselines.csv"):
    """Append a summary row to the baselines CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "run_id", "model_name", "param_count", "quantization",
        "dataset", "split", "subset_pct", "num_examples",
        "q_a_accuracy", "qa_r_accuracy", "q_ar_accuracy",  # VCR
        "accuracy",  # MMMU/MathVista
        "parse_failure_rate",
        "avg_inference_time_sec", "median_inference_time_sec",
        "min_inference_time_sec", "max_inference_time_sec",
        "total_inference_time_sec", "wall_time_sec",
        "load_time_sec", "vram_gb", "vram_model_gb",
        "seed", "device", "timestamp",
    ]

    row = {k: results.get(k, "") for k in fieldnames}
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    log.info(f"Summary row appended to: {csv_path}")


def run_sweep(
    model_group: str,
    dataset_name: str,
    **kwargs,
):
    """Run evaluation for a group of models."""
    if model_group == "small":
        models = SMALL_MODELS
    elif model_group == "large":
        models = LARGE_MODELS
    elif model_group == "all":
        models = ALL_MODELS
    else:
        raise ValueError(f"Unknown sweep group: {model_group}")

    log.info(f"{'*'*70}")
    log.info(f"  SWEEP: {model_group} models on {dataset_name}")
    log.info(f"  Models: {', '.join(models)}")
    log.info(f"{'*'*70}")

    all_results = []
    sweep_start = time.time()

    for model_idx, model_name in enumerate(models, 1):
        log.info(f"\n--- Sweep progress: model {model_idx}/{len(models)} ({model_name}) ---\n")

        # Auto-quantize large models
        quant = kwargs.get("quantization")
        if quant is None and model_name in ("llava-1.5-13b", "internvl2-26b", "llava-1.6-34b"):
            quant = "4bit"

        # API models don't need GPU
        device = kwargs.get("device", "cuda")
        if model_name in ("gpt-4o", "claude"):
            device = "cpu"

        try:
            results = run_single_evaluation(
                model_name=model_name,
                dataset_name=dataset_name,
                vcr_dir=kwargs.get("vcr_dir", "data/vcr"),
                subset_pct=kwargs.get("subset_pct"),
                max_samples=kwargs.get("max_samples"),
                subject=kwargs.get("subject"),
                quantization=quant,
                seed=kwargs.get("seed", 42),
                device=device,
                output_dir=kwargs.get("output_dir", "results"),
                restart=kwargs.get("restart", False),
                batch_size=kwargs.get("batch_size", 0),
            )
            append_to_csv(results, kwargs.get("csv_path", "results/baselines.csv"))
            all_results.append(results)

        except Exception as e:
            log.error(f"FAILED: {model_name} -- {e}", exc_info=True)
            continue

    # ---- Sweep summary ----
    sweep_time = time.time() - sweep_start
    log.info("")
    log.info(f"{'*'*70}")
    log.info(f"  SWEEP COMPLETE")
    log.info(f"  Total time: {format_time(sweep_time)}")
    log.info(f"  Models evaluated: {len(all_results)}/{len(models)}")
    log.info(f"{'*'*70}")
    log.info("")

    if dataset_name == "vcr":
        log.info(f"  {'Model':<22} {'Params':<8} {'Q->A':<8} {'QA->R':<8} {'Q->AR':<8} {'Parse%':<8} {'Avg(s)':<8}")
        log.info(f"  {'-'*78}")
        for r in all_results:
            log.info(
                f"  {r['model_name']:<22} {r['param_count']:<8} "
                f"{r['q_a_accuracy']:<8.4f} {r['qa_r_accuracy']:<8.4f} "
                f"{r['q_ar_accuracy']:<8.4f} {r['parse_failure_rate']:<8.3f} "
                f"{r['avg_inference_time_sec']:<8.2f}"
            )
    else:
        log.info(f"  {'Model':<22} {'Params':<8} {'Accuracy':<10} {'Parse%':<8} {'Avg(s)':<8}")
        log.info(f"  {'-'*60}")
        for r in all_results:
            log.info(
                f"  {r['model_name']:<22} {r['param_count']:<8} "
                f"{r['accuracy']:<10.4f} {r['parse_failure_rate']:<8.3f} "
                f"{r['avg_inference_time_sec']:<8.2f}"
            )

    log.info("")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument("--model", type=str, help="Model name (see --list_models)")
    parser.add_argument("--sweep", type=str, choices=["small", "large", "all"],
                        help="Run sweep over model group")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--dataset", type=str, default="vcr",
                        choices=["vcr", "mmmu", "mathvista"])
    parser.add_argument("--vcr_dir", type=str, default="data/vcr")
    parser.add_argument("--subset_pct", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Exact number of examples to evaluate (overrides --subset_pct)")
    parser.add_argument("--subject", type=str, default=None,
                        help="MMMU only: subject config to load, e.g. 'Math'")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=[None, "4bit", "8bit"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--csv_path", type=str, default="results/baselines.csv")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Explicit log file path (default: auto-generated in logs/)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Batch size for GPU inference. 0=auto-detect from VRAM, 1=sequential")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore existing checkpoint and start fresh")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-example prompts/responses in console")
    parser.add_argument("--list_models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name in list_evaluators():
            print(f"  {name}")
        return

    # ---- Setup logging ----
    from src.logger import TRACE
    console_level = TRACE if args.verbose else logging.INFO
    setup_logger(
        name="vlm-eval",
        log_file=args.log_file,
        console_level=console_level,
    )

    log_system_info()

    # ---- Load from YAML config if provided ----
    if args.config:
        log.info(f"Loading config from: {args.config}")
        with open(args.config) as f:
            config = yaml.safe_load(f)
        log.info(f"Config contents: {json.dumps(config, indent=2)}")
        for key, val in config.items():
            if not getattr(args, key, None):
                setattr(args, key, val)

    # ---- Run ----
    if args.sweep:
        run_sweep(
            model_group=args.sweep,
            dataset_name=args.dataset,
            vcr_dir=args.vcr_dir,
            subset_pct=args.subset_pct,
            max_samples=args.max_samples,
            subject=args.subject,
            quantization=args.quantization,
            seed=args.seed,
            device=args.device,
            output_dir=args.output_dir,
            csv_path=args.csv_path,
            restart=args.restart,
            batch_size=args.batch_size,
        )
    elif args.model:
        results = run_single_evaluation(
            model_name=args.model,
            dataset_name=args.dataset,
            vcr_dir=args.vcr_dir,
            subset_pct=args.subset_pct,
            max_samples=args.max_samples,
            subject=args.subject,
            quantization=args.quantization,
            seed=args.seed,
            device=args.device,
            output_dir=args.output_dir,
            restart=args.restart,
            batch_size=args.batch_size,
        )
        append_to_csv(results, args.csv_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
