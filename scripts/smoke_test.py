#!/usr/bin/env python3
"""
Smoke test: load each model, run a single inference, report pass/fail.

Logs full details (prompt, response, timing) to logs/smoke_test_<timestamp>.log.
Console shows a concise pass/fail summary.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --models moondream2 qwen2-vl-2b
    python scripts/smoke_test.py --skip-api
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from src.logger import setup_logger, format_time, TRACE
from src.models import get_evaluator, list_evaluators


log = logging.getLogger("vlm-eval")


def create_test_image() -> Image.Image:
    """Create a simple test image (red square on white background)."""
    img = Image.new("RGB", (336, 336), (255, 255, 255))
    for x in range(100, 236):
        for y in range(100, 236):
            img.putpixel((x, y), (255, 0, 0))
    return img


TEST_PROMPT = (
    "What color is the shape in the center of this image?\n"
    "A. Blue\nB. Red\nC. Green\nD. Yellow\n\n"
    "Answer with only the letter."
)

QUANTIZE_MODELS = {"llava-1.5-13b": "8bit", "internvl2-26b": "4bit", "llava-1.6-34b": "4bit"}
API_MODELS = {"gpt-4o", "claude"}


def smoke_test_model(
    model_name: str,
    image: Image.Image,
    device: str = "cuda",
) -> dict:
    """Test a single model. Returns status dict."""
    result = {
        "model": model_name,
        "status": "FAIL",
        "output": "",
        "time_sec": 0,
        "vram_gb": 0,
    }

    quant = QUANTIZE_MODELS.get(model_name)
    model_device = "cpu" if model_name in API_MODELS else device

    log.info(f"--- Testing: {model_name} ---")
    log.info(f"  Device: {model_device}  Quantization: {quant or 'none'}")
    log.log(TRACE, f"  Prompt being sent:\n{TEST_PROMPT}")

    try:
        evaluator = get_evaluator(model_name)

        log.info(f"  Loading model...")
        t0 = time.time()
        evaluator.load_model(device=model_device, quantization=quant)
        load_time = time.time() - t0
        vram_after_load = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        log.info(f"  Loaded in {format_time(load_time)} | VRAM: {vram_after_load:.2f} GB")

        log.info(f"  Running inference...")
        t1 = time.time()
        output = evaluator.generate(image, TEST_PROMPT, max_new_tokens=64)
        gen_time = time.time() - t1

        vram = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0

        result["status"] = "PASS"
        result["output"] = output.strip()[:200]
        result["time_sec"] = round(load_time + gen_time, 2)
        result["load_time"] = round(load_time, 2)
        result["gen_time"] = round(gen_time, 2)
        result["vram_gb"] = round(vram, 2)
        result["quantization"] = quant or "none"

        log.info(f"  PASS | Inference: {gen_time:.2f}s | VRAM: {vram:.2f} GB")
        log.info(f"  Response: \"{output.strip()[:100]}\"")
        log.log(TRACE, f"  Full response:\n{output}")

        del evaluator.model
        if hasattr(evaluator, "processor") and evaluator.processor:
            del evaluator.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        result["status"] = "FAIL"
        result["output"] = str(e)[:300]
        log.error(f"  FAIL: {e}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific models to test. Default: all.")
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip API-based models (GPT-4o, Claude)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full prompts/responses in console")
    args = parser.parse_args()

    console_level = TRACE if args.verbose else logging.INFO
    setup_logger(name="vlm-eval", console_level=console_level)

    models = args.models or list_evaluators()
    if args.skip_api:
        models = [m for m in models if m not in API_MODELS]

    image = create_test_image()

    log.info(f"{'='*70}")
    log.info(f"  SMOKE TEST")
    log.info(f"  Models: {len(models)}")
    log.info(f"  Device: {args.device}")
    log.info(f"  CUDA  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"  GPU   : {torch.cuda.get_device_name()}")
        log.info(f"  VRAM  : {torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f} GB")
    log.info(f"  Test prompt: \"{TEST_PROMPT[:60]}...\"")
    log.info(f"{'='*70}")

    results = []
    total_start = time.time()

    for i, model_name in enumerate(models, 1):
        log.info(f"\n[{i}/{len(models)}] {model_name}")
        r = smoke_test_model(model_name, image, device=args.device)
        results.append(r)

    total_time = time.time() - total_start
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = len(results) - passed

    log.info("")
    log.info(f"{'='*70}")
    log.info(f"  SMOKE TEST SUMMARY")
    log.info(f"  Passed: {passed}/{len(results)}  Failed: {failed}/{len(results)}")
    log.info(f"  Total time: {format_time(total_time)}")
    log.info(f"{'='*70}")
    log.info("")
    log.info(
        f"  {'Model':<22} {'Status':<8} {'VRAM(GB)':<10} {'Quant':<8} "
        f"{'Load(s)':<9} {'Infer(s)':<9} {'Response'}"
    )
    log.info(f"  {'-'*100}")
    for r in results:
        log.info(
            f"  {r['model']:<22} {r['status']:<8} "
            f"{r.get('vram_gb', '-'):<10} "
            f"{r.get('quantization', '-'):<8} "
            f"{r.get('load_time', '-'):<9} "
            f"{r.get('gen_time', '-'):<9} "
            f"{r['output'][:40]}"
        )

    # Save results JSON
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"smoke_test_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
