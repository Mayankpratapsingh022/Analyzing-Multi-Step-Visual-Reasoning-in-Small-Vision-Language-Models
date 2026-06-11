"""
Quantify VCR visual grounding with occlusion/box overlap metrics.

This script uses the existing Qwen2-VL occlusion extractor and VCR metadata
boxes to compare Q->A and QA->R on the same selected examples. It does not run
VCR validation; it consumes the existing Wave-2 per-example details JSON for
example selection and correctness strata.

Small validation run:
  python cloud_gpu_vcr/scripts/quantify_occlusion_overlap.py \
    --prepare-from-hf \
    --num-examples 20 \
    --model qwen2-vl-2b \
    --hf-id Qwen/Qwen2-VL-2B-Instruct \
    --out-dir results/occlusion_overlap_qwen2b_n20 \
    --dtype bf16 \
    --occlusion-grid 5 \
    --seed 42

Scale once diagnostics look aligned:
  python cloud_gpu_vcr/scripts/quantify_occlusion_overlap.py \
    --prepare-from-hf \
    --num-examples 500 \
    --model qwen2-vl-2b \
    --hf-id Qwen/Qwen2-VL-2B-Instruct \
    --out-dir results/occlusion_overlap_qwen2b_n500 \
    --dtype bf16 \
    --occlusion-grid 5 \
    --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_HF_DATASET_ID = "JaydeepR/vcr-mirror"
TARGET_REF_MODES = ("union", "rationale_only")
PRIMARY_METRIC = "activation_mass_in_target_boxes"
SECONDARY_METRICS = ("active_mask_iou_top20", "box_coverage_top20")
METRIC_COLUMNS = (PRIMARY_METRIC, *SECONDARY_METRICS)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def median(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else None


def correctness_stratum(record: dict) -> str:
    qa = bool(record.get("qa_correct"))
    r = bool(record.get("r_correct", record.get("rationale_correct")))
    if qa and r:
        return "both_succeeded"
    if (not qa) and (not r):
        return "both_failed"
    if (not qa) and r:
        return "qa_failed_qar_succeeded"
    return "qa_succeeded_qar_failed"


def valid_selection_record(record: dict) -> bool:
    if "annot_id" not in record:
        return False
    answer_label = record.get("answer_label")
    rationale_label = record.get("rationale_label")
    return isinstance(answer_label, int) and isinstance(rationale_label, int)


def pick_examples(
    details: dict,
    num_examples: int,
    seed: int,
    selection_strategy: str,
) -> list[dict]:
    per_example = [ex for ex in details.get("per_example", []) if valid_selection_record(ex)]
    if not per_example:
        raise SystemExit("details JSON has no usable per_example records")
    if num_examples <= 0:
        raise SystemExit("--num-examples must be a positive integer")

    rng = random.Random(seed)
    k = min(num_examples, len(per_example))
    if selection_strategy == "random":
        return rng.sample(per_example, k)

    groups: dict[str, list[dict]] = defaultdict(list)
    for ex in per_example:
        groups[correctness_stratum(ex)].append(ex)
    for records in groups.values():
        rng.shuffle(records)

    order = [
        "qa_failed_qar_succeeded",
        "both_succeeded",
        "both_failed",
        "qa_succeeded_qar_failed",
    ]
    selected: list[dict] = []
    while len(selected) < k and any(groups.values()):
        for name in order:
            if groups[name] and len(selected) < k:
                selected.append(groups[name].pop())
    return selected


def find_details_file(results_dir: Path, model_name: str, num_examples: int) -> Path | None:
    candidates = sorted(results_dir.glob(f"{model_name}_*_details.json"))
    matches = []
    for path in candidates:
        try:
            details = load_json(path)
        except Exception:
            continue
        if details.get("num_examples") == num_examples:
            matches.append(path)
    return matches[-1] if matches else None


def require_details_file(results_dir: Path, model_name: str, num_examples: int) -> Path:
    details_path = find_details_file(results_dir, model_name, num_examples)
    if details_path is None:
        raise SystemExit(
            f"no {model_name} details JSON with num_examples={num_examples} in {results_dir}. "
            "Run with --prepare-from-hf or download the details JSON first."
        )
    return details_path


def download_explicit_hf_details_file(
    *,
    hf_details_path: str,
    results_dir: Path,
    hf_dataset_id: str,
    hf_raw_dir: Path,
    token: str | None,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for --hf-details-path. "
            "Run cloud_gpu_vcr/scripts/bootstrap_cloud.sh first."
        ) from exc

    print(f"[occ-overlap] downloading explicit HF details file: {hf_details_path}")
    downloaded = Path(
        hf_hub_download(
            repo_id=hf_dataset_id,
            repo_type="dataset",
            filename=hf_details_path,
            local_dir=str(hf_raw_dir),
            token=token,
        )
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / downloaded.name
    if not target.exists():
        import shutil

        shutil.copy2(downloaded, target)
    return target


def parse_torch_dtype(name: str, device: str):
    import torch

    value = name.lower()
    if value == "auto":
        return None
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise SystemExit(f"unsupported --dtype {name!r}; use auto, bf16, fp16, or fp32")


def build_annot_index(dataset, annot_ids: set[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for i, ann in enumerate(dataset.annotations):
        annot_id = ann.get("annot_id")
        if annot_id in annot_ids:
            index[annot_id] = i
    return index


def build_prompts(example: dict) -> tuple[str, str]:
    from src.models.vlm_evaluator import VLMEvaluator

    qa_prompt = VLMEvaluator.format_vcr_prompt(
        example["question"], example["answer_choices"], task="qa"
    )
    answer_label = int(example["answer_label"])
    correct_answer_text = example["answer_choices"][answer_label]
    qar_prompt = VLMEvaluator.format_vcr_prompt(
        f"{example['question']} Answer: {correct_answer_text}. Why?",
        example["rationale_choices"],
        task="qar",
    )
    return qa_prompt, qar_prompt


def refs_for_mode(example: dict, target_ref_mode: str) -> list[int]:
    metadata = example.get("metadata", {})
    answer_label = int(example["answer_label"])
    rationale_label = int(example["rationale_label"])
    answer_refs = metadata.get("answer_choice_object_refs", [])
    rationale_refs = metadata.get("rationale_choice_object_refs", [])

    if rationale_label < 0 or rationale_label >= len(rationale_refs):
        return []
    rationale_only = list(rationale_refs[rationale_label])
    if target_ref_mode == "rationale_only":
        return sorted({int(ref) for ref in rationale_only if isinstance(ref, int)})

    refs: list[int] = []
    refs.extend(metadata.get("question_object_refs", []))
    if 0 <= answer_label < len(answer_refs):
        refs.extend(answer_refs[answer_label])
    refs.extend(rationale_only)
    return sorted({int(ref) for ref in refs if isinstance(ref, int)})


def parse_box(raw_box) -> tuple[float, float, float, float] | None:
    if isinstance(raw_box, dict):
        if {"x1", "y1", "x2", "y2"}.issubset(raw_box):
            vals = (raw_box["x1"], raw_box["y1"], raw_box["x2"], raw_box["y2"])
        elif {"xmin", "ymin", "xmax", "ymax"}.issubset(raw_box):
            vals = (raw_box["xmin"], raw_box["ymin"], raw_box["xmax"], raw_box["ymax"])
        elif {"left", "top", "right", "bottom"}.issubset(raw_box):
            vals = (raw_box["left"], raw_box["top"], raw_box["right"], raw_box["bottom"])
        elif {"x", "y", "w", "h"}.issubset(raw_box):
            try:
                x = float(raw_box["x"])
                y = float(raw_box["y"])
                w = float(raw_box["w"])
                h = float(raw_box["h"])
            except (TypeError, ValueError):
                return None
            vals = (x, y, x + w, y + h)
        else:
            return None
    elif isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
        vals = raw_box[:4]
    else:
        return None

    try:
        x1, y1, x2, y2 = (float(v) for v in vals)
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def clip_box(
    box: tuple[float, float, float, float],
    image_size_wh: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    w_img, h_img = image_size_wh
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), float(w_img))
    x2 = min(max(x2, 0.0), float(w_img))
    y1 = min(max(y1, 0.0), float(h_img))
    y2 = min(max(y2, 0.0), float(h_img))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def boxes_for_refs(
    metadata: dict,
    refs: list[int],
    image_size_wh: tuple[int, int],
) -> tuple[list[tuple[float, float, float, float]], list[int]]:
    boxes = metadata.get("boxes", [])
    selected: list[tuple[float, float, float, float]] = []
    invalid_refs: list[int] = []
    for ref in refs:
        if ref < 0 or ref >= len(boxes):
            invalid_refs.append(ref)
            continue
        parsed = parse_box(boxes[ref])
        clipped = clip_box(parsed, image_size_wh) if parsed is not None else None
        if clipped is None:
            invalid_refs.append(ref)
            continue
        selected.append(clipped)
    return selected, invalid_refs


def rect_union_area(rects: list[tuple[float, float, float, float]]) -> float:
    valid = [(x1, y1, x2, y2) for x1, y1, x2, y2 in rects if x2 > x1 and y2 > y1]
    if not valid:
        return 0.0
    xs = sorted({x1 for x1, _, x2, _ in valid} | {x2 for _, _, x2, _ in valid})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = [
            (y1, y2)
            for x1, y1, x2, y2 in valid
            if x1 < right and x2 > left
        ]
        if not intervals:
            continue
        intervals.sort()
        merged_len = 0.0
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged_len += cur_end - cur_start
                cur_start, cur_end = start, end
        merged_len += cur_end - cur_start
        area += (right - left) * merged_len
    return float(area)


def intersection_with_box_union(
    rect: tuple[float, float, float, float],
    boxes: list[tuple[float, float, float, float]],
) -> float:
    x1, y1, x2, y2 = rect
    intersections: list[tuple[float, float, float, float]] = []
    for bx1, by1, bx2, by2 in boxes:
        ix1 = max(x1, bx1)
        iy1 = max(y1, by1)
        ix2 = min(x2, bx2)
        iy2 = min(y2, by2)
        if ix2 > ix1 and iy2 > iy1:
            intersections.append((ix1, iy1, ix2, iy2))
    return rect_union_area(intersections)


def grid_cell_rects(
    grid_hw: tuple[int, int],
    image_size_wh: tuple[int, int],
) -> list[tuple[float, float, float, float]]:
    rows, cols = grid_hw
    w_img, h_img = image_size_wh
    rects = []
    for r in range(rows):
        for c in range(cols):
            y1 = int(round(r * h_img / rows))
            y2 = int(round((r + 1) * h_img / rows))
            x1 = int(round(c * w_img / cols))
            x2 = int(round((c + 1) * w_img / cols))
            rects.append((float(x1), float(y1), float(x2), float(y2)))
    return rects


def top_cell_mask(heatmap: np.ndarray, fraction: float) -> np.ndarray:
    flat = np.asarray(heatmap, dtype=np.float64).reshape(-1)
    n = flat.size
    k = max(1, int(math.ceil(n * fraction)))
    order = np.argsort(flat)[::-1]
    active = np.zeros(n, dtype=bool)
    active[order[:k]] = True
    return active


def compute_overlap_metrics(
    heatmap: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    image_size_wh: tuple[int, int],
    active_fraction: float,
) -> dict:
    if not boxes:
        return {
            PRIMARY_METRIC: None,
            "active_mask_iou_top20": None,
            "box_coverage_top20": None,
            "quality_warning": "no_target_boxes",
        }

    heat = np.nan_to_num(heatmap.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    heat = np.maximum(heat, 0.0)
    total_mass = float(heat.sum())
    if total_mass <= 1e-12:
        return {
            PRIMARY_METRIC: None,
            "active_mask_iou_top20": None,
            "box_coverage_top20": None,
            "quality_warning": "blank_or_zero_occlusion_heatmap",
        }

    rects = grid_cell_rects(tuple(heat.shape), image_size_wh)
    overlap_areas = np.array(
        [intersection_with_box_union(rect, boxes) for rect in rects],
        dtype=np.float64,
    )
    cell_areas = np.array(
        [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in rects],
        dtype=np.float64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        target_fraction_per_cell = np.divide(
            overlap_areas,
            cell_areas,
            out=np.zeros_like(overlap_areas),
            where=cell_areas > 0,
        )

    mass_in_target = float((heat.reshape(-1) * target_fraction_per_cell).sum())
    activation_mass = mass_in_target / total_mass

    active = top_cell_mask(heat, active_fraction)
    active_area = float(cell_areas[active].sum())
    active_target_intersection = float(overlap_areas[active].sum())
    target_area = rect_union_area(boxes)
    union_area = active_area + target_area - active_target_intersection
    active_iou = active_target_intersection / union_area if union_area > 0 else None
    box_coverage = active_target_intersection / target_area if target_area > 0 else None

    return {
        PRIMARY_METRIC: float(activation_mass),
        "active_mask_iou_top20": float(active_iou) if active_iou is not None else None,
        "box_coverage_top20": float(box_coverage) if box_coverage is not None else None,
        "quality_warning": None,
        "target_area_fraction": float(target_area / (image_size_wh[0] * image_size_wh[1])),
        "active_area_fraction_top20": float(active_area / (image_size_wh[0] * image_size_wh[1])),
    }


def combined_warning(*parts: str | None) -> str | None:
    values = [p for p in parts if p]
    return ";".join(values) if values else None


def occlusion_row(
    *,
    pick: dict,
    annot_id: str,
    hf_id: str,
    prompt_mode: str,
    target_ref_mode: str,
    target_refs: list[int],
    target_boxes: list[tuple[float, float, float, float]],
    invalid_target_refs: list[int],
    image: Image.Image,
    occlusion_result,
    active_fraction: float,
) -> dict:
    metrics = compute_overlap_metrics(
        occlusion_result.heatmap,
        target_boxes,
        image.size,
        active_fraction,
    )
    diag = occlusion_result.diagnostics or {}
    return {
        "annot_id": annot_id,
        "hf_id": hf_id,
        "prompt_mode": prompt_mode,
        "qa_correct": bool(pick.get("qa_correct")),
        "r_correct": bool(pick.get("r_correct", pick.get("rationale_correct"))),
        "correctness_stratum": correctness_stratum(pick),
        "answer_label": pick.get("answer_label"),
        "rationale_label": pick.get("rationale_label"),
        "pred_answer": pick.get("pred_answer"),
        "pred_rationale": pick.get("pred_rationale"),
        "target_ref_mode": target_ref_mode,
        "target_refs": json.dumps(target_refs),
        "invalid_target_refs": json.dumps(invalid_target_refs),
        "num_target_boxes": len(target_boxes),
        "image_width": image.size[0],
        "image_height": image.size[1],
        "occlusion_grid": f"{occlusion_result.grid_hw[0]}x{occlusion_result.grid_hw[1]}",
        "target_token": occlusion_result.target_token,
        "target_token_id": occlusion_result.target_token_id,
        "base_logprob": occlusion_result.base_logprob,
        PRIMARY_METRIC: metrics.get(PRIMARY_METRIC),
        "active_mask_iou_top20": metrics.get("active_mask_iou_top20"),
        "box_coverage_top20": metrics.get("box_coverage_top20"),
        "target_area_fraction": metrics.get("target_area_fraction"),
        "active_area_fraction_top20": metrics.get("active_area_fraction_top20"),
        "occlusion_entropy": diag.get("normalized_entropy"),
        "occlusion_border_mass": diag.get("border_mass"),
        "occlusion_peak_to_mean": diag.get("peak_to_mean"),
        "quality_warning": combined_warning(
            metrics.get("quality_warning"),
            diag.get("quality_warning"),
            "invalid_target_refs" if invalid_target_refs else None,
        ),
    }


def paired_values(
    rows: list[dict],
    target_ref_mode: str,
    metric: str,
) -> list[dict]:
    by_annot: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row.get("target_ref_mode") != target_ref_mode:
            continue
        value = safe_float(row.get(metric))
        if value is None:
            continue
        by_annot[row["annot_id"]][row["prompt_mode"]] = row

    pairs = []
    for annot_id, modes in by_annot.items():
        if "qa" not in modes or "qar" not in modes:
            continue
        qa_value = safe_float(modes["qa"].get(metric))
        qar_value = safe_float(modes["qar"].get(metric))
        if qa_value is None or qar_value is None:
            continue
        pairs.append({
            "annot_id": annot_id,
            "qa": qa_value,
            "qar": qar_value,
            "delta": qar_value - qa_value,
            "stratum": modes["qa"].get("correctness_stratum", "unknown"),
        })
    return pairs


def bootstrap_ci(
    deltas: np.ndarray,
    seed: int,
    n_bootstrap: int,
) -> list[float | None]:
    if deltas.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    n = deltas.size
    for i in range(n_bootstrap):
        means[i] = rng.choice(deltas, size=n, replace=True).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def paired_signflip_p_value(
    deltas: np.ndarray,
    seed: int,
    n_samples: int,
) -> float | None:
    if deltas.size == 0:
        return None
    nonzero = deltas[np.abs(deltas) > 1e-12]
    if nonzero.size == 0:
        return 1.0
    observed = abs(float(nonzero.mean()))
    rng = np.random.default_rng(seed)
    if nonzero.size <= 18:
        count = 0
        total = 2 ** nonzero.size
        for mask in range(total):
            signs = np.array(
                [1.0 if (mask >> i) & 1 else -1.0 for i in range(nonzero.size)],
                dtype=np.float64,
            )
            if abs(float((nonzero * signs).mean())) >= observed - 1e-12:
                count += 1
        return float(count / total)

    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_samples, nonzero.size))
    random_means = np.abs((signs * nonzero).mean(axis=1))
    return float((np.count_nonzero(random_means >= observed - 1e-12) + 1) / (n_samples + 1))


def summarize_pairs(
    pairs: list[dict],
    seed: int,
    n_bootstrap: int,
    n_permutation: int,
) -> dict:
    qa_values = np.array([p["qa"] for p in pairs], dtype=np.float64)
    qar_values = np.array([p["qar"] for p in pairs], dtype=np.float64)
    deltas = np.array([p["delta"] for p in pairs], dtype=np.float64)
    return {
        "n_valid_pairs": int(len(pairs)),
        "mean_qa": mean(qa_values),
        "median_qa": median(qa_values),
        "mean_qar": mean(qar_values),
        "median_qar": median(qar_values),
        "mean_delta_qar_minus_qa": mean(deltas),
        "median_delta_qar_minus_qa": median(deltas),
        "bootstrap_ci95_delta": bootstrap_ci(deltas, seed, n_bootstrap),
        "p_value_paired_signflip": paired_signflip_p_value(deltas, seed, n_permutation),
    }


def build_summary(
    *,
    rows: list[dict],
    selected: list[dict],
    skipped: list[dict],
    args: argparse.Namespace,
    details_path: Path,
    elapsed_sec: float,
) -> dict:
    metric_summaries: dict[str, dict[str, dict]] = {}
    for target_ref_mode in TARGET_REF_MODES:
        metric_summaries[target_ref_mode] = {}
        for metric in METRIC_COLUMNS:
            pairs = paired_values(rows, target_ref_mode, metric)
            metric_summaries[target_ref_mode][metric] = summarize_pairs(
                pairs,
                args.seed,
                args.bootstrap_samples,
                args.permutation_samples,
            )

    primary_pairs = paired_values(rows, "union", PRIMARY_METRIC)
    primary = summarize_pairs(
        primary_pairs,
        args.seed,
        args.bootstrap_samples,
        args.permutation_samples,
    )

    stratified_results = {}
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for pair in primary_pairs:
        by_stratum[pair["stratum"]].append(pair)
    for stratum, pairs in sorted(by_stratum.items()):
        stratified_results[stratum] = summarize_pairs(
            pairs,
            args.seed,
            args.bootstrap_samples,
            args.permutation_samples,
        )

    skip_reasons = Counter(item["reason"] for item in skipped)
    valid_annot_ids = sorted({p["annot_id"] for p in primary_pairs})
    return {
        "model": args.model,
        "hf_id": args.hf_id,
        "source_details": str(details_path),
        "n_total": len(selected),
        "n_valid": len(valid_annot_ids),
        "n_skipped": len(selected) - len(valid_annot_ids),
        "n_skipped_before_occlusion": len(skipped),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "target_ref_mode_primary": "union",
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": list(SECONDARY_METRICS),
        "mean_qa_overlap": primary["mean_qa"],
        "mean_qar_overlap": primary["mean_qar"],
        "median_qa_overlap": primary["median_qa"],
        "median_qar_overlap": primary["median_qar"],
        "mean_delta_qar_minus_qa": primary["mean_delta_qar_minus_qa"],
        "median_delta_qar_minus_qa": primary["median_delta_qar_minus_qa"],
        "bootstrap_ci95_delta": primary["bootstrap_ci95_delta"],
        "p_value_paired_test": primary["p_value_paired_signflip"],
        "stratified_results": stratified_results,
        "metric_summaries": metric_summaries,
        "selection": {
            "strategy": args.selection_strategy,
            "seed": args.seed,
            "num_examples_requested": args.num_examples,
            "full_val_n": args.full_val_n,
        },
        "occlusion": {
            "grid": args.occlusion_grid,
            "fill": args.occlusion_fill,
            "active_fraction": args.active_fraction,
        },
        "runtime": {
            "elapsed_sec": elapsed_sec,
            "elapsed_min": elapsed_sec / 60.0,
        },
    }


def draw_boxes(ax, boxes, color: str, linewidth: float = 2.0) -> None:
    import matplotlib.patches as patches

    for x1, y1, x2, y2 in boxes:
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
            )
        )


def draw_top_cells(ax, heatmap: np.ndarray, image_size_wh: tuple[int, int], active_fraction: float) -> None:
    import matplotlib.patches as patches

    active = top_cell_mask(heatmap, active_fraction)
    rects = grid_cell_rects(tuple(heatmap.shape), image_size_wh)
    for is_active, (x1, y1, x2, y2) in zip(active, rects):
        if not is_active:
            continue
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="cyan",
                linewidth=1.8,
            )
        )


def render_paired_diagnostic(
    *,
    image: Image.Image,
    qa_heatmap: np.ndarray,
    qar_heatmap: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    annot_id: str,
    out_path: Path,
    active_fraction: float,
) -> None:
    import matplotlib.pyplot as plt
    from src.analysis.attention_extractor import normalize_heatmap

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mode, heatmap in zip(axes, ["Q->A", "QA->R"], [qa_heatmap, qar_heatmap]):
        ax.imshow(image)
        overlay = Image.fromarray((normalize_heatmap(heatmap) * 255).astype(np.uint8))
        overlay = overlay.resize(image.size, Image.BILINEAR)
        ax.imshow(np.array(overlay), cmap="jet", alpha=0.45)
        draw_boxes(ax, boxes, color="lime")
        draw_top_cells(ax, heatmap, image.size, active_fraction)
        ax.set_title(mode, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{annot_id} | green=target boxes | cyan=top occlusion cells", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def csv_fieldnames() -> list[str]:
    return [
        "annot_id",
        "hf_id",
        "prompt_mode",
        "qa_correct",
        "r_correct",
        "correctness_stratum",
        "answer_label",
        "rationale_label",
        "pred_answer",
        "pred_rationale",
        "target_ref_mode",
        "target_refs",
        "invalid_target_refs",
        "num_target_boxes",
        "image_width",
        "image_height",
        "occlusion_grid",
        "target_token",
        "target_token_id",
        "base_logprob",
        PRIMARY_METRIC,
        "active_mask_iou_top20",
        "box_coverage_top20",
        "target_area_fraction",
        "active_area_fraction_top20",
        "occlusion_entropy",
        "occlusion_border_mass",
        "occlusion_peak_to_mean",
        "quality_warning",
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = csv_fieldnames()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2-vl-2b")
    p.add_argument("--hf-id", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--vcr-dir", default="/workspace/data/vcr")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="results/occlusion_overlap_qwen2b_n50")
    p.add_argument("--hf-dataset-id", default=DEFAULT_HF_DATASET_ID)
    p.add_argument("--hf-raw-dir", default="/workspace/data/vcr_raw")
    p.add_argument("--hf-token", default=None, help="Optional HF token. Defaults to HF_TOKEN.")
    p.add_argument(
        "--details-file",
        default=None,
        help=(
            "Use this local details JSON directly for annot_id selection and correctness labels. "
            "Overrides --model/--full-val-n details discovery."
        ),
    )
    p.add_argument(
        "--hf-details-path",
        default=None,
        help=(
            "Download and use this exact details JSON path from --hf-dataset-id. "
            "Example: wave2_latest/results/qwen2-vl-7b_vcr_20260517_151410_details.json"
        ),
    )
    p.add_argument("--prepare-from-hf", action="store_true")
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Download selected annotations/images/metadata and exit before model loading.",
    )
    p.add_argument("--full-val-n", type=int, default=26534)
    p.add_argument("--num-examples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--selection-strategy",
        choices=["random", "balanced_strata"],
        default="random",
        help="Use random for representative n=500 runs; balanced_strata is useful for smoke checks.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    p.add_argument("--device-map", default=None)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--min-pixels", type=int, default=None)
    p.add_argument("--max-pixels", type=int, default=401408)
    p.add_argument("--occlusion-grid", type=int, default=5)
    p.add_argument("--occlusion-fill", default="mean", choices=["mean", "gray", "black"])
    p.add_argument(
        "--active-fraction",
        type=float,
        default=0.20,
        help="Fraction of occlusion cells used for active-mask secondary metrics.",
    )
    p.add_argument("--diagnostic-overlays", type=int, default=5)
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--permutation-samples", type=int, default=10000)
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_only and not args.prepare_from_hf:
        raise SystemExit("--prepare-only requires --prepare-from-hf")
    if args.details_file and args.hf_details_path:
        raise SystemExit("Use only one of --details-file or --hf-details-path")
    if args.occlusion_grid <= 0:
        raise SystemExit("--occlusion-grid must be positive for this metric")
    if not (0.0 < args.active_fraction <= 1.0):
        raise SystemExit("--active-fraction must be in (0, 1]")
    if args.bootstrap_samples <= 0 or args.permutation_samples <= 0:
        raise SystemExit("--bootstrap-samples and --permutation-samples must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    vcr_dir = Path(args.vcr_dir)
    hf_raw_dir = Path(args.hf_raw_dir)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    print(f"[occ-overlap] selecting examples from details JSON")
    prepare_picked_vcr_files_from_hf = None
    if args.details_file:
        details_path = Path(args.details_file)
        if not details_path.exists():
            raise SystemExit(f"--details-file does not exist: {details_path}")
    elif args.hf_details_path:
        details_path = download_explicit_hf_details_file(
            hf_details_path=args.hf_details_path,
            results_dir=results_dir,
            hf_dataset_id=args.hf_dataset_id,
            hf_raw_dir=hf_raw_dir,
            token=hf_token,
        )
    elif args.prepare_from_hf:
        from extract_attention_sample import (
            ensure_details_from_hf,
            prepare_picked_vcr_files_from_hf,
        )

        details_path = ensure_details_from_hf(
            results_dir=results_dir,
            model_name=args.model,
            num_examples=args.full_val_n,
            hf_dataset_id=args.hf_dataset_id,
            hf_raw_dir=hf_raw_dir,
            token=hf_token,
        )
    else:
        details_path = require_details_file(results_dir, args.model, args.full_val_n)
    details = load_json(details_path)
    details_model = details.get("model_name")
    details_n = details.get("num_examples")
    print(
        f"[occ-overlap] source details: {details_path} "
        f"(model_name={details_model}, num_examples={details_n})"
    )
    selected = pick_examples(details, args.num_examples, args.seed, args.selection_strategy)
    selected_annot_ids = [ex["annot_id"] for ex in selected]
    print(
        f"[occ-overlap] selected {len(selected)} examples "
        f"(strategy={args.selection_strategy}, seed={args.seed})"
    )

    selected_payload = {
        "model": args.model,
        "hf_id": args.hf_id,
        "source_details": str(details_path),
        "selection_strategy": args.selection_strategy,
        "seed": args.seed,
        "num_examples_requested": args.num_examples,
        "selected_annot_ids": selected_annot_ids,
        "selected_records": [
            {
                "annot_id": ex.get("annot_id"),
                "qa_correct": ex.get("qa_correct"),
                "r_correct": ex.get("r_correct", ex.get("rationale_correct")),
                "answer_label": ex.get("answer_label"),
                "rationale_label": ex.get("rationale_label"),
                "correctness_stratum": correctness_stratum(ex),
            }
            for ex in selected
        ],
    }
    write_json(out_dir / "selected_annot_ids.json", selected_payload)

    if args.prepare_from_hf:
        if prepare_picked_vcr_files_from_hf is None:
            from extract_attention_sample import prepare_picked_vcr_files_from_hf

        prepare_picked_vcr_files_from_hf(
            picks=selected,
            vcr_dir=vcr_dir,
            hf_dataset_id=args.hf_dataset_id,
            hf_raw_dir=hf_raw_dir,
            split="val",
            token=hf_token,
        )
        if args.prepare_only:
            print("[occ-overlap] prepare-only complete; exiting before model load.")
            return

    import torch
    from src.analysis.attention_extractor import (
        extract_occlusion_qwen2vl,
        load_qwen2vl_eager,
    )
    from src.data_loader import VCRDataset

    ds_full = VCRDataset(vcr_dir=args.vcr_dir, split="val")
    by_annot = build_annot_index(ds_full, set(selected_annot_ids))
    print(f"[occ-overlap] resolved {len(by_annot)}/{len(selected_annot_ids)} annot_ids")

    dtype = parse_torch_dtype(args.dtype, args.device)
    dtype_name = "auto" if dtype is None else str(dtype).replace("torch.", "")
    load_mode = "4bit" if args.load_in_4bit else "8bit" if args.load_in_8bit else "full"
    print(
        f"[occ-overlap] loading {args.hf_id} on {args.device}, dtype={dtype_name}, "
        f"load_mode={load_mode}, device_map={args.device_map}, max_pixels={args.max_pixels}"
    )
    model, processor = load_qwen2vl_eager(
        args.hf_id,
        device=args.device,
        dtype=dtype,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    rows: list[dict] = []
    skipped: list[dict] = []
    diagnostics_written = 0
    start = time.time()

    for ordinal, pick in enumerate(selected, 1):
        annot_id = pick["annot_id"]
        idx = by_annot.get(annot_id)
        if idx is None:
            skipped.append({"annot_id": annot_id, "reason": "annot_id_not_found"})
            print(f"[occ-overlap] {ordinal}/{len(selected)} {annot_id}: skip annot_id_not_found")
            continue

        try:
            example = ds_full[idx]
            image = example["image"]
            union_refs = refs_for_mode(example, "union")
            union_boxes, union_invalid_refs = boxes_for_refs(
                example.get("metadata", {}),
                union_refs,
                image.size,
            )
            if not union_boxes:
                skipped.append({"annot_id": annot_id, "reason": "no_union_target_boxes"})
                print(f"[occ-overlap] {ordinal}/{len(selected)} {annot_id}: skip no target boxes")
                continue

            qa_prompt, qar_prompt = build_prompts(example)
            print(
                f"[occ-overlap] {ordinal}/{len(selected)} {annot_id}: "
                f"Q->A then QA->R occlusion"
            )
            occ_qa = extract_occlusion_qwen2vl(
                model,
                processor,
                image,
                qa_prompt,
                grid_hw=(args.occlusion_grid, args.occlusion_grid),
                fill=args.occlusion_fill,
            )
            occ_qar = extract_occlusion_qwen2vl(
                model,
                processor,
                image,
                qar_prompt,
                grid_hw=(args.occlusion_grid, args.occlusion_grid),
                fill=args.occlusion_fill,
            )

            union_rows_by_mode: dict[str, dict] = {}
            for target_ref_mode in TARGET_REF_MODES:
                refs = refs_for_mode(example, target_ref_mode)
                boxes, invalid_refs = boxes_for_refs(example.get("metadata", {}), refs, image.size)
                if target_ref_mode == "union":
                    invalid_refs = sorted(set(invalid_refs + union_invalid_refs))
                for prompt_mode, occ in (("qa", occ_qa), ("qar", occ_qar)):
                    row = occlusion_row(
                        pick=pick,
                        annot_id=annot_id,
                        hf_id=args.hf_id,
                        prompt_mode=prompt_mode,
                        target_ref_mode=target_ref_mode,
                        target_refs=refs,
                        target_boxes=boxes,
                        invalid_target_refs=invalid_refs,
                        image=image,
                        occlusion_result=occ,
                        active_fraction=args.active_fraction,
                    )
                    rows.append(row)
                    if target_ref_mode == "union":
                        union_rows_by_mode[prompt_mode] = row

            if diagnostics_written < args.diagnostic_overlays:
                diag_path = out_dir / "diagnostics" / f"{annot_id}_union_overlay.png"
                render_paired_diagnostic(
                    image=image,
                    qa_heatmap=occ_qa.heatmap,
                    qar_heatmap=occ_qar.heatmap,
                    boxes=union_boxes,
                    annot_id=annot_id,
                    out_path=diag_path,
                    active_fraction=args.active_fraction,
                )
                diagnostics_written += 1

            qa_overlap = union_rows_by_mode.get("qa", {}).get(PRIMARY_METRIC)
            qar_overlap = union_rows_by_mode.get("qar", {}).get(PRIMARY_METRIC)
            print(
                f"[occ-overlap] {annot_id}: union overlap "
                f"qa={safe_float(qa_overlap)} qar={safe_float(qar_overlap)}"
            )
        except Exception as exc:
            skipped.append({"annot_id": annot_id, "reason": type(exc).__name__, "error": str(exc)})
            print(f"[occ-overlap] {ordinal}/{len(selected)} {annot_id}: error {type(exc).__name__}: {exc}")
            if args.fail_fast:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed = time.time() - start
    write_csv(out_dir / "overlap_metrics.csv", rows)
    summary = build_summary(
        rows=rows,
        selected=selected,
        skipped=skipped,
        args=args,
        details_path=details_path,
        elapsed_sec=elapsed,
    )
    write_json(out_dir / "summary.json", summary)
    selected_payload["processed_annot_ids"] = sorted({row["annot_id"] for row in rows})
    selected_payload["skipped"] = skipped
    write_json(out_dir / "selected_annot_ids.json", selected_payload)

    print(f"[occ-overlap] metrics: {out_dir / 'overlap_metrics.csv'}")
    print(f"[occ-overlap] summary: {out_dir / 'summary.json'}")
    print(
        "[occ-overlap] primary union delta "
        f"QA->R - Q->A = {summary.get('mean_delta_qar_minus_qa')} "
        f"over n={summary.get('n_valid')} valid pairs"
    )


if __name__ == "__main__":
    main()
