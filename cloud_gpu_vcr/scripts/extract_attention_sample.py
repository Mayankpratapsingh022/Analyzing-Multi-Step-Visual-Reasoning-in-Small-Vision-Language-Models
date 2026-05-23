"""
Generate a small attention-map sample for mentor review.

Picks 5 correct + 5 incorrect VCR val examples from the Wave-2 details JSON
(qwen2-vl-2b run), runs each through `extract_attention_qwen2vl` for both the
Q->A and QA->R prompts, and saves visualizations as a single multi-panel PNG.

The picks are deterministic (seed=42) so the same 10 examples come back if
the script is re-run.

Outputs:
  results/attention_sample/attention_sample.png       — grid of all 10 examples
  results/attention_sample/<annot_id>_qa.png          — per-example Q->A panel
  results/attention_sample/<annot_id>_qar.png         — per-example QA->R panel
  results/attention_sample/sample_metadata.json       — what was picked + per-example metrics

Usage:
  python cloud_gpu_vcr/scripts/extract_attention_sample.py
  python cloud_gpu_vcr/scripts/extract_attention_sample.py --n-correct 5 --n-incorrect 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.attention_extractor import (  # noqa: E402
    extract_attention_qwen2vl,
    load_qwen2vl_eager,
    normalize_heatmap,
    upsample_heatmap_to_image,
)
from src.data_loader import VCRDataset  # noqa: E402
from src.models.vlm_evaluator import VLMEvaluator  # noqa: E402


def find_details_file(results_dir: Path, model_name: str, num_examples: int) -> Path:
    candidates = sorted(results_dir.glob(f"{model_name}_*_details.json"))
    matches = []
    for p in candidates:
        try:
            with open(p) as f:
                d = json.load(f)
            if d.get("num_examples") == num_examples:
                matches.append(p)
        except Exception:
            continue
    if not matches:
        raise SystemExit(
            f"no {model_name} details JSON with num_examples={num_examples} in {results_dir}"
        )
    return matches[-1]


def pick_examples(details: dict, n_correct: int, n_incorrect: int, seed: int) -> list[dict]:
    per_ex = details["per_example"]
    correct = [ex for ex in per_ex if ex.get("qa_correct")]
    wrong = [ex for ex in per_ex if not ex.get("qa_correct")]
    rng = random.Random(seed)
    pick_c = rng.sample(correct, min(n_correct, len(correct)))
    pick_w = rng.sample(wrong, min(n_incorrect, len(wrong)))
    return pick_c + pick_w


def render_panel(image: Image.Image, heatmap: np.ndarray, title: str, subtitle: str) -> plt.Figure:
    overlay = upsample_heatmap_to_image(heatmap, image.size)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("input image", fontsize=9)
    axes[1].imshow(image)
    axes[1].imshow(overlay, cmap="jet", alpha=0.55)
    axes[1].axis("off")
    axes[1].set_title("attention overlay", fontsize=9)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2-vl-2b")
    p.add_argument("--hf-id", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--vcr-dir", default="/workspace/data/vcr")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="results/attention_sample")
    p.add_argument("--full-val-n", type=int, default=26534)
    p.add_argument("--n-correct", type=int, default=5)
    p.add_argument("--n-incorrect", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    print(f"[attn-sample] picking examples from {args.model} Wave-2 details JSON")
    details_path = find_details_file(results_dir, args.model, args.full_val_n)
    print(f"[attn-sample]   {details_path.name}")
    with open(details_path) as f:
        details = json.load(f)
    picks = pick_examples(details, args.n_correct, args.n_incorrect, args.seed)
    print(f"[attn-sample] picked {len(picks)} examples")

    annot_ids = [ex["annot_id"] for ex in picks]
    ds_full = VCRDataset(vcr_dir=args.vcr_dir, split="val")
    by_annot = {}
    for i in range(len(ds_full)):
        a = ds_full.annotations[i]
        if a.get("annot_id") in annot_ids:
            by_annot[a["annot_id"]] = i
    print(f"[attn-sample] resolved {len(by_annot)}/{len(annot_ids)} annot_ids in val split")

    print(f"[attn-sample] loading {args.model} (eager attention) on {args.device}")
    model, processor = load_qwen2vl_eager(args.hf_id, device=args.device)

    sample_records = []
    panel_paths = []

    for pick in picks:
        annot_id = pick["annot_id"]
        idx = by_annot.get(annot_id)
        if idx is None:
            print(f"  [skip] annot_id {annot_id} not found in loader")
            continue
        ex = ds_full[idx]
        image = ex["image"]
        qa_prompt = VLMEvaluator.format_vcr_prompt(
            ex["question"], ex["answer_choices"], task="qa"
        )
        gt_letter_qa = "ABCD"[ex["answer_label"]]
        correct_answer_text = ex["answer_choices"][ex["answer_label"]]
        qar_prompt = VLMEvaluator.format_vcr_prompt(
            f"{ex['question']} Answer: {correct_answer_text}. Why?",
            ex["rationale_choices"],
            task="qar",
        )
        gt_letter_qar = "ABCD"[ex["rationale_label"]]

        print(f"[attn-sample] {annot_id}  Q->A  (correct={pick.get('qa_correct')})")
        res_qa = extract_attention_qwen2vl(model, processor, image, qa_prompt)
        print(f"  pred_token={res_qa.predicted_token!r}  patch={res_qa.patch_grid_hw}  layers_used={res_qa.num_layers_used}")
        print(f"[attn-sample] {annot_id}  QA->R (correct={pick.get('r_correct')})")
        res_qar = extract_attention_qwen2vl(model, processor, image, qar_prompt)
        print(f"  pred_token={res_qar.predicted_token!r}  patch={res_qar.patch_grid_hw}  layers_used={res_qar.num_layers_used}")

        subtitle_qa = (
            f"{annot_id} | Q->A | pred={pick.get('pred_answer')} truth={gt_letter_qa}"
            f" | {'CORRECT' if pick.get('qa_correct') else 'INCORRECT'}"
        )
        subtitle_qar = (
            f"{annot_id} | QA->R | pred={pick.get('pred_rationale')} truth={gt_letter_qar}"
            f" | {'CORRECT' if pick.get('r_correct') else 'INCORRECT'}"
        )

        qa_path = out_dir / f"{annot_id}_qa.png"
        qar_path = out_dir / f"{annot_id}_qar.png"
        fig = render_panel(image, res_qa.heatmap, ex["question"], subtitle_qa)
        fig.savefig(qa_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        fig = render_panel(image, res_qar.heatmap, ex["question"], subtitle_qar)
        fig.savefig(qar_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        panel_paths.append((qa_path, qar_path))

        sample_records.append({
            "annot_id": annot_id,
            "qa_correct": pick.get("qa_correct"),
            "r_correct": pick.get("r_correct"),
            "pred_answer": pick.get("pred_answer"),
            "answer_label": pick.get("answer_label"),
            "pred_rationale": pick.get("pred_rationale"),
            "rationale_label": pick.get("rationale_label"),
            "qa": {
                "predicted_next_token": res_qa.predicted_token,
                "patch_grid_hw": list(res_qa.patch_grid_hw),
                "num_image_tokens": res_qa.num_image_tokens,
                "num_layers_used": res_qa.num_layers_used,
                "heatmap_min": float(res_qa.heatmap.min()),
                "heatmap_max": float(res_qa.heatmap.max()),
            },
            "qar": {
                "predicted_next_token": res_qar.predicted_token,
                "patch_grid_hw": list(res_qar.patch_grid_hw),
                "num_image_tokens": res_qar.num_image_tokens,
                "num_layers_used": res_qar.num_layers_used,
                "heatmap_min": float(res_qar.heatmap.min()),
                "heatmap_max": float(res_qar.heatmap.max()),
            },
            "panel_files": [str(qa_path.name), str(qar_path.name)],
        })

    grid_path = out_dir / "attention_sample.png"
    n_rows = len(panel_paths)
    if n_rows > 0:
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
        if n_rows == 1:
            axes = np.array([axes])
        for r, (qa_p, qar_p) in enumerate(panel_paths):
            axes[r, 0].imshow(Image.open(qa_p))
            axes[r, 0].axis("off")
            axes[r, 1].imshow(Image.open(qar_p))
            axes[r, 1].axis("off")
        fig.tight_layout()
        fig.savefig(grid_path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        print(f"[attn-sample] grid: {grid_path}")

    meta_path = out_dir / "sample_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "hf_id": args.hf_id,
                "n_correct_requested": args.n_correct,
                "n_incorrect_requested": args.n_incorrect,
                "seed": args.seed,
                "source_details": str(details_path.name),
                "samples": sample_records,
            },
            f,
            indent=2,
        )
    print(f"[attn-sample] metadata: {meta_path}")
    print(f"[attn-sample] done. {len(sample_records)} examples extracted.")


if __name__ == "__main__":
    main()
