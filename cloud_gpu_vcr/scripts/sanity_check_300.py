"""
Wave-1 sanity check on the 300-example VCR sweep.

Reads the CSV at results/vcr_300_comparable.csv and the matching *_details.json
files. Prints, per model:
  - accuracies with bootstrap 95% CIs
  - parse failure rate
  - peak VRAM and load time
  - Wave-2 wall-time extrapolation (multiply by 26534/300)
  - three sample raw outputs (to eyeball whether the parser is masking weird
    behavior like refusals or off-format completions)

Usage:
  python cloud_gpu_vcr/scripts/sanity_check_300.py
  python cloud_gpu_vcr/scripts/sanity_check_300.py --csv results/vcr_300_comparable.csv --samples 3
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from pathlib import Path


def bootstrap_ci(correct: list[int], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    if not correct:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(correct)
    samples = []
    for _ in range(n_boot):
        s = sum(correct[rng.randrange(n)] for _ in range(n))
        samples.append(s / n)
    samples.sort()
    lo = samples[int(0.025 * n_boot)]
    hi = samples[int(0.975 * n_boot)]
    return (lo, hi)


def fmt_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo * 100:.1f}, {hi * 100:.1f}]"


def find_details_for(run_id: str, results_dir: Path) -> Path | None:
    candidates = list(results_dir.glob(f"{run_id}_details.json"))
    return candidates[0] if candidates else None


def peek_examples(details: dict, n: int) -> list[dict]:
    examples = details.get("per_example", [])
    if not examples:
        return []
    rng = random.Random(42)
    return rng.sample(examples, min(n, len(examples)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/vcr_300_comparable.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--samples", type=int, default=3, help="raw-output samples per model")
    parser.add_argument("--target-n", type=int, default=26534, help="Wave 2 example count for extrapolation")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    results_dir = Path(args.results_dir)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    rows_300 = [r for r in rows if r.get("max_samples", "") == "300"]
    if not rows_300:
        raise SystemExit("no n=300 rows in CSV")

    print(f"=== Wave-1 sanity check ===  CSV: {csv_path}  rows: {len(rows_300)}")
    print()

    header = f"{'model':<16}  {'Q->A (95% CI)':<22}  {'QA->R (95% CI)':<22}  {'Q->AR (95% CI)':<22}  {'parse':<6}  {'vram_gb':<8}  {'wall_s':<7}  Wave-2 hrs"
    print(header)
    print("-" * len(header))

    for r in rows_300:
        model = r["model_name"]
        run_id = r["run_id"]
        details_path = find_details_for(run_id, results_dir)

        ci_qa = ci_r = ci_qar = ("?", "?")
        if details_path and details_path.exists():
            with open(details_path) as f:
                details = json.load(f)
            examples = details.get("per_example", [])
            qa_correct = [int(bool(ex.get("qa_correct"))) for ex in examples]
            r_correct = [int(bool(ex.get("r_correct") or ex.get("rationale_correct"))) for ex in examples]
            qar_correct = [int(qa and rr) for qa, rr in zip(qa_correct, r_correct)]
            ci_qa = bootstrap_ci(qa_correct)
            ci_r = bootstrap_ci(r_correct)
            ci_qar = bootstrap_ci(qar_correct)

        def safe_float(s: str) -> float:
            try:
                return float(s)
            except (TypeError, ValueError):
                return float("nan")

        q_a = safe_float(r.get("q_a_accuracy", ""))
        qa_r = safe_float(r.get("qa_r_accuracy", ""))
        q_ar = safe_float(r.get("q_ar_accuracy", ""))
        parse = safe_float(r.get("parse_failure_rate", ""))
        vram = safe_float(r.get("vram_gb", ""))
        wall = safe_float(r.get("wall_time_sec", ""))
        scale = args.target_n / 300.0
        eta_hrs = wall * scale / 3600.0 if wall == wall else float("nan")

        qa_str = f"{fmt_pct(q_a)} {fmt_ci(*ci_qa) if isinstance(ci_qa[0], float) else ''}"
        r_str = f"{fmt_pct(qa_r)} {fmt_ci(*ci_r) if isinstance(ci_r[0], float) else ''}"
        qar_str = f"{fmt_pct(q_ar)} {fmt_ci(*ci_qar) if isinstance(ci_qar[0], float) else ''}"

        print(
            f"{model:<16}  {qa_str:<22}  {r_str:<22}  {qar_str:<22}  "
            f"{fmt_pct(parse):<6}  {vram:<8.2f}  {wall:<7.0f}  {eta_hrs:.1f}"
        )

    print()
    print(f"=== Raw outputs (sample of {args.samples} per model) ===")
    for r in rows_300:
        model = r["model_name"]
        run_id = r["run_id"]
        details_path = find_details_for(run_id, results_dir)
        print()
        print(f"--- {model} ({details_path.name if details_path else 'NO DETAILS JSON FOUND'}) ---")
        if not details_path or not details_path.exists():
            continue
        with open(details_path) as f:
            details = json.load(f)
        for ex in peek_examples(details, args.samples):
            qa_correct = ex.get("qa_correct")
            r_correct = ex.get("r_correct", ex.get("rationale_correct"))
            print(f"  annot_id={ex.get('annot_id')}  qa_correct={qa_correct}  r_correct={r_correct}")
            qa_out = ex.get("qa_raw_output", "")
            r_out = ex.get("r_raw_output", ex.get("qar_raw_output", ""))
            print(f"    Q->A raw : {qa_out!r}")
            print(f"    Q->A pred={ex.get('pred_answer')}  truth={ex.get('answer_label')}")
            if r_out:
                print(f"    QA->R raw: {r_out!r}")
                print(f"    QA->R pred={ex.get('pred_rationale')}  truth={ex.get('rationale_label')}")

    print()
    print(f"Wave-2 extrapolation uses target_n={args.target_n} (= full val split).")
    print("CIs are non-parametric bootstrap (1000 resamples). At n=300 expect ~10pp wide.")


if __name__ == "__main__":
    main()
