
**Subject:** Milestone 3 update — VCR full validation complete, moving to attention extraction

---

Hi Dr Sreedath,

**the full VCR validation is done.**

## Results

All on the full val split, n=26,534, seed=42, zero-shot direct prompting, 0% parse failure rate across the board.

| Model | Q→A | QA→R | Q→AR |
|---|---|---|---|
| qwen2-vl-2b | 63.4% | 62.1% | 40.9% |
| **qwen2-vl-7b** | **70.5%** | **71.9%** | **52.1%** |
| llava-next-7b | 63.7% | 64.6% | 42.4% |
| llava-1.5-13b (8-bit) | 63.5% | 61.4% | 40.4% |
| gpt-4o (reference) | 74.7% | 75.3% | 55.7% (n=300) |


- The pipeline captures per-example prompts, raw outputs, parsed answers, correctness flags, and timing — so the failure-attribution work can proceed on the existing details JSONs without re-running inference.
- n=300 pilot accuracies predicted the full-val numbers within ±2 pp for every model.

## Next

1. **Attention extraction (PyTorch forward hooks)** — starting with qwen2-vl-2b for fast dev iteration, then porting to qwen2-vl-7b and llava-next-7b. Since all three are decoder-only, "cross-attention to visual tokens" maps to the slice of self-attention from text-token positions to visual-token positions; I will extract both that slice and the full self-attention pattern across the multi-step Q→A and QA→R prompts.
2. **Failure attribution (visual recognition vs. logical reasoning)** — proceeding on the Wave 2 details JSONs (no new GPU runs needed for the analysis phase; new runs only for the attention-capture phase).

I will send a sample of attention maps once the extractor is built and validated on a handful of examples.

## Links

- **W&B dashboard:** https://wandb.ai/j-raijada25-personal/small-vlm-reasoning-vcr
- **Results CSV + per-example JSONs:** `JaydeepR/vcr-mirror/wave2_latest/` (HF private dataset)
- **Code:** `feature/mmmu-mathvista-baselines` branch

Thanks,
Jaydeep Raijada

