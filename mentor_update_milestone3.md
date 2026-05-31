**Subject:** Milestone 3 update — VCR full validation complete, moving to attention extraction

---

Hi Dr Sreedath,

**the full VCR validation is done.**

## Results

All on the full val split, n=26,534, seed=42, zero-shot direct prompting, 0% parse failure rate across the board.

| Model                 | Q→A       | QA→R      | Q→AR          |
| --------------------- | --------- | --------- | ------------- |
| qwen2-vl-2b           | 63.4%     | 62.1%     | 40.9%         |
| **qwen2-vl-7b**       | **70.5%** | **71.9%** | **52.1%**     |
| llava-next-7b         | 63.7%     | 64.6%     | 42.4%         |
| llava-1.5-13b (8-bit) | 63.5%     | 61.4%     | 40.4%         |
| gpt-4o (reference)    | 74.7%     | 75.3%     | 55.7% (n=300) |

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


Hi Jaydeep,

This is Sreedath. The direction I gave previously stands. Proceed with the attention extraction and failure attribution as outlined.

Can you summarize your results in a 1-2 page PDF (not raw results, your findings) and share with me?

Look specifically at whether the attention weights on the visual tokens remain focused or if they scatter across the multi-step reasoning prompts (QA→R) compared to direct answering (Q→A). Tell me if you see a noticeable difference in the attention patterns in the later layers when the model fails.

PS: This is not a chatbot, but I often use an LLM to refine my replies. I speak my thoughts to wispr flow, which converts it to text, then LLM converts it to a better format with scientific notations if needed (which I cannot type).

You · Mon 4:51 PM
Got it, I have sent across a few attention maps form the 7b model, but they don't seem to be very accurate, could you just look at a few, and tell me if it's the right thing to do. And should we be extracting attention maps using a bigger model for understanding?

You · Tue 6:43 PM
Let me know as soon as possible since I have two days off this week and can work on this promptly
