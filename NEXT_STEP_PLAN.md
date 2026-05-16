# Next Step Plan: Cloud-First VCR Baselines and Attention Analysis

## Current Status

The project is ready for Milestone 3, but the immediate missing piece is **VCR baseline evaluation**. The baselines completed so far are small-sample MMMU and MathVista runs; there are no completed VCR baseline runs yet.

Local context reviewed:

- `../Context/1772022293951_jaydeep_roadmap.pdf` defines VCR as the primary dataset and Milestone 3 as full model evaluation, ablations, and attention analysis.
- `../Context/Converstations_with_mentor.md` confirms the mentor's May 4 direction: proceed to core experiments, run full evaluations, distinguish visual recognition failures from logical reasoning failures, and extract attention maps.
- `../Context/baselines.csv` contains small-sample MMMU results.
- `../Context/baselines_mathvista_main.csv` and `../Context/baselines_math_vista_half.csv` contain small-sample MathVista results.
- The repo already has loaders and wrappers for VCR, MMMU, MathVista, Moondream2, Qwen2-VL, InternVL2, Phi-3-Vision, LLaVA, GPT-4o, and Claude.

Important gaps:

- VCR has not been downloaded or evaluated locally.
- The VCR dataset should be downloaded on the cloud GPU machine, preferably onto persistent storage.
- Existing baseline results are useful pipeline checks, but they are not the VCR results needed for the paper.
- The current pipeline logs accuracy and per-example outputs, but it does not yet implement attention extraction, Attention Object Overlap, failure taxonomy, or final analysis scripts.
- Some existing MathVista results have parse/evaluation concerns, so MathVista should remain secondary until VCR is stable.

## Research Basis

VCR remains the best primary dataset for this project because it directly tests answer selection plus rationale selection and reports Q->A, QA->R, and Q->AR accuracy. It also includes object metadata that can support grounding analysis.

Supporting sources:

- VCR project page: https://visualcommonsense.com/
- VCR paper: https://arxiv.org/abs/1811.10830
- VCR leaderboard and metric definitions: https://visualcommonsense.com/leaderboard/
- MMMU paper: https://arxiv.org/abs/2311.16502
- MathVista paper: https://arxiv.org/abs/2310.02255
- Qwen2.5-VL technical report: https://arxiv.org/abs/2502.13923
- Qwen3-VL technical report: https://arxiv.org/abs/2511.21631
- InternVL3.5 technical report: https://arxiv.org/abs/2508.18265
- MiniCPM-V 4.5 technical report: https://arxiv.org/abs/2509.18154
- Attention caution: https://arxiv.org/abs/1902.10186

Key interpretation:

- VCR should drive the main paper results.
- MMMU and MathVista are useful secondary stress tests, especially for expert and mathematical visual reasoning.
- Newer small VLMs such as Qwen3-VL, Qwen2.5-VL, InternVL3.5, and MiniCPM-V 4.5 are worth considering after the core VCR pipeline works.
- Attention maps should be presented as grounding evidence, not as definitive proof of model reasoning.

## Cloud GPU Setup

All VCR baselines and core experiments should run on the cloud GPU instance, not on the local machine.

Recommended cloud layout:

```text
/workspace/small-vlm-reasoning/      # cloned repo
/workspace/data/vcr/                 # VCR annotations and images
/workspace/results/                  # experiment outputs
/workspace/.cache/huggingface/       # HF model and dataset cache
```

Set persistent cache paths before running downloads or models:

```bash
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
```

Dataset path decision:

- Preferred: download VCR to `/workspace/data/vcr` and pass `--vcr_dir /workspace/data/vcr`.
- Alternative: symlink `/workspace/small-vlm-reasoning/data/vcr` to `/workspace/data/vcr`.
- Do not download VCR locally unless there is a specific debugging need; it is large and the real runs require cloud GPU anyway.

Cloud bootstrap steps:

1. Clone or sync the repo to the cloud GPU machine.
2. Create the conda environment from `environment.yml`.
3. Set Hugging Face cache paths on persistent storage.
4. Download VCR on the cloud machine.
5. Run a VCR loader validation before launching model baselines.
6. Run a 5-example model smoke test.
7. Run a 100-example VCR pilot.
8. Run full VCR validation after the pilot passes.

## Next-Step Plan

### 1. Download and Validate VCR on Cloud

Run on the cloud GPU machine:

```bash
bash scripts/download_vcr.sh /workspace/data/vcr
```

Validate:

- `/workspace/data/vcr/train.jsonl`, `/workspace/data/vcr/val.jsonl`, and `/workspace/data/vcr/test.jsonl` exist.
- `/workspace/data/vcr/vcr1images/` exists and contains image files.
- A small `VCRDataset` instance loads images, questions, answers, rationales, object labels, metadata filenames, and boxes.
- Bounding boxes are available from metadata JSON files for a representative sample.

Implementation adjustment:

- Preserve object-reference indices from VCR token lists, not only resolved object names.
- Store question, answer, and rationale object references in the normalized dataset item so Attention Object Overlap can map referenced objects to bounding boxes.

### 2. Freeze the Evaluation Protocol

Use VCR as the primary dataset and standardize every run.

Main model grid:

- Small models:
  - `moondream2`
  - `qwen2-vl-2b`
  - `phi3-vision`
  - `qwen2-vl-7b`
  - `llava-next-7b`
  - `internvl2-8b`
- Reference models:
  - `llava-1.5-13b`
  - `gpt-4o`
- Optional, if GPU memory and time allow:
  - `llava-1.6-34b` with 4-bit quantization
  - `internvl2-26b` with 4-bit quantization

Generation settings:

- Direct MCQ runs:
  - `do_sample=False`
  - `temperature=0`
  - `max_new_tokens=8`
  - prompt asks for only `A`, `B`, `C`, or `D`
- CoT ablation runs:
  - `do_sample=False`
  - `temperature=0`
  - larger `max_new_tokens`
  - require a final answer line that the parser can extract reliably

Every run should log:

- model name
- parameter count
- quantization
- dataset and split
- seed
- prompt strategy
- generation parameters
- example id
- prompt
- raw output
- parsed output
- ground-truth label
- correctness
- parse status
- latency
- GPU memory

### 3. Run Experiments in Three Waves

Wave 1: VCR pilot

- Run 100 VCR validation examples.
- Use all small models.
- Use zero-shot direct prompting only.
- Goal: verify dataset, prompts, parsing, model wrappers, output schema, checkpointing, and runtime.

Example command:

```bash
python -m src.baselines.evaluate \
  --model qwen2-vl-2b \
  --dataset vcr \
  --vcr_dir /workspace/data/vcr \
  --max_samples 100 \
  --seed 42 \
  --device cuda \
  --csv_path results/vcr_pilot.csv
```

Wave 2: VCR full validation

- Run the full VCR validation split for the main model grid.
- Use seed `42`.
- Treat Q->AR as the primary metric.
- Keep Q->A and QA->R separately so answer failures and rationale failures can be analyzed independently.

Wave 3: controlled ablations

- Use a fixed 500-example VCR validation subset.
- Compare:
  - zero-shot direct prompting
  - 3-shot few-shot prompting
  - zero-shot CoT with final-letter parsing
  - standard image resolution vs higher image resolution where wrapper support is practical
- Run at least the best small model, weakest small model, and one reference model.

Do not expand to Qwen3-VL or MiniCPM-V 4.5 until Wave 1 and Wave 2 are stable end to end.

### 4. Implement Attention Extraction for RQ3

Add an attention extraction path that supports two model families:

- explicit cross-attention, when available
- decoder self-attention over visual tokens, for Qwen/LLaVA-style models that place image tokens in the language sequence

Implementation requirements:

- Use eager attention where needed because optimized kernels often do not return attention tensors.
- Capture attention only for a bounded number of examples to avoid huge artifacts.
- Save compact artifacts, not full raw tensors by default.
- Store aggregated heatmaps, layer/head metadata, token spans, predicted labels, ground-truth labels, referenced object boxes, and correctness.

Extraction schedule:

1. Start with `qwen2-vl-2b` on 50 VCR examples.
2. Verify that visual-token spans and heatmaps align with image coordinates.
3. Scale to 500 balanced examples:
   - 250 correct predictions
   - 250 incorrect predictions
4. Run one small model and one reference model for comparison.

### 5. Add Analysis Outputs

Metrics:

- Q->A accuracy
- QA->R accuracy
- Q->AR accuracy
- parse failure rate
- average and median latency
- bootstrap 95% confidence intervals
- Attention Object Overlap

Attention Object Overlap:

- Use VCR object references from the question, selected answer, and selected rationale.
- Map referenced object ids to VCR metadata boxes.
- Convert attention over visual tokens into an image-space heatmap.
- Threshold the heatmap consistently.
- Compute IoU or overlap between thresholded heatmap regions and referenced object boxes.
- Report AOO separately for correct and incorrect predictions.

Failure taxonomy:

- visual recognition error
- spatial or relation error
- counting error
- commonsense or logical-chain error
- rationale-only error
- parse or output-format error

Manual review:

- Review 100 to 200 failed examples from the best small model.
- Store the review in `analysis/error_analysis.md` or an equivalent structured CSV/Markdown file.

Expected final artifacts:

- `results/vcr_pilot.csv`
- `results/baselines_vcr.csv`
- `results/final_results.json`
- `results/ablations/`
- `results/attention/`
- `figures/`
- `tables/`
- `analysis/error_analysis.md`

## Test Plan

Cloud validation tests:

- Confirm VCR files exist on persistent cloud storage.
- Confirm Hugging Face model cache is on persistent cloud storage.
- Confirm a stopped/restarted instance can still see `/workspace/data/vcr` and `/workspace/.cache/huggingface`.

Unit tests:

- VCR loader preserves object ids and boxes.
- Answer parser handles direct letters, `Answer: B`, parenthesized choices, and CoT final-answer text.
- Attention Object Overlap works on synthetic boxes and synthetic heatmaps.
- Result aggregation handles duplicate runs and parse failures correctly.

Smoke tests:

- Run 5 VCR examples through `qwen2-vl-2b`.
- Run 5 VCR examples with attention extraction enabled.
- Interrupt and resume a checkpointed run.
- Render a small attention heatmap and visually inspect alignment.

Acceptance checks:

- 100-example VCR pilot completes on cloud GPU.
- Full VCR validation produces complete CSV and JSON outputs for the main model grid.
- Direct MCQ parse failure rate is below 5%.
- At least one small model and one reference model produce usable attention artifacts.
- Analysis scripts produce final metrics, confidence intervals, and first-pass figures without manual editing.

## Assumptions

- This phase is inference-only. No fine-tuning is planned.
- VCR is the paper's primary dataset because it best matches the roadmap and supports grounding analysis.
- All VCR data download and all model inference happen on the cloud GPU machine.
- Local development is only for code edits, lightweight inspection, and documentation.
- MMMU and MathVista stay as secondary stress tests after VCR results are stable.
- Existing model wrappers should be used first. New wrappers for Qwen3-VL, Qwen2.5-VL, InternVL3.5, or MiniCPM-V 4.5 should be added only after the VCR pipeline is reliable.
- Large open models should use 4-bit quantization if GPU memory is limited.
- GPT-4o can serve as the closed-model reference if 34B open-weight models are too expensive or unstable.
