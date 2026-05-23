# Session Handoff — Milestone 3 (Attention Extraction)

This document hands off context to a new Claude session on a different machine.
Read this top-to-bottom before doing anything.

Today's date format example: handoff written 2026-05-18.

## 1. Project in one paragraph

**"Analyzing Multi-Step Visual Reasoning in Small Vision-Language Models."**
Primary dataset: VCR v1.0. We evaluate small VLMs (1B–8B) against larger
references on VCR's three metrics (Q→A, QA→R, Q→AR). The novel contribution
of the paper is **RQ3 — Attention Object Overlap (AOO)**: distinguish
visual-grounding failures (model attended to the wrong region) from logical
reasoning failures (model attended correctly but reasoned wrong), by
extracting attention from decoder layers and comparing against VCR's
object bounding boxes.

## 2. Repo, branch, paths

- GitHub: `Mayankpratapsingh022/Analyzing-Multi-Step-Visual-Reasoning-in-Small-Vision-Language-Models`
- Working branch: `feature/mmmu-mathvista-baselines`
- Latest commit at handoff time: `56015cd` ("Add attention extractor + 10-example sample script")
- User is a collaborator with push access.
- Local Windows clone:
  `C:\Users\acer\Desktop\Multi-Step-VisualReasoning-in-sVLMs\Analyzing-Multi-Step-Visual-Reasoning-in-Small-Vision-Language-Models`
- The parent folder `Multi-Step-VisualReasoning-in-sVLMs` is a DIFFERENT git
  repo (master branch). The actual project repo lives in the nested folder
  above. Don't run `git` commands from the parent.

## 3. What's done

### Wave 1 — pilot, n=300 per model

5 models on a deterministic 300-example val sample (seed=42):
qwen2-vl-2b, qwen2-vl-7b, llava-next-7b, llava-1.5-13b (8-bit), gpt-4o.
Results in `results/vcr_300_comparable.csv`. Parse failure rate 0.0% for
every model.

### Wave 2 — full validation, n=26,534

4 GPU models on the full VCR val split (gpt-4o skipped — too expensive at scale).

| Model | Q→A | QA→R | Q→AR | parse | peak VRAM | wall |
|---|---|---|---|---|---|---|
| qwen2-vl-2b | 63.4% | 62.1% | 40.9% | 0.0% | 4.15 GB | 4h 9m |
| qwen2-vl-7b | 70.5% | 71.9% | **52.1%** | 0.0% | 15.52 GB | 5h 39m |
| llava-next-7b | 63.7% | 64.6% | 42.4% | 0.0% | 14.13 GB | 3h 49m |
| llava-1.5-13b 8bit | 63.5% | 61.4% | 40.4% | 0.0% | 12.79 GB | 3h 45m |

Results CSV: `results/vcr_full_val.csv`. Per-example outputs (raw, parsed,
correctness flags) in `results/*_details.json` — used to pick examples for
the attention work without re-running inference.

### Mentor (Dr. Sreedath) approvals so far

1. "Proceed to Milestone 3" — pre-Wave-2 approval.
2. Post-Wave-2 reply: pipeline is good, n=300 pilot validated, approve
   attention extraction approach. Two gates:
   - **Attention work** — average across heads or focus on late-stage
     layers, otherwise matrices are too large to interpret.
   - **Failure attribution** — formally define visual-recognition vs.
     logical-reasoning criteria BEFORE running the analysis.
   - "Send a few sample maps once the hooks are working" (= a 10-example
     sample for sign-off before scaling to 500).

The current mentor update lives at `mentor_update_milestone3.md`. It was
already sent.

## 4. What's pending — the immediate next task

### Run the attention-extraction sample on a fresh GPU pod.

Code is already written and pushed at commit `56015cd`:

- `src/analysis/attention_extractor.py` — extracts late-layer, head-averaged
  attention from the last query position to visual-token positions in
  Qwen2-VL. Uses `attn_implementation="eager"` because flash/SDPA kernels
  do not expose attention probabilities.
- `cloud_gpu_vcr/scripts/extract_attention_sample.py` — picks 5 correct +
  5 incorrect VCR val examples from the qwen2-vl-2b Wave-2 details JSON
  (deterministic seed=42), runs the extractor on Q→A and QA→R prompts per
  example, saves visualization panels and a combined grid PNG into
  `results/attention_sample/`.

Expected runtime: ~5–10 min on any GPU pod with enough VRAM for
qwen2-vl-2b (≥ 6 GB).

### Step-by-step on a fresh pod

```bash
# 1. Clone + checkout
cd /workspace
git clone https://github.com/Mayankpratapsingh022/Analyzing-Multi-Step-Visual-Reasoning-in-Small-Vision-Language-Models.git
cd Analyzing-Multi-Step-Visual-Reasoning-in-Small-Vision-Language-Models
git checkout feature/mmmu-mathvista-baselines

# 2. Env vars (set HF_TOKEN with your own; W&B optional for this run)
export HF_TOKEN=<paste-or-`hf auth login` first>
export VCR_DIR=/workspace/data/vcr
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export USE_WANDB=0

# 3. Bootstrap (installs torch, transformers, matplotlib, etc.)
bash cloud_gpu_vcr/scripts/bootstrap_cloud.sh

# 4. Restore VCR + the qwen2-vl-2b details JSON from HF
mkdir -p /workspace/data/vcr_raw /workspace/data/vcr results
export HF_HUB_ENABLE_HF_TRANSFER=0   # avoids HF rate-limit issue at scale
hf download JaydeepR/vcr-mirror --repo-type dataset --local-dir /workspace/data/vcr_raw

# reorg into the layout the loader expects
mkdir -p /workspace/data/vcr_raw/vcr1annots /workspace/data/vcr_raw/vcr1images
mv /workspace/data/vcr_raw/vcr1annots/*.jsonl /workspace/data/vcr/ 2>/dev/null || true
mv /workspace/data/vcr_raw/vcr1images/vcr1images /workspace/data/vcr/vcr1images 2>/dev/null || true
ls /workspace/data/vcr  # expect: test.jsonl  train.jsonl  val.jsonl  vcr1images

# pull the qwen2-vl-2b full-val details JSON
hf download JaydeepR/vcr-mirror --repo-type dataset --include "wave2_latest/results/qwen2-vl-2b_*details*.json" --local-dir .
mv wave2_latest/results/qwen2-vl-2b_vcr_*_details.json results/

# 5. Run the sample
python cloud_gpu_vcr/scripts/extract_attention_sample.py
```

Outputs land in `results/attention_sample/`:
- `attention_sample.png` — the grid figure to send to the mentor
- `<annot_id>_qa.png` and `<annot_id>_qar.png` — per-example panels
- `sample_metadata.json` — record of picks, predictions, and layer stats

### After the run, back up to HF

```bash
hf upload JaydeepR/vcr-mirror results/attention_sample/ wave2_latest/results/attention_sample/ --repo-type dataset
```

## 5. What's pending after that

### A. Failure-attribution criteria spec (writing, no GPU)

Mentor's gate: formally define the rules separating "visual recognition
failure" from "logical reasoning failure" before running the analysis on
the existing details JSONs. Output should be a markdown spec the mentor
can sign off on.

A reasonable structure:

1. Definitions of each failure class (visual-recognition, logical-reasoning,
   parse, spatial, counting, commonsense, rationale-only).
2. Decision rules — when does an error count as visual-recognition vs.
   logical-reasoning? E.g., use attention-IoU thresholds against VCR
   object boxes (low IoU → visual; high IoU + wrong answer → reasoning).
3. Manual-review protocol for the ~150 errors that aren't programmatically
   classifiable.
4. Coding sheet structure for the manual review.

### B. Scale attention extraction (after mentor signs off on sample)

Per `NEXT_STEP_PLAN.md`:
- Validation step: 50 examples on qwen2-vl-2b (sanity check that
  visual-token spans map to image coordinates correctly)
- Full RQ3 study: 500 balanced examples (250 correct + 250 incorrect),
  on **two models** — one small (likely qwen2-vl-2b), one reference
  (likely qwen2-vl-7b or llava-1.5-13b — mentor may weigh in).
- Compute estimate: ~30–60 min of GPU per model. Whole RQ3 fits in
  ~2 hours of GPU once the code is validated on the sample.

### C. AOO (Attention Object Overlap) computation

Once attention extraction works at scale:
- Map referenced object IDs in each VCR example to their bounding boxes
  (boxes already extracted by the loader; available as `metadata["boxes"]`).
- Threshold the attention heatmap (e.g., top-20% pixels).
- Compute IoU between thresholded region and object boxes.
- Report AOO separately for correct vs incorrect predictions.

## 6. Where everything is backed up

| Artifact | Location | Access |
|---|---|---|
| Code | GitHub branch `feature/mmmu-mathvista-baselines` | Public repo, user has push |
| VCR dataset (raw) | HF `JaydeepR/vcr-mirror` | **PRIVATE**, user is owner |
| Wave 1 + 2 results + logs | HF `JaydeepR/vcr-mirror/wave2_latest/` | Same private dataset |
| W&B runs | `wandb.ai/j-raijada25-personal/small-vlm-reasoning-vcr` | User's W&B account |

Local Windows mirror of `wave2_latest/` may also exist at
`...\Multi-Step-VisualReasoning-in-sVLMs\wave2_backup\` if step 4 of the
prior backup was completed.

## 7. Known gotchas

1. **HF rate limit on raw-file dataset downloads.** `JaydeepR/vcr-mirror`
   is structured as a folder dump (not a parquet HF dataset), so
   `load_dataset(..., streaming=True)` does NOT work. Bulk downloading
   ~200k tiny files hits a `429 Too Many Requests` (3000 req per 5 min on
   the Team plan). Workarounds:
   - Set `export HF_HUB_ENABLE_HF_TRANSFER=0` (hf_transfer serializes
     small-file downloads — counter-intuitive but true).
   - Lower `--max-workers` to 4–8 to stay under the rate.
   - For sub-uses (like 300 examples for the sample run), the existing
     `cloud_gpu_vcr/scripts/fetch_300_val.py` script does targeted
     downloads. Pass `--samples 26534` to get all val files.

2. **Dataset layout dual-naming.** On HF the layout is
   `vcr1annots/*.jsonl` and `vcr1images/vcr1images/...` (doubly nested
   because of the upstream Kaggle mirror's structure). The loader expects
   `<vcr_dir>/{train,val,test}.jsonl` and `<vcr_dir>/vcr1images/*` (single
   nesting). Either reorg with `mv` (preferred) or symlink. Reorg
   commands are in the step-by-step above.

3. **Pyright stub warnings in attention_extractor.py.** Several "unknown
   attribute" warnings (Qwen2VLForConditionalGeneration, processor
   callable, processor.tokenizer, Image.BILINEAR). All work at runtime.
   Do not "fix" by adding shims; they are stub-incompleteness.

4. **Eager attention is slow.** For the sample (10 examples) it's fine.
   For the 500-example study you may want to switch back to flash/SDPA
   for the inference step and only re-run with eager attention on the
   500 chosen examples for attention capture.

5. **Pod might disappear.** All work that hasn't been pushed to HF or
   GitHub can vanish. Before terminating ANY pod, run
   `bash cloud_gpu_vcr/scripts/backup_to_hf.sh`.

## 8. Scripts you'll touch

All under `cloud_gpu_vcr/scripts/`:

- `bootstrap_cloud.sh` — pod env setup. Run once per pod.
- `fetch_300_val.py` — targeted VCR download (sample-based or full val).
- `extract_attention_sample.py` — the immediate next task (this script).
- `backup_to_hf.sh` — pre-termination snapshot of results/ + logs/.
- `preflight_audit.sh` — enumerate pod state before backup, sanity check.
- `run_wave2_sweep.sh` — only relevant if you need to re-run Wave 2.
- `run_300_sweep.sh`, `sanity_check_300.py` — Wave 1 era; keep for
  reference but not needed for current work.

## 9. Open decisions for the user

- Which reference model for the 500-example AOO comparison: qwen2-vl-7b
  or llava-1.5-13b? Mentor hasn't said yet. qwen2-vl-7b is the strongest
  open model; llava-1.5-13b is a different family with a quantization
  asterisk. Recommend qwen2-vl-7b unless mentor disagrees.
- Whether to also do attention on llava-next-7b. Plan said yes
  eventually; not blocking the immediate sample.
- Failure taxonomy: how much of the manual review the user will do vs.
  delegate to undergrad RAs (if any). Affects spec scope.

## 10. After the attention sample

When the sample PNG is generated and looks reasonable:
1. Append the figure to `mentor_update_milestone3.md` or write a new
   short update mentioning the sample is ready.
2. Send the figure + a one-paragraph note to Dr. Sreedath.
3. Move on to the failure-attribution criteria spec while waiting for
   mentor feedback on the sample.

---

Last updated: handoff written 2026-05-18 after pushing commit `56015cd`.
