# RunPod Guide: Qwen2-VL Attention Sample

This guide is for downloading a small selected VCR subset and running the 10-example attention-extraction sample for RQ3, not for rerunning Wave 2.

Goal: use `cloud_gpu_vcr/scripts/extract_attention_sample.py` to fetch only the selected examples, then generate `results/attention_sample/attention_sample.png` plus per-example panels and metadata.

## 0. Pod Requirements

Use a RunPod GPU instance with:

- CUDA-capable GPU
- At least 6 GB VRAM for `qwen2-vl-2b`
- Enough disk for the model cache and the selected VCR examples
- Hugging Face access to private dataset `JaydeepR/vcr-mirror`

Recommended pod image: a PyTorch CUDA image with Python 3.10+.

## Quick Copy-Paste Commands

Use this sequence on a fresh RunPod instance after opening a terminal.

```bash
cd /workspace

git clone https://github.com/Mayankpratapsingh022/Analyzing-Multi-Step-Visual-Reasoning-in-Small-Vision-Language-Models.git

cd Analyzing-Multi-Step-Visual-Reasoning-in-Small-Vision-Language-Models

git checkout feature/mmmu-mathvista-baselines

export HF_TOKEN=<paste-your-huggingface-token>
export VCR_DIR=/workspace/data/vcr
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export HF_HUB_ENABLE_HF_TRANSFER=0
export USE_WANDB=0

bash cloud_gpu_vcr/scripts/bootstrap_cloud.sh

mkdir -p /workspace/data/vcr_raw /workspace/data/vcr results
```

Download 100 selected examples only, without running attention extraction:

```bash
python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --prepare-only \
  --num-examples 100
```

Run the mentor-requested 10-example attention sample:

```bash
python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --num-examples 10
```

Check outputs:

```bash
ls -lh results/attention_sample/

python -m json.tool results/attention_sample/sample_metadata.json | head -120
```

Back up the generated sample:

```bash
hf upload JaydeepR/vcr-mirror \
  results/attention_sample/ \
  wave2_latest/results/attention_sample/ \
  --repo-type dataset
```

## 1. Clone The Repo

```bash
cd /workspace

git clone https://github.com/Mayankpratapsingh022/Analyzing-Multi-Step-Visual-Reasoning-in-Small-Vision-Language-Models.git

cd Analyzing-Multi-Step-Visual-Reasoning-in-Small-Vision-Language-Models

git checkout feature/mmmu-mathvista-baselines
```

Optional sanity check:

```bash
git log -1 --oneline
```

Expected branch:

```text
feature/mmmu-mathvista-baselines
```

## 2. Configure Environment Variables

Set `HF_TOKEN` to a token that can access the private Hugging Face dataset.

```bash
export HF_TOKEN=<paste-your-huggingface-token>

export VCR_DIR=/workspace/data/vcr
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

export USE_WANDB=0
```

Important HF rate-limit setting:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
```

This avoids the known rate-limit issue when downloading many small files from the raw-file VCR mirror.

Alternative login flow:

```bash
hf auth login
```

## 3. Bootstrap The Pod

Run the project bootstrap script once:

```bash
bash cloud_gpu_vcr/scripts/bootstrap_cloud.sh
```

This installs the runtime dependencies, including PyTorch, Transformers, Hugging Face tooling, and plotting libraries.

## 4. Minimal VCR Restore From Hugging Face

Do not redownload VCR from the original source. Also do not download the full HF mirror.

The attention sample script now supports a minimal restore mode:

```bash
mkdir -p /workspace/data/vcr_raw /workspace/data/vcr results
```

The Hugging Face mirror structure used by the script is:

```text
vcr1annots/val.jsonl
vcr1images/vcr1images/<movie_or_scene_dir>/<image>.jpg
vcr1images/vcr1images/<movie_or_scene_dir>/<metadata>.json
wave2_latest/results/qwen2-vl-2b_*details*.json
```

The script downloads only:

- `vcr1annots/val.jsonl`
- the matching `qwen2-vl-2b` Wave-2 details JSON
- `img_fn` and `metadata_fn` files for the selected examples

Use `--num-examples` to control the total number of selected examples. The script splits the count evenly across correct and incorrect Q->A outcomes.

Examples:

```bash
# 10 total examples: 5 correct + 5 incorrect
python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --prepare-only \
  --num-examples 10

# 50 total examples: 25 correct + 25 incorrect
python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --prepare-only \
  --num-examples 50

# 100 total examples: 50 correct + 50 incorrect
python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --prepare-only \
  --num-examples 100
```

Optional preflight to fetch only those local files without loading the model:

```bash
python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --prepare-only \
  --num-examples 100
```

Expected local layout after prepare:

```text
/workspace/data/vcr/val.jsonl
/workspace/data/vcr/vcr1images/<selected image files>
/workspace/data/vcr/vcr1images/<selected metadata files>
results/qwen2-vl-2b_*_details.json
```

Quick check:

```bash
find /workspace/data/vcr/vcr1images -type f | wc -l

ls results/*qwen2-vl-2b*details*.json
```

Expected selected asset count is usually up to `2 * num_examples`: one image and one metadata JSON per selected example, with fewer if examples share files.

## 5. Run The Attention Sample

```bash
python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --num-examples 10
```

This command performs the same minimal restore, then runs attention extraction.

If you already ran `--prepare-only`, this command will reuse the existing files and skip already-present downloads.

Expected runtime: about 5-10 minutes on a suitable GPU pod after model download/cache.

For mentor review, keep this at `--num-examples 10`. For only downloading more examples without running attention extraction, use `--prepare-only --num-examples 100`.

The extractor uses eager attention because flash/SDPA kernels do not expose attention probabilities.

Equivalent explicit command:

```bash
python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --hf-dataset-id JaydeepR/vcr-mirror \
  --hf-raw-dir /workspace/data/vcr_raw \
  --vcr-dir /workspace/data/vcr \
  --results-dir results \
  --num-examples 10 \
  --seed 42
```

## 6. Verify Outputs

Check generated files:

```bash
ls -lh results/attention_sample/
```

Expected outputs:

```text
attention_sample.png
sample_metadata.json
<annot_id>_qa.png
<annot_id>_qar.png
```

Quick metadata inspection:

```bash
python -m json.tool results/attention_sample/sample_metadata.json | head -120
```

Recommended visual checks before sending to the mentor:

- `attention_sample.png` exists and is non-empty.
- The grid contains the requested number of examples.
- Examples are split evenly across correct and incorrect cases.
- Both Q->A and QA->R panels are present where expected.
- Heatmaps align with plausible image regions rather than blank/constant maps.
- No obvious image-path failures or missing-image placeholders appear.

## 7. Back Up The Sample To Hugging Face

After verifying the sample looks reasonable:

```bash
hf upload JaydeepR/vcr-mirror \
  results/attention_sample/ \
  wave2_latest/results/attention_sample/ \
  --repo-type dataset
```

Before terminating the pod, optionally run the broader backup script:

```bash
bash cloud_gpu_vcr/scripts/backup_to_hf.sh
```

## 8. What Not To Run

Do not rerun Wave 2:

```bash
# Do not run:
# bash cloud_gpu_vcr/scripts/run_wave2_sweep.sh
```

Do not use Hugging Face streaming for this dataset mirror. The mirror is a raw folder dump, not a parquet HF dataset.

Do not run the old full mirror download for this sample:

```bash
# Do not run for the 10-example attention sample:
# hf download JaydeepR/vcr-mirror --repo-type dataset --local-dir /workspace/data/vcr_raw
```

## 9. Troubleshooting

If the run pauses for many minutes at this line:

```text
[attn-sample] downloading details JSON pattern from HF: wave2_latest/results/qwen2-vl-2b_*details*.json
```

stop it with `Ctrl+C`, pull the latest code, and rerun:

```bash
git pull

python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --num-examples 10
```

That older message came from a slower implementation that used `snapshot_download` and could spend a long time scanning the large raw-file HF dataset. The newer script lists only `wave2_latest/results/` and downloads the matching details JSON directly.

## 10. Next Deliverable After The Sample

After the sample is generated and visually checked, prepare a failure-attribution criteria spec for mentor sign-off.

Suggested output file:

```text
docs/failure_attribution_criteria.md
```

The spec should define:

- Visual-recognition failure
- Logical-reasoning failure
- Parse failure
- Spatial failure
- Counting failure
- Commonsense failure
- Rationale-only failure
- Decision rules using Attention Object Overlap thresholds
- Manual-review protocol for ambiguous cases
- Coding sheet fields for review
