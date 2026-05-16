# 300-Example VCR Baseline Setup

This guide is for running a **300-example VCR baseline slice** on cloud GPU. The goal is to get VCR results that are comparable to the existing MMMU and MathVista baseline runs without spending multiple days on full validation.

## Models to Run

Run the models that overlap best with the current MMMU/MathVista baseline results:

```text
qwen2-vl-2b
qwen2-vl-7b
llava-next-7b
llava-1.5-13b
gpt-4o
```

Optional if time remains:

```text
moondream2
```

This gives a comparable cross-dataset table:

```text
MMMU       -> qwen2-vl-2b, qwen2-vl-7b, llava-next-7b, llava-1.5-13b, gpt-4o
MathVista  -> qwen2-vl-2b, qwen2-vl-7b, llava-next-7b, llava-1.5-13b, gpt-4o
VCR 300    -> same model set
```

## Recommended Compute

Minimum practical setup:

```text
GPU: A100 40GB
Storage: 200GB persistent volume minimum
Preferred storage: 300GB persistent volume
```

If using a smaller GPU, skip `llava-1.5-13b` or run it with 8-bit quantization.

## RunPod Setup

Recommended RunPod template:

```text
Image: runpod/pytorch:2.3.1-py3.10-cuda12.1.1-devel-ubuntu22.04
Volume mount: /workspace
GPU: A100 40GB or A100 80GB
Disk: 200-300GB persistent storage
```

Clone the repo:

```bash
cd /workspace
git clone <YOUR_REPO_URL> small-vlm-reasoning
cd /workspace/small-vlm-reasoning
```

## Environment Variables

```bash
export VCR_DIR=/workspace/data/vcr
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

export WANDB_PROJECT=small-vlm-reasoning-vcr
export USE_WANDB=1
```

If using W&B:

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
# Optional:
export WANDB_ENTITY=<YOUR_WANDB_TEAM_OR_USERNAME>
```

If using GPT-4o:

```bash
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

If using gated Hugging Face models:

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
```

Disable W&B if needed:

```bash
export USE_WANDB=0
```

## Install Dependencies

Recommended:

```bash
bash cloud_gpu_vcr/scripts/bootstrap_cloud.sh
```

Manual equivalent:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Check the setup:

```bash
python cloud_gpu_vcr/scripts/check_cloud_state.py
```

## Download and Validate VCR

```bash
bash cloud_gpu_vcr/scripts/download_vcr_cloud.sh
python cloud_gpu_vcr/scripts/validate_vcr.py --vcr-dir /workspace/data/vcr --samples 5
```

## Run a Tiny Smoke Test First

Before the 300-example runs:

```bash
CSV_PATH=results/vcr_300_comparable.csv bash cloud_gpu_vcr/scripts/run_single_model.sh qwen2-vl-2b 5
```

Confirm:

- terminal logs show progress
- `logs/` has a log file
- `results/vcr_300_comparable.csv` is created
- W&B receives a run if `USE_WANDB=1`

## Run 300-Example Comparable Baselines

Run these from the repo root:

```bash
CSV_PATH=results/vcr_300_comparable.csv bash cloud_gpu_vcr/scripts/run_single_model.sh qwen2-vl-2b 300

CSV_PATH=results/vcr_300_comparable.csv bash cloud_gpu_vcr/scripts/run_single_model.sh qwen2-vl-7b 300

CSV_PATH=results/vcr_300_comparable.csv bash cloud_gpu_vcr/scripts/run_single_model.sh llava-next-7b 300
```

Run `llava-1.5-13b` with 8-bit quantization:

```bash
python -m src.baselines.evaluate \
  --model llava-1.5-13b \
  --dataset vcr \
  --vcr_dir /workspace/data/vcr \
  --max_samples 300 \
  --seed 42 \
  --device cuda \
  --batch_size 1 \
  --max_new_tokens 8 \
  --prompt_strategy zero_shot_direct \
  --quantization 8bit \
  --csv_path results/vcr_300_comparable.csv \
  --wandb \
  --wandb_project small-vlm-reasoning-vcr \
  --wandb_group vcr-300-comparable
```

Run GPT-4o:

```bash
CSV_PATH=results/vcr_300_comparable.csv bash cloud_gpu_vcr/scripts/run_single_model.sh gpt-4o 300
```

Optional Moondream2:

```bash
CSV_PATH=results/vcr_300_comparable.csv bash cloud_gpu_vcr/scripts/run_single_model.sh moondream2 300
```

## Expected Outputs

```text
results/vcr_300_comparable.csv
results/*_vcr_*_details.json
results/checkpoints/
logs/
```

The CSV should include:

```text
model_name
param_count
dataset
max_samples
q_a_accuracy
qa_r_accuracy
q_ar_accuracy
parse_failure_rate
avg_inference_time_sec
vram_model_gb
seed
timestamp
```

## Runtime Estimate

For 300 VCR examples, each example does two generations: Q->A and QA->R.

Approximate total on A100 40GB:

```text
qwen2-vl-2b      ~10-30 min after model download
qwen2-vl-7b      ~20-60 min after model download
llava-next-7b    ~20-60 min after model download
llava-1.5-13b    ~30-90 min after model download
gpt-4o           depends on API latency/rate limits
```

First run may take longer because models need to download into the Hugging Face cache.

Expected total:

```text
With model downloads: 4-8 hours
If models are cached: 2-5 hours
```

## Resume or Inspect

If a run stops, rerun the same command. Checkpoints are keyed by model, dataset, seed, sample count, prompt strategy, and token length.

Inspect checkpoints:

```bash
python scripts/checker.py --status
python scripts/checker.py --resume-cmd
```

## Notes

- Do not treat 300 examples as final paper results.
- Use this as a mentor update and pipeline validation baseline.
- Full VCR validation should be run later after confirming prompts, parsing, and runtime are acceptable.
- Keep `seed=42` fixed so the same 300 examples are used across all models.
