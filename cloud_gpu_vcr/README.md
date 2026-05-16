# Cloud GPU VCR Evaluation Runner

This folder contains the cloud-first runner for the next project phase: VCR baselines, full evaluation, and reproducible experiment logging.

It assumes you clone the full repository on a GPU machine, then run commands from the repository root.

## What This Runs

- VCR dataset download and validation on cloud storage.
- Small-model VCR pilot runs.
- Full VCR validation runs.
- Terminal logs plus file logs under `logs/`.
- Weights & Biases tracking when `USE_WANDB=1`.
- Deterministic seeding, stable checkpoint keys, and persistent Hugging Face caches.

## Cloud Layout

Recommended layout:

```text
/workspace/small-vlm-reasoning/      # cloned repository
/workspace/data/vcr/                 # VCR annotations and images
/workspace/results/                  # optional persistent result backup
/workspace/.cache/huggingface/       # model and dataset cache
```

The scripts default to:

```bash
export VCR_DIR=/workspace/data/vcr
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export WANDB_PROJECT=small-vlm-reasoning-vcr
```

## Quick Start

From the repository root on the cloud GPU machine:

```bash
bash cloud_gpu_vcr/scripts/bootstrap_cloud.sh
bash cloud_gpu_vcr/scripts/download_vcr_cloud.sh
python cloud_gpu_vcr/scripts/validate_vcr.py --vcr-dir /workspace/data/vcr --samples 5
bash cloud_gpu_vcr/scripts/run_single_model.sh qwen2-vl-2b 5
bash cloud_gpu_vcr/scripts/run_vcr_pilot.sh
```

After the pilot succeeds, run the full small-model evaluation:

```bash
bash cloud_gpu_vcr/scripts/run_vcr_full_small.sh
```

Run reference models separately because they are more expensive:

```bash
bash cloud_gpu_vcr/scripts/run_vcr_reference.sh
```

## Weights & Biases

By default the runner enables W&B if `USE_WANDB=1`.

Set these before launching runs:

```bash
export WANDB_API_KEY=...
export WANDB_PROJECT=small-vlm-reasoning-vcr
export WANDB_ENTITY=your-team-or-username   # optional
```

Disable W&B for a dry local/debug run:

```bash
export USE_WANDB=0
```

Each model run logs:

- run config
- final metrics
- timing and VRAM metrics
- per-example JSON as a W&B artifact

The terminal and `logs/` file still contain the detailed progress output.

## Reproducibility

The runner sets:

- fixed seed, default `42`
- deterministic PyTorch/CUDA flags
- persistent Hugging Face caches
- exact sample count for VCR pilot runs
- checkpoint keys that include dataset, sample size, prompt strategy, seed, and generation length
- CSV and per-example JSON outputs for every run

Important outputs:

```text
results/vcr_pilot.csv
results/baselines_vcr_small.csv
results/baselines_vcr_reference.csv
results/*_details.json
results/checkpoints/
logs/
```

## Config Files

- `configs/vcr_pilot.yml`: 100-example small-model pilot.
- `configs/vcr_full_small.yml`: full VCR validation for small models.
- `configs/vcr_reference.yml`: reference models, including GPT-4o.

You can also run a single model:

```bash
bash cloud_gpu_vcr/scripts/run_single_model.sh qwen2-vl-2b 100
```

Arguments:

- first argument: model name
- second argument: max samples, or `full`

Examples:

```bash
bash cloud_gpu_vcr/scripts/run_single_model.sh qwen2-vl-7b 100
bash cloud_gpu_vcr/scripts/run_single_model.sh llava-next-7b full
```
