# Cloud GPU Setup and Run Commands

This project should run VCR downloads and model evaluations on a cloud GPU machine, not locally.

## Recommended RunPod Settings

Use a persistent volume so downloaded models, datasets, logs, and results survive pod restarts.

Recommended template:

```text
Provider: RunPod
Image: runpod/pytorch:2.3.1-py3.10-cuda12.1.1-devel-ubuntu22.04
Volume mount: /workspace
GPU: start with 1x A100 40GB or 80GB if available
Disk: at least 120GB, preferably 200GB+
```

Expected storage needs:

- VCR images and annotations: about 25GB+
- Hugging Face model cache: can easily exceed 50GB depending on models
- Logs/results/checkpoints: depends on run size, keep several GB free

## Clone Repository

Run on the cloud GPU instance:

```bash
cd /workspace
git clone <YOUR_REPO_URL> small-vlm-reasoning
cd /workspace/small-vlm-reasoning
```

If you are copying code manually instead of cloning, make sure the repo root contains:

```text
cloud_gpu_vcr/
configs/
scripts/
src/
run_experiment.py
requirements.txt
environment.yml
setup.py
```

## Environment Variables

Set persistent cache paths:

```bash
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export VCR_DIR=/workspace/data/vcr
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$VCR_DIR"
```

Optional but recommended: add them to `~/.bashrc`.

```bash
cat >> ~/.bashrc <<'EOF'

# Small VLM reasoning project paths
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export VCR_DIR=/workspace/data/vcr
EOF
```

## API and Tracking Keys

For Hugging Face gated models:

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
```

For GPT-4o reference runs:

```bash
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

For Claude reference runs, if used:

```bash
export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
```

For Weights & Biases:

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export WANDB_PROJECT=small-vlm-reasoning-vcr
# Optional:
export WANDB_ENTITY=<YOUR_WANDB_TEAM_OR_USERNAME>
```

Disable W&B when debugging:

```bash
export USE_WANDB=0
```

Enable W&B:

```bash
export USE_WANDB=1
```

## Install Dependencies

Recommended one-command bootstrap:

```bash
bash cloud_gpu_vcr/scripts/bootstrap_cloud.sh
```

This installs Python dependencies, installs the repo in editable mode, sets cache paths, and checks the cloud state.

Manual install equivalent:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Optional flash attention install:

```bash
INSTALL_FLASH_ATTN=1 bash cloud_gpu_vcr/scripts/bootstrap_cloud.sh
```

Or manually:

```bash
python -m pip install flash-attn --no-build-isolation
```

Skip flash attention for first debug runs if installation is slow or fails.

## Check Cloud State

Run:

```bash
python cloud_gpu_vcr/scripts/check_cloud_state.py
```

Expected:

- CUDA available is `True`
- GPU name is shown
- `HF_HOME` path exists
- `VCR_DIR` path exists
- `wandb`, `transformers`, `datasets`, and project imports work

## Download VCR on Cloud

Run:

```bash
bash cloud_gpu_vcr/scripts/download_vcr_cloud.sh
```

This downloads VCR to:

```text
/workspace/data/vcr
```

Validate manually:

```bash
python cloud_gpu_vcr/scripts/validate_vcr.py --vcr-dir /workspace/data/vcr --samples 5
```

## Smoke Test One Model

Run a tiny VCR test before launching a sweep:

```bash
bash cloud_gpu_vcr/scripts/run_single_model.sh qwen2-vl-2b 5
```

Run a 100-example single-model test:

```bash
bash cloud_gpu_vcr/scripts/run_single_model.sh qwen2-vl-2b 100
```

Run a full single-model evaluation:

```bash
bash cloud_gpu_vcr/scripts/run_single_model.sh qwen2-vl-2b full
```

## Cloud Run Sequence

Recommended order from a fresh cloud GPU:

```bash
cd /workspace/small-vlm-reasoning

export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export VCR_DIR=/workspace/data/vcr
export WANDB_PROJECT=small-vlm-reasoning-vcr
export USE_WANDB=1

bash cloud_gpu_vcr/scripts/bootstrap_cloud.sh
bash cloud_gpu_vcr/scripts/download_vcr_cloud.sh
python cloud_gpu_vcr/scripts/validate_vcr.py --vcr-dir /workspace/data/vcr --samples 5
bash cloud_gpu_vcr/scripts/run_single_model.sh qwen2-vl-2b 5
bash cloud_gpu_vcr/scripts/run_vcr_pilot.sh
```

After the pilot succeeds:

```bash
bash cloud_gpu_vcr/scripts/run_vcr_full_small.sh
```

Run reference models separately:

```bash
bash cloud_gpu_vcr/scripts/run_vcr_reference.sh
```

## Outputs

Main outputs:

```text
logs/
results/vcr_single.csv
results/vcr_pilot.csv
results/baselines_vcr_small.csv
results/baselines_vcr_reference.csv
results/*_details.json
results/checkpoints/
```

W&B logs:

- metrics are logged under `eval/*`
- full per-example JSON is uploaded as a W&B artifact
- config includes seed, sample count, model, prompt strategy, generation length, device, and environment metadata

## Resume or Inspect Runs

List checkpoints:

```bash
python scripts/checker.py --status
```

Show resume commands:

```bash
python scripts/checker.py --resume-cmd
```

Inspect one checkpoint:

```bash
python scripts/checker.py --detail <checkpoint-file>
```

## Reproducibility Notes

The runner records:

- seed
- deterministic setting
- prompt strategy
- max samples
- generation token limit
- model name and quantization
- git commit and dirty status
- Python, PyTorch, CUDA, cuDNN, GPU name, and VRAM
- per-example prompts, raw outputs, parsed predictions, labels, correctness, and timing

Default deterministic settings:

```text
seed: 42
do_sample: False
max_new_tokens: 8
prompt_strategy: zero_shot_direct
batch_size: 1
```

## Common Issues

If `transformers` complains about `huggingface_hub>=0.30.0,<1.0`, reinstall dependencies:

```bash
python -m pip install -r requirements.txt --upgrade
```

If gated models fail to download, set `HF_TOKEN` and log in:

```bash
python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
```

If W&B is not wanted or API login fails:

```bash
export USE_WANDB=0
```

If a run crashes or is interrupted, rerun the same command. The checkpoint key includes seed, sample size, prompt strategy, and max token count, so matching runs resume safely.
