# Analyzing Multi-Step Visual Reasoning in Small Vision-Language Models

Research project evaluating multi-step visual reasoning capabilities of small VLMs (1B-8B parameters) compared against larger counterparts, using VCR, MMMU, and MathVista benchmarks.

## Setup

### 1. Create environment

```bash
conda env create -f environment.yml
conda activate small-vlm-reasoning
```

### 2. Download VCR dataset

```bash
bash scripts/download_vcr.sh data/vcr
```

MMMU and MathVista are downloaded automatically via HuggingFace `datasets`.

### 3. Smoke test models

```bash
# Test all open-weight models
python scripts/smoke_test.py --skip-api

# Test specific models
python scripts/smoke_test.py --models moondream2 qwen2-vl-2b
```

## Running Baselines

### Single model

```bash
# Quick test (5% subset)
python -m src.baselines.evaluate --model qwen2-vl-2b --dataset vcr --vcr_dir data/vcr --subset_pct 5

# Full validation set
python -m src.baselines.evaluate --model qwen2-vl-2b --dataset vcr --vcr_dir data/vcr
```

### Sweep all small models

```bash
python -m src.baselines.evaluate --sweep small --dataset vcr --vcr_dir data/vcr --subset_pct 5
```

### From config file

```bash
python run_experiment.py --config configs/baseline_small.yml
python run_experiment.py --config configs/baseline_large.yml
python run_experiment.py --config configs/baseline_mmmu.yml
```

### API models (GPT-4o, Claude)

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
python -m src.baselines.evaluate --model gpt-4o --dataset vcr --vcr_dir data/vcr --subset_pct 2
```

## Models

| # | Model | Params | Type |
|---|---|---|---|
| 1 | Moondream2 | 1.8B | Small |
| 2 | Qwen2-VL-2B | 2B | Small |
| 3 | InternVL2-2B | 2B | Small |
| 4 | Phi-3-Vision | 4.2B | Small |
| 5 | Qwen2-VL-7B | 7B | Small |
| 6 | LLaVA-NeXT-7B | 7B | Small |
| 7 | InternVL2-8B | 8B | Small |
| 8 | LLaVA-1.5-13B | 13B | Large |
| 9 | InternVL2-26B | 26B | Large |
| 10 | LLaVA-1.6-34B | 34B | Large |
| 11 | GPT-4o | Closed | API |
| 12 | Claude | Closed | API |

## GPU Requirements

| Model | FP16 VRAM | 8-bit | 4-bit |
|---|---|---|---|
| Moondream2 (1.8B) | ~4 GB | ~2 GB | - |
| Qwen2-VL-2B | ~5 GB | ~3 GB | - |
| InternVL2-2B | ~5 GB | ~3 GB | - |
| Phi-3-Vision (4.2B) | ~9 GB | ~5 GB | - |
| Qwen2-VL-7B | ~15 GB | ~9 GB | ~5 GB |
| LLaVA-NeXT-7B | ~15 GB | ~9 GB | ~5 GB |
| InternVL2-8B | ~17 GB | ~10 GB | ~6 GB |
| LLaVA-1.5-13B | ~27 GB | ~14 GB | ~8 GB |
| InternVL2-26B | ~54 GB | ~28 GB | ~15 GB |
| LLaVA-1.6-34B | ~70 GB | ~36 GB | ~19 GB |

## Results

Baseline results are written to `results/baselines.csv`. Per-example details are saved as JSON in `results/`.

## Project Structure

```
├── run_experiment.py          # Main experiment runner
├── environment.yml            # Conda environment
├── configs/                   # YAML experiment configs
├── data/                      # Datasets (gitignored)
├── src/
│   ├── data_loader.py         # Dataset loaders (VCR, MMMU, MathVista)
│   ├── models/                # Model wrappers
│   │   ├── vlm_evaluator.py   # Base class + registry
│   │   ├── moondream_wrapper.py
│   │   ├── qwen2vl_wrapper.py
│   │   ├── internvl2_wrapper.py
│   │   ├── phi3vision_wrapper.py
│   │   ├── llava_wrapper.py
│   │   └── api_wrapper.py     # GPT-4o + Claude
│   ├── baselines/
│   │   └── evaluate.py        # Evaluation script
│   └── analysis/
├── scripts/
│   ├── download_vcr.sh
│   └── smoke_test.py
├── results/
├── figures/
├── tables/
└── manuscript/
```
