# RunPod Guide: Qwen2-VL Attention And Occlusion Checks

This guide is for the current RQ3 workflow:

1. Restore only the selected VCR validation examples from the HF mirror.
2. Generate Qwen2-VL attention maps for Q→A and QA→R prompts.
3. Compare Qwen2-VL-2B and Qwen2-VL-7B on the same selected examples.
4. Optionally run occlusion-sensitivity maps as a more causal sanity check.
5. Send/backup the generated artifacts.

Do not rerun Wave 2 full validation for this task.

## 0. Requirements

Use a RunPod CUDA image with Python 3.10+ and enough VRAM:

- Qwen2-VL-2B attention: at least 8 GB VRAM recommended.
- Qwen2-VL-7B attention: at least 24 GB VRAM recommended.
- Qwen2-VL-7B + occlusion: use a larger GPU if possible because it performs many forward passes.

You need Hugging Face access to the private dataset:

```text
JaydeepR/vcr-mirror
```

## 1. Fresh Pod Setup

```bash
cd /workspace

git clone https://github.com/Mayankpratapsingh022/Analyzing-Multi-Step-Visual-Reasoning-in-Small-Vision-Language-Models.git

cd Analyzing-Multi-Step-Visual-Reasoning-in-Small-Vision-Language-Models

git checkout feature/mmmu-mathvista-baselines
git pull origin feature/mmmu-mathvista-baselines
```

Set environment variables:

```bash
export HF_TOKEN=<paste-your-huggingface-token>
export VCR_DIR=/workspace/data/vcr
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export HF_HUB_ENABLE_HF_TRANSFER=0
export USE_WANDB=0

mkdir -p /workspace/data/vcr_raw /workspace/data/vcr results
```

Bootstrap dependencies:

```bash
bash cloud_gpu_vcr/scripts/bootstrap_cloud.sh
```

Optional sanity check:

```bash
git log -1 --oneline
python -m py_compile src/analysis/attention_extractor.py cloud_gpu_vcr/scripts/extract_attention_sample.py
```

## 2. Optional Prepare-Only Download

This downloads only:

- `vcr1annots/val.jsonl`
- selected images and metadata JSON files
- matching Wave-2 details JSON used to pick examples

Run this if you want to verify HF access before loading any model:

```bash
python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --prepare-only \
  --num-examples 10
```

For a larger future sample:

```bash
python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --prepare-only \
  --num-examples 50
```

Expected local files:

```text
/workspace/data/vcr/val.jsonl
/workspace/data/vcr/vcr1images/<selected image files>
/workspace/data/vcr/vcr1images/<selected metadata files>
results/qwen2-vl-2b_*_details.json
```

Check:

```bash
ls -lh results/*qwen2-vl-2b*details*.json
find /workspace/data/vcr/vcr1images -type f | wc -l
```

## 3. Main 10-Example Qwen2-VL-2B Attention Run

This is the default mentor-review sample for the 2B model.

```bash
rm -rf results/attention_sample_qwen2b

python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --num-examples 10 \
  --model qwen2-vl-2b \
  --hf-id Qwen/Qwen2-VL-2B-Instruct \
  --out-dir results/attention_sample_qwen2b \
  --dtype bf16
```

The default attention method is:

```text
contrastive_rollout
```

It performs late-layer attention rollout and subtracts a neutral same-image prompt map to reduce positional/border priors.

## 4. Matching 10-Example Qwen2-VL-7B Attention Run

This uses the same selected examples as the 2B run, but loads Qwen2-VL-7B.

```bash
rm -rf results/attention_sample_qwen7b

python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --num-examples 10 \
  --model qwen2-vl-2b \
  --hf-id Qwen/Qwen2-VL-7B-Instruct \
  --out-dir results/attention_sample_qwen7b \
  --dtype bf16
```

Why `--model qwen2-vl-2b` here:

- `--model` selects the existing Wave-2 details JSON used to pick examples.
- `--hf-id` selects the actual model loaded for attention extraction.
- We keep `--model qwen2-vl-2b` so the 2B and 7B attention runs use the same 10 examples.

## 5. Larger 50-Example Comparison

Run this after the 10-example sample works. This is better for the mentor’s question about Q→A vs QA→R focus/scatter.

Qwen2-VL-2B:

```bash
rm -rf results/attention_sample_qwen2b_n50

python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --num-examples 50 \
  --model qwen2-vl-2b \
  --hf-id Qwen/Qwen2-VL-2B-Instruct \
  --out-dir results/attention_sample_qwen2b_n50 \
  --dtype bf16
```

Qwen2-VL-7B:

```bash
rm -rf results/attention_sample_qwen7b_n50

python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --num-examples 50 \
  --model qwen2-vl-2b \
  --hf-id Qwen/Qwen2-VL-7B-Instruct \
  --out-dir results/attention_sample_qwen7b_n50 \
  --dtype bf16
```

## 6. Occlusion Sanity Checks

Attention weights are not guaranteed causal explanations. Use occlusion checks on a small subset to see whether masking high-attention regions actually changes the model’s predicted letter probability.

Start small. A 5x5 grid with 4 examples runs 25 occlusion forwards per prompt, plus the normal attention forwards.

Qwen2-VL-2B occlusion check:

```bash
rm -rf results/attention_occlusion_check_qwen2b

python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --num-examples 4 \
  --model qwen2-vl-2b \
  --hf-id Qwen/Qwen2-VL-2B-Instruct \
  --out-dir results/attention_occlusion_check_qwen2b \
  --dtype bf16 \
  --occlusion-grid 5
```

Qwen2-VL-7B occlusion check:

```bash
rm -rf results/attention_occlusion_check_qwen7b

python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --num-examples 4 \
  --model qwen2-vl-2b \
  --hf-id Qwen/Qwen2-VL-7B-Instruct \
  --out-dir results/attention_occlusion_check_qwen7b \
  --dtype bf16 \
  --occlusion-grid 5
```

Expected extra files when occlusion is enabled:

```text
<annot_id>_qa_occlusion.png
<annot_id>_qar_occlusion.png
```

Use these as validation:

- If attention and occlusion highlight similar regions, the attention map is more credible.
- If attention highlights a region but occlusion shows no prediction drop there, treat the attention map as weak evidence.
- If both maps are scattered, report that the model’s visual evidence is not spatially focused for that prompt.

## 7. Raw And Diagnostic Ablations

Only run these for debugging. Do not use them as primary mentor results.

Old raw last-token attention:

```bash
python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --num-examples 10 \
  --model qwen2-vl-2b \
  --hf-id Qwen/Qwen2-VL-2B-Instruct \
  --out-dir results/attention_sample_raw_qwen2b \
  --dtype bf16 \
  --attention-method raw
```

Border-suppressed ablation:

```bash
python cloud_gpu_vcr/scripts/extract_attention_sample.py \
  --prepare-from-hf \
  --num-examples 10 \
  --model qwen2-vl-2b \
  --hf-id Qwen/Qwen2-VL-2B-Instruct \
  --out-dir results/attention_sample_no_border_qwen2b \
  --dtype bf16 \
  --suppress-border-patches 1
```

Do not use `--suppress-border-patches` for primary results. It is only for diagnosing border artifacts.

## 8. Verify Outputs

Check files:

```bash
ls -lh results/attention_sample_qwen2b/
ls -lh results/attention_sample_qwen7b/
```

Expected attention files:

```text
attention_sample.png
sample_metadata.json
<annot_id>_qa.png
<annot_id>_qar.png
```

Check metadata:

```bash
python -m json.tool results/attention_sample_qwen2b/sample_metadata.json | head -120
python -m json.tool results/attention_sample_qwen7b/sample_metadata.json | head -120
```

Check warnings and border metrics:

```bash
python -m json.tool results/attention_sample_qwen2b/sample_metadata.json | \
  grep -n "quality_warning\\|border_mass\\|normalized_entropy"

python -m json.tool results/attention_sample_qwen7b/sample_metadata.json | \
  grep -n "quality_warning\\|border_mass\\|normalized_entropy"
```

Quick numeric summary:

```bash
python - <<'PY'
import json
from pathlib import Path

for name, path in {
    "qwen2b": Path("results/attention_sample_qwen2b/sample_metadata.json"),
    "qwen7b": Path("results/attention_sample_qwen7b/sample_metadata.json"),
}.items():
    if not path.exists():
        print(f"{name}: missing {path}")
        continue
    data = json.loads(path.read_text())
    print(f"== {name} ==")
    print("hf_id:", data.get("hf_id"))
    print("method:", data.get("attention_method"))
    for phase in ["qa", "qar"]:
        rows = [s[phase]["diagnostics"] for s in data["samples"]]
        entropy = [r["normalized_entropy"] for r in rows if r.get("normalized_entropy") is not None]
        border = [r["border_mass"] for r in rows if r.get("border_mass") is not None]
        warnings = [r.get("quality_warning") for r in rows if r.get("quality_warning")]
        print(
            f"{phase}: n={len(rows)} "
            f"entropy_mean={sum(entropy)/len(entropy):.3f} "
            f"border_mean={sum(border)/len(border):.3f} "
            f"warnings={len(warnings)}"
        )
    print()
PY
```

Visual checks:

- Open `attention_sample.png`.
- Confirm Q→A and QA→R panels are both present.
- Confirm heatmaps are not blank, constant, or all pinned to one corner.
- Compare `quality_warning` entries against the visual grid.
- For occlusion runs, compare `*_occlusion.png` against the corresponding attention map.

## 9. Send Results Back To Local Machine

Using `runpodctl`, send each folder after verifying it is non-empty:

```bash
du -sh results/attention_sample_qwen2b results/attention_sample_qwen7b

runpodctl send results/attention_sample_qwen2b
runpodctl send results/attention_sample_qwen7b
```

For occlusion checks:

```bash
du -sh results/attention_occlusion_check_qwen2b results/attention_occlusion_check_qwen7b

runpodctl send results/attention_occlusion_check_qwen2b
runpodctl send results/attention_occlusion_check_qwen7b
```

On your local machine, receive with the code printed by RunPod:

```bash
runpodctl receive <code>
```

## 10. Back Up To Hugging Face

Upload the verified outputs:

```bash
hf upload JaydeepR/vcr-mirror \
  results/attention_sample_qwen2b/ \
  wave2_latest/results/attention_sample_qwen2b/ \
  --repo-type dataset

hf upload JaydeepR/vcr-mirror \
  results/attention_sample_qwen7b/ \
  wave2_latest/results/attention_sample_qwen7b/ \
  --repo-type dataset
```

For occlusion checks:

```bash
hf upload JaydeepR/vcr-mirror \
  results/attention_occlusion_check_qwen2b/ \
  wave2_latest/results/attention_occlusion_check_qwen2b/ \
  --repo-type dataset

hf upload JaydeepR/vcr-mirror \
  results/attention_occlusion_check_qwen7b/ \
  wave2_latest/results/attention_occlusion_check_qwen7b/ \
  --repo-type dataset
```

Before terminating the pod:

```bash
bash cloud_gpu_vcr/scripts/backup_to_hf.sh
```

## 11. What To Report

For the mentor PDF, do not send raw results only. Summarize:

- VCR full-validation accuracies from Wave 2.
- Attention method: late-layer visual-token `contrastive_rollout`.
- Q→A vs QA→R focus/scatter metrics:
  - `normalized_entropy`
  - `border_mass`
  - `quality_warning` count
- Correct vs failed example comparison.
- 2B vs 7B comparison.
- Occlusion sanity-check agreement/disagreement with attention.
- Caveat: decoder-only self-attention is a visual-token attribution proxy, not definitive causal evidence.

## 12. What Not To Run

Do not rerun Wave 2 full validation:

```bash
# Do not run for this task:
# bash cloud_gpu_vcr/scripts/run_wave2_sweep.sh
```

Do not bulk download the full VCR mirror unless necessary. The script’s `--prepare-from-hf` mode downloads only the needed files.
