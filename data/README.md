# Dataset Setup and Preprocessing

This repository evaluates VLMs on VCR, MMMU, and MathVista. The evaluation code
does not create large preprocessed dataset files. Instead, `src/data_loader.py`
normalizes each dataset on demand into the dictionary format consumed by
`src.models.vlm_evaluator.VLMEvaluator`.

## Common Layout

Keep raw datasets and caches under `data/`:

```text
data/
├── vcr/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── vcr1images/
└── README.md
```

`data/vcr/`, `data/mmmu/`, and `data/mathvista/` are gitignored. Hugging Face
datasets normally live in the Hugging Face cache, not directly in this folder.
Set `HF_HOME` or `HF_DATASETS_CACHE` if you want those downloads on a persistent
volume:

```bash
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
```

## VCR v1.0

Source: https://visualcommonsense.com/

VCR is the only benchmark in this repo that requires a manual local download.
The evaluation path is configured with `--vcr_dir` or `vcr_dir` in the YAML
configs. The repo defaults to `data/vcr`.

### Step 1: Download and Extract

Run the provided script from the repository root:

```bash
bash scripts/download_vcr.sh data/vcr
```

The script:

1. Creates the target directory.
2. Downloads `vcr1annots.zip`.
3. Extracts `train.jsonl`, `val.jsonl`, and `test.jsonl`.
4. Downloads `vcr1images.zip`.
5. Extracts the image tree into `vcr1images/`.
6. Prints line counts and image counts as a basic validation pass.

Expected structure:

```text
data/vcr/
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── vcr1images/
    ├── movieclip_00001/
    │   ├── 00001.jpg
    │   └── 00001.metadata.json
    └── ...
```

### Step 2: Keep the Raw Annotation Format

No offline conversion is required. Each VCR JSONL line is read directly by
`VCRDataset`:

```json
{
  "annot_id": "val-0",
  "img_fn": "movieclip_00001/00001.jpg",
  "metadata_fn": "movieclip_00001/00001.metadata.json",
  "objects": ["person", "person", "table"],
  "question": ["Why", "is", [0], "looking", "at", [1], "?"],
  "answer_choices": [
    ["Because", [0], "is", "surprised"],
    ["Because", [0], "wants", "to", "talk"]
  ],
  "rationale_choices": [],
  "answer_label": 1,
  "rationale_label": 0
}
```

VCR text stores object mentions as list tokens, for example `[0]`. The loader
replaces those references with names from the `objects` field.

### Step 3: Runtime Preprocessing in `VCRDataset`

For every example, `src/data_loader.py` performs these transformations:

1. Reads the selected split from `data/vcr/{split}.jsonl`.
2. Applies deterministic sampling when `subset_pct` is set.
3. Opens `vcr1images/{img_fn}` with Pillow and converts it to RGB.
4. Converts the tokenized question into plain text.
5. Converts all answer choices into plain text.
6. Converts all rationale choices into plain text.
7. Preserves `answer_label` and `rationale_label` as zero-based class ids.
8. Copies `annot_id`, `img_fn`, `objects`, and `metadata_fn` into `metadata`.
9. If the metadata JSON exists, loads object bounding boxes into
   `metadata["boxes"]`.

The normalized VCR item has this shape:

```python
{
    "image": PIL.Image.Image,
    "question": str,
    "answer_choices": list[str],
    "rationale_choices": list[str],
    "answer_label": int,
    "rationale_label": int,
    "metadata": {
        "annot_id": str,
        "img_fn": str,
        "objects": list[str],
        "metadata_fn": str,
        "boxes": list,  # present when metadata JSON is available
    },
}
```

### Step 4: Validate Locally

Use a small loader check before launching model runs:

```bash
python - <<'PY'
from src.data_loader import get_dataset

ds = get_dataset("vcr", vcr_dir="data/vcr", split="val", subset_pct=1, seed=42)
ex = ds[0]
print(len(ds))
print(ex["image"].size)
print(ex["question"])
print(ex["answer_choices"])
print(ex["metadata"].keys())
PY
```

The baseline evaluator uses VCR `val` by default:

```bash
python -m src.baselines.evaluate \
  --model qwen2-vl-2b \
  --dataset vcr \
  --vcr_dir data/vcr \
  --subset_pct 5
```

## MMMU

Source: Hugging Face dataset `MMMU/MMMU`

MMMU is downloaded automatically by the `datasets` library. There is no manual
preprocessing step and no required `data/mmmu` directory.

### Step 1: Choose Subject and Split

The repo uses the `validation` split. `MMMUDataset` accepts an optional
`subject` argument:

```python
from src.data_loader import get_dataset

ds = get_dataset("mmmu", split="validation", subject="Math", max_samples=100)
```

When `subject` is omitted, the loader calls
`get_dataset_config_names("MMMU/MMMU")`, loads every subject config, and
concatenates them.

The provided config `configs/baseline_mmmu.yml` uses:

```yaml
dataset: mmmu
subject: Math
max_samples: 100
subset_pct: null
```

### Step 2: Runtime Preprocessing in `MMMUDataset`

For every example, `src/data_loader.py` performs these transformations:

1. Downloads or reads `MMMU/MMMU` from the Hugging Face cache.
2. Applies `max_samples` first when set.
3. Otherwise applies deterministic `subset_pct` sampling when set.
4. Collects image columns `image_1` through `image_7`.
5. Converts PIL images to RGB.
6. Parses the `options` column when it is stored as a Python-list string.
7. Falls back to `option_A` through `option_E` or `A` through `E` fields when
   `options` is absent.
8. Converts question and choice values to plain strings.
9. Converts answer letters `A` through `E` into zero-based `answer_label`.
10. Copies id, subject, subfield, and question type into `metadata`.

The normalized MMMU item has this shape:

```python
{
    "image": PIL.Image.Image | None,
    "images": list[PIL.Image.Image],
    "question": str,
    "answer_choices": list[str],
    "answer_label": int,
    "metadata": {
        "id": str,
        "subject": str,
        "subfield": str,
        "question_type": str,
    },
}
```

Run the MMMU baseline with:

```bash
python run_experiment.py --config configs/baseline_mmmu.yml
```

or directly:

```bash
python -m src.baselines.evaluate \
  --model qwen2-vl-2b \
  --dataset mmmu \
  --subject Math \
  --max_samples 100
```

## MathVista

Source: Hugging Face dataset `AI4Math/MathVista`

MathVista is downloaded automatically by the `datasets` library. The repo uses
the `testmini` split by default.

### Step 1: Load the Split

```python
from src.data_loader import get_dataset

ds = get_dataset("mathvista", split="testmini", max_samples=100)
```

The provided config `configs/baseline_mathvista.yml` uses:

```yaml
dataset: mathvista
max_samples: 100
subset_pct: null
```

### Step 2: Runtime Preprocessing in `MathVistaDataset`

For every example, `src/data_loader.py` performs these transformations:

1. Downloads or reads `AI4Math/MathVista` from the Hugging Face cache.
2. Applies `max_samples` first when set.
3. Otherwise applies deterministic `subset_pct` sampling when set.
4. Reads `image` or `decoded_image` and converts PIL images to RGB.
5. Reads `choices`; if choices are stored as a JSON string, parses them into a
   Python list.
6. Reads `answer`.
7. Sets `answer_label` to the matching choice index when the answer exactly
   matches one of the choices.
8. Sets `answer_label` to `-1` when the example is free-form or cannot be mapped
   to a choice.
9. Preserves the raw answer in `answer_text`.
10. Copies pid, question type, answer type, and category into `metadata`.

The normalized MathVista item has this shape:

```python
{
    "image": PIL.Image.Image | None,
    "question": str,
    "answer_choices": list[str],
    "answer_label": int,
    "answer_text": str,
    "metadata": {
        "pid": str,
        "question_type": str,
        "answer_type": str,
        "category": str,
    },
}
```

Run the MathVista baseline with:

```bash
python run_experiment.py --config configs/baseline_mathvista.yml
```

or directly:

```bash
python -m src.baselines.evaluate \
  --model qwen2-vl-2b \
  --dataset mathvista \
  --max_samples 100
```

## Sampling Rules

All three loaders use the same sampling convention:

1. `max_samples` is an exact sample count and takes priority when supported.
2. `subset_pct` samples a percentage of the loaded split.
3. `seed` controls deterministic sampling.
4. `subset_pct: null` and `max_samples: null` run the full loaded split.

VCR currently supports `subset_pct` in the evaluation path. MMMU and MathVista
support both `max_samples` and `subset_pct`.

## Output Files

Dataset preprocessing happens at read time; evaluation outputs are written under
`results/`:

```text
results/
├── baselines.csv
├── baselines_mmmu.csv
├── baselines_mathvista.csv
├── *_details.json
└── checkpoints/
```

Per-example result JSON files include prompts, raw model responses, parsed
answers, labels, correctness flags, metadata, and timing. Checkpoints can be
inspected with:

```bash
python scripts/checker.py --status
```
