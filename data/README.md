# Datasets

## VCR v1.0 (Primary)

**Source**: https://visualcommonsense.com/

### Download

```bash
bash scripts/download_vcr.sh data/vcr
```

### Structure

```
data/vcr/
├── train.jsonl        # ~213k examples
├── val.jsonl          # ~26.5k examples
├── test.jsonl         # ~25.3k examples (no labels)
└── vcr1images/        # ~110k images (~25GB)
    ├── movieclip_00001/
    │   ├── 00001.jpg
    │   └── 00001.metadata.json   # bounding boxes, segmentations
    └── ...
```

### Annotation Format (per line in JSONL)

```json
{
  "annot_id": "val-0",
  "img_fn": "movieclip_00001/00001.jpg",
  "metadata_fn": "movieclip_00001/00001.metadata.json",
  "objects": ["person", "person", "table"],
  "question": ["Why", "is", [0], "looking", "at", [1], "?"],
  "answer_choices": [
    ["Because", [0], "is", "surprised"],
    ["Because", [0], "wants", "to", "talk"],
    ...
  ],
  "rationale_choices": [...],
  "answer_label": 1,
  "rationale_label": 0
}
```

Object references in text are encoded as `[index]` pointing into the `objects` list.
Metadata JSON files contain bounding boxes for each object (used for AOO metric).

### Evaluation Tasks

- **Q->A**: Given image + question, pick correct answer (4 choices)
- **QA->R**: Given image + question + correct answer, pick correct rationale (4 choices)
- **Q->AR**: Both correct (primary metric)

---

## MMMU

**Source**: HuggingFace `MMMU/MMMU`

Downloaded automatically via `datasets` library. No manual setup needed.

```python
from datasets import load_dataset
ds = load_dataset("MMMU/MMMU", "all", split="validation")
```

~900 validation examples across 30+ subjects. Multiple choice with optional images.

---

## MathVista

**Source**: HuggingFace `AI4Math/MathVista`

Downloaded automatically via `datasets` library. No manual setup needed.

```python
from datasets import load_dataset
ds = load_dataset("AI4Math/MathVista", split="testmini")
```

~1000 testmini examples. Mix of multiple choice and free-form answers.
Covers geometry, algebra, statistics, and scientific reasoning with visual context.
