"""
Data loaders for VCR, MMMU, and MathVista datasets.
Each loader returns a standardized dict per example for the evaluation pipeline.
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names


class VCRDataset(Dataset):
    """
    Visual Commonsense Reasoning (VCR) v1.0 dataset.

    Each example contains:
    - image from a movie scene
    - question with object references ([person1], [object2], etc.)
    - 4 answer choices + 4 rationale choices
    - ground truth labels for answer and rationale
    - object bounding boxes (metadata)

    Data format on disk:
        vcr_dir/
            val.jsonl          # annotations
            train.jsonl
            test.jsonl
            vcr1images/        # images organized by movie
                movieclip_xxx/
                    xxx.jpg
    """

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        vcr_dir: str,
        split: str = "val",
        subset_pct: Optional[float] = None,
        seed: int = 42,
    ):
        assert split in self.SPLITS, f"split must be one of {self.SPLITS}"
        self.vcr_dir = Path(vcr_dir)
        self.image_dir = self.vcr_dir / "vcr1images"
        self.split = split
        self.seed = seed

        annot_path = self.vcr_dir / f"{split}.jsonl"
        if not annot_path.exists():
            raise FileNotFoundError(
                f"VCR annotations not found at {annot_path}. "
                f"Run scripts/download_vcr.sh first."
            )

        self.annotations = []
        with open(annot_path) as f:
            for line in f:
                self.annotations.append(json.loads(line.strip()))

        if subset_pct is not None and 0 < subset_pct < 100:
            rng = random.Random(seed)
            k = max(1, int(len(self.annotations) * subset_pct / 100))
            self.annotations = rng.sample(self.annotations, k)

    def __len__(self):
        return len(self.annotations)

    def _resolve_references(self, tokens: list, objects: list[str]) -> str:
        """Convert token list with object indices into readable text.

        VCR stores text as lists like: ["Why", "is", [0], "looking", "at", [1], "?"]
        where integers refer to object names.
        """
        parts = []
        for token in tokens:
            if isinstance(token, list):
                idx = token[0]
                if idx < len(objects):
                    parts.append(objects[idx])
                else:
                    parts.append(f"[object{idx}]")
            else:
                parts.append(str(token))
        return " ".join(parts)

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]

        img_path = self.image_dir / ann["img_fn"]
        image = Image.open(img_path).convert("RGB")

        objects = ann.get("objects", [])
        metadata_fn = ann.get("metadata_fn", "")

        question = self._resolve_references(ann["question"], objects)

        answer_choices = [
            self._resolve_references(choice, objects)
            for choice in ann["answer_choices"]
        ]
        rationale_choices = [
            self._resolve_references(choice, objects)
            for choice in ann["rationale_choices"]
        ]

        result = {
            "image": image,
            "question": question,
            "answer_choices": answer_choices,
            "rationale_choices": rationale_choices,
            "answer_label": ann["answer_label"],
            "rationale_label": ann["rationale_label"],
            "metadata": {
                "annot_id": ann.get("annot_id", idx),
                "img_fn": ann["img_fn"],
                "objects": objects,
                "metadata_fn": metadata_fn,
            },
        }

        # Bounding boxes if available (for AOO metric later)
        if metadata_fn:
            meta_path = self.image_dir / metadata_fn
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                result["metadata"]["boxes"] = meta.get("boxes", [])

        return result


class MMMUDataset(Dataset):
    """
    MMMU dataset via HuggingFace datasets.
    Massive Multi-discipline Multimodal Understanding benchmark.

    Args:
        subject: HuggingFace config name for a specific subject, e.g. "Math",
                 "Physics", "Chemistry". Pass None (default) to load all subjects.
                 Full list: https://huggingface.co/datasets/MMMU/MMMU
        max_samples: Exact number of examples to use. Takes priority over
                     subset_pct when both are provided.
        subset_pct: Percentage of examples to sample (0-100). Used only when
                    max_samples is None.
    """

    def __init__(
        self,
        split: str = "validation",
        subject: Optional[str] = None,
        subset_pct: Optional[float] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
    ):
        if subject:
            self.ds = load_dataset("MMMU/MMMU", subject, split=split, cache_dir=cache_dir)
        else:
            # MMMU has no "all" config — load each subject config and concatenate
            configs = get_dataset_config_names("MMMU/MMMU")
            parts = [
                load_dataset("MMMU/MMMU", cfg, split=split, cache_dir=cache_dir)
                for cfg in configs
            ]
            self.ds = concatenate_datasets(parts)

        # max_samples takes priority over subset_pct
        if max_samples is not None:
            k = min(max_samples, len(self.ds))
            rng = random.Random(seed)
            indices = rng.sample(range(len(self.ds)), k)
            self.ds = self.ds.select(indices)
        elif subset_pct is not None and 0 < subset_pct < 100:
            k = max(1, int(len(self.ds) * subset_pct / 100))
            rng = random.Random(seed)
            indices = rng.sample(range(len(self.ds)), k)
            self.ds = self.ds.select(indices)

    def __len__(self):
        return len(self.ds)

    @staticmethod
    def _to_str(val) -> str:
        """Coerce any MMMU field value to a plain string.

        MMMU stores some fields (options, questions) as lists of strings
        rather than plain strings in certain examples.
        """
        if isinstance(val, list):
            return " ".join(str(v) for v in val if v is not None)
        return str(val) if val is not None else ""

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]

        # MMMU can have multiple images (image_1 through image_7)
        images = []
        for i in range(1, 8):
            img = row.get(f"image_{i}")
            if img is not None:
                if isinstance(img, Image.Image):
                    images.append(img.convert("RGB"))

        # MMMU stores choices in a single `options` column as a Python-literal
        # list string, e.g. "['Aurelia', 'Matilda', 'Hermione', 'Juno']".
        # Some older schemas use per-letter `option_A`..`option_E` fields.
        raw_opts = row.get("options")
        choices = []
        if raw_opts is not None:
            if isinstance(raw_opts, str):
                try:
                    import ast
                    parsed = ast.literal_eval(raw_opts)
                    if isinstance(parsed, list):
                        raw_opts = parsed
                except (ValueError, SyntaxError):
                    raw_opts = [raw_opts]
            if isinstance(raw_opts, list):
                choices = [self._to_str(o) for o in raw_opts if o is not None and o != ""]
        if not choices:
            for opt_key in ["A", "B", "C", "D", "E"]:
                val = row.get(f"option_{opt_key}", row.get(opt_key))
                if val is not None and val != "":
                    choices.append(self._to_str(val))

        question = self._to_str(row.get("question", ""))

        ans = row.get("answer", "")
        ans = self._to_str(ans).strip()
        answer_label = (
            ord(ans.upper()) - ord("A")
            if len(ans) == 1 and ans.upper() in "ABCDE"
            else -1
        )

        return {
            "image": images[0] if images else None,
            "images": images,
            "question": question,
            "answer_choices": choices,
            "answer_label": answer_label,
            "metadata": {
                "id": row.get("id", idx),
                "subject": row.get("subject", ""),
                "subfield": row.get("subfield", ""),
                "question_type": row.get("question_type", ""),
            },
        }


class MathVistaDataset(Dataset):
    """
    MathVista dataset via HuggingFace datasets.
    Mathematical reasoning in visual contexts.

    Args:
        max_samples: Exact number of examples to use. Takes priority over
                     subset_pct when both are provided.
        subset_pct: Percentage of examples to sample (0-100). Used only when
                    max_samples is None.
    """

    def __init__(
        self,
        split: str = "testmini",
        subset_pct: Optional[float] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
    ):
        self.ds = load_dataset("AI4Math/MathVista", split=split, cache_dir=cache_dir)

        # max_samples takes priority over subset_pct
        if max_samples is not None:
            k = min(max_samples, len(self.ds))
            rng = random.Random(seed)
            indices = rng.sample(range(len(self.ds)), k)
            self.ds = self.ds.select(indices)
        elif subset_pct is not None and 0 < subset_pct < 100:
            k = max(1, int(len(self.ds) * subset_pct / 100))
            rng = random.Random(seed)
            indices = rng.sample(range(len(self.ds)), k)
            self.ds = self.ds.select(indices)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]

        image = row.get("image") or row.get("decoded_image")
        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        choices = row.get("choices", [])
        if isinstance(choices, str):
            try:
                choices = json.loads(choices)
            except json.JSONDecodeError:
                choices = []

        answer = row.get("answer", "")
        answer_label = -1
        if choices and answer in choices:
            answer_label = choices.index(answer)

        return {
            "image": image,
            "question": row.get("question", ""),
            "answer_choices": choices,
            "answer_label": answer_label,
            "answer_text": answer,
            "metadata": {
                "pid": row.get("pid", idx),
                "question_type": row.get("question_type", ""),
                "answer_type": row.get("answer_type", ""),
                "category": row.get("metadata", {}).get("category", "")
                if isinstance(row.get("metadata"), dict)
                else "",
            },
        }


def get_dataset(name: str, **kwargs) -> Dataset:
    """Factory function to get a dataset by name.

    Common kwargs (forwarded to each dataset class):
        max_samples (int): exact number of examples to load
        subset_pct (float): percentage of examples to sample (ignored when max_samples set)
        seed (int): random seed for sampling
        subject (str): MMMU only — subject config name, e.g. "Math"
    """
    datasets_map = {
        "vcr": VCRDataset,
        "mmmu": MMMUDataset,
        "mathvista": MathVistaDataset,
    }
    if name not in datasets_map:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(datasets_map.keys())}")
    return datasets_map[name](**kwargs)
