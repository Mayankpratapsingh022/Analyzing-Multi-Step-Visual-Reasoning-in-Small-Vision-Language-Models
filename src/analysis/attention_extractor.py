"""
Attention extraction for decoder-only vision-language models.

Captures the slice of self-attention from the last (next-prediction) token
position to visual-token positions. In decoder-only VLMs like Qwen2-VL and
LLaVA there is no separate cross-attention module — visual tokens live in the
main sequence and attention to them is the operational equivalent of
"vision-language cross-attention."

Per the mentor's guidance, we (a) average across attention heads and
(b) optionally focus on the late-stage layers, since these matrices become
massive and the late layers tend to carry the most interpretable signal for
prediction-time grounding.

Currently implemented: Qwen2-VL family (qwen2-vl-2b, qwen2-vl-7b). The same
pattern extends to LLaVA-NEXT but is left for a follow-up — start with one
clean family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


DEFAULT_LATE_LAYER_FRACTION = 0.75
"""Fraction of decoder layers (counted from the start) to skip when averaging.

0.75 means: take only the last 25% of layers. Mentor's guidance was to focus
on late-stage layers for interpretability. Override per experiment.
"""


@dataclass
class AttentionResult:
    """One example's attention extraction output."""

    heatmap: np.ndarray
    """(H_patch, W_patch) attention from the last query position to visual tokens.

    Already averaged across heads and (the selected subset of) layers, summed
    over the temporal axis if the input has T>1 (single-image VCR has T=1).
    """

    patch_grid_hw: tuple[int, int]
    """(H_patch, W_patch) — the visual patch grid shape after Qwen2-VL's 2x2 merge."""

    predicted_token: str
    """Top-1 next-token prediction from the forward pass (decoded as a string)."""

    last_query_pos: int
    """Index of the prediction position in the input sequence (last non-padding token)."""

    num_image_tokens: int
    """Number of visual tokens fed into the LLM after the vision tower + merge."""

    num_layers_used: int
    """How many late-stage layers were averaged."""


def load_qwen2vl_eager(
    hf_id: str,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> tuple[Qwen2VLForConditionalGeneration, "AutoProcessor"]:
    """Load Qwen2-VL with eager attention so `output_attentions=True` works.

    Flash / SDPA attention kernels do not return the attention probabilities,
    so we must use the slower eager implementation for extraction. Memory is
    fine for qwen2-vl-2b on a single-example forward pass.
    """
    if dtype is None:
        if "cuda" in device and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

    processor_kwargs = {}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(hf_id, **processor_kwargs)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        hf_id,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    return model, processor


def _build_chat_text(processor, image: Image.Image, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_attention_qwen2vl(
    model: Qwen2VLForConditionalGeneration,
    processor: "AutoProcessor",
    image: Image.Image,
    prompt: str,
    late_layer_fraction: float = DEFAULT_LATE_LAYER_FRACTION,
) -> AttentionResult:
    """Run one forward pass and pull out the prediction-position attention to image tokens.

    The forward pass uses `output_attentions=True`. We compute the attention
    averaged across heads, averaged across the late layers, slice the last
    query row, and project that onto the visual-token positions.
    """
    text = _build_chat_text(processor, image, prompt)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)

    attentions = outputs.attentions
    num_layers = len(attentions)
    late_start = int(num_layers * late_layer_fraction)
    late_start = min(late_start, num_layers - 1)
    selected_layers = attentions[late_start:]

    image_token_id = model.config.image_token_id
    input_ids = inputs["input_ids"][0]
    image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    if image_positions.numel() == 0:
        raise RuntimeError(
            "no image tokens found in input_ids; processor or model id may be wrong"
        )

    last_query_pos = int(input_ids.shape[0]) - 1

    # average each layer over heads, then average the late layers, then take the
    # last query row. doing the head-mean per layer keeps the temporary tensor small.
    head_means = []
    for attn in selected_layers:
        head_means.append(attn[0].mean(dim=0).to(torch.float32))
    layer_stack = torch.stack(head_means, dim=0)
    avg_attn = layer_stack.mean(dim=0)
    img_attn = avg_attn[last_query_pos, image_positions].cpu().numpy()

    grid_thw = inputs["image_grid_thw"][0].cpu().numpy()
    t_grid = int(grid_thw[0])
    h_grid = int(grid_thw[1])
    w_grid = int(grid_thw[2])
    # Qwen2-VL applies a 2x2 spatial merge before the LLM, so the visual token
    # grid the LLM sees is half the resolution of the ViT patch grid in each
    # spatial dim.
    h_out = h_grid // 2
    w_out = w_grid // 2
    expected = t_grid * h_out * w_out
    if img_attn.size != expected:
        raise RuntimeError(
            f"image-token count mismatch: got {img_attn.size}, "
            f"expected t_grid*h_merged*w_merged = {t_grid}*{h_out}*{w_out} = {expected}"
        )

    heatmap = img_attn.reshape(t_grid, h_out, w_out)
    if t_grid == 1:
        heatmap = heatmap[0]
    else:
        heatmap = heatmap.sum(axis=0)
    heatmap = heatmap.astype(np.float32)

    finite_mask = np.isfinite(heatmap)
    finite_fraction = float(finite_mask.mean())
    if not finite_mask.any():
        raise RuntimeError(
            "attention heatmap contains no finite values. This usually happens when "
            "eager attention is run in fp16; rerun with bf16 or fp32."
        )
    if finite_fraction < 0.99:
        raise RuntimeError(
            f"attention heatmap has too many non-finite values "
            f"({finite_fraction:.3%} finite). Rerun with bf16 or fp32."
        )
    if finite_fraction < 1.0:
        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)

    logits = outputs.logits[0, -1]
    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "model logits contain non-finite values. This usually happens when "
            "eager attention is run in fp16; rerun with bf16 or fp32."
        )
    pred_id = int(logits.argmax().item())
    predicted_token = processor.tokenizer.decode([pred_id])

    return AttentionResult(
        heatmap=heatmap,
        patch_grid_hw=(h_out, w_out),
        predicted_token=predicted_token,
        last_query_pos=last_query_pos,
        num_image_tokens=int(image_positions.numel()),
        num_layers_used=len(selected_layers),
    )


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Min-max scale into [0, 1] for visualization. Returns float32."""
    finite = heatmap[np.isfinite(heatmap)]
    if finite.size == 0:
        return np.zeros_like(heatmap, dtype=np.float32)
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-12:
        return np.zeros_like(heatmap, dtype=np.float32)
    normalized = (heatmap - lo) / (hi - lo)
    return np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def upsample_heatmap_to_image(heatmap: np.ndarray, image_size_wh: tuple[int, int]) -> np.ndarray:
    """Bilinear-upsample a (H_patch, W_patch) heatmap to image (H, W) resolution.

    image_size_wh is (W, H) matching PIL's Image.size convention.
    Returns float32 in the same value range as the input.
    """
    w_img, h_img = image_size_wh
    h_norm = normalize_heatmap(heatmap)
    heatmap_pil = Image.fromarray((h_norm * 255).astype(np.uint8))
    heatmap_pil = heatmap_pil.resize((w_img, h_img), Image.BILINEAR)
    return np.array(heatmap_pil).astype(np.float32) / 255.0
