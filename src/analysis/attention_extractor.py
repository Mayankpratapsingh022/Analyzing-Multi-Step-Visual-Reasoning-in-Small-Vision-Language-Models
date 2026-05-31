"""
Attention extraction for decoder-only vision-language models.

Captures self-attention-derived scores from the last (next-prediction) token
position to visual-token positions. In decoder-only VLMs like Qwen2-VL and
LLaVA there is no separate cross-attention module; visual tokens live in the
main sequence, so text-to-visual self-attention is the available proxy for
vision-language grounding.

Per the mentor's guidance, we (a) average across attention heads and
(b) optionally focus on the late-stage layers, since these matrices become
massive and the late layers tend to carry the most interpretable signal for
prediction-time grounding.

Currently implemented: Qwen2-VL family (qwen2-vl-2b, qwen2-vl-7b). The same
pattern extends to LLaVA-NEXT but is left for a follow-up: start with one
clean family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)


DEFAULT_LATE_LAYER_FRACTION = 0.75
"""Fraction of decoder layers (counted from the start) to skip when averaging.

0.75 means: take only the last 25% of layers. Mentor's guidance was to focus
on late-stage layers for interpretability. Override per experiment.
"""

DEFAULT_BASELINE_PROMPT = (
    "Look at the image. Answer with only the letter A, B, C, or D."
)

ATTENTION_METHODS = {"raw", "rollout", "contrastive_rollout"}


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

    method: str
    """Attention scoring method used to produce `heatmap`."""

    image_attention_mass: float
    """Total unnormalized attention mass assigned to image tokens."""

    diagnostics: dict
    """Quality checks for suspicious artifacts such as border-dominated maps."""


@dataclass
class OcclusionResult:
    """Grid occlusion-sensitivity output for one image/prompt pair."""

    heatmap: np.ndarray
    """(H_grid, W_grid) log-probability drop when each image cell is masked."""

    grid_hw: tuple[int, int]
    """Occlusion grid shape used for the sensitivity scan."""

    predicted_token: str
    """Top-1 next-token prediction on the unmasked image."""

    target_token_id: int
    """Token id whose log-probability is measured under occlusion."""

    target_token: str
    """Decoded token corresponding to `target_token_id`."""

    base_logprob: float
    """Log-probability of the target token on the unmasked image."""

    diagnostics: dict
    """Quality checks for suspicious artifacts such as border-dominated maps."""


def load_qwen2vl_eager(
    hf_id: str,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    device_map: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
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
    if load_in_4bit and load_in_8bit:
        raise ValueError("Only one of load_in_4bit/load_in_8bit can be enabled.")

    processor = AutoProcessor.from_pretrained(hf_id, **processor_kwargs)
    model_kwargs = {
        "dtype": dtype,
        "attn_implementation": "eager",
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = device_map or "auto"
    elif load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["device_map"] = device_map or "auto"
    elif device_map:
        model_kwargs["device_map"] = device_map

    model = Qwen2VLForConditionalGeneration.from_pretrained(hf_id, **model_kwargs)
    if "device_map" not in model_kwargs:
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


def _forward_qwen2vl(
    model: Qwen2VLForConditionalGeneration,
    processor: "AutoProcessor",
    image: Image.Image,
    prompt: str,
):
    text = _build_chat_text(processor, image, prompt)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)
    return inputs, outputs


def _forward_qwen2vl_logits(
    model: Qwen2VLForConditionalGeneration,
    processor: "AutoProcessor",
    image: Image.Image,
    prompt: str,
) -> torch.Tensor:
    text = _build_chat_text(processor, image, prompt)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=False, use_cache=False)
    logits = outputs.logits[0, -1].to(torch.float32)
    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "model logits contain non-finite values. Rerun with bf16 or fp32."
        )
    return logits


def _selected_attention_layers(
    attentions: tuple[torch.Tensor, ...],
    late_layer_fraction: float,
) -> tuple[list[torch.Tensor], int]:
    num_layers = len(attentions)
    late_start = int(num_layers * late_layer_fraction)
    late_start = min(late_start, num_layers - 1)
    selected_layers = attentions[late_start:]

    head_means = []
    for attn in selected_layers:
        layer = attn[0].mean(dim=0).to(torch.float32)
        head_means.append(torch.nan_to_num(layer, nan=0.0, posinf=0.0, neginf=0.0))
    return head_means, late_start


def _visual_token_layout(
    model: Qwen2VLForConditionalGeneration,
    inputs: dict,
) -> tuple[torch.Tensor, tuple[int, int], int]:
    image_token_id = model.config.image_token_id
    input_ids = inputs["input_ids"][0]
    image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    if image_positions.numel() == 0:
        raise RuntimeError(
            "no image tokens found in input_ids; processor or model id may be wrong"
        )

    grid_thw = inputs["image_grid_thw"][0].cpu().numpy()
    t_grid = int(grid_thw[0])
    h_grid = int(grid_thw[1])
    w_grid = int(grid_thw[2])
    vision_config = getattr(model.config, "vision_config", None)
    if isinstance(vision_config, dict):
        spatial_merge_size = vision_config.get("spatial_merge_size", 2)
    else:
        spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
    h_out = h_grid // spatial_merge_size
    w_out = w_grid // spatial_merge_size
    expected = t_grid * h_out * w_out
    if int(image_positions.numel()) != expected:
        raise RuntimeError(
            f"image-token count mismatch: got {int(image_positions.numel())}, "
            f"expected t_grid*h_merged*w_merged = {t_grid}*{h_out}*{w_out} = {expected}"
        )
    return image_positions, (h_out, w_out), t_grid


def _rollout_scores(
    head_means: list[torch.Tensor],
    target_pos: int,
    image_positions: torch.Tensor,
) -> torch.Tensor:
    seq_len = head_means[0].shape[-1]
    eye = torch.eye(seq_len, device=head_means[0].device, dtype=torch.float32)
    rollout = eye
    for attn in head_means:
        attn = attn + eye
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = attn @ rollout
    return rollout[target_pos, image_positions]


def _raw_scores(
    head_means: list[torch.Tensor],
    target_pos: int,
    image_positions: torch.Tensor,
) -> torch.Tensor:
    avg_attn = torch.stack(head_means, dim=0).mean(dim=0)
    return avg_attn[target_pos, image_positions]


def _scores_to_heatmap(
    scores: torch.Tensor,
    patch_grid_hw: tuple[int, int],
    t_grid: int,
    spatial_normalize: bool = True,
) -> tuple[np.ndarray, float]:
    scores = torch.nan_to_num(scores.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scores = scores.clamp_min(0.0)
    image_attention_mass = float(scores.sum().item())
    if spatial_normalize and image_attention_mass > 0:
        scores = scores / image_attention_mass

    h_out, w_out = patch_grid_hw
    heatmap = scores.detach().cpu().numpy().reshape(t_grid, h_out, w_out)
    if t_grid == 1:
        heatmap = heatmap[0]
    else:
        heatmap = heatmap.sum(axis=0)
    return heatmap.astype(np.float32), image_attention_mass


def _attention_heatmap_from_forward(
    model: Qwen2VLForConditionalGeneration,
    inputs: dict,
    outputs,
    method: str,
    late_layer_fraction: float,
    spatial_normalize: bool,
) -> tuple[np.ndarray, tuple[int, int], int, int, float, int]:
    if method not in {"raw", "rollout"}:
        raise ValueError(f"internal method must be raw or rollout, got {method!r}")

    image_positions, patch_grid_hw, t_grid = _visual_token_layout(model, inputs)
    input_ids = inputs["input_ids"][0]
    target_pos = int(input_ids.shape[0]) - 1
    head_means, _ = _selected_attention_layers(outputs.attentions, late_layer_fraction)

    if method == "rollout":
        scores = _rollout_scores(head_means, target_pos, image_positions)
    else:
        scores = _raw_scores(head_means, target_pos, image_positions)

    heatmap, image_attention_mass = _scores_to_heatmap(
        scores,
        patch_grid_hw,
        t_grid,
        spatial_normalize=spatial_normalize,
    )
    return (
        heatmap,
        patch_grid_hw,
        target_pos,
        len(head_means),
        image_attention_mass,
        int(image_positions.numel()),
    )


def _validate_heatmap(heatmap: np.ndarray) -> np.ndarray:
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
    return heatmap.astype(np.float32)


def heatmap_diagnostics(heatmap: np.ndarray, border_width: int = 1) -> dict:
    """Return lightweight quality checks for a spatial attention map."""
    finite = np.nan_to_num(heatmap.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    finite = finite - finite.min()
    total = float(finite.sum())
    h, w = finite.shape
    result = {
        "heatmap_min": float(np.nanmin(heatmap)),
        "heatmap_max": float(np.nanmax(heatmap)),
        "heatmap_finite_fraction": float(np.isfinite(heatmap).mean()),
        "border_width": int(border_width),
    }
    if total <= 0:
        result.update({
            "border_mass": None,
            "border_area_fraction": None,
            "normalized_entropy": None,
            "peak_to_mean": None,
            "top_patch_yx": None,
            "quality_warning": "empty_or_constant_heatmap",
        })
        return result

    bw = max(0, min(int(border_width), h // 2, w // 2))
    border_mask = np.zeros((h, w), dtype=bool)
    if bw > 0:
        border_mask[:bw, :] = True
        border_mask[-bw:, :] = True
        border_mask[:, :bw] = True
        border_mask[:, -bw:] = True
    border_mass = float(finite[border_mask].sum() / total) if bw > 0 else 0.0
    border_area_fraction = float(border_mask.mean()) if bw > 0 else 0.0

    probs = finite.reshape(-1) / total
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(probs.size))
    mean = float(finite.mean())
    peak_to_mean = float(finite.max() / mean) if mean > 0 else None
    top_y, top_x = np.unravel_index(int(finite.argmax()), finite.shape)

    warning = None
    if bw > 0 and border_mass > max(0.35, border_area_fraction * 2.5):
        warning = "border_dominated_heatmap"
    elif entropy > 0.95:
        warning = "nearly_uniform_heatmap"

    result.update({
        "border_mass": border_mass,
        "border_area_fraction": border_area_fraction,
        "normalized_entropy": entropy,
        "peak_to_mean": peak_to_mean,
        "top_patch_yx": [int(top_y), int(top_x)],
        "quality_warning": warning,
    })
    return result


def suppress_heatmap_border(heatmap: np.ndarray, border_width: int) -> np.ndarray:
    """Zero edge patches for visualization-only ablations."""
    bw = int(border_width)
    if bw <= 0:
        return heatmap
    h, w = heatmap.shape
    bw = min(bw, h // 2, w // 2)
    suppressed = heatmap.copy()
    suppressed[:bw, :] = 0
    suppressed[-bw:, :] = 0
    suppressed[:, :bw] = 0
    suppressed[:, -bw:] = 0
    if np.isfinite(suppressed).any() and float(np.nanmax(suppressed)) > 0:
        return suppressed
    return heatmap


def _occlude_cell(
    image: Image.Image,
    row: int,
    col: int,
    grid_hw: tuple[int, int],
    fill: str = "mean",
) -> Image.Image:
    arr = np.array(image.convert("RGB")).copy()
    h_img, w_img = arr.shape[:2]
    rows, cols = grid_hw
    y0 = int(round(row * h_img / rows))
    y1 = int(round((row + 1) * h_img / rows))
    x0 = int(round(col * w_img / cols))
    x1 = int(round((col + 1) * w_img / cols))

    if fill == "black":
        value = np.array([0, 0, 0], dtype=np.uint8)
    elif fill == "gray":
        value = np.array([127, 127, 127], dtype=np.uint8)
    elif fill == "mean":
        value = arr.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    else:
        raise ValueError(f"unsupported occlusion fill {fill!r}")

    arr[y0:y1, x0:x1] = value
    return Image.fromarray(arr)


def extract_occlusion_qwen2vl(
    model: Qwen2VLForConditionalGeneration,
    processor: "AutoProcessor",
    image: Image.Image,
    prompt: str,
    grid_hw: tuple[int, int] = (7, 7),
    fill: str = "mean",
    target_token_id: int | None = None,
) -> OcclusionResult:
    """Estimate visual importance by masking image cells and measuring output drop.

    This is slower than attention extraction but more faithful as a sanity
    check: a cell is important if masking it lowers the model's own next-token
    log-probability for the unmasked prediction.
    """
    if grid_hw[0] <= 0 or grid_hw[1] <= 0:
        raise ValueError(f"grid_hw must be positive, got {grid_hw}")

    base_logits = _forward_qwen2vl_logits(model, processor, image, prompt)
    if target_token_id is None:
        target_token_id = int(base_logits.argmax().item())
    target_token = processor.tokenizer.decode([target_token_id])
    predicted_token = processor.tokenizer.decode([int(base_logits.argmax().item())])
    base_logprob = torch.log_softmax(base_logits, dim=-1)[target_token_id]

    rows, cols = grid_hw
    heatmap = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            masked = _occlude_cell(image, r, c, grid_hw, fill=fill)
            masked_logits = _forward_qwen2vl_logits(model, processor, masked, prompt)
            masked_logprob = torch.log_softmax(masked_logits, dim=-1)[target_token_id]
            drop = float((base_logprob - masked_logprob).item())
            heatmap[r, c] = max(0.0, drop)

    heatmap = _validate_heatmap(heatmap)
    return OcclusionResult(
        heatmap=heatmap,
        grid_hw=grid_hw,
        predicted_token=predicted_token,
        target_token_id=int(target_token_id),
        target_token=target_token,
        base_logprob=float(base_logprob.item()),
        diagnostics={
            **heatmap_diagnostics(heatmap),
            "occlusion_fill": fill,
        },
    )


def extract_attention_qwen2vl(
    model: Qwen2VLForConditionalGeneration,
    processor: "AutoProcessor",
    image: Image.Image,
    prompt: str,
    late_layer_fraction: float = DEFAULT_LATE_LAYER_FRACTION,
    method: str = "contrastive_rollout",
    baseline_prompt: str = DEFAULT_BASELINE_PROMPT,
    spatial_normalize: bool = True,
    suppress_border_patches: int = 0,
) -> AttentionResult:
    """Run one forward pass and pull out the prediction-position attention to image tokens.

    The forward pass uses `output_attentions=True`. `raw` averages heads/layers
    and slices the last query row. `rollout` composes attention through the
    selected layers with residual connections. `contrastive_rollout` subtracts
    a same-image neutral-prompt rollout map to reduce image-position priors.
    """
    if method not in ATTENTION_METHODS:
        raise ValueError(
            f"unsupported attention method {method!r}; "
            f"expected one of {sorted(ATTENTION_METHODS)}"
        )

    inputs, outputs = _forward_qwen2vl(model, processor, image, prompt)
    base_method = "rollout" if method == "contrastive_rollout" else method
    (
        heatmap,
        patch_grid_hw,
        last_query_pos,
        num_layers_used,
        image_attention_mass,
        num_image_tokens,
    ) = (
        _attention_heatmap_from_forward(
            model,
            inputs,
            outputs,
            base_method,
            late_layer_fraction,
            spatial_normalize,
        )
    )

    baseline_mass = None
    if method == "contrastive_rollout":
        baseline_inputs, baseline_outputs = _forward_qwen2vl(
            model, processor, image, baseline_prompt
        )
        baseline_heatmap, baseline_grid_hw, _, _, baseline_mass, _ = (
            _attention_heatmap_from_forward(
                model,
                baseline_inputs,
                baseline_outputs,
                "rollout",
                late_layer_fraction,
                spatial_normalize,
            )
        )
        if baseline_grid_hw != patch_grid_hw:
            raise RuntimeError(
                f"baseline grid mismatch: prompt grid {patch_grid_hw}, "
                f"baseline grid {baseline_grid_hw}"
            )
        heatmap = np.maximum(heatmap - baseline_heatmap, 0.0).astype(np.float32)

    heatmap = _validate_heatmap(heatmap)
    heatmap = suppress_heatmap_border(heatmap, suppress_border_patches)

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
        patch_grid_hw=patch_grid_hw,
        predicted_token=predicted_token,
        last_query_pos=last_query_pos,
        num_image_tokens=num_image_tokens,
        num_layers_used=num_layers_used,
        method=method,
        image_attention_mass=image_attention_mass,
        diagnostics={
            **heatmap_diagnostics(heatmap),
            "baseline_image_attention_mass": baseline_mass,
            "spatial_normalized": spatial_normalize,
            "suppressed_border_patches": int(suppress_border_patches),
        },
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
