"""LLaVA family wrapper (LLaVA-1.5, LLaVA-NeXT / 1.6)."""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration

from .vlm_evaluator import VLMEvaluator, register_evaluator


class LLaVABase(VLMEvaluator):
    """Shared logic for LLaVA-family models via HuggingFace."""

    HF_ID: str = ""
    MODEL_CLASS = LlavaForConditionalGeneration

    def load_model(self, device: str = "cuda", quantization: Optional[str] = None):
        self.device = device
        dtype = torch.float16 if "cuda" in device else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.HF_ID)

        load_kwargs = {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }

        if quantization:
            load_kwargs["device_map"] = "auto"
            load_kwargs["quantization_config"] = self.get_quantization_config(quantization)
        else:
            load_kwargs["device_map"] = device

        self.model = self.MODEL_CLASS.from_pretrained(self.HF_ID, **load_kwargs)
        self.model.eval()

    def _build_chat_text(self, prompt: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
        text = self._build_chat_text(prompt)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]

    def generate_batch(self, images: list, prompts: list, max_new_tokens: int = 512) -> list:
        texts = [self._build_chat_text(p) for p in prompts]
        inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)


@register_evaluator("llava-1.5-7b")
class LLaVA15_7BEvaluator(LLaVABase):
    model_name = "llava-1.5-7b"
    param_count = "7B"
    HF_ID = "llava-hf/llava-1.5-7b-hf"
    MODEL_CLASS = LlavaForConditionalGeneration


@register_evaluator("llava-1.5-13b")
class LLaVA15_13BEvaluator(LLaVABase):
    model_name = "llava-1.5-13b"
    param_count = "13B"
    HF_ID = "llava-hf/llava-1.5-13b-hf"
    MODEL_CLASS = LlavaForConditionalGeneration


@register_evaluator("llava-next-7b")
class LLaVANext7BEvaluator(LLaVABase):
    model_name = "llava-next-7b"
    param_count = "7B"
    HF_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
    MODEL_CLASS = LlavaNextForConditionalGeneration


@register_evaluator("llava-1.6-34b")
class LLaVA16_34BEvaluator(LLaVABase):
    model_name = "llava-1.6-34b"
    param_count = "34B"
    HF_ID = "llava-hf/llava-v1.6-34b-hf"
    MODEL_CLASS = LlavaNextForConditionalGeneration
