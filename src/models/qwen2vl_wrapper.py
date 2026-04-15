"""Qwen2-VL (2B and 7B) wrapper."""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .vlm_evaluator import VLMEvaluator, register_evaluator


class Qwen2VLBase(VLMEvaluator):
    """Shared logic for Qwen2-VL variants."""

    HF_ID: str = ""

    def load_model(self, device: str = "cuda", quantization: Optional[str] = None):
        self.device = device
        dtype = torch.float16 if "cuda" in device else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.HF_ID)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.HF_ID,
            dtype=dtype,
            device_map=device if quantization else None,
            quantization_config=self.get_quantization_config(quantization),
        )

        if not quantization:
            self.model.to(device)
        self.model.eval()

    def _build_chat_text(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
        text = self._build_chat_text(image, prompt)
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def generate_batch(self, images: list, prompts: list, max_new_tokens: int = 512) -> list:
        texts = [self._build_chat_text(img, p) for img, p in zip(images, prompts)]
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)


@register_evaluator("qwen2-vl-2b")
class Qwen2VL2BEvaluator(Qwen2VLBase):
    model_name = "qwen2-vl-2b"
    param_count = "2B"
    HF_ID = "Qwen/Qwen2-VL-2B-Instruct"


@register_evaluator("qwen2-vl-7b")
class Qwen2VL7BEvaluator(Qwen2VLBase):
    model_name = "qwen2-vl-7b"
    param_count = "7B"
    HF_ID = "Qwen/Qwen2-VL-7B-Instruct"
