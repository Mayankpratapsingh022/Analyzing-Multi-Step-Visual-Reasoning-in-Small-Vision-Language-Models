"""Moondream2 (1.8B) wrapper."""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from .vlm_evaluator import VLMEvaluator, register_evaluator


@register_evaluator("moondream2")
class Moondream2Evaluator(VLMEvaluator):
    """vikhyatk/moondream2 -- 1.8B edge-device VLM."""

    model_name = "moondream2"
    param_count = "1.8B"
    HF_ID = "vikhyatk/moondream2"

    def load_model(self, device: str = "cuda", quantization: Optional[str] = None):
        self.device = device
        dtype = torch.float16 if "cuda" in device else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.HF_ID,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device if quantization else None,
            quantization_config=self.get_quantization_config(quantization),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.HF_ID, trust_remote_code=True
        )

        if not quantization:
            self.model.to(device)
        self.model.eval()

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
        enc_image = self.model.encode_image(image)
        answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
        return answer
