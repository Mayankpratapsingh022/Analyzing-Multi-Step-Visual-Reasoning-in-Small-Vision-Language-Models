"""Phi-3-Vision (4.2B) wrapper."""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .vlm_evaluator import VLMEvaluator, register_evaluator


@register_evaluator("phi3-vision")
class Phi3VisionEvaluator(VLMEvaluator):
    """microsoft/Phi-3-vision-128k-instruct -- 4.2B reasoning-optimized VLM."""

    model_name = "phi3-vision"
    param_count = "4.2B"
    HF_ID = "microsoft/Phi-3-vision-128k-instruct"

    def load_model(self, device: str = "cuda", quantization: Optional[str] = None):
        self.device = device
        dtype = torch.float16 if "cuda" in device else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            self.HF_ID, trust_remote_code=True
        )

        load_kwargs = {
            "dtype": dtype,
            "trust_remote_code": True,
            "_attn_implementation": "eager",  # fallback if flash-attn missing
        }

        if quantization:
            load_kwargs["device_map"] = "auto"
            load_kwargs["quantization_config"] = self.get_quantization_config(quantization)
        else:
            load_kwargs["device_map"] = device

        self.model = AutoModelForCausalLM.from_pretrained(self.HF_ID, **load_kwargs)
        self.model.eval()

    def _build_chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
        return self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
        formatted = self._build_chat_text(prompt)
        inputs = self.processor(formatted, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]

    def generate_batch(self, images: list, prompts: list, max_new_tokens: int = 512) -> list:
        texts = [self._build_chat_text(p) for p in prompts]
        inputs = self.processor(texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)
