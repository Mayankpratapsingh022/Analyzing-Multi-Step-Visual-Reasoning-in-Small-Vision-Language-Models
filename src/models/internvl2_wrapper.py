"""InternVL2 (2B, 8B, 26B) wrapper."""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .vlm_evaluator import VLMEvaluator, register_evaluator


class InternVL2Base(VLMEvaluator):
    """Shared logic for InternVL2 variants.

    InternVL2 uses a custom chat interface with dynamic resolution.
    The model's `chat` method handles image preprocessing internally.
    """

    HF_ID: str = ""

    def load_model(self, device: str = "cuda", quantization: Optional[str] = None):
        self.device = device
        dtype = torch.float16 if "cuda" in device else torch.float32

        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if quantization:
            load_kwargs["device_map"] = "auto"
            load_kwargs["quantization_config"] = self.get_quantization_config(quantization)
        else:
            load_kwargs["device_map"] = device

        self.model = AutoModel.from_pretrained(self.HF_ID, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.HF_ID, trust_remote_code=True
        )
        self.model.eval()

    def _preprocess_image(self, image: Image.Image):
        """Use InternVL2's dynamic preprocessing.

        Falls back to basic resize if the custom function isn't available.
        """
        try:
            from transformers import AutoImageProcessor
            img_processor = AutoImageProcessor.from_pretrained(
                self.HF_ID, trust_remote_code=True
            )
            return img_processor(images=image, return_tensors="pt")
        except Exception:
            # Fallback: use model's built-in image handling
            return image

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
        # InternVL2 models typically expose a .chat() method
        if hasattr(self.model, "chat"):
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
            response = self.model.chat(
                self.tokenizer,
                image,
                prompt,
                generation_config,
            )
            return response

        # Fallback: manual generation
        messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        pixel_values = self._preprocess_image(image)

        if isinstance(pixel_values, dict):
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            inputs.update({k: v.to(self.model.device) for k, v in pixel_values.items()})
        else:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


@register_evaluator("internvl2-2b")
class InternVL2_2BEvaluator(InternVL2Base):
    model_name = "internvl2-2b"
    param_count = "2B"
    HF_ID = "OpenGVLab/InternVL2-2B"


@register_evaluator("internvl2-8b")
class InternVL2_8BEvaluator(InternVL2Base):
    model_name = "internvl2-8b"
    param_count = "8B"
    HF_ID = "OpenGVLab/InternVL2-8B"


@register_evaluator("internvl2-26b")
class InternVL2_26BEvaluator(InternVL2Base):
    model_name = "internvl2-26b"
    param_count = "26B"
    HF_ID = "OpenGVLab/InternVL2-26B"
