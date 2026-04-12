"""API-based model wrappers for GPT-4o and Claude.

Supports concurrent batch requests via ThreadPoolExecutor.
"""

import base64
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image

from .vlm_evaluator import VLMEvaluator, register_evaluator

CACHE_DIR = Path("results/.api_cache")


def _image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _cache_key(model: str, image: Image.Image, prompt: str) -> str:
    img_hash = hashlib.md5(image.tobytes()[:4096]).hexdigest()[:12]
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
    return f"{model}_{img_hash}_{prompt_hash}"


def _load_cached(key: str) -> Optional[str]:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["response"]
    return None


def _save_cache(key: str, response: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps({"response": response}))


@register_evaluator("gpt-4o")
class GPT4oEvaluator(VLMEvaluator):
    """OpenAI GPT-4o via API. Requires OPENAI_API_KEY env var."""

    model_name = "gpt-4o"
    param_count = "closed"

    def load_model(self, device: str = "cpu", quantization: Optional[str] = None):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)
        self.device = "api"

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
        key = _cache_key("gpt4o", image, prompt)
        cached = _load_cached(key)
        if cached is not None:
            return cached

        b64 = _image_to_base64(image)

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    max_tokens=max_new_tokens,
                    temperature=0,
                )
                text = response.choices[0].message.content
                _save_cache(key, text)
                return text
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return f"[API_ERROR: {e}]"

    def generate_batch(self, images: list, prompts: list, max_new_tokens: int = 512) -> list:
        """Concurrent API calls via thread pool (up to 8 parallel)."""
        results = [None] * len(images)
        with ThreadPoolExecutor(max_workers=min(8, len(images))) as pool:
            futures = {
                pool.submit(self.generate, img, p, max_new_tokens): i
                for i, (img, p) in enumerate(zip(images, prompts))
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def auto_batch_size(self) -> int:
        return 8  # API concurrency, not VRAM-bound


@register_evaluator("claude")
class ClaudeEvaluator(VLMEvaluator):
    """Anthropic Claude via API. Requires ANTHROPIC_API_KEY env var."""

    model_name = "claude"
    param_count = "closed"

    def load_model(self, device: str = "cpu", quantization: Optional[str] = None):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("Set ANTHROPIC_API_KEY environment variable.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.device = "api"

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
        key = _cache_key("claude", image, prompt)
        cached = _load_cached(key)
        if cached is not None:
            return cached

        b64 = _image_to_base64(image)

        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=max_new_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": b64,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )
                text = response.content[0].text
                _save_cache(key, text)
                return text
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return f"[API_ERROR: {e}]"

    def generate_batch(self, images: list, prompts: list, max_new_tokens: int = 512) -> list:
        """Concurrent API calls via thread pool (up to 8 parallel)."""
        results = [None] * len(images)
        with ThreadPoolExecutor(max_workers=min(8, len(images))) as pool:
            futures = {
                pool.submit(self.generate, img, p, max_new_tokens): i
                for i, (img, p) in enumerate(zip(images, prompts))
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def auto_batch_size(self) -> int:
        return 8  # API concurrency, not VRAM-bound
