from .vlm_evaluator import VLMEvaluator, get_evaluator, list_evaluators

# Import all wrappers so they self-register via @register_evaluator
from . import moondream_wrapper   # noqa: F401
from . import qwen2vl_wrapper     # noqa: F401
from . import internvl2_wrapper   # noqa: F401
from . import phi3vision_wrapper  # noqa: F401
from . import llava_wrapper       # noqa: F401
from . import api_wrapper         # noqa: F401

__all__ = ["VLMEvaluator", "get_evaluator", "list_evaluators"]
