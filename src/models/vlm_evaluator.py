"""
Unified VLM evaluation interface.

Each model wrapper subclasses VLMEvaluator and implements:
- load_model()
- generate()

The base class provides shared logic for VCR evaluation, answer parsing,
metric computation, and detailed logging.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Optional

import torch
from PIL import Image

from src.logger import format_time, format_progress_bar, TRACE
from src.checkpoint import CheckpointManager

# ---------------------------------------------------------------------------
# Compatibility patch for transformers >= 4.50
# Models loaded with trust_remote_code=True may not define all_tied_weights_keys,
# which transformers 4.50 now requires on every PreTrainedModel subclass.
# ---------------------------------------------------------------------------
try:
    from transformers import PreTrainedModel as _PTM
    if not hasattr(_PTM, "all_tied_weights_keys"):
        _PTM.all_tied_weights_keys = []
except Exception:
    pass


log = logging.getLogger("vlm-eval")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type["VLMEvaluator"]] = {}


def register_evaluator(name: str):
    """Decorator to register a model wrapper."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_evaluator(name: str, **kwargs) -> "VLMEvaluator":
    """Factory: instantiate a registered evaluator by name."""
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_evaluators() -> list[str]:
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


def _truncate(text: str, max_len: int = 150) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class VLMEvaluator(ABC):
    """Base class for all VLM evaluation wrappers."""

    model = None
    processor = None
    model_name: str = ""
    param_count: str = ""
    device: str = "cuda"

    @abstractmethod
    def load_model(
        self,
        device: str = "cuda",
        quantization: Optional[str] = None,
    ) -> None:
        """Load model and processor onto device.

        Args:
            device: 'cuda', 'cpu', or 'cuda:N'
            quantization: None, '4bit', or '8bit'
        """

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Run inference on a single image+prompt pair. Return generated text."""

    def generate_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
        max_new_tokens: int = 512,
    ) -> list[str]:
        """Run batched inference on multiple image+prompt pairs.

        Override in subclasses for true GPU batching.
        Default: sequential fallback (works for any model).
        """
        return [self.generate(img, p, max_new_tokens) for img, p in zip(images, prompts)]

    def auto_batch_size(self) -> int:
        """Estimate a safe batch size based on available VRAM.

        Heuristic: after loading the model, each extra batch item needs
        roughly 1-2 GB for image encoding + KV cache. We leave 2 GB headroom.
        Returns 1 if not on CUDA.
        """
        if not torch.cuda.is_available():
            return 1
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        used = torch.cuda.memory_allocated() / (1024 ** 3)
        free = total - used
        # ~1.5 GB per extra item, 2 GB headroom
        bs = max(1, int((free - 2.0) / 1.5))
        return min(bs, 32)  # cap at 32

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def get_quantization_config(self, quantization: Optional[str]):
        """Return a BitsAndBytesConfig or None."""
        if quantization is None:
            return None
        from transformers import BitsAndBytesConfig
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError(f"Unknown quantization: {quantization}")

    @staticmethod
    def format_vcr_prompt(question: str, choices: list[str], task: str = "qa") -> str:
        """Build a zero-shot multiple-choice prompt for VCR."""
        letters = "ABCD"
        choices_text = "\n".join(
            f"{letters[i]}. {c}" for i, c in enumerate(choices[:4])
        )

        if task == "qa":
            return (
                f"Look at the image and answer the following multiple-choice question.\n\n"
                f"Question: {question}\n\n"
                f"{choices_text}\n\n"
                f"Answer with only the letter (A, B, C, or D)."
            )
        else:  # qar
            return (
                f"Look at the image and select the best rationale for the answer "
                f"to the following question.\n\n"
                f"Question: {question}\n\n"
                f"{choices_text}\n\n"
                f"Answer with only the letter (A, B, C, or D)."
            )

    @staticmethod
    def parse_answer(text: str) -> int:
        """Extract a letter choice (A-D) from model output. Returns 0-3 or -1."""
        text = text.strip()

        # Direct single letter
        if text and text[0].upper() in "ABCD" and (len(text) == 1 or not text[1].isalpha()):
            return ord(text[0].upper()) - ord("A")

        # Patterns: "The answer is B", "Answer: C", "(A)", etc.
        patterns = [
            r"(?:answer|choice)\s*(?:is|:)\s*\(?([A-Da-d])\)?",
            r"^\(?([A-Da-d])\)?[\.\,\s]",
            r"\(([A-Da-d])\)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return ord(m.group(1).upper()) - ord("A")

        return -1

    # ------------------------------------------------------------------
    # VCR evaluation with full logging + checkpointing
    # ------------------------------------------------------------------

    def evaluate_vcr(
        self,
        dataset,
        split: str = "val",
        subset_pct: Optional[float] = None,
        seed: int = 42,
        log_every: int = 25,
        run_id: str = "",
        restart: bool = False,
        save_interval_sec: float = 300,
        batch_size: int = 0,
    ) -> dict:
        """Run full VCR evaluation: Q->A, QA->R, Q->AR.

        Args:
            batch_size: number of examples per GPU batch.
                        0 = auto-detect from VRAM, 1 = sequential.
            restart: if True, ignore existing checkpoint and start fresh
            save_interval_sec: how often to save checkpoint (seconds)
        """
        total = len(dataset)
        letters = "ABCD"

        # ---- Batch size ----
        if batch_size <= 0:
            batch_size = self.auto_batch_size()
        log.info(f"  Batch size  : {batch_size}" + (" (auto)" if batch_size > 1 else ""))

        # ---- Checkpoint setup ----
        ckpt = CheckpointManager(
            model_name=self.model_name,
            dataset="vcr",
            split=split,
            seed=seed,
            total_examples=total,
            run_id=run_id,
            subset_pct=subset_pct,
            save_interval_sec=save_interval_sec,
        )

        if restart:
            ckpt.force_restart()
        elif ckpt.is_completed():
            log.info("This run was already completed. Use --restart to re-run.")
            existing = ckpt._existing
            return existing.get("final_metrics", {})

        start_idx = ckpt.resume_index()
        resumed_examples = ckpt.resumed_examples()

        log.info(f"{'='*70}")
        log.info(f"  VCR Evaluation {'(RESUMED)' if start_idx > 0 else 'Start'}")
        log.info(f"  Model       : {self.model_name} ({self.param_count})")
        log.info(f"  Dataset     : VCR {split}")
        log.info(f"  Examples    : {total}" + (f" ({subset_pct}% subset)" if subset_pct else ""))
        log.info(f"  Batch size  : {batch_size}")
        log.info(f"  Seed        : {seed}")
        log.info(f"  Device      : {self.device}")
        log.info(f"  VRAM at start: {_get_vram_gb():.2f} GB")
        log.info(f"  Checkpoint  : saves every {int(save_interval_sec)}s to {ckpt.path}")
        if start_idx > 0:
            log.info(f"  Resuming from example {start_idx}/{total}")
        log.info(f"{'='*70}")

        results = {
            "model_name": self.model_name,
            "param_count": self.param_count,
            "dataset": "vcr",
            "split": split,
            "subset_pct": subset_pct,
            "seed": seed,
            "batch_size": batch_size,
            "per_example": list(resumed_examples),
        }

        # Rebuild running counters from resumed data
        correct_qa = sum(1 for e in resumed_examples if e.get("qa_correct"))
        correct_r = sum(1 for e in resumed_examples if e.get("r_correct"))
        correct_qar = sum(1 for e in resumed_examples if e.get("qar_correct"))
        parse_failures_qa = sum(1 for e in resumed_examples if e.get("pred_answer") == -1)
        parse_failures_r = sum(1 for e in resumed_examples if e.get("pred_rationale") == -1)
        total_time = sum(e.get("total_time_sec", 0) for e in resumed_examples)
        inference_times = [e.get("total_time_sec", 0) for e in resumed_examples]
        wall_start = time.time()

        # ---- Batched evaluation loop ----
        for batch_start in range(start_idx, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_indices = list(range(batch_start, batch_end))
            batch_examples = [dataset[i] for i in batch_indices]
            bs = len(batch_examples)

            # -- Collect Q->A batch --
            qa_images = [ex["image"] for ex in batch_examples]
            qa_prompts = [
                self.format_vcr_prompt(ex["question"], ex["answer_choices"], task="qa")
                for ex in batch_examples
            ]

            for j, p in enumerate(qa_prompts):
                idx = batch_indices[j]
                aid = batch_examples[j]["metadata"].get("annot_id", idx)
                log.log(TRACE, f"[Example {idx+1}/{total} | ID: {aid}] Q->A PROMPT:\n{p}")

            t0 = time.time()
            qa_outputs = self.generate_batch(qa_images, qa_prompts)
            qa_batch_time = time.time() - t0

            # -- Collect QA->R batch --
            qar_prompts = []
            for ex in batch_examples:
                correct_answer_text = ex["answer_choices"][ex["answer_label"]]
                qar_prompts.append(self.format_vcr_prompt(
                    f"{ex['question']} Answer: {correct_answer_text}. Why?",
                    ex["rationale_choices"],
                    task="qar",
                ))

            for j, p in enumerate(qar_prompts):
                idx = batch_indices[j]
                aid = batch_examples[j]["metadata"].get("annot_id", idx)
                log.log(TRACE, f"[Example {idx+1}/{total} | ID: {aid}] QA->R PROMPT:\n{p}")

            t2 = time.time()
            qar_outputs = self.generate_batch(qa_images, qar_prompts)
            qar_batch_time = time.time() - t2

            # Distribute time evenly across batch items for per-example stats
            qa_time_each = qa_batch_time / bs
            qar_time_each = qar_batch_time / bs

            # -- Process results for each example in batch --
            for j in range(bs):
                idx = batch_indices[j]
                ex = batch_examples[j]
                annot_id = ex["metadata"].get("annot_id", idx)
                qa_output = qa_outputs[j]
                qar_output = qar_outputs[j]

                pred_answer = self.parse_answer(qa_output)
                qa_correct = pred_answer == ex["answer_label"]
                qa_parse_fail = pred_answer == -1

                pred_rationale = self.parse_answer(qar_output)
                r_correct = pred_rationale == ex["rationale_label"]
                r_parse_fail = pred_rationale == -1

                log.log(TRACE,
                    f"[Example {idx+1}/{total} | ID: {annot_id}] Q->A: "
                    f"pred={letters[pred_answer] if pred_answer >= 0 else 'FAIL'} "
                    f"gt={letters[ex['answer_label']]} {'OK' if qa_correct else 'WRONG'} | "
                    f"QA->R: pred={letters[pred_rationale] if pred_rationale >= 0 else 'FAIL'} "
                    f"gt={letters[ex['rationale_label']]} {'OK' if r_correct else 'WRONG'} | "
                    f"Raw Q->A: \"{_truncate(qa_output, 60)}\" | "
                    f"Raw QA->R: \"{_truncate(qar_output, 60)}\""
                )
                if qa_parse_fail:
                    log.warning(f"[{idx+1}/{total}] Q->A parse fail ID {annot_id}: \"{_truncate(qa_output, 80)}\"")
                if r_parse_fail:
                    log.warning(f"[{idx+1}/{total}] QA->R parse fail ID {annot_id}: \"{_truncate(qar_output, 80)}\"")

                example_time = qa_time_each + qar_time_each
                total_time += example_time
                inference_times.append(example_time)
                if qa_correct: correct_qa += 1
                if r_correct: correct_r += 1
                if qa_correct and r_correct: correct_qar += 1
                if qa_parse_fail: parse_failures_qa += 1
                if r_parse_fail: parse_failures_r += 1

                done = idx + 1
                example_result = {
                    "idx": idx,
                    "annot_id": annot_id,
                    "qa_prompt": qa_prompts[j],
                    "qa_raw_output": qa_output[:500],
                    "pred_answer": pred_answer,
                    "answer_label": ex["answer_label"],
                    "qa_correct": qa_correct,
                    "qa_inference_time_sec": round(qa_time_each, 3),
                    "qar_prompt": qar_prompts[j],
                    "qar_raw_output": qar_output[:500],
                    "pred_rationale": pred_rationale,
                    "rationale_label": ex["rationale_label"],
                    "r_correct": r_correct,
                    "qar_correct": qa_correct and r_correct,
                    "qar_inference_time_sec": round(qar_time_each, 3),
                    "total_time_sec": round(example_time, 3),
                }
                results["per_example"].append(example_result)

                running_metrics = {
                    "q_a_accuracy": correct_qa / done,
                    "qa_r_accuracy": correct_r / done,
                    "q_ar_accuracy": correct_qar / done,
                    "parse_failures_qa": parse_failures_qa,
                    "parse_failures_r": parse_failures_r,
                    "avg_inference_time_sec": total_time / done,
                }
                ckpt.record(idx, example_result, running_metrics)

            # ---- Periodic progress log (after each batch) ----
            done = batch_end
            if done % log_every < batch_size or done == total:
                avg_time = total_time / done
                eta = avg_time * (total - done)
                elapsed = time.time() - wall_start
                qa_acc = correct_qa / done
                r_acc = correct_r / done
                qar_acc = correct_qar / done

                throughput = done / elapsed if elapsed > 0 else 0
                log.info(
                    f"{format_progress_bar(done, total)}  "
                    f"Q->A: {qa_acc:.3f}  QA->R: {r_acc:.3f}  Q->AR: {qar_acc:.3f}  |  "
                    f"Avg: {avg_time:.2f}s/ex  {throughput:.1f} ex/s  "
                    f"ETA: {format_time(eta)}  "
                    f"Elapsed: {format_time(elapsed)}  VRAM: {_get_vram_gb():.1f}GB"
                )
                if parse_failures_qa > 0 or parse_failures_r > 0:
                    log.info(
                        f"  Parse failures -> Q->A: {parse_failures_qa} ({parse_failures_qa/done:.1%})  "
                        f"QA->R: {parse_failures_r} ({parse_failures_r/done:.1%})"
                    )

        # ---- Final summary ----
        wall_elapsed = time.time() - wall_start
        total_parse_failures = parse_failures_qa + parse_failures_r

        results["q_a_accuracy"] = correct_qa / total if total else 0
        results["qa_r_accuracy"] = correct_r / total if total else 0
        results["q_ar_accuracy"] = correct_qar / total if total else 0
        results["parse_failure_rate"] = total_parse_failures / (total * 2) if total else 0
        results["parse_failures_qa"] = parse_failures_qa
        results["parse_failures_r"] = parse_failures_r
        results["num_examples"] = total
        results["avg_inference_time_sec"] = total_time / total if total else 0
        results["median_inference_time_sec"] = sorted(inference_times)[len(inference_times) // 2] if inference_times else 0
        results["min_inference_time_sec"] = min(inference_times) if inference_times else 0
        results["max_inference_time_sec"] = max(inference_times) if inference_times else 0
        results["total_inference_time_sec"] = total_time
        results["wall_time_sec"] = wall_elapsed

        # Mark checkpoint as completed
        ckpt.mark_completed(results)

        log.info("")
        log.info(f"{'='*70}")
        log.info(f"  FINAL RESULTS -- {self.model_name} on VCR {split}")
        log.info(f"{'='*70}")
        log.info(f"  Examples evaluated : {total}")
        log.info(f"")
        log.info(f"  Q->A  Accuracy     : {results['q_a_accuracy']:.4f}  ({correct_qa}/{total})")
        log.info(f"  QA->R Accuracy     : {results['qa_r_accuracy']:.4f}  ({correct_r}/{total})")
        log.info(f"  Q->AR Accuracy     : {results['q_ar_accuracy']:.4f}  ({correct_qar}/{total})")
        log.info(f"")
        log.info(f"  Parse failures (Q->A) : {parse_failures_qa}/{total} ({parse_failures_qa/total:.1%})")
        log.info(f"  Parse failures (QA->R): {parse_failures_r}/{total} ({parse_failures_r/total:.1%})")
        log.info(f"")
        log.info(f"  Inference time (per example, both Q->A + QA->R):")
        log.info(f"    Average : {results['avg_inference_time_sec']:.3f}s")
        log.info(f"    Median  : {results['median_inference_time_sec']:.3f}s")
        log.info(f"    Min     : {results['min_inference_time_sec']:.3f}s")
        log.info(f"    Max     : {results['max_inference_time_sec']:.3f}s")
        log.info(f"  Total inference time   : {format_time(total_time)}")
        log.info(f"  Total wall time        : {format_time(wall_elapsed)}")
        log.info(f"  VRAM at end            : {_get_vram_gb():.2f} GB")
        log.info(f"{'='*70}")

        return results

    # ------------------------------------------------------------------
    # Generic MCQ evaluation with full logging + checkpointing
    # ------------------------------------------------------------------

    def evaluate_mcq(
        self,
        dataset,
        dataset_name: str = "mmmu",
        log_every: int = 20,
        seed: int = 42,
        run_id: str = "",
        restart: bool = False,
        save_interval_sec: float = 300,
        batch_size: int = 0,
    ) -> dict:
        """Generic MCQ evaluation for MMMU / MathVista with batching + checkpointing."""
        total = len(dataset)
        letters = "ABCDE"

        if batch_size <= 0:
            batch_size = self.auto_batch_size()

        # ---- Checkpoint setup ----
        ckpt = CheckpointManager(
            model_name=self.model_name,
            dataset=dataset_name,
            split="val",
            seed=seed,
            total_examples=total,
            run_id=run_id,
            save_interval_sec=save_interval_sec,
        )

        if restart:
            ckpt.force_restart()
        elif ckpt.is_completed():
            log.info("This run was already completed. Use --restart to re-run.")
            return ckpt._existing.get("final_metrics", {})

        start_idx = ckpt.resume_index()
        resumed_examples = ckpt.resumed_examples()

        log.info(f"{'='*70}")
        log.info(f"  MCQ Evaluation {'(RESUMED)' if start_idx > 0 else 'Start'}")
        log.info(f"  Model       : {self.model_name} ({self.param_count})")
        log.info(f"  Dataset     : {dataset_name}")
        log.info(f"  Examples    : {total}")
        log.info(f"  Batch size  : {batch_size}")
        log.info(f"  Device      : {self.device}")
        log.info(f"  VRAM at start: {_get_vram_gb():.2f} GB")
        log.info(f"  Checkpoint  : saves every {int(save_interval_sec)}s to {ckpt.path}")
        if start_idx > 0:
            log.info(f"  Resuming from example {start_idx}/{total}")
        log.info(f"{'='*70}")

        correct = sum(1 for e in resumed_examples if e.get("correct"))
        parse_failures = sum(1 for e in resumed_examples if e.get("pred") == -1)
        total_time = sum(e.get("inference_time_sec", 0) for e in resumed_examples)
        inference_times = [e.get("inference_time_sec", 0) for e in resumed_examples]
        per_example = list(resumed_examples)
        wall_start = time.time()

        for batch_start in range(start_idx, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_indices = list(range(batch_start, batch_end))
            batch_examples = [dataset[i] for i in batch_indices]
            bs = len(batch_examples)

            # Collect images and prompts
            images = []
            prompts = []
            for ex in batch_examples:
                img = ex.get("image") or (ex.get("images", [None])[0])
                images.append(img)
                choices = ex.get("answer_choices", [])
                if choices:
                    prompts.append(self.format_vcr_prompt(ex["question"], choices, task="qa"))
                else:
                    prompts.append(
                        f"Look at the image and answer the following question.\n\n"
                        f"Question: {ex['question']}\n\nProvide a short answer."
                    )

            for j, p in enumerate(prompts):
                log.log(TRACE, f"[Example {batch_indices[j]+1}/{total}] PROMPT:\n{p}")

            t0 = time.time()
            # Filter out None images for batch call
            valid = [(img, p) for img, p in zip(images, prompts) if img is not None]
            if valid:
                valid_imgs, valid_prompts = zip(*valid)
                outputs = self.generate_batch(list(valid_imgs), list(valid_prompts))
            else:
                outputs = [""] * bs

            # Re-insert empty strings for None-image examples
            out_iter = iter(outputs)
            full_outputs = [next(out_iter) if img is not None else "" for img in images]
            batch_time = time.time() - t0
            time_each = batch_time / bs

            for j in range(bs):
                idx = batch_indices[j]
                ex = batch_examples[j]
                output = full_outputs[j]
                meta = ex.get("metadata", {})
                example_id = meta.get("id", meta.get("pid", idx))
                choices = ex.get("answer_choices", [])
                label = ex.get("answer_label", -1)

                total_time += time_each
                inference_times.append(time_each)

                if choices:
                    pred = self.parse_answer(output)
                    is_correct = pred == label
                    if pred == -1:
                        parse_failures += 1
                        log.warning(f"[{idx+1}/{total}] Parse fail ID {example_id}: \"{_truncate(output, 80)}\"")
                else:
                    pred = output.strip()
                    is_correct = pred.lower() == str(ex.get("answer_text", "")).lower()

                if is_correct:
                    correct += 1

                log.log(TRACE,
                    f"[Example {idx+1}/{total} | ID: {example_id}] "
                    f"Pred: {pred}  GT: {label}  Correct: {is_correct}  "
                    f"Raw: \"{_truncate(output, 60)}\""
                )

                done = idx + 1
                example_result = {
                    "idx": idx, "id": example_id, "prompt": prompts[j],
                    "raw_output": output[:500],
                    "pred": pred if not choices else int(pred) if isinstance(pred, int) else pred,
                    "label": label, "correct": is_correct,
                    "inference_time_sec": round(time_each, 3), "metadata": meta,
                }
                per_example.append(example_result)

                running_metrics = {
                    "accuracy": correct / done,
                    "parse_failures": parse_failures,
                    "avg_inference_time_sec": total_time / done,
                }
                ckpt.record(idx, example_result, running_metrics)

            done = batch_end
            if done % log_every < batch_size or done == total:
                avg_time = total_time / done
                eta = avg_time * (total - done)
                elapsed = time.time() - wall_start
                throughput = done / elapsed if elapsed > 0 else 0
                log.info(
                    f"{format_progress_bar(done, total)}  "
                    f"Acc: {correct/done:.3f}  |  "
                    f"Avg: {avg_time:.2f}s/ex  {throughput:.1f} ex/s  "
                    f"ETA: {format_time(eta)}  "
                    f"Elapsed: {format_time(elapsed)}  VRAM: {_get_vram_gb():.1f}GB"
                )
                if parse_failures > 0:
                    log.info(f"  Parse failures: {parse_failures} ({parse_failures/done:.1%})")

        # ---- Final summary ----
        wall_elapsed = time.time() - wall_start

        result = {
            "model_name": self.model_name,
            "param_count": self.param_count,
            "dataset": dataset_name,
            "accuracy": correct / total if total else 0,
            "parse_failure_rate": parse_failures / total if total else 0,
            "parse_failures": parse_failures,
            "num_examples": total,
            "avg_inference_time_sec": total_time / total if total else 0,
            "median_inference_time_sec": sorted(inference_times)[len(inference_times) // 2] if inference_times else 0,
            "min_inference_time_sec": min(inference_times) if inference_times else 0,
            "max_inference_time_sec": max(inference_times) if inference_times else 0,
            "total_inference_time_sec": total_time,
            "wall_time_sec": wall_elapsed,
            "per_example": per_example,
        }

        ckpt.mark_completed(result)

        log.info("")
        log.info(f"{'='*70}")
        log.info(f"  FINAL RESULTS -- {self.model_name} on {dataset_name}")
        log.info(f"{'='*70}")
        log.info(f"  Examples evaluated : {total}")
        log.info(f"  Accuracy           : {result['accuracy']:.4f}  ({correct}/{total})")
        log.info(f"  Parse failures     : {parse_failures}/{total} ({result['parse_failure_rate']:.1%})")
        log.info(f"")
        log.info(f"  Inference time (per example):")
        log.info(f"    Average : {result['avg_inference_time_sec']:.3f}s")
        log.info(f"    Median  : {result['median_inference_time_sec']:.3f}s")
        log.info(f"    Min     : {result['min_inference_time_sec']:.3f}s")
        log.info(f"    Max     : {result['max_inference_time_sec']:.3f}s")
        log.info(f"  Total inference time: {format_time(total_time)}")
        log.info(f"  Total wall time    : {format_time(wall_elapsed)}")
        log.info(f"  VRAM at end        : {_get_vram_gb():.2f} GB")
        log.info(f"{'='*70}")

        return result
