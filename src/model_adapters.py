"""Adapters that expose local seq2seq checkpoints through pipeline interfaces.

The adapters keep model loading optional at import time.  Unit tests and CPU
smoke tests can inject a small ``infer_text`` callable, while formal runs can
use :class:`TransformersTextGenerator` with local Hugging Face checkpoints.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from generation_trace_v9.state_format import build_trace_prompt
from src.models.output_parser import parse_plan, parse_suffix, parse_trace, parse_verification
from src.models.prompts import (
    build_feedback_localized_repair_prompt,
    build_suffix_repair_prompt,
    build_generation_prompt,
    build_repair_prompt,
    build_verification_prompt,
    format_repair_trace_prompt,
    validate_grid,
)


InferText = Callable[[str, int], str]


class TransformersTextGenerator:
    """Greedy text generation from one local seq2seq checkpoint."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cuda:0") -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on model environment
            raise RuntimeError(
                "PyTorch and Transformers are required for checkpoint inference"
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} was requested but is unavailable")

        checkpoint_path = str(checkpoint)
        self.checkpoint = checkpoint_path
        self.device = device
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, local_files_only=True
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint_path, local_files_only=True
        ).to(device)
        self._model.eval()

    def __call__(self, prompt: str, max_new_tokens: int) -> str:
        batch = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        ).to(self.device)
        with self._torch.inference_mode():
            token_ids = self._model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=0,
                early_stopping=False,
            )[0]
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)


def _parse_path(raw_output: str, grid: dict[str, Any], output_format: str) -> dict[str, Any]:
    if output_format == "plan":
        return parse_plan(raw_output)
    if output_format == "trace_only":
        return parse_trace(raw_output, grid)
    raise ValueError(f"unsupported output format: {output_format!r}")


class Seq2SeqGeneratorAdapter:
    """Generate and strictly parse one candidate plan."""

    def __init__(
        self,
        infer_text: InferText,
        *,
        output_format: str = "plan",
        max_new_tokens: int = 128,
        checkpoint_id: str | None = None,
    ) -> None:
        self.infer_text = infer_text
        self.output_format = output_format
        self.max_new_tokens = max_new_tokens
        self.checkpoint_id = checkpoint_id

    def generate(self, grid: dict[str, Any]) -> dict[str, Any]:
        if self.output_format == "trace_only":
            validate_grid(grid)
            prompt = build_trace_prompt(grid)
        else:
            prompt = build_generation_prompt(grid)
        raw_output = self.infer_text(prompt, self.max_new_tokens)
        parsed = _parse_path(raw_output, grid, self.output_format)
        return {
            **parsed,
            "prompt": prompt,
            "output_format": self.output_format,
            "checkpoint": self.checkpoint_id,
        }


class Seq2SeqVerifierAdapter:
    """Predict validity, first error step, and error type."""

    def __init__(
        self,
        infer_text: InferText,
        *,
        max_new_tokens: int = 128,
        checkpoint_id: str | None = None,
    ) -> None:
        self.infer_text = infer_text
        self.max_new_tokens = max_new_tokens
        self.checkpoint_id = checkpoint_id

    def verify(self, grid: dict[str, Any], plan: list[str]) -> dict[str, Any]:
        prompt = build_verification_prompt(grid, plan)
        raw_output = self.infer_text(prompt, self.max_new_tokens)
        parsed = parse_verification(raw_output, len(plan))
        return {
            **parsed,
            "prompt": prompt,
            "checkpoint": self.checkpoint_id,
        }


class Seq2SeqRepairerAdapter:
    """Repair a candidate plan using one of the frozen feedback conditions."""

    def __init__(
        self,
        infer_text: InferText,
        *,
        output_format: str = "plan",
        prompt_protocol: str = "standard",
        max_new_tokens: int = 128,
        checkpoint_id: str | None = None,
    ) -> None:
        self.infer_text = infer_text
        self.output_format = output_format
        if prompt_protocol not in {"standard", "feedback_localized_v1", "suffix_v1"}:
            raise ValueError(f"unsupported Repair prompt protocol: {prompt_protocol!r}")
        self.prompt_protocol = prompt_protocol
        self.max_new_tokens = max_new_tokens
        self.checkpoint_id = checkpoint_id

    def repair(
        self,
        environment_id: str,
        grid: dict[str, Any],
        candidate_plan: list[str],
        feedback_source: str,
        verifier_feedback: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del environment_id
        if self.prompt_protocol == "suffix_v1":
            if self.output_format != "suffix_v1":
                raise ValueError("suffix_v1 prompt requires suffix_v1 output")
            prompt = build_suffix_repair_prompt(
                grid, candidate_plan, feedback_source, verifier_feedback
            )
        elif self.prompt_protocol == "feedback_localized_v1":
            if self.output_format != "trace_only":
                raise ValueError("feedback_localized_v1 requires trace_only output")
            prompt = build_feedback_localized_repair_prompt(
                grid, candidate_plan, feedback_source, verifier_feedback
            )
        else:
            prompt = build_repair_prompt(
                grid,
                candidate_plan,
                feedback_source,
                verifier_feedback,
            )
        if self.output_format == "trace_only" and self.prompt_protocol == "standard":
            prompt = format_repair_trace_prompt(prompt)
        raw_output = self.infer_text(prompt, self.max_new_tokens)
        parsed = (
            parse_suffix(raw_output)
            if self.output_format == "suffix_v1"
            else _parse_path(raw_output, grid, self.output_format)
        )
        repaired_plan = (
            candidate_plan[:verifier_feedback["first_error_step"]] + parsed["plan"]
            if self.output_format == "suffix_v1" and parsed["parse_success"]
            else parsed["plan"]
        )
        return {
            "raw_output": raw_output,
            "repaired_plan": repaired_plan,
            "parse_success": parsed["parse_success"],
            "trace_positions": parsed.get("trace_positions"),
            "prompt": prompt,
            "output_format": self.output_format,
            "prompt_protocol": self.prompt_protocol,
            "checkpoint": self.checkpoint_id,
        }


class ConditionRoutingRepairer:
    """Route the no-feedback baseline and feedback conditions to matched models."""

    def __init__(self, no_feedback_repairer: Any, feedback_repairer: Any) -> None:
        self.no_feedback_repairer = no_feedback_repairer
        self.feedback_repairer = feedback_repairer

    def repair(self, **request: Any) -> dict[str, Any]:
        repairer = (
            self.no_feedback_repairer
            if request.get("feedback_source") == "no_feedback"
            else self.feedback_repairer
        )
        result = repairer.repair(**request)
        return {**result, "condition_routed": True}
