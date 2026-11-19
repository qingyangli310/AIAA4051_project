"""Replay Generation predictions through the official Executor and Repair model.

This is an integration diagnostic, not a held-out model-quality result.  A
candidate that already succeeds bypasses Repair.  Failed candidates are
evaluated under both no-feedback and executor-feedback Repair conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.executor.executor import execute_plan
from src.models.output_parser import parse_plan, parse_trace
from src.models.prompts import build_repair_prompt


FEEDBACK_SOURCES = ("no_feedback", "executor_feedback")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def structured_feedback_prompt(
    prompt: str, feedback_source: str, executor_result: dict[str, Any]
) -> str:
    if feedback_source == "no_feedback":
        step, reason = "unknown", "unknown"
    else:
        step = executor_result["first_error_step"]
        reason = executor_result["reason"]
    return (
        "Repair control fields:\n"
        f"ERROR_STEP={step}\n"
        f"ERROR_REASON={reason}\n"
        f"FEEDBACK_SOURCE={feedback_source}\n"
        + prompt
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--repair-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--repair-output-format", choices=("plan", "trace_only"), default="plan"
    )
    parser.add_argument("--structured-feedback", action="store_true")
    parser.add_argument("--include-shuffled-feedback", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("PyTorch and Transformers are required") from exc
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    rows = read_jsonl(args.predictions)
    initial: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        grid = row["grid"]
        candidate = row.get("plan", row.get("candidate_plan"))
        if not isinstance(candidate, list):
            raise ValueError(f"missing candidate plan: {row.get('base_sample_id')}")
        result = execute_plan(grid, candidate)
        item = {
            "base_sample_id": row["base_sample_id"],
            "environment_id": row["environment_id"],
            "grid": grid,
            "candidate_plan": candidate,
            "initial_executor": result,
            "local_diagnostic_reason": row.get("diagnostic_reason"),
        }
        initial.append(item)
        if not result["success"]:
            failures.append(item)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.repair_checkpoint), local_files_only=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(args.repair_checkpoint), local_files_only=True
    ).to(args.device)
    model.eval()

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    conditions = list(FEEDBACK_SOURCES)
    if args.include_shuffled_feedback:
        conditions.append("shuffled_executor_feedback")
    for condition in conditions:
        feedback_source = (
            "no_feedback" if condition == "no_feedback" else "executor_feedback"
        )
        feedback_items = failures
        if condition == "shuffled_executor_feedback" and failures:
            feedback_items = failures[1:] + failures[:1]
        prompts = [
            build_repair_prompt(
                item["grid"],
                item["candidate_plan"],
                feedback_source,
                feedback_item["initial_executor"]
                if feedback_source != "no_feedback"
                else None,
            )
            for item, feedback_item in zip(failures, feedback_items, strict=True)
        ]
        if args.structured_feedback:
            prompts = [
                structured_feedback_prompt(
                    prompt, feedback_source, feedback_item["initial_executor"]
                )
                for prompt, feedback_item in zip(prompts, feedback_items, strict=True)
            ]
        raw_outputs: list[str] = []
        for offset in range(0, len(prompts), args.batch_size):
            batch = tokenizer(
                prompts[offset : offset + args.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(args.device)
            with torch.inference_mode():
                token_ids = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    no_repeat_ngram_size=0,
                    early_stopping=False,
                )
            raw_outputs.extend(tokenizer.batch_decode(token_ids, skip_special_tokens=True))

        for item, feedback_item, prompt, raw_output in zip(
            failures, feedback_items, prompts, raw_outputs, strict=True
        ):
            parsed = (
                parse_trace(raw_output, item["grid"])
                if args.repair_output_format == "trace_only"
                else parse_plan(raw_output)
            )
            repaired_plan = parsed["plan"] if parsed["parse_success"] else []
            final_result = execute_plan(item["grid"], repaired_plan)
            records.append(
                {
                    **item,
                    "feedback_source": condition,
                    "provided_feedback": (
                        None
                        if feedback_source == "no_feedback"
                        else {
                            "first_error_step": feedback_item["initial_executor"][
                                "first_error_step"
                            ],
                            "reason": feedback_item["initial_executor"]["reason"],
                        }
                    ),
                    "repair_prompt": prompt,
                    "repair_raw_output": raw_output,
                    "repair_parse_success": parsed["parse_success"],
                    "repaired_plan": repaired_plan,
                    "final_executor": final_result,
                }
            )

    elapsed = time.perf_counter() - started
    initial_successes = sum(item["initial_executor"]["success"] for item in initial)
    branch_metrics: dict[str, Any] = {}
    for feedback_source in conditions:
        branch = [row for row in records if row["feedback_source"] == feedback_source]
        repaired_successes = sum(row["final_executor"]["success"] for row in branch)
        parse_successes = sum(row["repair_parse_success"] for row in branch)
        branch_metrics[feedback_source] = {
            "failed_candidates_sent_to_repair": len(branch),
            "repair_parse_success": [parse_successes, len(branch)],
            "conditional_repair_success": [repaired_successes, len(branch)],
            "conditional_repair_success_rate": rate(repaired_successes, len(branch)),
            "end_to_end_success": [initial_successes + repaired_successes, len(initial)],
            "end_to_end_success_rate": rate(initial_successes + repaired_successes, len(initial)),
        }

    diagnostic_agreement = sum(
        (item["local_diagnostic_reason"] == "success")
        == item["initial_executor"]["success"]
        for item in initial
    )
    report = {
        "status": "complete",
        "scope": "Dev diagnostic; not held-out test evidence",
        "predictions": str(args.predictions),
        "predictions_sha256": sha256(args.predictions),
        "repair_checkpoint": str(args.repair_checkpoint),
        "device": args.device,
        "repair_output_format": args.repair_output_format,
        "structured_feedback": args.structured_feedback,
        "include_shuffled_feedback": args.include_shuffled_feedback,
        "examples": len(initial),
        "initial_generator_success": [initial_successes, len(initial)],
        "initial_generator_success_rate": rate(initial_successes, len(initial)),
        "local_diagnostic_official_executor_agreement": [diagnostic_agreement, len(initial)],
        "branches": branch_metrics,
        "inference_seconds": elapsed,
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "no_repeat_ngram_size": 0,
            "early_stopping": False,
            "max_new_tokens": args.max_new_tokens,
        },
    }

    args.output.mkdir(parents=True)
    (args.output / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
