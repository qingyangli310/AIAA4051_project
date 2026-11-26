"""Task-aware scoring for Part 1 model outputs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from src.evaluation.metrics import evaluate_records
from src.executor.executor import execute_plan
from src.part1.baselines import shortest_path
from src.models.output_parser import parse_plan, parse_suffix, parse_trace, parse_verification


def _collapse(raw_outputs: Sequence[str]) -> dict[str, Any]:
    counts = Counter(raw_outputs)
    total = len(raw_outputs)
    return {
        "unique_predictions": len(counts),
        "most_common_predictions": counts.most_common(5),
        "most_common_share": counts.most_common(1)[0][1] / total if total else None,
    }


def score_plan_task(
    task: str, rows: Sequence[dict[str, Any]], raw_outputs: Sequence[str]
) -> dict[str, Any]:
    if task not in {"generation", "repair"}:
        raise ValueError("plan scoring supports generation or repair")
    if len(rows) != len(raw_outputs):
        raise ValueError("row and prediction counts differ")
    flattened = []
    exact = parsed_count = 0
    for row, raw in zip(rows, raw_outputs):
        if row.get("output_format") == "suffix_v1":
            parsed = parse_suffix(raw)
            if parsed["parse_success"]:
                parsed["plan"] = row["valid_prefix"] + parsed["plan"]
        elif row.get("output_format") == "trace_only":
            parsed = parse_trace(raw, row["grid"])
        else:
            parsed = parse_plan(raw)
        parsed_count += int(parsed["parse_success"])
        exact += int(raw.strip().lower() == row["target_text"].strip().lower())
        result = execute_plan(row["grid"], parsed["plan"]) if parsed["parse_success"] else {
            "success": False,
            "reason": "parse_failure",
        }
        predicted_length = len(parsed["plan"])
        target_plan = row["target_plan"]
        flattened.append(
            {
                "candidate_valid": False if task == "repair" else True,
                "repair_success": result["success"] if task == "repair" else False,
                "candidate_parse_success": parsed["parse_success"],
                "repair_parse_success": parsed["parse_success"] if task == "repair" else None,
                "final_parse_success": parsed["parse_success"],
                "final_success": result["success"],
                "final_reason": result.get("reason"),
                "optimal_length": (
                    len(shortest_path(row["grid"]))
                    if task == "repair"
                    else len(target_plan)
                ),
                "predicted_length": predicted_length,
            }
        )
    metrics = evaluate_records(flattened)
    return {
        "task": task,
        "examples": len(rows),
        "parse_rate": parsed_count / len(rows) if rows else None,
        "exact_match_rate": exact / len(rows) if rows else None,
        **_collapse(raw_outputs),
        "metrics": metrics,
    }


def score_verification(
    rows: Sequence[dict[str, Any]], raw_outputs: Sequence[str]
) -> dict[str, Any]:
    if len(rows) != len(raw_outputs):
        raise ValueError("row and prediction counts differ")
    flattened = []
    exact = parsed_count = 0
    for row, raw in zip(rows, raw_outputs):
        parsed = parse_verification(raw, len(row["candidate_plan"]))
        parsed_count += int(parsed["parse_success"])
        expected = row.get("verdict_text", row["target_text"]).strip().lower()
        exact += int(parsed.get("normalized_output") == expected)
        truth = row["executor_label"]
        flattened.append(
            {
                "ground_truth_valid": truth["success"],
                "predicted_valid": parsed["valid"],
                "ground_truth_error_step": truth["first_error_step"],
                "predicted_error_step": parsed["first_error_step"],
                "final_success": False,
                "final_reason": "parse_failure" if not parsed["parse_success"] else None,
            }
        )
    metrics = evaluate_records(flattened)
    reason_rows = [
        (row["executor_label"]["reason"], parse_verification(raw, len(row["candidate_plan"]))["reason"])
        for row, raw in zip(rows, raw_outputs)
        if row["executor_label"]["reason"] is not None
    ]
    reason_accuracy = (
        sum(truth == predicted for truth, predicted in reason_rows) / len(reason_rows)
        if reason_rows
        else None
    )
    return {
        "task": "verification",
        "examples": len(rows),
        "parse_rate": parsed_count / len(rows) if rows else None,
        "exact_match_rate": exact / len(rows) if rows else None,
        "error_reason_accuracy": reason_accuracy,
        **_collapse(raw_outputs),
        "metrics": metrics,
    }


def score_predictions(
    task: str, rows: Sequence[dict[str, Any]], raw_outputs: Sequence[str]
) -> dict[str, Any]:
    if task == "verification":
        return score_verification(rows, raw_outputs)
    return score_plan_task(task, rows, raw_outputs)
