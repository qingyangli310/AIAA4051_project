"""Run and report C's reproducible November 15-16 stage acceptance smoke test."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.metrics import evaluate_records
from src.pipeline import run_pipeline
from src.smoke_integration import (
    DeterministicRepairPlaceholder,
    ExecutorVerifierAdapter,
    FixedPlanGenerator,
    RealExecutorAdapter,
)


BRANCHES = ("no_feedback", "self_feedback", "executor_feedback")


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == limit:
                    break
    if len(rows) != limit:
        raise ValueError(f"requested {limit} rows, found {len(rows)} in {path}")
    return rows


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    verification = record["verification"]
    repair = record["repair"] or {}
    final = record["final_result"]
    return {
        "status": record["status"],
        "candidate_parse_success": record["generation"]["parse_success"],
        "candidate_valid": verification["valid"],
        "ground_truth_valid": False,
        "predicted_valid": verification["valid"],
        "ground_truth_error_step": record["executor_label"]["first_error_step"],
        "predicted_error_step": verification["first_error_step"],
        "repair_parse_success": repair.get("parse_success"),
        "repair_success": final["success"],
        "final_parse_success": repair.get("parse_success"),
        "final_success": final["success"],
        "final_reason": final["reason"],
        "optimal_length": len(record["reference_plan"]),
        "predicted_length": len(record["repaired_plan"]),
    }


def run(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for branch in BRANCHES:
        for index, row in enumerate(rows):
            # Every tenth self/executor example intentionally remains unrepaired.
            # This validates failure accounting and provides a reproducible case.
            inject_failure = branch != "no_feedback" and index % 10 == 9
            try:
                result = run_pipeline(
                    row["grid"],
                    FixedPlanGenerator(row["corrupted_plan"]),
                    ExecutorVerifierAdapter(),
                    DeterministicRepairPlaceholder(
                        row["original_plan"],
                        keep_candidate=branch == "no_feedback",
                        inject_failure=inject_failure,
                    ),
                    RealExecutorAdapter(),
                    feedback_source=branch,
                )
                records.append(
                    {
                        "status": "ok",
                        "sample_id": row["sample_id"],
                        "base_sample_id": row["base_sample_id"],
                        "corruption_type": row["corruption_type"],
                        "executor_label": row["executor_label"],
                        "reference_plan": row["original_plan"],
                        "mock": True,
                        "mock_reason": "B/C trained checkpoints unavailable",
                        **result,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "status": "error",
                        "sample_id": row.get("sample_id"),
                        "environment_id": row.get("environment_id"),
                        "feedback_source": branch,
                        "corruption_type": row.get("corruption_type"),
                        "mock": True,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
    return records


def summarize(records: list[dict[str, Any]], sample_count: int) -> dict[str, Any]:
    by_branch: dict[str, Any] = {}
    by_error: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        selected = [r for r in records if r["feedback_source"] == branch]
        ok = [r for r in selected if r["status"] == "ok"]
        by_branch[branch] = {
            "run_kind": "deterministic_mock",
            "metrics": evaluate_records([flatten(r) for r in ok]),
            "batch_errors": len(selected) - len(ok),
        }
        error_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in ok:
            error_groups[record["corruption_type"]].append(record)
        by_error[branch] = {
            error_type: evaluate_records([flatten(r) for r in group])
            for error_type, group in sorted(error_groups.items())
        }
    return {
        "stage": "member_c_20261115_16_acceptance",
        "input_kind": "A frozen ID single-error condition",
        "sample_count_per_branch": sample_count,
        "total_pipeline_runs": len(records),
        "all_model_outputs_mock": True,
        "real_components": ["A frozen data", "A canonical Executor", "C metrics"],
        "unavailable_components": [
            "B November 15 Generator checkpoint/predictions",
            "B November 15 Verifier checkpoint/predictions",
            "trained Repair checkpoint",
        ],
        "error_type_counts": dict(sorted(Counter(r["corruption_type"] for r in records if r["feedback_source"] == "no_feedback").items())),
        "branches": by_branch,
        "metrics_by_error_type": by_error,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_cases(path: Path, records: list[dict[str, Any]]) -> None:
    success = next(r for r in records if r["feedback_source"] == "executor_feedback" and r["final_result"]["success"])
    failure = next(r for r in records if r["feedback_source"] == "executor_feedback" and not r["final_result"]["success"])
    text = f"""# Member C stage-acceptance cases

All model-like outputs below are deterministic mocks; the final checks use A's real Executor.

## Successful repair

- sample: `{success['sample_id']}`
- corruption: `{success['corruption_type']}`
- candidate: `{success['candidate_plan']}`
- feedback: `{success['verification']}`
- repaired: `{success['repaired_plan']}`
- final success: `{success['final_result']['success']}`

## Preserved failure

- sample: `{failure['sample_id']}`
- corruption: `{failure['corruption_type']}`
- candidate: `{failure['candidate_plan']}`
- feedback: `{failure['verification']}`
- repaired: `{failure['repaired_plan']}`
- final result: `{failure['final_result']}`
- explanation: deterministic failure injection verifies that an unsuccessful repair remains visible and is counted as failure.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/samples/nov15_paired/id_single_error.jsonl"))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--logs", type=Path, default=Path("results/member_c_day15_16_smoke_logs.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/member_c_day15_16_summary.json"))
    parser.add_argument("--cases", type=Path, default=Path("docs/MEMBER_C_20261115_16_CASES.md"))
    args = parser.parse_args()
    if not 20 <= args.limit <= 50:
        raise ValueError("stage smoke test limit must be between 20 and 50")
    rows = read_jsonl(args.input, args.limit)
    records = run(rows)
    summary = summarize(records, args.limit)
    write_jsonl(args.logs, records)
    write_json(args.summary, summary)
    write_cases(args.cases, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
