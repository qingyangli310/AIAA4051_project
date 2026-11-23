"""Run paired Direct, Self-feedback, and Executor-feedback experiments.

The Generator runs exactly once per grid.  Every branch reuses that candidate,
which prevents candidate drift from confounding the feedback comparison.
Formal test runs require an explicit confirmation flag so that frozen test data
is not consumed accidentally during development.
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

from src.evaluation.metrics import evaluate_records
from src.model_adapters import (
    Seq2SeqGeneratorAdapter,
    Seq2SeqRepairerAdapter,
    Seq2SeqVerifierAdapter,
    TransformersTextGenerator,
)
from src.part1.baselines import BFSGenerator, BFSRepairer, ExecutorVerifier
from src.pipeline import run_pipeline_comparison_batch
from src.protocol import load_protocol, validate_protocol
from src.smoke_integration import RealExecutorAdapter


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_record(record: dict[str, Any], condition: str) -> dict[str, Any]:
    branch = record["branches"][condition]
    generation = record["generation"]
    repair = branch["repair"]
    final = branch["final_result"]
    candidate = record["candidate_executor"]
    verification = branch["verification"]
    is_self_feedback = condition == "self_feedback"
    final_parse_success = (
        repair.get("parse_success")
        if isinstance(repair, dict)
        else generation.get("parse_success")
    )
    return {
        "status": record["status"],
        "candidate_parse_success": generation.get("parse_success"),
        "candidate_valid": candidate.get("success") is True,
        "repair_parse_success": (
            repair.get("parse_success") if isinstance(repair, dict) else None
        ),
        "final_parse_success": final_parse_success,
        "final_success": final.get("success") is True,
        "final_reason": final.get("reason"),
        "repair_success": (
            candidate.get("success") is not True and final.get("success") is True
        ),
        "optimal_length": record.get("optimal_length"),
        "predicted_length": len(branch["repaired_plan"]),
        "ground_truth_valid": (
            candidate.get("success") is True if is_self_feedback else None
        ),
        "ground_truth_error_step": (
            candidate.get("first_error_step") if is_self_feedback else None
        ),
        "predicted_valid": (
            verification.get("valid")
            if is_self_feedback and isinstance(verification, dict)
            else None
        ),
        "predicted_error_step": (
            verification.get("first_error_step")
            if is_self_feedback and isinstance(verification, dict)
            else None
        ),
        "ground_truth_reason": candidate.get("reason") if is_self_feedback else None,
        "predicted_reason": (
            verification.get("reason")
            if is_self_feedback and isinstance(verification, dict)
            else None
        ),
    }


def summarize(
    records: list[dict[str, Any]], conditions: tuple[str, ...]
) -> dict[str, Any]:
    successful_records = [record for record in records if record["status"] == "ok"]
    by_condition: dict[str, Any] = {}
    for condition in conditions:
        metric_rows = [metric_record(record, condition) for record in successful_records]
        reason_counts: dict[str, int] = {}
        reason_groups: dict[str, list[dict[str, Any]]] = {}
        for record in successful_records:
            reason = record["candidate_executor"].get("reason") or "success"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            reason_groups.setdefault(reason, []).append(metric_record(record, condition))
        reason_eligible = [
            row for row in metric_rows if row.get("ground_truth_reason") is not None
        ]
        reason_correct = sum(
            row.get("predicted_reason") == row.get("ground_truth_reason")
            for row in reason_eligible
        )
        by_condition[condition] = {
            "metrics": evaluate_records(metric_rows),
            "error_reason_accuracy": (
                reason_correct / len(reason_eligible) if reason_eligible else None
            ),
            "error_reason_counts": [reason_correct, len(reason_eligible)],
            "candidate_reason_counts": reason_counts,
            "metrics_by_candidate_reason": {
                reason: evaluate_records(rows) for reason, rows in sorted(reason_groups.items())
            },
        }
    return {
        "input_rows": len(records),
        "completed_rows": len(successful_records),
        "error_rows": len(records) - len(successful_records),
        "conditions": by_condition,
    }


def build_components(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    executor = RealExecutorAdapter()
    if args.symbolic:
        return BFSGenerator(), ExecutorVerifier(), BFSRepairer(), executor

    missing = [
        name
        for name in (
            "generator_checkpoint",
            "verifier_checkpoint",
            "repair_checkpoint",
        )
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "real-model mode requires checkpoints: " + ", ".join(sorted(missing))
        )
    generator_runner = TransformersTextGenerator(
        args.generator_checkpoint, device=args.device
    )
    verifier_runner = TransformersTextGenerator(
        args.verifier_checkpoint, device=args.device
    )
    repair_runner = TransformersTextGenerator(args.repair_checkpoint, device=args.device)
    return (
        Seq2SeqGeneratorAdapter(
            generator_runner,
            output_format=args.generator_output_format,
            max_new_tokens=args.generator_max_new_tokens,
            checkpoint_id=str(args.generator_checkpoint),
        ),
        Seq2SeqVerifierAdapter(
            verifier_runner,
            max_new_tokens=args.verifier_max_new_tokens,
            checkpoint_id=str(args.verifier_checkpoint),
        ),
        Seq2SeqRepairerAdapter(
            repair_runner,
            output_format=args.repair_output_format,
            max_new_tokens=args.repair_max_new_tokens,
            checkpoint_id=str(args.repair_checkpoint),
        ),
        executor,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("diagnostic", "dev", "test"), default="dev")
    parser.add_argument("--confirm-frozen-test", action="store_true")
    parser.add_argument("--protocol-config", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=(
            "direct",
            "no_feedback",
            "self_feedback",
            "executor_feedback",
            "shuffled_executor_feedback",
        ),
        default=("direct", "self_feedback", "executor_feedback"),
    )
    parser.add_argument("--symbolic", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generator-checkpoint", type=Path)
    parser.add_argument("--verifier-checkpoint", type=Path)
    parser.add_argument("--repair-checkpoint", type=Path)
    parser.add_argument(
        "--generator-output-format", choices=("plan", "trace_only"), default="trace_only"
    )
    parser.add_argument(
        "--repair-output-format", choices=("plan", "trace_only"), default="trace_only"
    )
    parser.add_argument("--generator-max-new-tokens", type=int, default=128)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=128)
    parser.add_argument("--repair-max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if args.split == "test" and not args.confirm_frozen_test:
        raise RuntimeError(
            "refusing to consume frozen test data without --confirm-frozen-test"
        )
    if args.split == "test" and args.protocol_config is None:
        raise RuntimeError("formal test requires --protocol-config")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    conditions = tuple(args.conditions)
    protocol = None
    if args.protocol_config is not None:
        protocol = validate_protocol(
            load_protocol(args.protocol_config),
            require_ready=args.split == "test",
        )
    rows = read_jsonl(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]

    generator, verifier, repairer, executor = build_components(args)
    started = time.perf_counter()
    records = run_pipeline_comparison_batch(
        rows,
        generator,
        verifier,
        repairer,
        executor,
        conditions=conditions,
    )
    elapsed = time.perf_counter() - started
    report = {
        "status": "complete",
        "scope": args.split,
        "formal_test": args.split == "test",
        "symbolic_reference": args.symbolic,
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "conditions": list(conditions),
        "protocol": protocol,
        "checkpoints": {
            "generator": str(args.generator_checkpoint) if args.generator_checkpoint else None,
            "verifier": str(args.verifier_checkpoint) if args.verifier_checkpoint else None,
            "repair": str(args.repair_checkpoint) if args.repair_checkpoint else None,
        },
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "generator_max_new_tokens": args.generator_max_new_tokens,
            "verifier_max_new_tokens": args.verifier_max_new_tokens,
            "repair_max_new_tokens": args.repair_max_new_tokens,
        },
        "elapsed_seconds": elapsed,
        "summary": summarize(records, conditions),
    }

    args.output.mkdir(parents=True)
    (args.output / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
