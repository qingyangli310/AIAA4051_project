"""Build a validated, diverse, distribution-balanced Repair training set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_generation_like_repair_data import trace_target, use_trace_output
from src.executor.executor import execute_plan
from src.models.prompts import PROTOCOL_VERSION, build_repair_prompt
from src.part1.baselines import shortest_path
from src.part1.data import read_jsonl, write_jsonl


# Candidate-level proportions. Each selected candidate becomes one no-feedback
# and one executor-feedback example. Corruptions are exactly balanced while the
# error-reason mix remains close to the known task-generating distribution.
QUALITY_PROPORTIONS = {
    ("premature_stopping", "goal_not_reached"): 0.250,
    ("action_deletion", "goal_not_reached"): 0.200,
    ("action_deletion", "obstacle_collision"): 0.050,
    ("action_insertion", "goal_not_reached"): 0.150,
    ("action_insertion", "obstacle_collision"): 0.056,
    ("action_insertion", "out_of_bounds"): 0.044,
    ("action_substitution", "goal_not_reached"): 0.150,
    ("action_substitution", "obstacle_collision"): 0.074,
    ("action_substitution", "out_of_bounds"): 0.026,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def stable_value(seed: int, value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big"
    )


def source_pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["base_sample_id"],
        row["provenance"]["corruption_type"],
        tuple(row["input"]["candidate_plan"]),
        tuple(row["output"]["repaired_plan"]),
    )


def semantic_key(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "grid": row["input"]["grid"],
            "candidate_plan": row["input"]["candidate_plan"],
            "target_plan": row["output"]["repaired_plan"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def validate_and_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[source_pair_key(row)].append(row)

    candidates = []
    seen_semantic: set[str] = set()
    for key, pair in sorted(groups.items(), key=lambda item: str(item[0])):
        feedback = Counter(row["feedback_source"] for row in pair)
        if feedback != {"no_feedback": 1, "executor_feedback": 1}:
            raise ValueError(f"source candidate is not feedback-paired: {key}: {feedback}")
        executor_row = next(row for row in pair if row["feedback_source"] == "executor_feedback")
        no_feedback_row = next(row for row in pair if row["feedback_source"] == "no_feedback")
        grid = executor_row["input"]["grid"]
        candidate_plan = executor_row["input"]["candidate_plan"]
        target_plan = executor_row["output"]["repaired_plan"]
        actual = execute_plan(grid, candidate_plan)
        target = execute_plan(grid, target_plan)
        supplied = executor_row["input"]["verifier_feedback"]
        if actual["success"]:
            raise ValueError(f"Repair candidate unexpectedly succeeds: {key}")
        if not target["success"]:
            raise ValueError(f"Repair target does not succeed: {key}")
        if len(target_plan) != len(shortest_path(grid)):
            raise ValueError(f"Repair target is not shortest: {key}")
        if (supplied["first_error_step"], supplied["reason"]) != (
            actual["first_error_step"],
            actual["reason"],
        ):
            raise ValueError(f"executor feedback mismatch: {key}")
        if no_feedback_row["input"]["verifier_feedback"] is not None:
            raise ValueError(f"no-feedback row leaks verifier feedback: {key}")
        signature = semantic_key(executor_row)
        if signature in seen_semantic:
            continue
        seen_semantic.add(signature)
        candidates.append(
            {
                "pair": pair,
                "signature": signature,
                "environment_id": executor_row["environment_id"],
                "corruption_type": executor_row["provenance"]["corruption_type"],
                "reason": actual["reason"],
                "error_step": actual["first_error_step"],
                "target_length": len(target_plan),
                "candidate_length": len(candidate_plan),
            }
        )
    return candidates


def allocate_quotas(pair_count: int) -> dict[tuple[str, str], int]:
    raw = {key: pair_count * value for key, value in QUALITY_PROPORTIONS.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remainder = pair_count - sum(quotas.values())
    order = sorted(raw, key=lambda key: (raw[key] - quotas[key], str(key)), reverse=True)
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def select_diverse(
    candidates: list[dict[str, Any]], pair_count: int, seed: int
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    quotas = allocate_quotas(pair_count)
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        pools[(candidate["corruption_type"], candidate["reason"])].append(candidate)
    for key, quota in quotas.items():
        if len(pools[key]) < quota:
            raise ValueError(f"quality stratum {key} has {len(pools[key])}; needs {quota}")
        pools[key].sort(key=lambda row: stable_value(seed, row["signature"]))

    selected = []
    environment_usage: Counter[str] = Counter()
    length_usage: Counter[tuple[int, int]] = Counter()
    remaining = dict(quotas)
    while len(selected) < pair_count:
        active = [key for key, count in remaining.items() if count]
        key = max(
            active,
            key=lambda value: (remaining[value] / quotas[value], stable_value(seed + 1, str(value))),
        )
        pool = pools[key]
        best_index = min(
            range(len(pool)),
            key=lambda index: (
                environment_usage[pool[index]["environment_id"]],
                length_usage[(pool[index]["target_length"], pool[index]["candidate_length"])],
                stable_value(seed + 2, pool[index]["signature"]),
            ),
        )
        candidate = pool.pop(best_index)
        selected.append(candidate)
        remaining[key] -= 1
        environment_usage[candidate["environment_id"]] += 1
        length_usage[(candidate["target_length"], candidate["candidate_length"])] += 1
    return selected, quotas


def model_row(source: dict[str, Any]) -> dict[str, Any]:
    grid = dict(source["input"]["grid"])
    candidate_plan = list(source["input"]["candidate_plan"])
    target_plan = list(source["output"]["repaired_plan"])
    feedback_source = source["feedback_source"]
    feedback = source["input"]["verifier_feedback"]
    corruption = source["provenance"]["corruption_type"]
    row = {
        "example_id": f"repair::hq::{source['base_sample_id']}::{corruption}::{feedback_source}",
        "task": "repair",
        "split": "train",
        "condition": "validated_balanced_quality_v1",
        "environment_id": source["environment_id"],
        "base_sample_id": source["base_sample_id"],
        "feedback_source": feedback_source,
        "corruption_type": corruption,
        "grid": grid,
        "candidate_plan": candidate_plan,
        "verifier_feedback": feedback,
        "input_text": build_repair_prompt(grid, candidate_plan, feedback_source, feedback),
        "target_text": ", ".join(target_plan),
        "target_plan": target_plan,
        "protocol_version": PROTOCOL_VERSION,
    }
    return use_trace_output(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.train_count % 2:
        raise ValueError("train-count must be even to preserve feedback pairs")

    source_rows = read_jsonl(args.source)
    candidates = validate_and_group(source_rows)
    selected, quotas = select_diverse(candidates, args.train_count // 2, args.seed)
    train_rows = [model_row(row) for candidate in selected for row in candidate["pair"]]
    random.Random(args.seed).shuffle(train_rows)
    dev_rows = [use_trace_output(row) if row.get("output_format") != "trace_only" else row for row in read_jsonl(args.dev)]

    train_environments = {row["environment_id"] for row in train_rows}
    dev_environments = {row["environment_id"] for row in dev_rows}
    if train_environments & dev_environments:
        raise AssertionError("environment leakage between high-quality Train and Dev")
    unique_model_io = {(row["input_text"], row["target_text"]) for row in train_rows}
    if len(unique_model_io) != len(train_rows):
        raise AssertionError("high-quality training set contains repeated model I/O")

    args.output.mkdir(parents=True)
    train_path = args.output / "repair_train.jsonl"
    dev_path = args.output / "repair_dev.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(dev_path, dev_rows)
    executor_rows = [row for row in train_rows if row["feedback_source"] == "executor_feedback"]
    manifest = {
        "protocol": "repair-high-quality-v1",
        "seed": args.seed,
        "source": str(args.source),
        "source_rows": len(source_rows),
        "validated_unique_candidate_pool": len(candidates),
        "train_count": len(train_rows),
        "candidate_pair_count": len(selected),
        "dev_count": len(dev_rows),
        "train_unique_model_io": len(unique_model_io),
        "train_unique_environments": len(train_environments),
        "dev_unique_environments": len(dev_environments),
        "train_dev_environment_overlap": 0,
        "feedback_counts": dict(Counter(row["feedback_source"] for row in train_rows)),
        "corruption_counts": dict(Counter(row["corruption_type"] for row in train_rows)),
        "error_reason_counts_by_candidate": dict(
            Counter(row["verifier_feedback"]["reason"] for row in executor_rows)
        ),
        "candidate_quotas": {f"{key[0]}::{key[1]}": value for key, value in quotas.items()},
        "target_length_counts": dict(Counter(len(row["target_plan"]) for row in executor_rows)),
        "candidate_length_counts": dict(Counter(len(row["candidate_plan"]) for row in executor_rows)),
        "quality_checks": {
            "feedback_paired": True,
            "candidate_fails_executor": True,
            "executor_feedback_matches": True,
            "target_succeeds_executor": True,
            "target_is_shortest": True,
            "no_feedback_does_not_leak_diagnosis": True,
            "semantic_duplicates_removed": True,
        },
        "train_sha256": digest(train_path),
        "dev_sha256": digest(dev_path),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
