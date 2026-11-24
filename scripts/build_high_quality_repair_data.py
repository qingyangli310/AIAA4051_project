"""Build a leak-free, cross-task-disjoint Repair dataset.

The formal Train set uses one semantic candidate per base sample and keeps the
no-feedback/executor-feedback pair as a controlled treatment. The formal Dev
set uses one row per base sample so its row count is also its independent
sample count.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_generation_like_repair_data import use_trace_output
from src.data.build_nov14_dataset import single_corruption
from src.executor.executor import execute_plan
from src.models.prompts import PROTOCOL_VERSION, build_repair_prompt
from src.part1.baselines import shortest_path
from src.part1.data import read_jsonl, write_jsonl


CORRUPTION_TYPES = (
    "action_substitution",
    "action_deletion",
    "action_insertion",
    "premature_stopping",
)
GRID_FIELDS = ("environment_id", "rows", "cols", "start", "goal", "obstacles")

# Candidate-level proportions. Every selected Train candidate becomes one
# no-feedback row and one executor-feedback row. Corruptions are balanced;
# failure reasons retain the validated distribution of the earlier dataset.
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


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def semantic_key(
    grid: dict[str, Any], candidate_plan: list[str], target_plan: list[str]
) -> str:
    return json.dumps(
        {
            "grid": grid,
            "candidate_plan": candidate_plan,
            "target_plan": target_plan,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def load_excluded_base_ids(paths: Iterable[Path]) -> tuple[set[str], dict[str, Any]]:
    base_ids: set[str] = set()
    evidence: dict[str, Any] = {}
    for path in paths:
        rows = read_jsonl(path)
        ids = {str(row["base_sample_id"]) for row in rows}
        base_ids.update(ids)
        evidence[portable_path(path)] = {
            "rows": len(rows),
            "unique_base_sample_ids": len(ids),
            "sha256": digest(path),
        }
    return base_ids, evidence


def build_candidate_pool(
    base_rows: list[dict[str, Any]], excluded_base_ids: set[str], seed: int
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_semantic: set[str] = set()
    for base in base_rows:
        base_id = str(base["base_sample_id"])
        if base_id in excluded_base_ids:
            continue
        grid = {field: base[field] for field in GRID_FIELDS}
        target_plan = list(base["shortest_plan"])
        target_result = execute_plan(grid, target_plan)
        if not target_result["success"] or len(target_plan) != len(shortest_path(grid)):
            raise ValueError(f"invalid shortest target for {base_id}")
        for corruption_type in CORRUPTION_TYPES:
            corruption_seed = stable_value(seed, f"{base_id}:{corruption_type}")
            corrupted = single_corruption(
                grid, target_plan, corruption_type, corruption_seed
            )
            if corrupted is None:
                continue
            candidate_plan, corruption_position = corrupted
            result = execute_plan(grid, candidate_plan)
            if result["success"]:
                continue
            signature = semantic_key(grid, candidate_plan, target_plan)
            if signature in seen_semantic:
                continue
            seen_semantic.add(signature)
            candidates.append(
                {
                    "signature": signature,
                    "environment_id": base["environment_id"],
                    "base_sample_id": base_id,
                    "grid": grid,
                    "candidate_plan": candidate_plan,
                    "target_plan": target_plan,
                    "corruption_type": corruption_type,
                    "corruption_position": corruption_position,
                    "corruption_seed": corruption_seed,
                    "reason": result["reason"],
                    "error_step": result["first_error_step"],
                    "target_length": len(target_plan),
                    "candidate_length": len(candidate_plan),
                }
            )
    return candidates


def allocate_quotas(candidate_count: int) -> dict[tuple[str, str], int]:
    raw = {key: candidate_count * value for key, value in QUALITY_PROPORTIONS.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remainder = candidate_count - sum(quotas.values())
    order = sorted(
        raw,
        key=lambda key: (raw[key] - quotas[key], str(key)),
        reverse=True,
    )
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def select_diverse(
    candidates: list[dict[str, Any]],
    candidate_count: int,
    seed: int,
    *,
    unique_environments: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    """Select exact strata while never reusing a base sample."""

    quotas = allocate_quotas(candidate_count)
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        pools[(candidate["corruption_type"], candidate["reason"])].append(candidate)
    for key, quota in quotas.items():
        if len(pools[key]) < quota:
            raise ValueError(f"quality stratum {key} has {len(pools[key])}; needs {quota}")
        pools[key].sort(key=lambda row: stable_value(seed, row["signature"]))

    selected: list[dict[str, Any]] = []
    used_base_ids: set[str] = set()
    used_environments: set[str] = set()
    environment_usage: Counter[str] = Counter()
    length_usage: Counter[tuple[int, int]] = Counter()
    remaining = dict(quotas)

    while len(selected) < candidate_count:
        choices: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for key, count in remaining.items():
            if not count:
                continue
            available = [
                row
                for row in pools[key]
                if row["base_sample_id"] not in used_base_ids
                and (
                    not unique_environments
                    or row["environment_id"] not in used_environments
                )
            ]
            if len(available) < count:
                raise ValueError(
                    f"quality stratum {key} has only {len(available)} eligible "
                    f"candidates for {count} remaining slots"
                )
            choices[key] = available

        key = min(
            choices,
            key=lambda value: (
                len(choices[value]) - remaining[value],
                stable_value(seed + 1, str(value)),
            ),
        )
        candidate = min(
            choices[key],
            key=lambda row: (
                environment_usage[row["environment_id"]],
                length_usage[(row["target_length"], row["candidate_length"])],
                stable_value(seed + 2, row["signature"]),
            ),
        )
        selected.append(candidate)
        remaining[key] -= 1
        used_base_ids.add(candidate["base_sample_id"])
        used_environments.add(candidate["environment_id"])
        environment_usage[candidate["environment_id"]] += 1
        length_usage[(candidate["target_length"], candidate["candidate_length"])] += 1

    return selected, quotas


def model_row(
    candidate: dict[str, Any], feedback_source: str, split: str
) -> dict[str, Any]:
    feedback = None
    if feedback_source == "executor_feedback":
        feedback = {
            "valid": False,
            "first_error_step": candidate["error_step"],
            "reason": candidate["reason"],
        }
    row = {
        "example_id": (
            f"repair::hq2::{candidate['base_sample_id']}::"
            f"{candidate['corruption_type']}::{feedback_source}"
        ),
        "task": "repair",
        "split": split,
        "condition": "cross_task_disjoint_quality_v2",
        "environment_id": candidate["environment_id"],
        "base_sample_id": candidate["base_sample_id"],
        "feedback_source": feedback_source,
        "corruption_type": candidate["corruption_type"],
        "corruption_position": candidate["corruption_position"],
        "corruption_seed": candidate["corruption_seed"],
        "grid": candidate["grid"],
        "candidate_plan": candidate["candidate_plan"],
        "verifier_feedback": feedback,
        "input_text": build_repair_prompt(
            candidate["grid"],
            candidate["candidate_plan"],
            feedback_source,
            feedback,
        ),
        "target_text": ", ".join(candidate["target_plan"]),
        "target_plan": candidate["target_plan"],
        "protocol_version": PROTOCOL_VERSION,
    }
    return use_trace_output(row)


def _assert_dataset(
    train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]
) -> None:
    train_environments = {row["environment_id"] for row in train_rows}
    dev_environments = {row["environment_id"] for row in dev_rows}
    if train_environments & dev_environments:
        raise AssertionError("environment leakage between Repair Train and Dev")
    if {row["base_sample_id"] for row in train_rows} & {
        row["base_sample_id"] for row in dev_rows
    }:
        raise AssertionError("base-sample leakage between Repair Train and Dev")
    for rows, name in ((train_rows, "Train"), (dev_rows, "Dev")):
        model_io = {(row["input_text"], row["target_text"]) for row in rows}
        if len(model_io) != len(rows):
            raise AssertionError(f"{name} contains repeated model I/O")
        for row in rows:
            if execute_plan(row["grid"], row["candidate_plan"])["success"]:
                raise AssertionError(f"{name} contains a successful candidate")
            if not execute_plan(row["grid"], row["target_plan"])["success"]:
                raise AssertionError(f"{name} contains a failing target")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-train",
        type=Path,
        default=ROOT / "data/processed/nov14/train.jsonl",
    )
    parser.add_argument(
        "--base-dev",
        type=Path,
        default=ROOT / "data/processed/nov14/dev.jsonl",
    )
    parser.add_argument("--exclude-train", type=Path, action="append", default=[])
    parser.add_argument("--exclude-dev", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--dev-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.train_count % 2:
        raise ValueError("train-count must be even to preserve feedback pairs")
    if args.dev_count % 2:
        raise ValueError("dev-count must be even to balance feedback sources")

    excluded_train, train_exclusion_evidence = load_excluded_base_ids(
        args.exclude_train
    )
    excluded_dev, dev_exclusion_evidence = load_excluded_base_ids(args.exclude_dev)
    train_pool = build_candidate_pool(
        read_jsonl(args.base_train), excluded_train, args.seed
    )
    dev_pool = build_candidate_pool(
        read_jsonl(args.base_dev), excluded_dev, args.seed + 10_000
    )
    train_candidates, train_quotas = select_diverse(
        train_pool,
        args.train_count // 2,
        args.seed,
        unique_environments=False,
    )
    dev_candidates, dev_quotas = select_diverse(
        dev_pool,
        args.dev_count,
        args.seed + 20_000,
        unique_environments=True,
    )

    train_rows = [
        model_row(candidate, feedback_source, "train")
        for candidate in train_candidates
        for feedback_source in ("no_feedback", "executor_feedback")
    ]
    random.Random(args.seed).shuffle(train_rows)
    dev_rows = [
        model_row(
            candidate,
            "no_feedback" if index < args.dev_count // 2 else "executor_feedback",
            "dev",
        )
        for index, candidate in enumerate(dev_candidates)
    ]
    random.Random(args.seed + 1).shuffle(dev_rows)
    _assert_dataset(train_rows, dev_rows)

    args.output.mkdir(parents=True)
    train_path = args.output / "repair_train.jsonl"
    dev_path = args.output / "repair_dev.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(dev_path, dev_rows)

    train_base_ids = {row["base_sample_id"] for row in train_rows}
    dev_base_ids = {row["base_sample_id"] for row in dev_rows}
    train_environments = {row["environment_id"] for row in train_rows}
    dev_environments = {row["environment_id"] for row in dev_rows}
    train_executor_rows = [
        row for row in train_rows if row["feedback_source"] == "executor_feedback"
    ]
    manifest = {
        "protocol": "repair-high-quality-v2-cross-task-disjoint",
        "seed": args.seed,
        "base_train": portable_path(args.base_train),
        "base_dev": portable_path(args.base_dev),
        "excluded_train_sources": train_exclusion_evidence,
        "excluded_dev_sources": dev_exclusion_evidence,
        "candidate_pool": {"train": len(train_pool), "dev": len(dev_pool)},
        "train": {
            "rows": len(train_rows),
            "semantic_candidates": len(train_candidates),
            "unique_base_sample_ids": len(train_base_ids),
            "unique_environments": len(train_environments),
            "unique_model_io": len(
                {(row["input_text"], row["target_text"]) for row in train_rows}
            ),
            "feedback_counts": dict(
                Counter(row["feedback_source"] for row in train_rows)
            ),
            "corruption_counts_by_candidate": dict(
                Counter(row["corruption_type"] for row in train_executor_rows)
            ),
            "failure_reason_counts_by_candidate": dict(
                Counter(row["verifier_feedback"]["reason"] for row in train_executor_rows)
            ),
            "candidate_quotas": {
                f"{key[0]}::{key[1]}": value for key, value in train_quotas.items()
            },
        },
        "dev": {
            "rows": len(dev_rows),
            "semantic_candidates": len(dev_candidates),
            "unique_base_sample_ids": len(dev_base_ids),
            "unique_environments": len(dev_environments),
            "unique_model_io": len(
                {(row["input_text"], row["target_text"]) for row in dev_rows}
            ),
            "feedback_counts": dict(
                Counter(row["feedback_source"] for row in dev_rows)
            ),
            "corruption_counts": dict(
                Counter(row["corruption_type"] for row in dev_rows)
            ),
            "candidate_quotas": {
                f"{key[0]}::{key[1]}": value for key, value in dev_quotas.items()
            },
        },
        "overlap_checks": {
            "train_with_excluded_base_ids": len(train_base_ids & excluded_train),
            "dev_with_excluded_base_ids": len(dev_base_ids & excluded_dev),
            "repair_train_dev_base_ids": len(train_base_ids & dev_base_ids),
            "repair_train_dev_environments": len(
                train_environments & dev_environments
            ),
        },
        "quality_checks": {
            "train_feedback_paired": True,
            "train_one_candidate_per_base_sample": True,
            "dev_one_row_per_base_sample": True,
            "dev_one_row_per_environment": True,
            "candidate_fails_executor": True,
            "executor_feedback_matches": True,
            "target_succeeds_executor": True,
            "target_is_shortest": True,
            "no_feedback_does_not_leak_diagnosis": True,
            "semantic_duplicates_removed": True,
        },
        "sha256": {
            "repair_train.jsonl": digest(train_path),
            "repair_dev.jsonl": digest(dev_path),
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
