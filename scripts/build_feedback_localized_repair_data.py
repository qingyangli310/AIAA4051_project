"""Build the Project-6-compliant feedback-localized Repair v4 data.

The materialized training budget is exactly 1,000 rows: 500 matched candidates
for a no-feedback baseline model and 500 views of the same candidates for an
executor-feedback model.  Dev contains 50 unique candidates, materialized once
per model solely to support a paired comparison.  Train and Dev environments
come from their respective frozen splits and never cross.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_generation_like_repair_data import (
    deduplicate_candidate_model_io,
    generation_like_candidates,
    shortest_suffix,
    take_environment_diverse,
    trace_target,
)
from src.executor.executor import execute_plan
from src.models.prompts import (
    PROTOCOL_VERSION,
    build_feedback_localized_repair_prompt,
)
from src.part1.data import read_jsonl, write_jsonl


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def controlled_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize frozen controlled corruptions to localized-suffix targets."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        grid = row["grid"]
        plan = row["candidate_plan"]
        result = execute_plan(grid, plan)
        if result["success"]:
            continue
        error_step = result["first_error_step"]
        valid_prefix = plan[:error_step]
        target_plan = valid_prefix + shortest_suffix(grid, result["final_position"])
        signature = json.dumps(
            {"grid": grid, "candidate_plan": plan, "target_plan": target_plan},
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(
            {
                "base_sample_id": row["base_sample_id"],
                "environment_id": row["environment_id"],
                "grid": grid,
                "candidate_plan": plan,
                "executor_label": result,
                "target_plan": target_plan,
                "corruption_type": row.get("corruption_type", "controlled_error"),
                "candidate_family": "controlled_error",
            }
        )
    return candidates


def select_candidates(
    base_rows: list[dict[str, Any]],
    controlled_rows: list[dict[str, Any]],
    *,
    collision_count: int,
    controlled_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    collision_pool = deduplicate_candidate_model_io(
        generation_like_candidates(base_rows, "localized_suffix")
    )
    collisions = take_environment_diverse(collision_pool, collision_count, seed)
    for candidate in collisions:
        candidate["corruption_type"] = "generation_like_manhattan_collision"
        candidate["candidate_family"] = "generation_like_collision"

    controlled_pool = controlled_candidates(controlled_rows)
    controlled = take_environment_diverse(
        controlled_pool, controlled_count, seed + 1
    )
    selected = collisions + controlled
    random.Random(seed).shuffle(selected)
    return selected


def materialize(
    candidates: list[dict[str, Any]], feedback_source: str, split: str
) -> list[dict[str, Any]]:
    rows = []
    for index, candidate in enumerate(candidates):
        feedback = (
            None
            if feedback_source == "no_feedback"
            else candidate["executor_label"]
        )
        error_step = candidate["executor_label"]["first_error_step"]
        valid_prefix = candidate["candidate_plan"][:error_step]
        rows.append(
            {
                "example_id": (
                    f"repair_v4::{split}::{index:04d}::{feedback_source}"
                ),
                "task": "repair",
                "split": split,
                "condition": "feedback_localized_suffix_v1",
                "candidate_family": candidate["candidate_family"],
                "environment_id": candidate["environment_id"],
                "base_sample_id": candidate["base_sample_id"],
                "feedback_source": feedback_source,
                "corruption_type": candidate["corruption_type"],
                "grid": candidate["grid"],
                "candidate_plan": candidate["candidate_plan"],
                "executor_label": candidate["executor_label"],
                "verifier_feedback": feedback,
                "valid_prefix": valid_prefix,
                "resume_position": candidate["executor_label"]["final_position"],
                "input_text": build_feedback_localized_repair_prompt(
                    candidate["grid"],
                    candidate["candidate_plan"],
                    feedback_source,
                    feedback,
                ),
                "target_text": trace_target(
                    candidate["grid"], candidate["target_plan"]
                ),
                "target_plan": candidate["target_plan"],
                "output_format": "trace_only",
                "prompt_protocol": "feedback_localized_v1",
                "protocol_version": PROTOCOL_VERSION,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-dev", type=Path, required=True)
    parser.add_argument("--repair-train", type=Path, required=True)
    parser.add_argument("--repair-dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    base_train = read_jsonl(args.base_train)
    base_dev = read_jsonl(args.base_dev)
    train_candidates = select_candidates(
        base_train,
        read_jsonl(args.repair_train),
        collision_count=375,
        controlled_count=125,
        seed=args.seed,
    )
    dev_candidates = select_candidates(
        base_dev,
        read_jsonl(args.repair_dev),
        collision_count=25,
        controlled_count=25,
        seed=args.seed + 1000,
    )
    train_environments = {row["environment_id"] for row in train_candidates}
    dev_environments = {row["environment_id"] for row in dev_candidates}
    if train_environments & dev_environments:
        raise AssertionError("Train and Dev environments overlap")

    view_manifest: dict[str, Any] = {}
    for feedback_source in ("no_feedback", "executor_feedback"):
        directory = args.output / feedback_source
        directory.mkdir(parents=True)
        train_rows = materialize(train_candidates, feedback_source, "train")
        dev_rows = materialize(dev_candidates, feedback_source, "dev")
        train_path = directory / "repair_train.jsonl"
        dev_path = directory / "repair_dev.jsonl"
        write_jsonl(train_path, train_rows)
        write_jsonl(dev_path, dev_rows)
        view_manifest[feedback_source] = {
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "train_sha256": digest(train_path),
            "dev_sha256": digest(dev_path),
        }

    manifest = {
        "seed": args.seed,
        "method": "feedback_localized_suffix_v1",
        "output_format": "trace_only",
        "prompt_protocol": "feedback_localized_v1",
        "train_unique_candidates": len(train_candidates),
        "train_materialized_rows_total": len(train_candidates) * 2,
        "dev_unique_candidates": len(dev_candidates),
        "dev_materialized_views_total": len(dev_candidates) * 2,
        "train_candidate_families": dict(
            Counter(row["candidate_family"] for row in train_candidates)
        ),
        "dev_candidate_families": dict(
            Counter(row["candidate_family"] for row in dev_candidates)
        ),
        "train_unique_environments": len(train_environments),
        "dev_unique_environments": len(dev_environments),
        "train_dev_environment_overlap": 0,
        "views": view_manifest,
        "budget_interpretation": {
            "repair_train_examples": 1000,
            "repair_dev_unique_examples": 50,
            "note": (
                "Dev candidates are materialized under both conditions for a "
                "paired ablation; they remain the same 50 unique examples."
            ),
        },
        "data_policy": [
            "Training candidates use frozen Train environments only.",
            "Dev candidates use frozen Dev environments only.",
            "No current Generator Dev failure or frozen Test row is used for training.",
            "The no-feedback prompt contains no error step, reason, prefix, or resume position.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
