"""Build a fixed-budget Repair set with Generation-like collision failures.

The output keeps 1,500 training examples: 750 frozen Part 1 Repair rows and
750 paired no-feedback/executor-feedback rows made from Train-only Manhattan
paths that collide with an obstacle.  The existing Repair Dev set is copied
unchanged and must not be used to construct the hard training candidates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.executor.executor import execute_plan
from src.models.prompts import PROTOCOL_VERSION, build_repair_prompt
from src.part1.data import read_jsonl, write_jsonl


GRID_FIELDS = ("environment_id", "rows", "cols", "start", "goal", "obstacles")


def trace_target(grid: dict[str, Any], plan: list[str]) -> str:
    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    row, col = grid["start"]
    parts = [f"({row},{col})"]
    for action in plan:
        delta_row, delta_col = deltas[action]
        row, col = row + delta_row, col + delta_col
        parts.extend((action, f"({row},{col})"))
    if [row, col] != grid["goal"]:
        raise ValueError("Repair target does not reach the goal")
    return "Trace: " + " ".join(parts)


def use_trace_output(row: dict[str, Any]) -> dict[str, Any]:
    marker = "Output only the complete comma-separated repaired path using: "
    if marker not in row["input_text"]:
        raise ValueError(f"Repair prompt marker missing: {row['example_id']}")
    prefix = row["input_text"].split(marker, 1)[0]
    updated = dict(row)
    updated["input_text"] = (
        prefix
        + "Output exactly one Trace: line with the start coordinate, then each action "
        "and its resulting coordinate. Use coordinates formatted (row,col). Stop at "
        "the goal.\nAnswer:"
    )
    updated["target_text"] = trace_target(row["grid"], row["target_plan"])
    updated["output_format"] = "trace_only"
    return updated


def use_structured_feedback(row: dict[str, Any]) -> dict[str, Any]:
    feedback = row.get("verifier_feedback")
    if feedback is None:
        step, reason = "unknown", "unknown"
    else:
        step, reason = feedback["first_error_step"], feedback["reason"]
    updated = dict(row)
    updated["input_text"] = (
        "Repair control fields:\n"
        f"ERROR_STEP={step}\n"
        f"ERROR_REASON={reason}\n"
        f"FEEDBACK_SOURCE={row['feedback_source']}\n"
        + row["input_text"]
    )
    updated["feedback_format"] = "structured_prefix_v1"
    return updated


def direct_plans(grid: dict[str, Any]) -> list[list[str]]:
    start_row, start_col = grid["start"]
    goal_row, goal_col = grid["goal"]
    vertical = ["down" if goal_row > start_row else "up"] * abs(goal_row - start_row)
    horizontal = ["right" if goal_col > start_col else "left"] * abs(goal_col - start_col)
    plans = [vertical + horizontal, horizontal + vertical]
    return [plan for index, plan in enumerate(plans) if plan and plan not in plans[:index]]


def shortest_suffix(grid: dict[str, Any], start: list[int]) -> list[str]:
    """Return one deterministic shortest valid suffix from start to the goal."""
    goal = tuple(grid["goal"])
    origin = tuple(start)
    obstacles = {tuple(cell) for cell in grid["obstacles"]}
    deltas = (
        ("up", -1, 0),
        ("down", 1, 0),
        ("left", 0, -1),
        ("right", 0, 1),
    )
    queue = deque([(origin, [])])
    seen = {origin}
    while queue:
        position, path = queue.popleft()
        if position == goal:
            return path
        for action, delta_row, delta_col in deltas:
            nxt = (position[0] + delta_row, position[1] + delta_col)
            if (
                0 <= nxt[0] < grid["rows"]
                and 0 <= nxt[1] < grid["cols"]
                and nxt not in obstacles
                and nxt not in seen
            ):
                seen.add(nxt)
                queue.append((nxt, path + [action]))
    raise ValueError(f"goal is unreachable from {start}")


def generation_like_candidates(
    base_rows: list[dict[str, Any]], hard_target: str
) -> list[dict[str, Any]]:
    candidates = []
    for row in base_rows:
        grid = {field: row[field] for field in GRID_FIELDS}
        for order, plan in enumerate(direct_plans(grid)):
            result = execute_plan(grid, plan)
            if result["reason"] != "obstacle_collision":
                continue
            if hard_target == "localized_suffix":
                error_step = result["first_error_step"]
                valid_prefix = plan[:error_step]
                target_plan = valid_prefix + shortest_suffix(grid, result["final_position"])
            else:
                target_plan = list(row["shortest_plan"])
            if not execute_plan(grid, target_plan)["success"]:
                raise AssertionError("constructed Repair target is not executable")
            candidates.append(
                {
                    "base_sample_id": row["base_sample_id"],
                    "environment_id": row["environment_id"],
                    "grid": grid,
                    "candidate_plan": plan,
                    "executor_label": result,
                    "target_plan": target_plan,
                    "direct_order": order,
                }
            )
    return candidates


def take_environment_diverse(
    rows: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["environment_id"]].append(row)
    rng = random.Random(seed)
    environment_ids = sorted(groups)
    rng.shuffle(environment_ids)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < count:
        progressed = False
        for environment_id in environment_ids:
            if groups[environment_id]:
                selected.append(groups[environment_id].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"requested {count} hard candidates, found {len(selected)}")
    return selected


def paired_repair_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        for feedback_source in ("no_feedback", "executor_feedback"):
            feedback = (
                None
                if feedback_source == "no_feedback"
                else candidate["executor_label"]
            )
            rows.append(
                {
                    "example_id": (
                        f"repair::generation_like::{candidate['base_sample_id']}::"
                        f"{candidate['direct_order']}::{feedback_source}"
                    ),
                    "task": "repair",
                    "split": "train",
                    "condition": "generation_like_collision",
                    "environment_id": candidate["environment_id"],
                    "base_sample_id": candidate["base_sample_id"],
                    "feedback_source": feedback_source,
                    "corruption_type": "generation_like_manhattan_collision",
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
            )
    return rows


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--repair-train", type=Path, required=True)
    parser.add_argument("--repair-dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-format", choices=("plan", "trace_only"), default="plan")
    parser.add_argument(
        "--hard-target",
        choices=("global_shortest", "localized_suffix"),
        default="global_shortest",
    )
    parser.add_argument("--structured-feedback", action="store_true")
    parser.add_argument("--original-per-feedback", type=int, default=375)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 0 <= args.original_per_feedback <= 750:
        raise ValueError("original-per-feedback must be between 0 and 750")

    base_rows = read_jsonl(args.base_train)
    frozen_repair = read_jsonl(args.repair_train)
    by_feedback = {
        source: [row for row in frozen_repair if row["feedback_source"] == source]
        for source in ("no_feedback", "executor_feedback")
    }
    rng = random.Random(args.seed)
    original_rows = []
    for source in ("no_feedback", "executor_feedback"):
        pool = list(by_feedback[source])
        rng.shuffle(pool)
        original_rows.extend(pool[: args.original_per_feedback])

    hard_pool = generation_like_candidates(base_rows, args.hard_target)
    hard_candidate_count = 750 - args.original_per_feedback
    hard_candidates = take_environment_diverse(hard_pool, hard_candidate_count, args.seed)
    hard_rows = paired_repair_rows(hard_candidates)
    train_rows = original_rows + hard_rows
    rng.shuffle(train_rows)
    if len(train_rows) != 1500:
        raise AssertionError(f"unexpected training count: {len(train_rows)}")
    if set(row["environment_id"] for row in hard_rows) - set(
        row["environment_id"] for row in base_rows
    ):
        raise AssertionError("hard examples escaped the Train environment pool")

    dev_rows = read_jsonl(args.repair_dev)
    if args.structured_feedback:
        train_rows = [use_structured_feedback(row) for row in train_rows]
        dev_rows = [use_structured_feedback(row) for row in dev_rows]
    if args.output_format == "trace_only":
        train_rows = [use_trace_output(row) for row in train_rows]
        dev_rows = [use_trace_output(row) for row in dev_rows]

    args.output.mkdir(parents=True)
    train_path = args.output / "repair_train.jsonl"
    dev_path = args.output / "repair_dev.jsonl"
    write_jsonl(train_path, train_rows)
    if args.output_format == "plan":
        shutil.copy2(args.repair_dev, dev_path)
    else:
        write_jsonl(dev_path, dev_rows)
    manifest = {
        "seed": args.seed,
        "output_format": args.output_format,
        "hard_target": args.hard_target,
        "structured_feedback": args.structured_feedback,
        "original_per_feedback": args.original_per_feedback,
        "train_count": len(train_rows),
        "dev_count": len(read_jsonl(dev_path)),
        "original_repair_rows": len(original_rows),
        "generation_like_rows": len(hard_rows),
        "generation_like_unique_candidates": len(hard_candidates),
        "generation_like_unique_environments": len(
            {row["environment_id"] for row in hard_candidates}
        ),
        "feedback_sources": dict(Counter(row["feedback_source"] for row in train_rows)),
        "hard_failure_reasons": dict(
            Counter(row["executor_label"]["reason"] for row in hard_candidates)
        ),
        "train_sha256": digest(train_path),
        "dev_sha256": digest(dev_path),
        "notes": [
            "Hard candidates use Train environments only.",
            "The frozen Repair Dev file is copied unchanged.",
            "Generation v9 Dev predictions are not read by this builder.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
