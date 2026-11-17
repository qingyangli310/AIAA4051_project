"""Build the frozen Part 1 text-to-text datasets within the 4,000-example cap."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.models.prompts import (
    PROTOCOL_VERSION,
    build_generation_prompt,
    build_repair_prompt,
    build_verification_prompt,
)


GRID_FIELDS = ("environment_id", "rows", "cols", "start", "goal", "obstacles")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL file: {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def grid_from_base(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in GRID_FIELDS}


def grid_from_error(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row["grid"])


def verification_target(label: dict[str, Any]) -> str:
    if label["success"]:
        return "valid"
    return (
        f"invalid | step={label['first_error_step']} | "
        f"reason={label['reason']}"
    )


def error_sample_id(row: dict[str, Any]) -> str:
    if row.get("sample_id"):
        return str(row["sample_id"])
    digest = hashlib.sha256(
        json.dumps(
            {
                "base_sample_id": row["base_sample_id"],
                "corruption_type": row["corruption_type"],
                "corruption_seed": row["corruption_seed"],
                "corrupted_plan": row["corrupted_plan"],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return f"error_{digest}"


def _shuffle(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    result = list(rows)
    random.Random(seed).shuffle(result)
    return result


def take_diverse(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Round-robin environment groups before reusing an environment."""
    if count > len(rows):
        raise ValueError(f"requested {count} rows from only {len(rows)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["environment_id"]].append(row)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def generation_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    grid = grid_from_base(row)
    target_plan = list(row["shortest_plan"])
    return {
        "example_id": f"generation::{row['base_sample_id']}",
        "task": "generation",
        "split": split,
        "environment_id": row["environment_id"],
        "base_sample_id": row["base_sample_id"],
        "grid": grid,
        "input_text": build_generation_prompt(grid),
        "target_text": ", ".join(target_plan),
        "target_plan": target_plan,
        "optimal_length": len(target_plan),
        "protocol_version": PROTOCOL_VERSION,
    }


def verification_valid_rows(row: dict[str, Any], split: str) -> list[dict[str, Any]]:
    grid = grid_from_base(row)
    result = []
    for kind in ("shortest", "nonoptimal"):
        plan = list(row[f"{kind}_plan"])
        label = dict(row[f"{kind}_executor_label"])
        result.append(
            {
                "example_id": f"verification::{row['base_sample_id']}::{kind}",
                "task": "verification",
                "split": split,
                "environment_id": row["environment_id"],
                "base_sample_id": row["base_sample_id"],
                "plan_kind": kind,
                "grid": grid,
                "candidate_plan": plan,
                "executor_label": label,
                "input_text": build_verification_prompt(grid, plan),
                "target_text": verification_target(label),
                "protocol_version": PROTOCOL_VERSION,
            }
        )
    return result


def verification_error_row(row: dict[str, Any], split: str, condition: str) -> dict[str, Any]:
    grid, plan, label = grid_from_error(row), list(row["corrupted_plan"]), dict(row["executor_label"])
    sample_id = error_sample_id(row)
    return {
        "example_id": f"verification::{sample_id}",
        "task": "verification",
        "split": split,
        "condition": condition,
        "environment_id": row["environment_id"],
        "base_sample_id": row["base_sample_id"],
        "sample_id": sample_id,
        "corruption_type": row.get("corruption_type"),
        "grid": grid,
        "candidate_plan": plan,
        "executor_label": label,
        "input_text": build_verification_prompt(grid, plan),
        "target_text": verification_target(label),
        "protocol_version": PROTOCOL_VERSION,
    }


def repair_structured_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    inp, out = row["input"], row["output"]
    grid = dict(inp["grid"])
    plan = list(inp["candidate_plan"])
    feedback_source = row["feedback_source"]
    feedback = inp.get("verifier_feedback")
    corruption = row.get("provenance", {}).get("corruption_type", "unknown")
    return {
        "example_id": (
            f"repair::{row['base_sample_id']}::{corruption}::{feedback_source}"
        ),
        "task": "repair",
        "split": split,
        "environment_id": row["environment_id"],
        "base_sample_id": row["base_sample_id"],
        "feedback_source": feedback_source,
        "corruption_type": corruption,
        "grid": grid,
        "candidate_plan": plan,
        "verifier_feedback": feedback,
        "input_text": build_repair_prompt(grid, plan, feedback_source, feedback),
        "target_text": ", ".join(out["repaired_plan"]),
        "target_plan": list(out["repaired_plan"]),
        "protocol_version": PROTOCOL_VERSION,
    }


def repair_error_row(
    row: dict[str, Any], split: str, condition: str, feedback_source: str
) -> dict[str, Any]:
    grid, plan = grid_from_error(row), list(row["corrupted_plan"])
    label = dict(row["executor_label"])
    feedback = {
        "valid": False,
        "first_error_step": label["first_error_step"],
        "reason": label["reason"],
    }
    if feedback_source == "no_feedback":
        feedback = None
    sample_id = error_sample_id(row)
    return {
        "example_id": f"repair::{sample_id}::{feedback_source}",
        "task": "repair",
        "split": split,
        "condition": condition,
        "environment_id": row["environment_id"],
        "base_sample_id": row["base_sample_id"],
        "sample_id": sample_id,
        "feedback_source": feedback_source,
        "corruption_type": row.get("corruption_type"),
        "grid": grid,
        "candidate_plan": plan,
        "verifier_feedback": feedback,
        "input_text": build_repair_prompt(grid, plan, feedback_source, feedback),
        "target_text": ", ".join(row["original_plan"]),
        "target_plan": list(row["original_plan"]),
        "protocol_version": PROTOCOL_VERSION,
    }


def _balanced_repair(rows: list[dict[str, Any]], count: int, seed: int, split: str) -> list[dict[str, Any]]:
    if count % 2:
        raise ValueError("balanced Repair count must be even")
    per_source = count // 2
    selected = []
    for offset, feedback_source in enumerate(("no_feedback", "executor_feedback")):
        pool = [row for row in rows if row["feedback_source"] == feedback_source]
        selected.extend(
            repair_structured_row(row, split)
            for row in take_diverse(pool, per_source, seed + offset)
        )
    return _shuffle(selected, seed + 10)


def build_part1_datasets(root: Path, output: Path, seed: int = 42) -> dict[str, Any]:
    train_base = read_jsonl(root / "data/processed/nov14/train.jsonl")
    dev_base = read_jsonl(root / "data/processed/nov14/dev.jsonl")
    train_errors = read_jsonl(root / "data/samples/nov14_pilot/single_error_train.jsonl")
    dev_errors = read_jsonl(root / "data/samples/nov14_pilot/single_error_dev.jsonl")
    train_repairs = read_jsonl(root / "data/samples/member_c_day14/train_repair.jsonl")
    dev_repairs = read_jsonl(root / "data/samples/member_c_day14/dev_repair.jsonl")
    heldout_base = read_jsonl(root / "data/samples/nov14_pilot/heldout_test.jsonl")
    id_errors = read_jsonl(root / "data/samples/nov15_paired/id_single_error.jsonl")
    ood_errors = read_jsonl(root / "data/samples/nov15_paired/ood_double_error.jsonl")

    generation_train = [
        generation_row(row, "train")
        for row in take_diverse(train_base, 1000, seed)
    ]
    generation_dev = [generation_row(row, "dev") for row in take_diverse(dev_base, 20, seed)]

    train_valid_pool = [
        item for row in train_base for item in verification_valid_rows(row, "train")
    ]
    train_invalid_pool = [
        verification_error_row(row, "train", "id_single_error") for row in train_errors
    ]
    verification_train = _shuffle(
        take_diverse(train_valid_pool, 750, seed)
        + take_diverse(train_invalid_pool, 750, seed + 1),
        seed + 2,
    )
    dev_valid_pool = [
        item for row in dev_base for item in verification_valid_rows(row, "dev")
    ]
    dev_invalid_pool = [
        verification_error_row(row, "dev", "id_single_error") for row in dev_errors
    ]
    verification_dev = _shuffle(
        take_diverse(dev_valid_pool, 45, seed)
        + take_diverse(dev_invalid_pool, 45, seed + 1),
        seed + 2,
    )

    repair_train = _balanced_repair(train_repairs, 1500, seed, "train")
    repair_dev = _balanced_repair(dev_repairs, 90, seed, "dev")

    heldout_by_base = {row["base_sample_id"]: row for row in heldout_base}
    generation_id = [
        generation_row(heldout_by_base[row["base_sample_id"]], "test")
        for row in id_errors
    ]
    id_valid = [
        verification_valid_rows(heldout_by_base[row["base_sample_id"]], "test")[0]
        for row in id_errors[:100]
    ]
    for row in id_valid:
        row["condition"] = "id_valid"
    verification_id = _shuffle(
        id_valid
        + [
            verification_error_row(row, "test", "id_single_error")
            for row in id_errors[:100]
        ],
        seed,
    )
    verification_ood = [
        verification_error_row(row, "test", "ood_double_error") for row in ood_errors
    ]
    repair_id = [
        repair_error_row(row, "test", "id_single_error", "executor_feedback")
        for row in id_errors
    ]
    repair_ood = [
        repair_error_row(row, "test", "ood_double_error", "executor_feedback")
        for row in ood_errors
    ]

    datasets = {
        "generation_train": generation_train,
        "generation_dev": generation_dev,
        "verification_train": verification_train,
        "verification_dev": verification_dev,
        "repair_train": repair_train,
        "repair_dev": repair_dev,
        "generation_id_test": generation_id,
        "verification_id_test": verification_id,
        "verification_ood_test": verification_ood,
        "repair_id_test": repair_id,
        "repair_ood_test": repair_ood,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in datasets.items():
        write_jsonl(output / f"{name}.jsonl", rows)

    train_total = sum(len(datasets[f"{task}_train"]) for task in ("generation", "verification", "repair"))
    dev_total = sum(len(datasets[f"{task}_dev"]) for task in ("generation", "verification", "repair"))
    if train_total != 4000 or dev_total != 200:
        raise AssertionError("Part 1 budget construction failed")
    train_envs = {row["environment_id"] for task in ("generation", "verification", "repair") for row in datasets[f"{task}_train"]}
    dev_envs = {row["environment_id"] for task in ("generation", "verification", "repair") for row in datasets[f"{task}_dev"]}
    test_envs = {row["environment_id"] for row in generation_id}
    if train_envs & dev_envs or train_envs & test_envs or dev_envs & test_envs:
        raise AssertionError("environment leakage detected")

    hashes = {
        f"{name}.jsonl": hashlib.sha256((output / f"{name}.jsonl").read_bytes()).hexdigest()
        for name in datasets
    }
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "counts": {name: len(rows) for name, rows in datasets.items()},
        "train_total": train_total,
        "dev_total": dev_total,
        "train_environment_count": len(train_envs),
        "dev_environment_count": len(dev_envs),
        "test_environment_count": len(test_envs),
        "environment_intersections": {
            "train_dev": len(train_envs & dev_envs),
            "train_test": len(train_envs & test_envs),
            "dev_test": len(dev_envs & test_envs),
        },
        "sha256": hashes,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
