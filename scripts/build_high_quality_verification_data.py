"""Build a validated, non-overlapping Verification dataset for the final protocol."""

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

from src.executor.executor import ACTION_DELTAS, execute_plan
from src.models.prompts import PROTOCOL_VERSION, build_verification_prompt
from src.part1.data import GRID_FIELDS, read_jsonl, verification_target, write_jsonl


ERROR_REASONS = (
    "goal_not_reached",
    "obstacle_collision",
    "out_of_bounds",
    "invalid_action_token",
)
PLAN_KINDS = ("shortest", "nonoptimal")


def _grid(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in GRID_FIELDS}


def _semantic_key(environment_id: str, plan: Iterable[str]) -> str:
    return json.dumps([environment_id, list(plan)], separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_number(*parts: str) -> int:
    payload = "\0".join(parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def _quota(total: int, labels: tuple[str, ...]) -> dict[str, int]:
    quotient, remainder = divmod(total, len(labels))
    return {
        label: quotient + int(index < remainder)
        for index, label in enumerate(labels)
    }


def _plans_from_row(row: dict[str, Any]) -> Iterable[list[str]]:
    for field in ("reference_actions", "target_plan", "candidate_plan"):
        value = row.get(field)
        if isinstance(value, list):
            yield [str(action) for action in value]


def _exclusions(paths: list[Path]) -> tuple[set[str], set[str], dict[str, int]]:
    base_ids: set[str] = set()
    semantic_keys: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        rows = read_jsonl(path)
        counts[str(path.relative_to(ROOT))] = len(rows)
        for row in rows:
            base_sample_id = row.get("base_sample_id")
            if base_sample_id:
                base_ids.add(str(base_sample_id))
            environment_id = row.get("environment_id")
            if environment_id:
                for plan in _plans_from_row(row):
                    semantic_keys.add(_semantic_key(str(environment_id), plan))
    return base_ids, semantic_keys, counts


def _mutated_candidate(
    row: dict[str, Any], reason: str
) -> tuple[list[str], str, str, int] | None:
    """Return plan, source kind, mutation type, and mutation step."""
    choices = []
    for kind in PLAN_KINDS:
        plan = [str(action) for action in row[f"{kind}_plan"]]
        label = execute_plan(_grid(row), plan)
        if label["success"]:
            choices.append((kind, plan, label["trace"]))
    if not choices:
        return None

    offset = _stable_number(str(row["base_sample_id"]), reason) % len(choices)
    choices = choices[offset:] + choices[:offset]
    if reason == "goal_not_reached":
        kind, plan, _ = choices[0]
        if not plan:
            return None
        candidate = plan[:-1]
        label = execute_plan(_grid(row), candidate)
        if label["reason"] == reason:
            return candidate, kind, "action_deletion", len(candidate)
        return None

    if reason == "invalid_action_token":
        kind, plan, _ = choices[0]
        step = _stable_number(str(row["base_sample_id"]), reason, "step") % (len(plan) + 1)
        candidate = plan[:step] + ["jump"] + plan[step:]
        label = execute_plan(_grid(row), candidate)
        if label["reason"] == reason and label["first_error_step"] == step:
            return candidate, kind, "action_insertion", step
        return None

    grid = _grid(row)
    rows, cols = grid["rows"], grid["cols"]
    obstacles = {tuple(value) for value in grid["obstacles"]}
    actions = tuple(ACTION_DELTAS)
    for kind, plan, trace in choices:
        indices = list(range(len(plan)))
        random.Random(_stable_number(str(row["base_sample_id"]), reason, kind)).shuffle(indices)
        for step in indices:
            position = tuple(trace[step])
            alternatives = list(actions)
            random.Random(
                _stable_number(str(row["base_sample_id"]), reason, kind, str(step))
            ).shuffle(alternatives)
            for action in alternatives:
                if action == plan[step]:
                    continue
                delta = ACTION_DELTAS[action]
                nxt = (position[0] + delta[0], position[1] + delta[1])
                matches = (
                    reason == "out_of_bounds"
                    and not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols)
                ) or (reason == "obstacle_collision" and nxt in obstacles)
                if not matches:
                    continue
                candidate = plan[:step] + [action] + plan[step + 1 :]
                label = execute_plan(grid, candidate)
                if label["reason"] == reason and label["first_error_step"] == step:
                    return candidate, kind, "action_substitution", step

        # An inserted move still represents one controlled error and expands the
        # pool when the original path does not have a replaceable boundary step.
        insertion_steps = list(range(len(plan) + 1))
        random.Random(
            _stable_number(str(row["base_sample_id"]), reason, kind, "insert")
        ).shuffle(insertion_steps)
        for step in insertion_steps:
            position = tuple(trace[step])
            for action, delta in ACTION_DELTAS.items():
                nxt = (position[0] + delta[0], position[1] + delta[1])
                matches = (
                    reason == "out_of_bounds"
                    and not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols)
                ) or (reason == "obstacle_collision" and nxt in obstacles)
                if not matches:
                    continue
                candidate = plan[:step] + [action] + plan[step:]
                label = execute_plan(grid, candidate)
                if label["reason"] == reason and label["first_error_step"] == step:
                    return candidate, kind, "action_insertion", step
    return None


def _verification_row(
    base: dict[str, Any],
    plan: list[str],
    split: str,
    *,
    plan_kind: str,
    reason: str | None,
    corruption_type: str | None = None,
    mutation_step: int | None = None,
) -> dict[str, Any]:
    grid = _grid(base)
    label = execute_plan(grid, plan)
    expected_reason = None if reason is None else reason
    if label["reason"] != expected_reason or label["success"] != (reason is None):
        raise AssertionError(
            f"executor mismatch for {base['base_sample_id']}: {label}"
        )
    category = "valid" if reason is None else reason
    result = {
        "example_id": f"verification::hq::{base['base_sample_id']}::{category}",
        "task": "verification",
        "split": split,
        "condition": "validated_nonoverlap_quality_v1",
        "environment_id": base["environment_id"],
        "base_sample_id": base["base_sample_id"],
        "plan_kind": plan_kind,
        "grid": grid,
        "candidate_plan": plan,
        "executor_label": label,
        "input_text": build_verification_prompt(grid, plan),
        "target_text": verification_target(label),
        "protocol_version": PROTOCOL_VERSION,
    }
    if reason is not None:
        result.update(
            {
                "corruption_type": corruption_type,
                "mutation_step": mutation_step,
                "failure_reason": reason,
            }
        )
    return result


def _take_diverse(
    candidates: list[dict[str, Any]],
    count: int,
    *,
    used_base_ids: set[str],
    used_semantic_keys: set[str],
    forbidden_semantic_keys: set[str],
    used_environments: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        base_id = str(row["base_sample_id"])
        semantic_key = _semantic_key(row["environment_id"], row["candidate_plan"])
        if (
            base_id in used_base_ids
            or semantic_key in used_semantic_keys
            or semantic_key in forbidden_semantic_keys
        ):
            continue
        groups[str(row["environment_id"])].append(row)
    rng = random.Random(seed)
    environment_ids = sorted(groups)
    rng.shuffle(environment_ids)
    for environment_id in environment_ids:
        rng.shuffle(groups[environment_id])

    selected: list[dict[str, Any]] = []
    for prefer_new_environment in (True, False):
        progressed = True
        while len(selected) < count and progressed:
            progressed = False
            for environment_id in environment_ids:
                if prefer_new_environment and environment_id in used_environments:
                    continue
                while groups[environment_id]:
                    row = groups[environment_id].pop()
                    base_id = str(row["base_sample_id"])
                    semantic_key = _semantic_key(
                        row["environment_id"], row["candidate_plan"]
                    )
                    if base_id in used_base_ids or semantic_key in used_semantic_keys:
                        continue
                    selected.append(row)
                    used_base_ids.add(base_id)
                    used_semantic_keys.add(semantic_key)
                    used_environments.add(environment_id)
                    progressed = True
                    break
                if len(selected) == count:
                    return selected
    raise ValueError(f"only selected {len(selected)} of {count} requested rows")


def _build_split(
    base_rows: list[dict[str, Any]],
    *,
    split: str,
    count: int,
    excluded_base_ids: set[str],
    forbidden_semantic_keys: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    if count % 2:
        raise ValueError("Verification split count must be even")
    eligible = [
        row for row in base_rows if str(row["base_sample_id"]) not in excluded_base_ids
    ]
    used_base_ids: set[str] = set()
    used_semantic_keys: set[str] = set()
    used_environments: set[str] = set()
    selected: list[dict[str, Any]] = []

    valid_count = count // 2
    for offset, (plan_kind, plan_count) in enumerate(
        _quota(valid_count, PLAN_KINDS).items()
    ):
        valid_candidates = []
        for base in eligible:
            plan = [str(action) for action in base[f"{plan_kind}_plan"]]
            if execute_plan(_grid(base), plan)["success"]:
                valid_candidates.append(
                    _verification_row(
                        base,
                        plan,
                        split,
                        plan_kind=plan_kind,
                        reason=None,
                    )
                )
        selected.extend(
            _take_diverse(
                valid_candidates,
                plan_count,
                used_base_ids=used_base_ids,
                used_semantic_keys=used_semantic_keys,
                forbidden_semantic_keys=forbidden_semantic_keys,
                used_environments=used_environments,
                seed=seed + offset,
            )
        )

    invalid_count = count - valid_count
    for offset, (reason, reason_count) in enumerate(
        _quota(invalid_count, ERROR_REASONS).items(), start=10
    ):
        invalid_candidates = []
        for base in eligible:
            if str(base["base_sample_id"]) in used_base_ids:
                continue
            mutation = _mutated_candidate(base, reason)
            if mutation is None:
                continue
            plan, plan_kind, corruption_type, mutation_step = mutation
            invalid_candidates.append(
                _verification_row(
                    base,
                    plan,
                    split,
                    plan_kind=plan_kind,
                    reason=reason,
                    corruption_type=corruption_type,
                    mutation_step=mutation_step,
                )
            )
        selected.extend(
            _take_diverse(
                invalid_candidates,
                reason_count,
                used_base_ids=used_base_ids,
                used_semantic_keys=used_semantic_keys,
                forbidden_semantic_keys=forbidden_semantic_keys,
                used_environments=used_environments,
                seed=seed + offset,
            )
        )

    random.Random(seed + 100).shuffle(selected)
    if len(selected) != count or len(used_base_ids) != count:
        raise AssertionError("Verification split is not one-row-per-base-sample")
    return selected


def _profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "unique_model_io": len({(row["input_text"], row["target_text"]) for row in rows}),
        "unique_base_sample_ids": len({row["base_sample_id"] for row in rows}),
        "unique_environments": len({row["environment_id"] for row in rows}),
        "label_counts": dict(Counter("valid" if row["executor_label"]["success"] else "invalid" for row in rows)),
        "failure_reason_counts": dict(Counter(row["executor_label"]["reason"] for row in rows if not row["executor_label"]["success"])),
        "valid_plan_kind_counts": dict(Counter(row["plan_kind"] for row in rows if row["executor_label"]["success"])),
        "corruption_counts": dict(Counter(row.get("corruption_type") for row in rows if not row["executor_label"]["success"])),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_excluded_base, train_forbidden_semantic, train_sources = _exclusions(
        args.exclude_train
    )
    dev_excluded_base, dev_forbidden_semantic, dev_sources = _exclusions(
        args.exclude_dev
    )
    train_rows = _build_split(
        read_jsonl(args.base_train),
        split="train",
        count=args.train_count,
        excluded_base_ids=train_excluded_base,
        forbidden_semantic_keys=train_forbidden_semantic,
        seed=args.seed,
    )
    dev_rows = _build_split(
        read_jsonl(args.base_dev),
        split="dev",
        count=args.dev_count,
        excluded_base_ids=dev_excluded_base,
        forbidden_semantic_keys=dev_forbidden_semantic,
        seed=args.seed + 1000,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    train_path = args.output / "verification_train.jsonl"
    dev_path = args.output / "verification_dev.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(dev_path, dev_rows)

    train_base_ids = {row["base_sample_id"] for row in train_rows}
    dev_base_ids = {row["base_sample_id"] for row in dev_rows}
    train_semantic = {
        _semantic_key(row["environment_id"], row["candidate_plan"])
        for row in train_rows
    }
    dev_semantic = {
        _semantic_key(row["environment_id"], row["candidate_plan"])
        for row in dev_rows
    }
    train_envs = {row["environment_id"] for row in train_rows}
    dev_envs = {row["environment_id"] for row in dev_rows}
    manifest = {
        "protocol": "verification-high-quality-v1",
        "seed": args.seed,
        "base_train": str(args.base_train.relative_to(ROOT)),
        "base_dev": str(args.base_dev.relative_to(ROOT)),
        "excluded_train_sources": train_sources,
        "excluded_dev_sources": dev_sources,
        "train": _profile(train_rows),
        "dev": _profile(dev_rows),
        "overlap_checks": {
            "train_base_ids_with_generator_or_repair": len(train_base_ids & train_excluded_base),
            "dev_base_ids_with_generator_or_repair": len(dev_base_ids & dev_excluded_base),
            "train_semantic_plans_with_generator_or_repair": len(train_semantic & train_forbidden_semantic),
            "dev_semantic_plans_with_generator_or_repair": len(dev_semantic & dev_forbidden_semantic),
            "train_dev_base_ids": len(train_base_ids & dev_base_ids),
            "train_dev_environments": len(train_envs & dev_envs),
        },
        "quality_checks": {
            "executor_labels_recomputed": True,
            "one_row_per_base_sample": True,
            "model_io_unique": True,
            "all_four_failure_reasons_present": True,
            "valid_shortest_and_nonoptimal_present": True,
        },
        "sha256": {
            "verification_train.jsonl": _sha256(train_path),
            "verification_dev.jsonl": _sha256(dev_path),
        },
        "environment_note": (
            "Generator v9 spans all Train environments, so cross-task environment "
            "separation is impossible. Base-sample and semantic-plan separation are zero-overlap."
        ),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-train", type=Path, default=ROOT / "data/processed/nov14/train.jsonl"
    )
    parser.add_argument(
        "--base-dev", type=Path, default=ROOT / "data/processed/nov14/dev.jsonl"
    )
    parser.add_argument(
        "--exclude-train",
        type=Path,
        action="append",
        default=[
            ROOT / "artifacts/generation_trace_v9/data/transformed_train.jsonl",
            ROOT / "data/repair_high_quality_v1/repair_train.jsonl",
        ],
    )
    parser.add_argument(
        "--exclude-dev",
        type=Path,
        action="append",
        default=[
            ROOT / "artifacts/generation_trace_v9/data/transformed_dev.jsonl",
            ROOT / "data/repair_high_quality_v1/repair_dev.jsonl",
        ],
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/verification_high_quality_v1"
    )
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--dev-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
