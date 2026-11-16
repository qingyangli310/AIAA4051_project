"""Build paired ID single-error and OOD double-error held-out conditions."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.build_nov14_dataset import single_corruption
from src.data.ppnl_io import load_json, load_jsonl, write_json, write_jsonl
from src.executor.executor import ACTION_DELTAS, execute_plan

SEED = 42
ERROR_KINDS = ("action_substitution", "action_deletion", "action_insertion", "premature_stopping")
ACTIONS = tuple(ACTION_DELTAS)


def _grid(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row[k] for k in ("environment_id", "rows", "cols", "start", "goal", "obstacles")}


def _stable_int(seed: int, base_id: str, salt: str = "") -> int:
    digest = hashlib.sha256(f"{seed}:{base_id}:{salt}".encode()).hexdigest()
    return int(digest[:16], 16)


def _apply_operation(plan: list[str], operation: dict[str, Any]) -> list[str]:
    result = list(plan)
    kind, pos = operation["corruption_type"], operation["position"]
    if kind == "action_substitution":
        result[pos] = operation["replacement_action"]
    elif kind == "action_deletion":
        result.pop(pos)
    elif kind == "action_insertion":
        result.insert(pos, operation["inserted_action"])
    elif kind == "premature_stopping":
        result = result[:pos]
    else:
        raise ValueError(f"unknown corruption type: {kind}")
    return result


def _operation_candidates(grid: dict[str, Any], plan: list[str], seed: int) -> list[dict[str, Any]]:
    candidates = []
    for pos in range(len(plan) + 1):
        for kind in ERROR_KINDS:
            if kind in ("action_substitution", "action_deletion", "premature_stopping") and pos >= len(plan):
                continue
            choices = ACTIONS if kind in ("action_substitution", "action_insertion") else (None,)
            for action in choices:
                if kind == "action_substitution" and action == plan[pos]:
                    continue
                operation = {"corruption_type": kind, "position": pos}
                if kind == "action_substitution":
                    operation["replacement_action"] = action
                    operation["logical_position"] = f"token:{pos}"
                elif kind == "action_deletion":
                    operation["logical_position"] = f"token:{pos}"
                elif kind == "action_insertion":
                    operation["inserted_action"] = action
                    operation["logical_position"] = f"gap:{pos}"
                else:
                    operation["logical_position"] = f"stop:{pos}"
                independent = _apply_operation(plan, operation)
                label = execute_plan(grid, independent)
                if label["valid"] and not label["success"] and label["reason"] == "goal_not_reached":
                    operation["independent_plan"] = independent
                    operation["independent_executor_label"] = label
                    operation["seed"] = _stable_int(seed, grid["environment_id"],
                                                       f"{operation['logical_position']}:{kind}:{action}")
                    candidates.append(operation)
    return candidates


def find_double_corruption(grid: dict[str, Any], plan: list[str], seed: int) -> dict[str, Any] | None:
    """Find two independently failing edits with visible, reached effects."""
    candidates = _operation_candidates(grid, plan, seed)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    pairs = []
    for idx, first in enumerate(candidates):
        for second in candidates[idx + 1 :]:
            if first["position"] == second["position"]:
                continue
            # Search different corruption types first; same-type pairs remain
            # a deterministic fallback as allowed by the project task.
            pairs.append((first, second))
    pairs.sort(key=lambda pair: pair[0]["corruption_type"] == pair[1]["corruption_type"])
    for first, second in pairs:
            # Higher original positions are edited first so lower positions
            # retain their original indexes. A stop boundary must be after the
            # other edit, otherwise it would remove that edit from the plan.
            if "premature_stopping" in (first["corruption_type"], second["corruption_type"]):
                stop = first if first["corruption_type"] == "premature_stopping" else second
                other = second if stop is first else first
                if other["position"] >= stop["position"]:
                    continue
            ordered = sorted((first, second), key=lambda op: op["position"], reverse=True)
            combined = list(plan)
            provenance = []
            for op_index, operation in enumerate(ordered, start=1):
                before = list(combined)
                combined = _apply_operation(combined, operation)
                provenance.append({
                    "operation_id": f"mutation_{op_index}",
                    "corruption_type": operation["corruption_type"],
                    "logical_position": operation["logical_position"],
                    "position": operation["position"],
                    "seed": operation["seed"],
                    "replacement_action": operation.get("replacement_action"),
                    "inserted_action": operation.get("inserted_action"),
                    "before_plan": before,
                    "after_plan": list(combined),
                    "independent_plan": list(operation["independent_plan"]),
                    "independent_executor_label": operation["independent_executor_label"],
                })
            label = execute_plan(grid, combined)
            # Requiring a valid, fully executed non-goal plan proves both action
            # positions were reached rather than hidden by Executor early-stop.
            if label["success"] or not label["valid"] or label["executed_steps"] != len(combined):
                continue
            visible = True
            for operation in (first, second):
                other = second if operation is first else first
                without_this = execute_plan(grid, _apply_operation(plan, other))
                # The combined trace must differ from replay with this mutation
                # removed; this catches a later edit that is overwritten or inert.
                if label["trace"] == without_this["trace"] and label["final_position"] == without_this["final_position"]:
                    visible = False
                    break
            if not visible:
                continue
            if [op["after_plan"] for op in provenance][-1] != combined:
                continue
            return {"plan": combined, "operations": provenance, "executor_label": label}
    return None


def _id_corruption(grid: dict[str, Any], plan: list[str], base_id: str, seed: int, counts: Counter):
    # Choose the currently least represented feasible kind for an even distribution.
    order = list(ERROR_KINDS)
    random.Random(_stable_int(seed, base_id, "id-kind")).shuffle(order)
    order.sort(key=lambda kind: counts[kind])
    for kind in order:
        op_seed = _stable_int(seed, base_id, kind)
        result = single_corruption(grid, plan, kind, op_seed)
        if result is None:
            continue
        corrupted, pos = result
        label = execute_plan(grid, corrupted)
        if label["success"]:
            continue
        return {
            "corruption_type": kind,
            "corruption_position": pos,
            "corruption_seed": op_seed,
            "original_plan": list(plan),
            "corrupted_plan": corrupted,
            "executor_label": label,
        }
    return None


def _validate_base(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 163:
        raise ValueError(f"expected exactly 163 held-out base samples, found {len(rows)}")
    bases = [r.get("base_sample_id") for r in rows]
    envs = [r.get("environment_id") for r in rows]
    if None in bases or len(set(bases)) != len(rows):
        raise ValueError("base_sample_id values must be present and unique")
    if None in envs or len(set(envs)) != len(rows):
        raise ValueError("environment_id values must be present and unique in held-out data")
    for row in rows:
        for key in ("shortest_plan", "nonoptimal_plan"):
            if key not in row:
                raise ValueError(f"{row['base_sample_id']} missing {key}")
            label = execute_plan(_grid(row), row[key])
            if not label["valid"] or not label["success"]:
                raise ValueError(f"invalid {key} for {row['base_sample_id']}: {label}")


def build(input_path: Path, output_dir: Path, manifests_dir: Path, seed: int = SEED) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    _validate_base(rows)
    train = load_json(manifests_dir / "train.json")
    dev = load_json(manifests_dir / "dev.json")
    forbidden_envs = set(train["environment_ids"]) | set(dev["environment_ids"])
    if any(r["environment_id"] in forbidden_envs for r in rows):
        raise ValueError("held-out environment overlaps Train or Dev")

    ood_ready: list[tuple[dict[str, Any], dict[str, Any]]] = []
    exceptions = []
    for row in rows:
        grid = _grid(row)
        source_plan = list(row["shortest_plan"])
        found = find_double_corruption(grid, source_plan, _stable_int(seed, row["base_sample_id"], "ood-search"))
        if found is None:
            exceptions.append({"base_sample_id": row["base_sample_id"], "environment_id": row["environment_id"],
                               "reason": "no_qualifying_distinct_double_corruption_found"})
            continue
        ood_ready.append((row, found))

    id_rows = []
    ood_rows = []
    pairs = []
    kind_counts: Counter = Counter()
    for row, found in ood_ready:
        grid = _grid(row)
        base_id = row["base_sample_id"]
        id_result = _id_corruption(grid, list(row["shortest_plan"]), base_id, seed, kind_counts)
        if id_result is None:
            exceptions.append({"base_sample_id": base_id, "environment_id": row["environment_id"],
                               "reason": "no_failing_id_single_corruption_found"})
            continue
        kind_counts[id_result["corruption_type"]] += 1
        common = {
            "environment_id": row["environment_id"], "base_sample_id": base_id,
            "source_plan_id": f"{base_id}:shortest", "grid": grid,
            "shortest_plan": list(row["shortest_plan"]), "nonoptimal_plan": list(row["nonoptimal_plan"]),
        }
        id_id = f"id_{base_id.removeprefix('base_')}"
        ood_id = f"ood_{base_id.removeprefix('base_')}"
        id_rows.append({"sample_id": id_id, **common, **id_result})
        ood_rows.append({"sample_id": ood_id, **common,
                         "corruption_type": [op["corruption_type"] for op in found["operations"]],
                         "corruption_position": [op["position"] for op in found["operations"]],
                         "corruption_seed": seed, "original_plan": list(row["shortest_plan"]),
                         "corrupted_plan": found["plan"], "executor_label": found["executor_label"],
                         "corruption_provenance": found["operations"]})
        pair_payload = f"{base_id}:{row['environment_id']}"
        pairs.append({"pair_id": "pair_" + hashlib.sha256(pair_payload.encode()).hexdigest(),
                      "base_sample_id": base_id, "environment_id": row["environment_id"],
                      "id_sample_id": id_id, "ood_sample_id": ood_id, "source_plan_id": common["source_plan_id"]})

    if len(id_rows) != len(ood_rows) or len(pairs) != len(id_rows):
        raise AssertionError("internal pairing mismatch")
    manifest = {
        "seed": seed, "source_file": str(input_path), "base_sample_count": len(rows),
        "eligible_pair_count": len(pairs), "id_count": len(id_rows), "ood_count": len(ood_rows),
        "exception_count": len(exceptions), "id_error_distribution": dict(sorted(Counter(r["corruption_type"] for r in id_rows).items())),
        "ood_error_combination_distribution": dict(sorted(Counter("+".join(r["corruption_type"]) for r in ood_rows).items())),
        "train_environment_count": len(train["environment_ids"]), "dev_environment_count": len(dev["environment_ids"]),
        "heldout_environment_count": len({r["environment_id"] for r in rows}),
        "paired_environment_count": len({r["environment_id"] for r in id_rows}),
        "environment_leakage": False,
        "double_corruption_criteria": {
            "two_independent_single_failures": True, "distinct_logical_positions": True,
            "sequential_replay_exact": True, "combined_executor_reaches_both": True,
            "combined_success_false": True,
        },
    }
    # Write only after all preflight checks and all records have been built.
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "id_single_error.jsonl", id_rows)
    write_jsonl(output_dir / "ood_double_error.jsonl", ood_rows)
    write_jsonl(output_dir / "pair_manifest.jsonl", pairs)
    write_json(output_dir / "manifest.json", manifest)
    write_jsonl(output_dir / "exceptions.jsonl", exceptions)
    write_jsonl(manifests_dir / "id_single_error.jsonl", id_rows)
    write_jsonl(manifests_dir / "ood_double_error.jsonl", ood_rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/samples/nov14_pilot/heldout_test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/samples/nov15_paired"))
    parser.add_argument("--manifests", type=Path, default=Path("data/manifests"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output, args.manifests, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
