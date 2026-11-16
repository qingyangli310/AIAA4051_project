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


def _substitution_candidates(plan: list[str]):
    for pos, old in enumerate(plan):
        for replacement in ACTIONS:
            if replacement == old:
                continue
            candidate = list(plan)
            candidate[pos] = replacement
            yield pos, replacement, candidate


def find_double_substitution(grid: dict[str, Any], plan: list[str], seed: int) -> dict[str, Any] | None:
    """Find two independently failing substitutions with visible, reached effects."""
    candidates = []
    for pos, replacement, candidate in _substitution_candidates(plan):
        label = execute_plan(grid, candidate)
        if not label["success"] and label["valid"] and label["executed_steps"] == len(candidate):
            candidates.append((pos, replacement, candidate, label))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    for idx, first in enumerate(candidates):
        p1, a1, independent1, label1 = first
        for p2, a2, independent2, label2 in candidates[idx + 1 :]:
            if p1 == p2:
                continue
            combined = list(plan)
            combined[p1] = a1
            combined[p2] = a2
            label = execute_plan(grid, combined)
            # Requiring a valid, fully executed non-goal plan proves both action
            # positions were reached rather than hidden by Executor early-stop.
            if label["success"] or not label["valid"] or label["executed_steps"] != len(combined):
                continue
            trace_original = execute_plan(grid, plan)["trace"]
            trace_first = execute_plan(grid, independent1)["trace"]
            trace_second = execute_plan(grid, independent2)["trace"]
            trace_combined = label["trace"]
            # Each changed token must alter its immediate resulting position;
            # the second must also differ from replaying only mutation one.
            if trace_combined[p1 + 1] == trace_original[p1 + 1]:
                continue
            if trace_combined[p2 + 1] == trace_first[p2 + 1]:
                continue
            if trace_second[p2 + 1] == trace_original[p2 + 1]:
                continue
            ops = []
            sequential = list(plan)
            for op_index, (pos, action) in enumerate(((p1, a1), (p2, a2)), start=1):
                before = list(sequential)
                sequential[pos] = action
                independent = list(plan)
                independent[pos] = action
                ops.append({
                    "operation_id": f"mutation_{op_index}",
                    "corruption_type": "action_substitution",
                    "logical_position": f"token:{pos}",
                    "position": pos,
                    "seed": _stable_int(seed, grid["environment_id"], f"{pos}:{action}:{op_index}"),
                    "original_action": plan[pos],
                    "replacement_action": action,
                    "before_plan": before,
                    "after_plan": list(sequential),
                    "independent_plan": independent,
                    "independent_executor_label": execute_plan(grid, independent),
                })
            if sequential != combined:
                continue
            return {"plan": combined, "operations": ops, "executor_label": label}
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
        found = find_double_substitution(grid, source_plan, _stable_int(seed, row["base_sample_id"], "ood-search"))
        if found is None:
            exceptions.append({"base_sample_id": row["base_sample_id"], "environment_id": row["environment_id"],
                               "reason": "no_distinct_reached_non_cancelling_double_substitution_found"})
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
