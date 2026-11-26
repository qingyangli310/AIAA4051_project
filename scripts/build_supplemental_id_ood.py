"""Build paired supplementary diagnostics without changing frozen Test files.

These examples reuse the already inspected held-out environments. They are
diagnostic data, not a new blind Test set.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.build_nov15_paired_conditions import (
    ACTIONS, ERROR_KINDS, _apply_operation, _grid, _stable_int,
)
from src.data.ppnl_io import load_jsonl, write_json, write_jsonl
from src.executor.executor import execute_plan


def operations(grid, plan, seed):
    options = []
    for position in range(len(plan) + 1):
        for kind in ERROR_KINDS:
            if kind != "action_insertion" and position == len(plan):
                continue
            choices = ACTIONS if kind in ("action_substitution", "action_insertion") else (None,)
            for action in choices:
                if kind == "action_substitution" and action == plan[position]:
                    continue
                op = {"corruption_type": kind, "position": position}
                if kind == "action_substitution":
                    op["replacement_action"] = action
                elif kind == "action_insertion":
                    op["inserted_action"] = action
                independently = _apply_operation(plan, op)
                label = execute_plan(grid, independently)
                if not label["success"]:
                    op["independent_plan"] = independently
                    op["independent_executor_label"] = label
                    options.append(op)
    random.Random(seed).shuffle(options)
    return options


def pairs(grid, plan, ops):
    """Retain pairs for which removing either edit changes the final execution."""
    found = {}
    for index, first in enumerate(ops):
        for second in ops[index + 1:]:
            if first["position"] == second["position"]:
                continue
            if first["corruption_type"] == second["corruption_type"]:
                continue
            ordered = sorted((first, second), key=lambda op: op["position"], reverse=True)
            if any(op["corruption_type"] == "premature_stopping" for op in ordered):
                stop = next(op for op in ordered if op["corruption_type"] == "premature_stopping")
                if any(op is not stop and op["position"] >= stop["position"] for op in ordered):
                    continue
            combined = list(plan)
            provenance = []
            for op in ordered:
                before = combined
                combined = _apply_operation(combined, op)
                provenance.append({**op, "before_plan": before, "after_plan": combined})
            label = execute_plan(grid, combined)
            if label["success"]:
                continue
            if any(
                (lambda single: label["trace"] == single["trace"]
                 and label["final_position"] == single["final_position"])(
                    execute_plan(grid, _apply_operation(plan, other))
                )
                for other in (first, second)
            ):
                continue
            found.setdefault(label["reason"], {
                "corrupted_plan": combined,
                "executor_label": label,
                "corruption_type": [op["corruption_type"] for op in provenance],
                "corruption_provenance": provenance,
            })
            if len(found) == 3:
                return found
    return found


def shuffled_content_changed(rows):
    """Mirror the pipeline's same-length feedback shuffle."""
    groups = {}
    for row in rows:
        groups.setdefault(len(row["corrupted_plan"]), []).append(row)
    changed = 0
    for group in groups.values():
        for index, target in enumerate(group):
            own = target["executor_label"]
            candidates = group[index + 1:] + group[:index + 1]
            source = next(
                (row for row in candidates if (
                    row["executor_label"]["first_error_step"],
                    row["executor_label"]["reason"],
                ) != (own["first_error_step"], own["reason"])),
                candidates[0],
            )
            other = source["executor_label"]
            changed += (other["first_error_step"], other["reason"]) != (
                own["first_error_step"], own["reason"]
            )
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/samples/nov14_pilot/heldout_test.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/supplemental_id_ood_v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = load_jsonl(args.source)
    rng = random.Random(42)
    rng.shuffle(rows)
    id_counts, ood_counts, kind_counts = Counter(), Counter(), Counter()
    id_rows, ood_rows = [], []
    for row in rows:
        grid, plan = _grid(row), row["shortest_plan"]
        ops = operations(grid, plan, _stable_int(42, row["base_sample_id"]))
        choices = pairs(grid, plan, ops)
        if not choices:
            continue
        ood_reason = min(choices, key=lambda reason: (ood_counts[reason], reason))
        ood = choices[ood_reason]
        id_op = min(
            ops,
            key=lambda op: (
                id_counts[op["independent_executor_label"]["reason"]],
                kind_counts[op["corruption_type"]],
            ),
        )
        id_label = id_op["independent_executor_label"]
        common = {
            "base_sample_id": row["base_sample_id"],
            "environment_id": row["environment_id"],
            "grid": grid,
            "shortest_plan": plan,
            "condition": "supplemental_diagnostic_not_blind_test",
        }
        id_rows.append({
            **common, "sample_id": f"supp_id_{row['base_sample_id']}",
            "corruption_type": id_op["corruption_type"],
            "corrupted_plan": id_op["independent_plan"],
            "executor_label": id_label,
            "corruption_provenance": id_op,
        })
        ood_rows.append({
            **common, "sample_id": f"supp_ood_{row['base_sample_id']}",
            **ood,
        })
        id_counts[id_label["reason"]] += 1
        kind_counts[id_op["corruption_type"]] += 1
        ood_counts[ood_reason] += 1

    if len(id_rows) < 150 or len(ood_counts) < 3:
        raise ValueError(f"insufficient paired diversity: {len(id_rows)} pairs, {ood_counts}")
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "id_single_error.jsonl", id_rows)
    write_jsonl(args.output / "ood_double_error.jsonl", ood_rows)
    manifest = {
        "status": "supplemental_diagnostic_not_blind_test",
        "seed": 42,
        "paired_count": len(id_rows),
        "id_corruption_types": dict(kind_counts),
        "id_executor_reasons": dict(id_counts),
        "ood_executor_reasons": dict(ood_counts),
        "id_shuffled_feedback_content_changed": shuffled_content_changed(id_rows),
        "ood_shuffled_feedback_content_changed": shuffled_content_changed(ood_rows),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "notes": [
            "Uses previously inspected held-out environments; preserve original frozen Test.",
            "OOD operations independently fail and each changes the combined execution.",
        ],
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
