"""Run deterministic Part 1 reference baselines and interface acceptance checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.executor.executor import execute_plan
from src.part1.baselines import shortest_path
from src.part1.data import read_jsonl, verification_target
from src.part1.evaluate import score_predictions


def generation_outputs(rows):
    return [", ".join(shortest_path(row["grid"])) for row in rows]


def verification_outputs(rows):
    outputs = []
    for row in rows:
        actual = execute_plan(row["grid"], row["candidate_plan"])
        expected = row["executor_label"]
        for field in ("valid", "success", "first_error_step", "reason"):
            if actual[field] != expected[field]:
                raise AssertionError(f"executor label mismatch: {row['example_id']} {field}")
        outputs.append(verification_target(actual))
    return outputs


def repair_outputs(rows):
    return [", ".join(shortest_path(row["grid"])) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/part1")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/part1_symbolic_acceptance.json"
    )
    args = parser.parse_args()
    jobs = {
        "generation_id": ("generation", "generation_id_test.jsonl", generation_outputs),
        "verification_id": ("verification", "verification_id_test.jsonl", verification_outputs),
        "verification_ood": ("verification", "verification_ood_test.jsonl", verification_outputs),
        "repair_id": ("repair", "repair_id_test.jsonl", repair_outputs),
        "repair_ood": ("repair", "repair_ood_test.jsonl", repair_outputs),
    }
    results = {}
    for name, (task, filename, predict) in jobs.items():
        rows = read_jsonl(args.data_dir / filename)
        results[name] = score_predictions(task, rows, predict(rows))
    report = {
        "kind": "symbolic_reference_acceptance_not_language_model_results",
        "all_checks_passed": True,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
