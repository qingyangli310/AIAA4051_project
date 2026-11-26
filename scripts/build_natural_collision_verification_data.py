"""Replace synthetic Train collisions with actual Generator Train failures."""

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

from scripts.build_trace_verification_data import trace_target
from src.executor.executor import execute_plan
from src.models.prompts import PROTOCOL_VERSION, build_verification_prompt
from src.part1.data import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/verification_trace_v1")
    parser.add_argument(
        "--generator-train-predictions",
        type=Path,
        default=ROOT / "results/generation_trace_v9/train_monitor_predictions.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/verification_natural_collision_v2"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    original = read_jsonl(args.source / "verification_train.jsonl")
    dev = read_jsonl(args.source / "verification_dev.jsonl")
    generator_rows = read_jsonl(args.generator_train_predictions)
    dev_environments = {row["environment_id"] for row in dev}
    natural = []
    seen = set()
    for row in generator_rows:
        grid, plan = row["grid"], row["candidate_plan"]
        label = execute_plan(grid, plan)
        if label["reason"] != "obstacle_collision":
            continue
        key = json.dumps([grid, plan], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        if row["environment_id"] in dev_environments:
            raise AssertionError("Generator Train collision overlaps Verification Dev")
        verdict = (
            f"invalid | step={label['first_error_step']} | "
            "reason=obstacle_collision"
        )
        candidate = {
            "example_id": f"verification::generator_train::{row['base_sample_id']}",
            "task": "verification",
            "split": "train",
            "condition": "generator_train_natural_collision",
            "environment_id": row["environment_id"],
            "base_sample_id": row["base_sample_id"],
            "grid": grid,
            "candidate_plan": plan,
            "executor_label": label,
            "failure_reason": "obstacle_collision",
            "input_text": build_verification_prompt(grid, plan),
            "verdict_text": verdict,
            "protocol_version": PROTOCOL_VERSION,
            "representation": "trace_then_verdict_v1",
        }
        candidate["target_text"] = trace_target(
            {**candidate, "target_text": verdict}
        )
        natural.append(candidate)

    synthetic = [
        row for row in original
        if row["executor_label"]["reason"] == "obstacle_collision"
    ]
    if not natural or len(natural) > len(synthetic):
        raise ValueError("natural collision count cannot replace synthetic pool")
    dropped = {row["example_id"] for row in synthetic[:len(natural)]}
    train = [row for row in original if row["example_id"] not in dropped] + natural
    random.Random(42).shuffle(train)
    if len(train) != len(original):
        raise AssertionError("Verification training budget changed")
    if Counter(row["executor_label"]["reason"] for row in train) != Counter(
        row["executor_label"]["reason"] for row in original
    ):
        raise AssertionError("Verification class balance changed")

    args.output.mkdir(parents=True)
    train_path = args.output / "verification_train.jsonl"
    dev_path = args.output / "verification_dev.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    manifest = {
        "seed": 42,
        "train_count": len(train),
        "dev_count": len(dev),
        "natural_generator_train_collisions": len(natural),
        "synthetic_collisions_replaced": len(dropped),
        "train_dev_environment_overlap": len(
            {row["environment_id"] for row in train} & dev_environments
        ),
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256(dev_path.read_bytes()).hexdigest(),
        "generator_predictions_sha256": hashlib.sha256(
            args.generator_train_predictions.read_bytes()
        ).hexdigest(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
