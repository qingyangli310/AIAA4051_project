"""Build 500 feedback suffix examples from Train and 50 from Dev."""

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

from scripts.build_generation_like_repair_data import (
    deduplicate_candidate_model_io, generation_like_candidates, shortest_suffix,
    take_environment_diverse,
)
from src.executor.executor import execute_plan
from src.models.prompts import PROTOCOL_VERSION, build_suffix_repair_prompt
from src.part1.data import read_jsonl, write_jsonl


def normalize(grid, plan, base_id, family, corruption):
    label = execute_plan(grid, plan)
    if label["success"]:
        raise ValueError("Repair candidate must fail")
    step = label["first_error_step"]
    suffix = shortest_suffix(grid, label["final_position"])
    target = plan[:step] + suffix
    if not execute_plan(grid, target)["success"]:
        raise AssertionError("Repair target is not executable")
    return {
        "grid": grid, "candidate_plan": plan, "executor_label": label,
        "target_plan": target, "suffix_plan": suffix,
        "environment_id": grid["environment_id"], "base_sample_id": base_id,
        "candidate_family": family, "corruption_type": corruption,
    }


def choose(base, verification, monitor, *, split, seed):
    generated = deduplicate_candidate_model_io(
        generation_like_candidates(base, "localized_suffix")
    )
    generated_count = 199 if split == "train" else 26
    selected = [normalize(
        row["grid"], row["candidate_plan"], row["base_sample_id"],
        "generation_like_collision", "manhattan_collision",
    ) for row in take_environment_diverse(generated, generated_count, seed)]
    if split == "train":
        natural = [normalize(
            row["grid"], row["candidate_plan"], row["base_sample_id"],
            "generator_train_failure", "generator_collision",
        ) for row in monitor if row["diagnostic_reason"] == "obstacle_collision"]
        if len(natural) != 37:
            raise ValueError(f"expected 37 archived Train failures, got {len(natural)}")
        selected.extend(natural)
    reasons = ("goal_not_reached", "out_of_bounds", "invalid_action_token") if split == "train" else (
        "goal_not_reached", "out_of_bounds", "invalid_action_token", "obstacle_collision"
    )
    per_reason = 88 if split == "train" else 6
    for index, reason in enumerate(reasons):
        pool = [row for row in verification if row["executor_label"]["reason"] == reason]
        chosen = take_environment_diverse(pool, per_reason, seed + 100 + index)
        selected.extend(normalize(
            row["grid"], row["candidate_plan"], row["base_sample_id"],
            "controlled_error", row.get("corruption_type", reason),
        ) for row in chosen)
    random.Random(seed).shuffle(selected)
    expected = 500 if split == "train" else 50
    if len(selected) != expected:
        raise AssertionError(f"unexpected {split} count {len(selected)}")
    signatures = [json.dumps([r["grid"],r["candidate_plan"]],sort_keys=True) for r in selected]
    if len(set(signatures)) != len(signatures):
        raise AssertionError(f"duplicate {split} model candidates")
    return selected


def materialize(candidates, split):
    rows = []
    for index, c in enumerate(candidates):
        step = c["executor_label"]["first_error_step"]
        rows.append({
            "example_id": f"repair_v5::{split}::{index:04d}",
            "task": "repair", "split": split, "condition": "feedback_suffix_v1",
            **c,
            "feedback_source": "executor_feedback",
            "verifier_feedback": c["executor_label"],
            "valid_prefix": c["candidate_plan"][:step],
            "input_text": build_suffix_repair_prompt(
                c["grid"], c["candidate_plan"], "executor_feedback", c["executor_label"]
            ),
            "target_text": "Suffix: " + (", ".join(c["suffix_plan"]) or "<done>"),
            "output_format": "suffix_v1", "prompt_protocol": "suffix_v1",
            "protocol_version": PROTOCOL_VERSION,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/repair_v5_suffix")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    train = choose(
        read_jsonl(ROOT / "data/processed/nov14/train.jsonl"),
        read_jsonl(ROOT / "data/verification_trace_v1/verification_train.jsonl"),
        read_jsonl(ROOT / "results/generation_trace_v9/train_monitor_predictions.jsonl"),
        split="train", seed=42,
    )
    dev = choose(
        read_jsonl(ROOT / "data/processed/nov14/dev.jsonl"),
        read_jsonl(ROOT / "data/verification_trace_v1/verification_dev.jsonl"),
        [], split="dev", seed=1042,
    )
    if {r["environment_id"] for r in train} & {r["environment_id"] for r in dev}:
        raise AssertionError("Train/Dev environment overlap")
    args.output.mkdir(parents=True)
    for split, candidates in (("train",train),("dev",dev)):
        write_jsonl(args.output / f"repair_{split}.jsonl", materialize(candidates,split))
    manifest = {
        "seed": 42, "train_count": len(train), "dev_count": len(dev),
        "combined_with_existing_no_feedback_baseline_train_count": 1000,
        "train_reasons": dict(Counter(r["executor_label"]["reason"] for r in train)),
        "dev_reasons": dict(Counter(r["executor_label"]["reason"] for r in dev)),
        "train_families": dict(Counter(r["candidate_family"] for r in train)),
        "train_dev_environment_overlap": 0,
        "train_sha256": hashlib.sha256((args.output / "repair_train.jsonl").read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256((args.output / "repair_dev.jsonl").read_bytes()).hexdigest(),
        "data_policy": "Only Train environments build Train examples; archived Generator Train monitor only; no Test tuning.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
