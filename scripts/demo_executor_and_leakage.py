"""Demonstrate the real Executor and frozen environment split checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ppnl_io import load_json, load_jsonl
from src.executor.executor import execute_plan


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train = load_json(root / "data/manifests/train.json")
    dev = load_json(root / "data/manifests/dev.json")
    heldout = load_json(root / "data/manifests/heldout_test.json")
    train_envs, dev_envs, heldout_envs = map(lambda x: set(x["environment_ids"]), (train, dev, heldout))
    if train_envs & dev_envs or train_envs & heldout_envs or dev_envs & heldout_envs:
        raise SystemExit("environment leakage found")

    good = load_jsonl(root / "data/samples/nov14_pilot/heldout_test.jsonl")[0]
    good_grid = {k: good[k] for k in ("environment_id", "rows", "cols", "start", "goal", "obstacles")}
    good_result = execute_plan(good_grid, good["shortest_plan"])
    bad = load_jsonl(root / "data/samples/nov15_paired/id_single_error.jsonl")[0]
    bad_result = execute_plan(bad["grid"], bad["corrupted_plan"])
    id_rows = load_jsonl(root / "data/samples/nov15_paired/id_single_error.jsonl")
    ood_rows = load_jsonl(root / "data/samples/nov15_paired/ood_double_error.jsonl")
    pairs = load_jsonl(root / "data/samples/nov15_paired/pair_manifest.jsonl")
    if not good_result["success"] or bad_result["success"]:
        raise SystemExit("Executor demo returned an unexpected result")
    if not ({r["base_sample_id"] for r in id_rows} == {r["base_sample_id"] for r in ood_rows} ==
            {p["base_sample_id"] for p in pairs}):
        raise SystemExit("ID/OOD pairing mismatch")

    print(json.dumps({
        "correct_path": {"sample_id": good["base_sample_id"], "plan": good["shortest_plan"],
                         "valid": good_result["valid"], "success": good_result["success"],
                         "final_position": good_result["final_position"]},
        "error_path": {"sample_id": bad["sample_id"], "plan": bad["corrupted_plan"],
                       "first_error_step": bad_result["first_error_step"], "reason": bad_result["reason"],
                       "valid": bad_result["valid"], "success": bad_result["success"]},
        "split_environment_intersections": {"train_dev": 0, "train_heldout": 0, "dev_heldout": 0},
        "id_count": len(id_rows), "ood_count": len(ood_rows), "pair_count": len(pairs),
        "executor": "src.executor.executor.execute_plan",
    }, indent=2))


if __name__ == "__main__":
    main()
