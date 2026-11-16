"""Audit the existing Project 6 seed-42 data and write freeze records.

This script reads source and derived data only. It never regenerates datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_nov15_paired_conditions import _apply_operation
from src.data.load_ppnl import FILES, load_ppnl
from src.data.ppnl_io import load_json, load_jsonl, write_json
from src.executor.executor import execute_plan

FREEZE_ID = "project6-6x6-seed42-20261116"
PPNL_COMMIT = "5fd0376fc50ae9860bc986085165ebac5675fd44"
GENERATION_COMMIT = "c2a223f310fb053e1583dc26c069f2a429445427"
EXPECTED_FILES = {
    "1_train_set_6x6_samples.json": (16032, "b20a44269cff617abd5132d3ad209f72fb0ce92f78b4eaefc5aa8a92fac093e2"),
    "1dev_set_6x6_samples.json": (2004, "aad6f181c4eff903c1f262ae009f2c3ac2d0f787abae4765c51976fd1ee8c574"),
    "1_goals_test_seen_6x6_samples.json": (2004, "ebdc305d207b0fda3d147b76ddd93fdecdb879dd5bd6acfad90bd242c15a2507"),
    "1goals_unseen_6x6_samples.json": (5040, "ee841e3adb907116471989331f0e4cffed16038c46592491abdd7bafd34d432e"),
    "1_goals_test_unseen_5x5_samples.json": (3750, "f34edfe112d248087ae76366c4f4e56cc0e0d31729d0bd981b6c8656344b32cd"),
    "1_goals_test_unseen_7x7_samples.json": (3750, "db358fd01da87004c2d62551e44cfcb506cb00e88aba29815ebc0fd1c288eb58"),
    "1_goals_test_unseen_6x6more_obstacles_samples.json": (4500, "8825fee0599751b4da946b9b6a75808bd9aeff8a5595e8db9e833ee38828b16d"),
}
EXPECTED_SPLITS = {
    "train": (572, 17510), "dev": (82, 2517), "heldout_test": (163, 4989)
}
DATA_FILES = [
    "data/processed/nov14/train.jsonl", "data/processed/nov14/dev.jsonl",
    "data/processed/nov14/heldout_test.jsonl",
    "data/manifests/train.json", "data/manifests/dev.json",
    "data/manifests/heldout_test.json", "data/manifests/excluded_manifest.json",
    "data/samples/nov14_pilot/train.jsonl", "data/samples/nov14_pilot/dev.jsonl",
    "data/samples/nov14_pilot/heldout_test.jsonl",
    "data/samples/nov14_pilot/single_error_train.jsonl",
    "data/samples/nov14_pilot/single_error_dev.jsonl",
    "data/samples/nov14_pilot/manifest.json",
    "data/samples/legacy_a4_pilot/train.jsonl", "data/samples/legacy_a4_pilot/dev.jsonl",
    "data/samples/legacy_a4_pilot/id_test.jsonl", "data/samples/legacy_a4_pilot/ood_test.jsonl",
    "data/samples/legacy_a4_pilot/manifest.json",
    "data/samples/ppnl_integration_20261113.jsonl",
    "data/samples/nov15_paired/id_single_error.jsonl",
    "data/samples/nov15_paired/ood_double_error.jsonl",
    "data/samples/nov15_paired/pair_manifest.jsonl",
    "data/samples/nov15_paired/manifest.json",
    "data/samples/nov15_paired/exceptions.jsonl",
    "data/manifests/id_single_error.jsonl", "data/manifests/ood_double_error.jsonl",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grid(record: dict[str, Any]) -> dict[str, Any]:
    if "grid" in record:
        return record["grid"]
    return {key: record[key] for key in ("environment_id", "rows", "cols", "start", "goal", "obstacles")}


def audit(repo: Path, ppnl_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = repo.resolve()
    ppnl_root = ppnl_root.resolve()
    ppnl_repo = ppnl_root.parents[1]
    actual_commit = subprocess.check_output(["git", "-C", str(ppnl_repo), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != PPNL_COMMIT:
        raise AssertionError(f"PPNL commit mismatch: {actual_commit}")

    source_files = []
    total_raw = total_converted = total_unreachable = 0
    for filename in FILES:
        path = ppnl_root / filename
        expected_count, expected_sha = EXPECTED_FILES[filename]
        if not path.is_file():
            raise AssertionError(f"missing official file: {path}")
        sha = _sha256(path)
        if sha != expected_sha:
            raise AssertionError(f"source checksum mismatch for {filename}: {sha}")
        loaded = load_ppnl(path)
        issues = loaded["issues"]
        if loaded["total"] != expected_count or any("unreachable_source" not in x["error"] for x in issues):
            raise AssertionError(f"unexpected parsing/conversion result for {filename}")
        if any("source_file" not in x or "source_index" not in x for x in issues):
            raise AssertionError(f"untraceable source issue in {filename}")
        source_files.append({"file": filename, "records": loaded["total"], "converted": len(loaded["records"]),
                             "unreachable_excluded": len(issues), "sha256": sha,
                             "json_parse_success": True, "required_fields_missing": 0})
        total_raw += loaded["total"]
        total_converted += len(loaded["records"])
        total_unreachable += len(issues)
    if (total_raw, total_converted, total_unreachable) != (37080, 36569, 511):
        raise AssertionError("PPNL grand totals do not match frozen source audit")
    standard_files = FILES[:4]
    standard = [x for x in source_files if x["file"] in standard_files]
    if (sum(x["records"] for x in standard), sum(x["converted"] for x in standard),
        sum(x["unreachable_excluded"] for x in standard)) != (25080, 25016, 64):
        raise AssertionError("standard 6x6 totals mismatch")

    split_records: dict[str, list[dict[str, Any]]] = {}
    split_envs: dict[str, set[str]] = {}
    split_summaries = {}
    for split, (expected_env_count, expected_samples) in EXPECTED_SPLITS.items():
        manifest = load_json(repo / f"data/manifests/{split}.json")
        records = load_jsonl(repo / f"data/processed/nov14/{split}.jsonl")
        if manifest["seed"] != 42 or len(records) != expected_samples or len(manifest["environment_ids"]) != expected_env_count:
            raise AssertionError(f"{split} counts or seed mismatch")
        if manifest["sample_count"] != len(records) or manifest["environment_count"] != expected_env_count:
            raise AssertionError(f"{split} manifest count mismatch")
        envs = {r["environment_id"] for r in records}
        if envs != set(manifest["environment_ids"]):
            raise AssertionError(f"{split} environment manifest mismatch")
        if len({r["base_sample_id"] for r in records}) != len(records):
            raise AssertionError(f"duplicate base_sample_id in {split}")
        for record in records:
            grid = _grid(record)
            shortest = execute_plan(grid, record["shortest_plan"])
            if not shortest["valid"] or not shortest["success"] or len(record["shortest_plan"]) != record["shortest_length"]:
                raise AssertionError(f"bad shortest path: {record['base_sample_id']}")
            nonoptimal = execute_plan(grid, record["nonoptimal_plan"])
            if not nonoptimal["valid"] or not nonoptimal["success"] or len(record["nonoptimal_plan"]) <= len(record["shortest_plan"]):
                raise AssertionError(f"bad nonoptimal path: {record['base_sample_id']}")
        split_records[split] = records
        split_envs[split] = envs
        split_summaries[split] = {"environment_count": len(envs), "sample_count": len(records)}
    if split_envs["train"] & split_envs["dev"] or split_envs["train"] & split_envs["heldout_test"] or split_envs["dev"] & split_envs["heldout_test"]:
        raise AssertionError("Train/Dev/Held-out environment leakage")

    excluded = load_json(repo / "data/manifests/excluded_manifest.json")
    excluded_rows = excluded["records"]
    if excluded["source_commit"] != PPNL_COMMIT or excluded["count"] != 511 or len(excluded_rows) != 511:
        raise AssertionError("excluded manifest count or commit mismatch")
    excluded_keys = [(x.get("source_file"), x.get("source_index")) for x in excluded_rows]
    if any(not f or i is None for f, i in excluded_keys) or len(set(excluded_keys)) != 511:
        raise AssertionError("excluded records missing or duplicating source provenance")

    id_rows = load_jsonl(repo / "data/samples/nov15_paired/id_single_error.jsonl")
    ood_rows = load_jsonl(repo / "data/samples/nov15_paired/ood_double_error.jsonl")
    pairs = load_jsonl(repo / "data/samples/nov15_paired/pair_manifest.jsonl")
    exceptions = load_jsonl(repo / "data/samples/nov15_paired/exceptions.jsonl")
    id_by_base = {r["base_sample_id"]: r for r in id_rows}
    ood_by_base = {r["base_sample_id"]: r for r in ood_rows}
    if len(id_rows) != 162 or len(ood_rows) != 162 or len(pairs) != 162 or len(exceptions) != 1:
        raise AssertionError("paired test row counts differ from frozen Nov 15 data")
    if len(id_by_base) != len(id_rows) or set(id_by_base) != set(ood_by_base) or set(id_by_base) != {x["base_sample_id"] for x in pairs}:
        raise AssertionError("ID/OOD base_sample_id pairing mismatch")
    id_envs = {r["environment_id"] for r in id_rows}
    ood_envs = {r["environment_id"] for r in ood_rows}
    if id_envs != ood_envs or id_envs & (split_envs["train"] | split_envs["dev"]):
        raise AssertionError("ID/OOD environment mismatch or Train/Dev leakage")
    for pair in pairs:
        base_id = pair["base_sample_id"]
        if pair["id_sample_id"] != id_by_base[base_id]["sample_id"] or pair["ood_sample_id"] != ood_by_base[base_id]["sample_id"]:
            raise AssertionError(f"pair row does not reference both conditions for {base_id}")
    id_counts = Counter()
    for row in id_rows:
        id_counts[row["corruption_type"]] += 1
        result = execute_plan(row["grid"], row["corrupted_plan"])
        if result != row["executor_label"] or result["success"]:
            raise AssertionError(f"ID executor label mismatch: {row['sample_id']}")
    combo_counts = Counter()
    for row in ood_rows:
        label = execute_plan(row["grid"], row["corrupted_plan"])
        ops = row["corruption_provenance"]
        if label != row["executor_label"] or label["success"] or not label["valid"] or label["executed_steps"] != len(row["corrupted_plan"]):
            raise AssertionError(f"OOD combined Executor result mismatch: {row['sample_id']}")
        if len(ops) != 2 or ops[0]["position"] == ops[1]["position"] or ops[0]["logical_position"] == ops[1]["logical_position"]:
            raise AssertionError(f"OOD mutation locations not distinct: {row['sample_id']}")
        sequential = list(row["original_plan"])
        for op in ops:
            if op["before_plan"] != sequential:
                raise AssertionError(f"OOD sequential before_plan mismatch: {row['sample_id']}")
            independent = execute_plan(row["grid"], op["independent_plan"])
            if independent != op["independent_executor_label"] or independent["success"] or not independent["valid"]:
                raise AssertionError(f"OOD independent mutation did not fail: {row['sample_id']}")
            sequential = _apply_operation(sequential, op)
            if sequential != op["after_plan"]:
                raise AssertionError(f"OOD sequential replay mismatch: {row['sample_id']}")
        if sequential != row["corrupted_plan"]:
            raise AssertionError(f"OOD final replay differs: {row['sample_id']}")
        for index in range(2):
            without = _apply_operation(row["original_plan"], ops[1-index])
            other_label = execute_plan(row["grid"], without)
            if label["trace"] == other_label["trace"] and label["final_position"] == other_label["final_position"]:
                raise AssertionError(f"OOD mutation was obscured or had no observable effect: {row['sample_id']}")
        combo_counts["+".join(op["corruption_type"] for op in ops)] += 1
    if set(id_counts) != {"action_substitution", "action_deletion", "action_insertion", "premature_stopping"}:
        raise AssertionError("ID corruption kinds are incomplete")

    paired_manifest = load_json(repo / "data/samples/nov15_paired/manifest.json")
    if paired_manifest["id_count"] != len(id_rows) or paired_manifest["ood_count"] != len(ood_rows) or paired_manifest["eligible_pair_count"] != len(pairs):
        raise AssertionError("Nov 15 manifest does not match its JSONL files")
    heldout_pilot = load_jsonl(repo / "data/samples/nov14_pilot/heldout_test.jsonl")
    for record in heldout_pilot:
        grid = _grid(record)
        for key in ("shortest_plan", "nonoptimal_plan"):
            result = execute_plan(grid, record[key])
            if not result["valid"] or not result["success"]:
                raise AssertionError(f"Nov 14 pilot path failed: {record['base_sample_id']} {key}")

    tracked = subprocess.check_output(["git", "-C", str(repo), "ls-files"], text=True).splitlines()
    tracked_raw = [p for p in tracked if p.startswith("data/raw/") and not p.endswith("README.md") and not p.endswith(".gitkeep")]
    tracked_ppnl = [p for p in tracked if Path(p).name in FILES]
    if tracked_raw or tracked_ppnl:
        raise AssertionError(f"raw PPNL files are tracked: {tracked_raw + tracked_ppnl}")

    derived_hashes = {}
    for relative in DATA_FILES:
        path = repo / relative
        if not path.is_file():
            raise AssertionError(f"missing derived dataset artifact: {relative}")
        derived_hashes[relative] = _sha256(path)

    result = {
        "freeze_id": FREEZE_ID, "seed": 42, "ppnl_commit": PPNL_COMMIT,
        "ppnl_source_url": "https://github.com/MohamedAghzal/llms-as-path-planners.git",
        "ppnl_files": source_files,
        "source_totals": {"raw_records": total_raw, "converted_records": total_converted, "excluded_unreachable": total_unreachable,
                          "standard_6x6_raw": 25080, "standard_6x6_converted": 25016, "standard_6x6_excluded": 64},
        "splits": split_summaries,
        "split_ratios": {k: v["environment_count"] / 817 for k, v in split_summaries.items()},
        "split_environment_leakage": False,
        "excluded_manifest": {"count": len(excluded_rows), "all_have_source_file_and_index": True,
                              "all_source_file_index_pairs_unique": True},
        "paired_test": {"id_count": len(id_rows), "ood_count": len(ood_rows), "pair_count": len(pairs),
                        "exception_count": len(exceptions), "id_error_distribution": dict(sorted(id_counts.items())),
                        "ood_combination_distribution": dict(sorted(combo_counts.items())),
                        "paired_environment_sets_equal": True, "no_train_dev_environment_overlap": True,
                        "executor_labels_rechecked": len(id_rows) + len(ood_rows),
                        "ood_independent_replays": 2 * len(ood_rows), "ood_sequential_replays": len(ood_rows),
                        "ood_combined_full_execution_and_unmasked": len(ood_rows)},
        "all_processed_shortest_and_nonoptimal_paths_rechecked": sum(len(x) for x in split_records.values()),
        "nov14_heldout_pilot_shortest_and_nonoptimal_paths_rechecked": len(heldout_pilot),
        "derived_file_sha256": derived_hashes,
        "raw_ppnl_tracked_in_git": False,
        "audit_passed": True,
    }
    manifest = {
        "freeze_id": FREEZE_ID, "freeze_date": "2026-11-16", "seed": 42,
        "ppnl": {"repository_url": result["ppnl_source_url"], "commit": PPNL_COMMIT, "files": source_files,
                 "raw_records": total_raw, "converted_records": total_converted, "excluded_unreachable": total_unreachable},
        "splits": split_summaries, "split_ratios": result["split_ratios"],
        "paired_conditions": result["paired_test"],
        "excluded_samples": {"count": len(excluded_rows), "manifest": "data/manifests/excluded_manifest.json"},
        "derived_file_sha256": derived_hashes,
        "code_version": {"data_generation_commit": GENERATION_COMMIT,
                         "freeze_audit_script": "scripts/audit_frozen_data.py"},
        "reproduction_commands": [
            "UV_CACHE_DIR=/tmp/uv-cache-p6 uv run --offline --python 3.12 scripts/audit_frozen_data.py",
            "UV_CACHE_DIR=/tmp/uv-cache-p6 uv run --offline --python 3.12 scripts/demo_executor_and_leakage.py",
        ],
    }
    return result, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ppnl-root", type=Path, default=Path("../ppnl_official/ppnl-spatial-temporal-reasoning/single_goal"))
    args = parser.parse_args()
    result, manifest = audit(args.repo, args.ppnl_root)
    write_json(args.repo / "results/data_freeze_audit_20261116.json", result)
    write_json(args.repo / "data/manifests/frozen_dataset_20261116.json", manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
