"""Read-only audit of the shared frozen Train and Dev JSONL files."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.environment import base_sample_id, environment_id
from src.data.ppnl_io import load_json, load_jsonl, write_json

FREEZE_ID = "project6-6x6-seed42-20261116"
PPNL_COMMIT = "5fd0376fc50ae9860bc986085165ebac5675fd44"
SOURCE_INFO = {
    "1_train_set_6x6_samples.json": ("data/manifests/train.json", 16032),
    "1dev_set_6x6_samples.json": ("data/manifests/dev.json", 2004),
    "1_goals_test_seen_6x6_samples.json": ("data/manifests/heldout_test.json", 2004),
    "1goals_unseen_6x6_samples.json": ("data/manifests/heldout_test.json", 5040),
}
REQUIRED = {
    "environment_id", "rows", "cols", "start", "goal", "obstacles", "plan",
    "source_file", "source_index", "base_sample_id", "official_plan", "shortest_plan",
    "shortest_length", "official_length", "official_optimal", "shortest_executor_label",
    "nonoptimal_plan", "nonoptimal_executor_label",
}
EXPECTED = {
    "train": {"rows": 17510, "environments": 572,
              "sha256": "765e66a975faffedc28893f23eb50dd5b7d1c68e4ee5f27ec08dcb3d842cdd4c"},
    "dev": {"rows": 2517, "environments": 82,
            "sha256": "7c3f490b65e21b005fdde4b2d70c1d74dd1db655b065d505c5ce1be28db2f20f"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(repo: Path, check_remote: bool = True) -> dict[str, Any]:
    repo = repo.resolve()
    freeze = load_json(repo / "data/manifests/frozen_dataset_20261116.json")
    if freeze["freeze_id"] != FREEZE_ID or freeze["seed"] != 42 or freeze["ppnl"]["commit"] != PPNL_COMMIT:
        raise AssertionError("frozen Manifest identity, seed, or PPNL commit mismatch")
    split_audits = {}
    environments: dict[str, set[str]] = {}
    for split, expected in EXPECTED.items():
        rel = f"data/processed/nov14/{split}.jsonl"
        path = repo / rel
        if not path.is_file():
            raise AssertionError(f"missing {rel}")
        digest = sha256(path)
        if digest != expected["sha256"] or digest != freeze["derived_file_sha256"][rel]:
            raise AssertionError(f"{split} SHA-256 differs from frozen record")
        records = load_jsonl(path)
        if len(records) != expected["rows"]:
            raise AssertionError(f"{split} row count mismatch: {len(records)}")
        split_manifest = load_json(repo / f"data/manifests/{split}.json")
        if split_manifest["seed"] != 42 or split_manifest["sample_count"] != len(records):
            raise AssertionError(f"{split} split manifest mismatch")
        expected_envs = set(split_manifest["environment_ids"])
        seen_envs: set[str] = set()
        seen_bases: set[str] = set()
        seen_sources: set[tuple[str, int]] = set()
        for index, record in enumerate(records):
            missing = REQUIRED - record.keys()
            if missing:
                raise AssertionError(f"{split}[{index}] missing fields: {sorted(missing)}")
            rows, cols = record["rows"], record["cols"]
            computed_env = environment_id(rows, cols, record["obstacles"])
            if record["environment_id"] != computed_env:
                raise AssertionError(f"{split}[{index}] environment_id mismatch")
            if record["environment_id"] not in expected_envs:
                raise AssertionError(f"{split}[{index}] environment absent from split manifest")
            if record["source_file"] not in SOURCE_INFO or not isinstance(record["source_index"], int):
                raise AssertionError(f"{split}[{index}] invalid source coordinates")
            source_manifest, source_limit = SOURCE_INFO[record["source_file"]]
            if record["source_index"] < 0 or record["source_index"] >= source_limit:
                raise AssertionError(f"{split}[{index}] source index outside original file")
            if split_manifest["split"] not in ("train", "dev", "heldout_test"):
                raise AssertionError(f"bad split label: {split_manifest['split']}")
            expected_base = base_sample_id(PPNL_COMMIT, record["source_file"], record["source_index"])
            if record["base_sample_id"] != expected_base:
                raise AssertionError(f"{split}[{index}] base_sample_id mismatch")
            source_key = (record["source_file"], record["source_index"])
            if source_key in seen_sources:
                raise AssertionError(f"duplicate source record in {split}: {source_key}")
            seen_sources.add(source_key)
            if record["base_sample_id"] in seen_bases:
                raise AssertionError(f"duplicate base_sample_id in {split}")
            seen_bases.add(record["base_sample_id"])
            if record["environment_id"] in seen_envs:
                # Multiple rows can legitimately share an environment in full
                # PPNL, so only IDs and manifest membership are uniqueness checks.
                pass
            seen_envs.add(record["environment_id"])
            if record["shortest_length"] != len(record["shortest_plan"]):
                raise AssertionError(f"{split}[{index}] shortest_length mismatch")
            if record["official_length"] != len(record["official_plan"]):
                raise AssertionError(f"{split}[{index}] official_length mismatch")
        actual_envs = {r["environment_id"] for r in records}
        if actual_envs != expected_envs or len(actual_envs) != expected["environments"]:
            raise AssertionError(f"{split} environment set/count differs from frozen split manifest")
        split_audits[split] = {"path": rel, "rows": len(records), "environment_count": len(actual_envs),
                               "base_sample_ids_unique": len(seen_bases) == len(records),
                               "source_indices_unique_within_split": len(seen_sources) == len(records),
                               "required_fields_complete": True, "environment_ids_recomputed": True,
                               "base_sample_ids_recomputed": True, "sha256": digest}
        environments[split] = actual_envs
    intersection = environments["train"] & environments["dev"]
    if intersection:
        raise AssertionError(f"Train/Dev environment leakage ({len(intersection)} IDs)")
    heldout = set(load_json(repo / "data/manifests/heldout_test.json")["environment_ids"])
    if environments["train"] & heldout or environments["dev"] & heldout:
        raise AssertionError("Train/Dev overlaps Held-out Test")
    heldout_path = repo / "data/processed/nov14/heldout_test.jsonl"
    heldout_sha = sha256(heldout_path) if heldout_path.is_file() else None
    if heldout_sha != freeze["derived_file_sha256"]["data/processed/nov14/heldout_test.jsonl"]:
        raise AssertionError("Held-out local artifact missing or differs from freeze hash")

    remote = {"a_branch_has_train_dev": None, "c_branch_has_train_dev": None,
              "b_branch_status": "not found/unknown", "heldout_present_on_github": None}
    if check_remote:
        refs = subprocess.check_output(["git", "-C", str(repo), "for-each-ref", "--format=%(refname) %(objectname)", "refs/remotes/origin"], text=True)
        sha_by_ref = {line.split()[0].replace("refs/remotes/origin/", "refs/heads/"): line.split()[1]
                      for line in refs.splitlines() if line.strip() and not line.split()[0].endswith("/HEAD")}
        a_sha = sha_by_ref.get("refs/heads/feature/a-executor")
        c_refs = [ref for ref in sha_by_ref if ref in ("refs/heads/feature/c-repair-pipeline", "refs/heads/feature/c-repair-evaluation")]
        c_sha = sha_by_ref.get("refs/heads/feature/c-repair-evaluation") or sha_by_ref.get("refs/heads/feature/c-repair-pipeline")
        b_refs = [ref for ref in sha_by_ref if "feature/b-" in ref or "feature/b_" in ref]
        remote["a_branch_sha"] = a_sha
        remote["c_branch_sha"] = c_sha
        remote["c_branches_found"] = c_refs
        remote["b_branches_found"] = b_refs
        remote["b_branch_status"] = "not found among remote heads" if not b_refs else "found; inspect named refs"
        if a_sha:
            tracked = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", a_sha,
                                                "--", "data/processed/nov14"], text=True).splitlines()
            remote["a_branch_has_train_dev"] = all(f"data/processed/nov14/{s}.jsonl" in tracked for s in ("train", "dev"))
            remote["heldout_present_on_github"] = "data/processed/nov14/heldout_test.jsonl" in tracked
        # c-branch tree is inspected without changing or checking out that branch.
        if c_sha:
            c_files = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", c_sha,
                                                 "--", "data/processed/nov14"], text=True).splitlines()
            remote["c_branch_has_train_dev"] = all(f"data/processed/nov14/{s}.jsonl" in c_files for s in ("train", "dev"))
            remote["c_branch_has_heldout"] = "data/processed/nov14/heldout_test.jsonl" in c_files
        for ref in b_refs:
            b_sha = sha_by_ref[ref]
            b_files = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", b_sha,
                                                "--", "data/processed/nov14"], text=True).splitlines()
            remote.setdefault("b_branch_files", {})[ref] = {
                "has_train_dev": all(f"data/processed/nov14/{s}.jsonl" in b_files for s in ("train", "dev")),
                "has_heldout": "data/processed/nov14/heldout_test.jsonl" in b_files,
            }

    return {"freeze_id": FREEZE_ID, "seed": 42, "split_audits": split_audits,
            "train_dev_environment_intersection_count": 0,
            "heldout_local_sha256_checked": True,
            "remote_status": remote,
            "training_budget": {"unlimited_file_rows": 17510,
                                "do_not_sample_or_change_budget_here": True,
                                "team_task_budget_limit": 4000}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/shared_training_data_audit_20261117.json"))
    parser.add_argument("--skip-remote", action="store_true", help="audit local files only")
    args = parser.parse_args()
    result = audit(args.repo, check_remote=not args.skip_remote)
    write_json(args.repo / args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
