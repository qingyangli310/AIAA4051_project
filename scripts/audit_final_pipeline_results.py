"""Read-only audit of frozen formal pipeline records."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.candidate_source import load_candidate_records, validate_candidate_grid
from src.data.build_nov14_dataset import shortest_path
from src.data.ppnl_io import load_jsonl
from src.executor.executor import execute_plan

CONDITIONS = {"direct", "no_feedback", "self_feedback", "executor_feedback",
              "shuffled_executor_feedback"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root=ROOT):
    frozen = json.loads((root / "data/manifests/frozen_dataset_20261116.json").read_text())
    part1 = json.loads((root / "data/part1/manifest.json").read_text())
    results, summaries, pairs = {}, {}, {}
    for split, condition in (("id", "id_single_error"), ("ood", "ood_double_error")):
        folder = root / f"results/final_pipeline_v4_{split}"
        report = json.loads((folder / "report.json").read_text())
        path = f"data/part1/repair_{split}_test.jsonl"
        require(report["scope"] == "test" and report["formal_test"] is True
                and report["symbolic_reference"] is False
                and report["status"] == "complete" and report["data"] == path
                and report["candidate_condition"] == condition
                and report["candidate_source"] == "dataset"
                and set(report["conditions"]) == CONDITIONS, f"{split}: report scope")
        require(digest(root / path) == report["data_sha256"]
                == part1["sha256"][Path(path).name], f"{split}: input hash")
        upstream = f"data/samples/nov15_paired/{condition}.jsonl"
        require(digest(root / upstream) == frozen["derived_file_sha256"][upstream],
                f"{split}: frozen input hash")
        require(all(r["split"] == "test" and r["condition"] == condition
                    for r in load_jsonl(root / path)), f"{split}: mixed input scopes")
        source = load_candidate_records(root / path, condition=condition)
        origin = {r["sample_id"]: r for r in load_candidate_records(root / upstream, condition=condition)}
        by_id = {r["sample_id"]: r for r in source}
        rows = load_jsonl(folder / "records.jsonl")
        require(len(rows) == len(by_id) == len(origin) == 162
                and {r.get("sample_id") for r in rows} == by_id.keys() == origin.keys(),
                f"{split}: missing or duplicate records / run exceptions")
        failures, changed = Counter(), 0
        pairs[split] = {}
        for row in rows:
            sid = row["sample_id"]
            expected = by_id[sid]
            require(row["status"] == "ok" and row["test_condition"] == condition
                    and row["candidate_source"] == "dataset" and row["generation"] is None,
                    f"{sid}: execution or scope error")
            validate_candidate_grid(expected, row["grid"])
            validate_candidate_grid(origin[sid], row["grid"])
            plan = row["candidate_plan"]
            require(plan == expected["candidate_plan"] == origin[sid]["candidate_plan"]
                    and all(row[k] == expected[k] == origin[sid][k]
                            for k in ("base_sample_id", "environment_id")),
                    f"{sid}: candidate or identity mismatch")
            sha = hashlib.sha256(json.dumps(plan, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            require(sha == row["candidate_plan_sha256"], f"{sid}: candidate hash")
            require(set(row["branches"]) == CONDITIONS, f"{sid}: incomplete conditions")
            require(row["candidate_executor"] == execute_plan(row["grid"], plan)
                    and row["branches"]["direct"]["repaired_plan"] == plan,
                    f"{sid}: direct candidate mismatch")
            require(row["optimal_length"] == len(shortest_path(row["grid"])), f"{sid}: optimal length")
            pairs[split][row["base_sample_id"]] = row["grid"]
            for name, branch in row["branches"].items():
                require(isinstance(branch["final_result"]["success"], bool)
                        and branch["final_result"] == execute_plan(row["grid"], branch["repaired_plan"]),
                        f"{sid}: {name} final execution")
                if "candidate_plan" in branch:
                    require(branch["candidate_plan"] == plan, f"{sid}: {name} candidate mismatch")
                for key in ("verification", "repair"):
                    if (branch.get(key) or {}).get("parse_success") is False:
                        failures[f"{name}.{key}"] += 1
            actual = row["branches"]["executor_feedback"]["verification"]
            shuffled = row["branches"]["shuffled_executor_feedback"]
            differs = any(actual[k] != shuffled["verification"][k]
                          for k in ("first_error_step", "reason"))
            require(differs == shuffled["feedback_content_changed"], f"{sid}: shuffle flag")
            changed += differs
        require(len(pairs[split]) == 162, f"{split}: duplicate base_sample_id")
        results[split] = rows
        summaries[split] = {"rows": len(rows), "run_errors": 0,
                            "parse_failures": dict(failures), "shuffle_changed": changed}
    require(pairs["id"] == pairs["ood"], "ID/OOD source grids or pairing differ")
    pair_path = "data/samples/nov15_paired/pair_manifest.jsonl"
    require(digest(root / pair_path) == frozen["derived_file_sha256"][pair_path],
            "pair manifest hash")
    manifest_rows = load_jsonl(root / pair_path)
    by_base = {r["base_sample_id"]: r for r in manifest_rows}
    require(len(manifest_rows) == len(by_base) == 162
            and by_base.keys() == pairs["id"].keys(), "frozen pair manifest coverage")
    for split, rows in results.items():
        for row in rows:
            pair = by_base[row["base_sample_id"]]
            require(pair[f"{split}_sample_id"] == row["sample_id"]
                    and pair["environment_id"] == row["environment_id"],
                    "record differs from frozen pair manifest")
    return results, summaries


if __name__ == "__main__":
    _, summary = audit()
    print(json.dumps(summary, indent=2))
