"""Offline roundtrip evidence using A's supplied labels and mock model outputs."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from src.models.api import GridModel
from src.models.day13_contract import LABEL_FIELDS, convert_sample
from src.models.prompts import PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ROOT / "data/samples/member_A_executor_integration_samples_20261113.jsonl"


class FixedTextBackend:
    def __init__(self, text):
        self.text = text

    def generate_text(self, prompt):
        return {"raw_output": self.text, "metadata": {"backend": "mock_label_roundtrip"}}


def check_sample(row):
    generation, verification = convert_sample(row)
    g = GridModel(backend=FixedTextBackend(generation["target_text"])).generate_plan(generation["grid"])
    v = GridModel(backend=FixedTextBackend(verification["target_text"])).verify_plan(
        verification["grid"], verification["candidate_plan"]
    )
    checks = {
        "generation_prompt_matches": g["metadata"]["prompt"] == generation["input_text"],
        "verification_prompt_matches": v["metadata"]["prompt"] == verification["input_text"],
        "generation_roundtrip": g["parse_success"] and g["plan"] == generation["expected_plan"],
        "verification_roundtrip": v["parse_success"] and all(
            v[key] == verification["expected_verification"][key] for key in LABEL_FIELDS
        ),
    }
    return generation, verification, {
        "sample_id": row["sample_id"], "environment_id": row["environment_id"],
        "passed": all(checks.values()), "checks": checks,
        "generation": g, "verification": v, "executor_label": row["executor_label"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/day13")
    args = parser.parse_args()
    content = args.samples.read_bytes()
    report = {
        "protocol_version": PROTOCOL_VERSION, "mode": "mock_label_roundtrip",
        "executor_rerun": False, "model_loaded": False,
        "source_filename": args.samples.name, "source_sha256": hashlib.sha256(content).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "cases": [],
    }
    generations, verifications, seen, coverage = [], [], set(), Counter()
    for line_number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            g, v, case = check_sample(row)
            if row["sample_id"] in seen:
                raise ValueError(f"duplicate sample_id: {row['sample_id']}")
            seen.add(row["sample_id"])
            generations.append(g)
            verifications.append(v)
            coverage[row["executor_label"]["reason"] or "success"] += 1
        except Exception as exc:
            case = {"passed": False, "error_type": type(exc).__name__, "error": str(exc)}
        report["cases"].append({"line_number": line_number, **case})
    required = {"success", "obstacle_collision", "out_of_bounds", "invalid_action_token", "goal_not_reached"}
    report["coverage"] = dict(coverage)
    report["missing_coverage"] = sorted(required - set(coverage))
    report["total"] = len(report["cases"])
    report["passed"] = sum(case["passed"] for case in report["cases"])
    report["all_passed"] = (
        report["total"] > 0 and report["passed"] == report["total"] and not report["missing_coverage"]
    )
    output = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    for name, records in (("generation_examples.jsonl", generations), ("verification_examples.jsonl", verifications)):
        with (output / name).open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / "roundtrip_report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Roundtrip: {report['passed']}/{report['total']}; all_passed={report['all_passed']}")
    print(f"Evidence: {output}")
    print("Mock output and supplied A labels only; no model inference or executor rerun.")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
