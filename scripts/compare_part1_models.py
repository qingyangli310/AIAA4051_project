"""Select the Part 1 primary model from matched Generation Dev reports only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("task") != "generation" or not report.get("formal_part1_run"):
        raise ValueError(f"not a formal Generation report: {path}")
    return report


def dev_key(report: dict) -> tuple[float, float, float, str]:
    scoring = report["dev_scoring"]
    metrics = scoring["metrics"]
    return (
        metrics["success_rate"] or 0.0,
        metrics["feasibility"] or 0.0,
        metrics["mean_optimality"] or 0.0,
        report["model_key"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [read_report(path) for path in args.reports]
    matched = (
        "train_sha256",
        "dev_sha256",
        "epochs",
        "learning_rate",
        "batch_size",
        "gradient_accumulation",
    )
    for field in matched:
        if reports[0][field] != reports[1][field]:
            raise ValueError(f"unmatched model budgets: {field}")
    both_collapsed = all(
        report["dev_scoring"]["metrics"]["success_rate"] == 0
        and report["dev_scoring"]["most_common_share"] >= 0.9
        for report in reports
    )
    if both_collapsed:
        selection = None
        status = "blocked_by_generation_collapse"
    else:
        selection = max(reports, key=dev_key)["model_key"]
        status = "selected_on_dev_only"
    result = {
        "status": status,
        "primary_model": selection,
        "selection_metric_order": ["success_rate", "feasibility", "mean_optimality"],
        "test_results_used_for_selection": False,
        "models": {
            report["model_key"]: report["dev_scoring"] for report in reports
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
