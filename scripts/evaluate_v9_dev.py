"""Run the fixed v9 Dev100 through the Generator adapter and canonical executor."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generation_trace_v9.state_format import diagnostic_execute
from src.model_adapters import Seq2SeqGeneratorAdapter, TransformersTextGenerator
from src.smoke_integration import RealExecutorAdapter


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_path = ROOT / "artifacts/generation_trace_v9/data/transformed_dev.jsonl"
    archive_path = ROOT / "results/generation_trace_v9/dev_predictions.jsonl"
    rows = read_jsonl(data_path)
    archive = {row["base_sample_id"]: row for row in read_jsonl(archive_path)}
    if len(rows) != 100 or len(archive) != 100:
        raise ValueError("Expected the fixed 100-row v9 Development set and archive")
    if {row["base_sample_id"] for row in rows} != set(archive):
        raise ValueError("Development sample IDs do not match the archive")
    args.output.mkdir(parents=True, exist_ok=False)

    print("Loading local v9 checkpoint on CPU...", flush=True)
    runner = TransformersTextGenerator(args.checkpoint, device="cpu")
    generator = Seq2SeqGeneratorAdapter(runner, output_format="trace_only", max_new_tokens=128)
    executor = RealExecutorAdapter()
    records = []
    started = time.perf_counter()
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, start=1):
            record = {"base_sample_id": row["base_sample_id"], "grid": row["grid"]}
            try:
                result = generator.generate(row["grid"])
                if result["prompt"] != row["input_text"]:
                    raise ValueError("Runtime prompt differs from the archived v9 input")
                execution = executor.execute(row["grid"], result["plan"]) if result["parse_success"] else None
                diagnostic = diagnostic_execute(row["grid"], result["plan"]) if result["parse_success"] else "parse_failure"
                old = archive[row["base_sample_id"]]
                record.update(
                    status="ok",
                    generation=result,
                    executor=execution,
                    diagnostic_reason=diagnostic,
                    success=bool(execution and execution["success"]),
                    action_exact_match=result["parse_success"] and result["plan"] == row["reference_actions"],
                    archived_plan_match=result["plan"] == old["plan"],
                    archived_raw_output_match=result["raw_output"] == old["raw_output"],
                    archived_diagnostic_reason=old["diagnostic_reason"],
                )
            except Exception as exc:
                record.update(status="error", success=False, error=f"{type(exc).__name__}: {exc}")
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            success_count = sum(item["success"] for item in records)
            print(f"[{index}/100] success={success_count}/{index} status={record['status']}", flush=True)

    report = {
        "scope": "fixed_v9_dev100_cpu",
        "checkpoint": str(args.checkpoint.resolve()),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "examples": len(records),
        "success_count": sum(item["success"] for item in records),
        "success_rate": sum(item["success"] for item in records) / len(records),
        "parse_success_count": sum(item.get("generation", {}).get("parse_success", False) for item in records),
        "exact_match_count": sum(item.get("action_exact_match", False) for item in records),
        "error_count": sum(item["status"] == "error" for item in records),
        "diagnostic_success_count": sum(item.get("diagnostic_reason") == "success" for item in records),
        "archived_plan_match_count": sum(item.get("archived_plan_match", False) for item in records),
        "archived_raw_output_match_count": sum(item.get("archived_raw_output_match", False) for item in records),
        "archived_diagnostic_success_count": sum(item["diagnostic_reason"] == "success" for item in archive.values()),
        "failure_reasons": dict(Counter(
            "runtime_error" if item["status"] == "error" else
            "parse_failure" if not item["generation"]["parse_success"] else
            item["executor"]["reason"]
            for item in records if not item["success"]
        )),
        "inference_and_scoring_seconds": time.perf_counter() - started,
        "device": "cpu",
        "decoding": {"max_new_tokens": 128, "num_beams": 1, "do_sample": False},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
