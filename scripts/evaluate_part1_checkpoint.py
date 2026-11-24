"""Evaluate a saved Part 1 checkpoint on one frozen JSONL task file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.part1.data import read_jsonl
from src.part1.evaluate import score_predictions
from src.model_adapters import Seq2SeqRepairerAdapter
from scripts.build_generation_like_repair_data import trace_target


def predict_trace_repair(row: dict, infer_text, max_output_length: int) -> dict:
    repairer = Seq2SeqRepairerAdapter(
        infer_text, output_format="trace_only", max_new_tokens=max_output_length
    )
    return repairer.repair(
        environment_id=row["environment_id"],
        grid=row["grid"],
        candidate_plan=row["candidate_plan"],
        feedback_source=row["feedback_source"],
        verifier_feedback=row["verifier_feedback"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("generation", "verification", "repair"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-output-length", type=int, default=128)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--repair-output-format", choices=("plan", "trace_only"), default="plan")
    args = parser.parse_args()
    if args.task != "repair" and args.repair_output_format != "plan":
        parser.error("--repair-output-format trace_only requires --task repair")
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install requirements-project6.txt in the frozen environment") from exc
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("formal evaluation requires CUDA; use --allow-cpu only for diagnostics")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = read_jsonl(args.data)
    print("Loading checkpoint...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(args.checkpoint), local_files_only=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    def infer_text(prompt: str, max_new_tokens: int) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
        with torch.inference_mode():
            token_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=0,
                early_stopping=False,
            )[0]
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    raw_outputs, predictions, scoring_rows = [], [], []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        scoring_row = row
        details = {}
        if args.task == "repair" and args.repair_output_format == "trace_only":
            details = predict_trace_repair(row, infer_text, args.max_output_length)
            raw = details["raw_output"]
            # Reference paths are used only for scoring after inference.
            scoring_row = {
                **row,
                "input_text": details["prompt"],
                "output_format": "trace_only",
                "target_text": trace_target(row["grid"], row["target_plan"]),
            }
        else:
            raw = infer_text(row["input_text"], args.max_output_length)
        raw_outputs.append(raw)
        scoring_rows.append(scoring_row)
        predictions.append(
            {
                "example_id": row["example_id"],
                "environment_id": row["environment_id"],
                "target_text": scoring_row["target_text"],
                "raw_output": raw,
                **details,
            }
        )
        print(f"[{index}/{len(rows)}] evaluated", flush=True)
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True)
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    report = {
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "device": device,
        "examples": len(rows),
        "inference_seconds": elapsed,
        "repair_output_format": args.repair_output_format if args.task == "repair" else None,
        "scoring": score_predictions(args.task, scoring_rows, raw_outputs),
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "no_repeat_ngram_size": 0,
            "early_stopping": False,
            "max_new_tokens": args.max_output_length,
        },
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
