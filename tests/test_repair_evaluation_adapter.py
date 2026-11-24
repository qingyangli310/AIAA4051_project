"""Exercise the evaluation CLI with archived outputs and legacy row formatting."""

from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from scripts.evaluate_part1_checkpoint import main, predict_trace_repair
from src.models.prompts import build_repair_prompt


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines()]


def test_trace_repair_does_not_read_reference_answers() -> None:
    row = read_rows("data/repair_high_quality_v2/repair_dev.jsonl")[0]
    expected_prompt = row["input_text"]
    row["input_text"] = "obsolete input"
    row.pop("target_plan")
    row.pop("target_text")

    def infer(prompt: str, limit: int) -> str:
        assert prompt == expected_prompt
        return "Trace: (2,4) right (2,5)"

    result = predict_trace_repair(row, infer, 128)
    assert result["parse_success"] is True
    assert result["repaired_plan"] == ["right"]


def test_evaluation_cli_handles_legacy_repair_rows() -> None:
    rows = read_rows("data/repair_high_quality_v2/repair_dev.jsonl")
    archive = read_rows("results/repair_flan_t5_base_high_quality_v2/dev_predictions.jsonl")
    outputs = {row["example_id"]: row["raw_output"] for row in archive}
    by_prompt = {row["input_text"]: outputs[row["example_id"]] for row in rows}

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, prompt, **kwargs):
            assert prompt in by_prompt
            return Batch(prompt=prompt)

        def decode(self, tokens, **kwargs):
            return tokens

    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, prompt, **kwargs):
            return [by_prompt[prompt]]

    legacy_rows = []
    for row in rows:
        legacy = dict(row)
        legacy.pop("output_format")
        legacy["input_text"] = build_repair_prompt(
            row["grid"], row["candidate_plan"], row["feedback_source"], row["verifier_feedback"]
        )
        legacy["target_text"] = ", ".join(row["target_plan"])
        legacy_rows.append(legacy)

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "legacy_dev.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in legacy_rows), encoding="utf-8")
        output = Path(directory) / "evaluation"
        argv = [
            "evaluate_part1_checkpoint.py", "--task", "repair",
            "--checkpoint", "unused-test-checkpoint", "--data", str(source),
            "--output", str(output), "--repair-output-format", "trace_only", "--allow-cpu",
        ]
        modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), inference_mode=nullcontext),
            "transformers": SimpleNamespace(AutoTokenizer=Tokenizer, AutoModelForSeq2SeqLM=Model),
        }
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            assert main() == 0
        report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
        predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        assert report["examples"] == 50
        assert report["scoring"]["parse_rate"] == 47 / 50
        assert report["scoring"]["metrics"]["counts"]["success"] == [38, 50]
        assert all(row["raw_output"] == outputs[row["example_id"]] for row in predictions)
        assert all("prompt" in row and "repaired_plan" in row for row in predictions)
        assert source.read_text(encoding="utf-8") == "".join(json.dumps(row) + "\n" for row in legacy_rows)
