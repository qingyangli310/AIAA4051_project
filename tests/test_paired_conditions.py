import json
import shutil
from pathlib import Path

from src.data.build_nov15_paired_conditions import build
from src.data.ppnl_io import load_jsonl
from src.executor.executor import execute_plan


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data/samples/nov15_paired"


def test_id_and_ood_labels_are_executor_results_and_failed():
    for filename in ("id_single_error.jsonl", "ood_double_error.jsonl"):
        for row in load_jsonl(PILOT / filename):
            result = execute_plan(row["grid"], row["corrupted_plan"])
            assert row["executor_label"] == result
            assert result["success"] is False
            assert result["first_error_step"] >= 0


def test_generation_is_byte_reproducible(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for name in ("train.json", "dev.json"):
        shutil.copy(ROOT / "data/manifests" / name, manifests / name)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    build(ROOT / "data/samples/nov14_pilot/heldout_test.jsonl", out_a, manifests, 42)
    build(ROOT / "data/samples/nov14_pilot/heldout_test.jsonl", out_b, manifests, 42)
    names = ("id_single_error.jsonl", "ood_double_error.jsonl", "pair_manifest.jsonl", "manifest.json", "exceptions.jsonl")
    assert {n: (out_a / n).read_bytes() for n in names} == {n: (out_b / n).read_bytes() for n in names}


def test_manifest_counts_match_files_and_limits():
    manifest = json.loads((PILOT / "manifest.json").read_text())
    assert manifest["id_count"] == len(load_jsonl(PILOT / "id_single_error.jsonl"))
    assert manifest["ood_count"] == len(load_jsonl(PILOT / "ood_double_error.jsonl"))
    assert manifest["eligible_pair_count"] == len(load_jsonl(PILOT / "pair_manifest.jsonl"))
    assert manifest["id_count"] <= 200
    assert manifest["ood_count"] <= 200
