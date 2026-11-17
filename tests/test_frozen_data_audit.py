import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_freeze_manifest_matches_full_audit_output_and_available_artifacts():
    manifest = json.loads((ROOT / "data/manifests/frozen_dataset_20261116.json").read_text())
    audit = json.loads((ROOT / "results/data_freeze_audit_20261116.json").read_text())
    assert manifest["freeze_id"] == audit["freeze_id"] == "project6-6x6-seed42-20261116"
    assert manifest["seed"] == audit["seed"] == 42
    assert audit["audit_passed"] is True
    assert manifest["derived_file_sha256"] == audit["derived_file_sha256"]
    for relative, expected in audit["derived_file_sha256"].items():
        path = ROOT / relative
        # A's freeze intentionally keeps complete processed splits out of Git.
        # Validate them when locally present without making a clean clone fail.
        if not path.exists():
            assert relative.startswith("data/processed/nov14/")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected


def test_frozen_source_and_split_totals_are_documented():
    manifest = json.loads((ROOT / "data/manifests/frozen_dataset_20261116.json").read_text())
    assert manifest["ppnl"]["commit"] == "5fd0376fc50ae9860bc986085165ebac5675fd44"
    assert len(manifest["ppnl"]["files"]) == 7
    assert manifest["ppnl"]["raw_records"] == 37080
    assert manifest["ppnl"]["converted_records"] == 36569
    assert manifest["ppnl"]["excluded_unreachable"] == 511
    assert manifest["splits"] == {
        "train": {"environment_count": 572, "sample_count": 17510},
        "dev": {"environment_count": 82, "sample_count": 2517},
        "heldout_test": {"environment_count": 163, "sample_count": 4989},
    }
    assert manifest["paired_conditions"]["id_count"] == 162
    assert manifest["paired_conditions"]["ood_count"] == 162
    assert manifest["paired_conditions"]["pair_count"] == 162
