import json
from pathlib import Path

from src.data.ppnl_io import load_jsonl


ROOT = Path(__file__).resolve().parents[1]
PAIRED = ROOT / "data/samples/nov15_paired"


def test_id_ood_have_exact_pair_and_environment_sets_without_train_dev_leakage():
    id_rows = load_jsonl(PAIRED / "id_single_error.jsonl")
    ood_rows = load_jsonl(PAIRED / "ood_double_error.jsonl")
    pairs = load_jsonl(PAIRED / "pair_manifest.jsonl")
    ids = {r["base_sample_id"]: r for r in id_rows}
    oods = {r["base_sample_id"]: r for r in ood_rows}
    assert len(ids) == len(id_rows) == len(oods) == len(ood_rows)
    assert set(ids) == set(oods)
    assert {r["environment_id"] for r in id_rows} == {r["environment_id"] for r in ood_rows}
    assert len(pairs) == len(ids)
    for pair in pairs:
        assert pair["base_sample_id"] in ids and pair["base_sample_id"] in oods
        assert pair["id_sample_id"] == ids[pair["base_sample_id"]]["sample_id"]
        assert pair["ood_sample_id"] == oods[pair["base_sample_id"]]["sample_id"]
        assert pair["environment_id"] == ids[pair["base_sample_id"]]["environment_id"]
    train = json.loads((ROOT / "data/manifests/train.json").read_text())
    dev = json.loads((ROOT / "data/manifests/dev.json").read_text())
    held = {r["environment_id"] for r in id_rows}
    assert held.isdisjoint(train["environment_ids"])
    assert held.isdisjoint(dev["environment_ids"])


def test_id_type_distribution_is_balanced_and_ood_has_two_types():
    id_rows = load_jsonl(PAIRED / "id_single_error.jsonl")
    ood_rows = load_jsonl(PAIRED / "ood_double_error.jsonl")
    assert {r["corruption_type"] for r in id_rows} == {
        "action_substitution", "action_deletion", "action_insertion", "premature_stopping"
    }
    assert max(sum(r["corruption_type"] == kind for r in id_rows) for kind in {
        "action_substitution", "action_deletion", "action_insertion", "premature_stopping"
    }) - min(sum(r["corruption_type"] == kind for r in id_rows) for kind in {
        "action_substitution", "action_deletion", "action_insertion", "premature_stopping"
    }) <= 1
    assert all(len(r["corruption_provenance"]) == 2 for r in ood_rows)
