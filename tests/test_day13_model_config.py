from copy import deepcopy

import pytest

from scripts.prepare_day13_model_config import validate_decision


def decision():
    # Synthetic test commit, never use as a real checkpoint revision.
    return {
        "status": "confirmed", "confirmed_by": "test_owner",
        "confirmed_at": "2026-11-14", "confirmation_evidence": "test fixture",
        "checkpoint": "google-t5/t5-small", "tokenizer": "google-t5/t5-small",
        "revision": "a" * 40, "tokenizer_revision": "a" * 40,
        "proposed_settings": {"device": "cpu", "seed": 42, "max_input_length": 512,
                              "max_new_tokens": 64, "do_sample": False, "num_beams": 1},
    }


def test_pinned_config_is_compatible_with_existing_backend():
    from src.models.backend import ModelConfig
    config = ModelConfig(**validate_decision(decision()))
    assert config.revision == "a" * 40


@pytest.mark.parametrize("field,value", [
    ("status", "pending_owner_confirmation"), ("confirmed_by", None),
    ("revision", "main"), ("tokenizer_revision", "b" * 40),
    ("checkpoint", "facebook/bart-base"),
])
def test_unconfirmed_or_ambiguous_model_is_rejected(field, value):
    record = deepcopy(decision())
    record[field] = value
    with pytest.raises(ValueError):
        validate_decision(record)
