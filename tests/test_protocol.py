import pytest

from src.protocol import validate_protocol


def config():
    return {
        "protocol_id": "test",
        "status": "locked",
        "primary_model": {"family": "facebook/bart-base"},
        "training_examples": {
            "generation": 2000,
            "verification": 500,
            "repair": 1500,
        },
        "dev_examples": {
            "generation": 100,
            "verification": 50,
            "repair": 50,
        },
        "test_examples": {"id": 162, "ood": 162},
        "epochs": {"generation": 3, "verification": 3, "repair": 3},
        "task_models": {
            task: {
                "base_model_family": "facebook/bart-base",
                "checkpoint": f"checkpoints/{task}",
                "checkpoint_sha256": "a" * 64,
                "revision": "example-revision",
            }
            for task in ("generation", "verification", "repair")
        },
    }


def test_2000_500_1500_budget_is_valid() -> None:
    result = validate_protocol(config(), require_ready=True)
    assert result["training_total"] == 4000
    assert result["dev_total"] == 200
    assert result["ready_for_formal_test"] is True


def test_budget_over_4000_is_rejected() -> None:
    invalid = config()
    invalid["training_examples"]["verification"] = 501
    with pytest.raises(ValueError, match="exceeds 4000"):
        validate_protocol(invalid)


def test_mixed_primary_model_families_are_rejected() -> None:
    invalid = config()
    invalid["task_models"]["repair"]["base_model_family"] = "google-t5/t5-small"
    with pytest.raises(ValueError, match="primary model family"):
        validate_protocol(invalid, require_ready=True)
