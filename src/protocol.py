"""Validation for the final Project 6 experiment protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any


TRAIN_TASKS = ("generation", "verification", "repair")


def load_protocol(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency environment issue
        raise RuntimeError("PyYAML is required to load the protocol config") from exc
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("protocol config must contain a mapping")
    return data


def _counts(section: Any, name: str) -> dict[str, int]:
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping")
    result: dict[str, int] = {}
    for task in TRAIN_TASKS:
        value = section.get(task)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name}.{task} must be a non-negative integer")
        result[task] = value
    return result


def validate_protocol(
    config: dict[str, Any], *, require_ready: bool = False
) -> dict[str, Any]:
    """Validate data budgets and primary-model consistency."""
    train = _counts(config.get("training_examples"), "training_examples")
    dev = _counts(config.get("dev_examples"), "dev_examples")
    epochs = _counts(config.get("epochs"), "epochs")
    train_total = sum(train.values())
    dev_total = sum(dev.values())
    if train_total > 4000:
        raise ValueError(f"training example budget exceeds 4000: {train_total}")
    if dev_total > 200:
        raise ValueError(f"development example budget exceeds 200: {dev_total}")
    if any(value > 3 for value in epochs.values()):
        raise ValueError("each applicable training subset is limited to three epochs")

    test_examples = config.get("test_examples")
    if not isinstance(test_examples, dict):
        raise ValueError("test_examples must be a mapping")
    for condition in ("id", "ood"):
        value = test_examples.get(condition)
        if type(value) is not int or not 0 <= value <= 200:
            raise ValueError(f"test_examples.{condition} must be between 0 and 200")

    primary = config.get("primary_model")
    if not isinstance(primary, dict):
        raise ValueError("primary_model must be a mapping")
    family = primary.get("family")
    status = config.get("status")
    task_models = config.get("task_models")
    if not isinstance(task_models, dict):
        raise ValueError("task_models must be a mapping")
    for task in TRAIN_TASKS:
        if not isinstance(task_models.get(task), dict):
            raise ValueError(f"task_models.{task} must be a mapping")
    configured_families = {
        task: task_models.get(task, {}).get("base_model_family")
        for task in TRAIN_TASKS
    }
    nonempty_families = {value for value in configured_families.values() if value}
    if family and any(value != family for value in nonempty_families):
        raise ValueError("all formal task checkpoints must use the primary model family")
    if require_ready:
        if status != "locked":
            raise ValueError("formal test requires protocol status=locked")
        if not isinstance(family, str) or not family.strip():
            raise ValueError("formal test requires a confirmed primary model family")
        if set(configured_families.values()) != {family}:
            raise ValueError(
                "formal test requires Generation, Verification, and Repair task models "
                "to declare the primary model family"
            )
        for task in TRAIN_TASKS:
            task_model = task_models[task]
            checkpoint = task_model.get("checkpoint")
            checkpoint_sha256 = task_model.get("checkpoint_sha256")
            revision = task_model.get("revision")
            if not isinstance(checkpoint, str) or not checkpoint.strip():
                raise ValueError(f"formal test requires task_models.{task}.checkpoint")
            if (
                not isinstance(checkpoint_sha256, str)
                or len(checkpoint_sha256) != 64
                or any(char not in "0123456789abcdef" for char in checkpoint_sha256.lower())
            ):
                raise ValueError(
                    f"formal test requires a SHA-256 for task_models.{task}"
                )
            if not isinstance(revision, str) or not revision.strip():
                raise ValueError(f"formal test requires task_models.{task}.revision")

    return {
        "protocol_id": config.get("protocol_id"),
        "status": status,
        "primary_model_family": family,
        "training_examples": train,
        "training_total": train_total,
        "dev_examples": dev,
        "dev_total": dev_total,
        "test_examples": {
            "id": test_examples["id"],
            "ood": test_examples["ood"],
        },
        "epochs": epochs,
        "task_model_families": configured_families,
        "ready_for_formal_test": status == "locked" and bool(family),
    }
