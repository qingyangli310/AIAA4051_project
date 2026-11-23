from scripts.train_part1 import MODELS


def test_flan_checkpoints_are_explicit_and_distinct_from_plain_t5() -> None:
    assert MODELS["flan-t5-small"] == (
        "google/flan-t5-small",
        "ec9123e078f63797d82dbe0812dc176c7008fc29",
    )
    assert MODELS["flan-t5-base"] == (
        "google/flan-t5-base",
        "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    )
    assert MODELS["flan-t5-small"] != MODELS["t5-small"]
