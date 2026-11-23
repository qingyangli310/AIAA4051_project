from src.model_adapters import (
    Seq2SeqGeneratorAdapter,
    Seq2SeqRepairerAdapter,
    Seq2SeqVerifierAdapter,
)


GRID = {
    "environment_id": "grid_adapter",
    "rows": 3,
    "cols": 3,
    "start": [0, 0],
    "goal": [2, 2],
    "obstacles": [[0, 2]],
}


def test_generator_adapter_parses_trace_output() -> None:
    infer = lambda prompt, limit: "Trace: (0,0) down (1,0) right (1,1)"  # noqa: E731
    result = Seq2SeqGeneratorAdapter(infer, output_format="trace_only").generate(GRID)
    assert result["parse_success"] is True
    assert result["plan"] == ["down", "right"]
    assert result["prompt"].startswith("Task: generate")


def test_verifier_adapter_preserves_parse_failure() -> None:
    limits = []

    def infer(prompt, limit):
        limits.append(limit)
        return "probably invalid"

    result = Seq2SeqVerifierAdapter(infer).verify(GRID, ["right"])
    assert result["parse_success"] is False
    assert result["valid"] is None
    assert result["raw_output"] == "probably invalid"
    assert limits == [128]


def test_repair_adapter_hides_feedback_in_no_feedback_prompt() -> None:
    prompts = []

    def infer(prompt, limit):
        prompts.append(prompt)
        return "down, down, right, right"

    result = Seq2SeqRepairerAdapter(infer, output_format="plan").repair(
        environment_id=GRID["environment_id"],
        grid=GRID,
        candidate_plan=["right", "right"],
        feedback_source="no_feedback",
        verifier_feedback=None,
    )
    assert result["parse_success"] is True
    assert result["repaired_plan"] == ["down", "down", "right", "right"]
    assert "Verifier feedback: none" in prompts[0]

