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
    def infer(prompt: str, limit: int) -> str:
        assert "Output exactly one Trace:" in prompt
        assert "Output only comma-separated actions" not in prompt
        return "Trace: (0,0) down (1,0) right (1,1)"

    result = Seq2SeqGeneratorAdapter(infer, output_format="trace_only").generate(GRID)
    assert result["parse_success"] is True
    assert result["plan"] == ["down", "right"]


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


def test_trace_repair_prompt_and_feedback() -> None:
    for source in ("no_feedback", "executor_feedback", "self_feedback"):
        feedback = None if source == "no_feedback" else {
            "valid": False, "first_error_step": 1, "reason": "obstacle_collision"
        }

        def infer(prompt: str, limit: int) -> str:
            assert "Output exactly one Trace:" in prompt
            assert "comma-separated repaired path" not in prompt
            assert f"Feedback source: {source}" in prompt
            if source == "no_feedback":
                assert "Verifier feedback: none" in prompt
                assert "reason=obstacle_collision" not in prompt
            else:
                assert "invalid | step=1 | reason=obstacle_collision" in prompt
            return "Trace: (0,0) down (1,0) right (1,1) down (2,1) right (2,2)"

        result = Seq2SeqRepairerAdapter(infer, output_format="trace_only").repair(
            environment_id=GRID["environment_id"],
            grid=GRID,
            candidate_plan=["right", "right"],
            feedback_source=source,
            verifier_feedback=feedback,
        )
        assert result["parse_success"] is True
        assert result["repaired_plan"] == ["down", "right", "down", "right"]

