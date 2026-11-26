from src.model_adapters import (
    ConditionRoutingRepairer,
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


def test_feedback_localized_repair_prompt_exposes_only_authorized_localization() -> None:
    prompts = []

    def infer(prompt: str, limit: int) -> str:
        prompts.append(prompt)
        return "Trace: (0,0) down (1,0) right (1,1) down (2,1) right (2,2)"

    adapter = Seq2SeqRepairerAdapter(
        infer,
        output_format="trace_only",
        prompt_protocol="feedback_localized_v1",
    )
    adapter.repair(
        environment_id=GRID["environment_id"],
        grid=GRID,
        candidate_plan=["down", "right", "right"],
        feedback_source="executor_feedback",
        verifier_feedback={
            "valid": False,
            "first_error_step": 2,
            "reason": "obstacle_collision",
        },
    )
    assert "Keep this valid prefix unchanged: down, right" in prompts[-1]
    assert "Resume from last valid position: (1,1)" in prompts[-1]

    adapter.repair(
        environment_id=GRID["environment_id"],
        grid=GRID,
        candidate_plan=["down", "right", "right"],
        feedback_source="no_feedback",
        verifier_feedback=None,
    )
    assert "Repair localization" not in prompts[-1]
    assert "obstacle_collision" not in prompts[-1]
    assert "Resume from last valid position" not in prompts[-1]


def test_condition_routing_repairer_uses_separate_no_feedback_model() -> None:
    calls = []

    class Repairer:
        def __init__(self, name):
            self.name = name

        def repair(self, **request):
            calls.append((self.name, request["feedback_source"]))
            return {"repaired_plan": [], "parse_success": True}

    router = ConditionRoutingRepairer(Repairer("baseline"), Repairer("feedback"))
    router.repair(feedback_source="no_feedback")
    router.repair(feedback_source="executor_feedback")
    router.repair(feedback_source="self_feedback")
    assert calls == [
        ("baseline", "no_feedback"),
        ("feedback", "executor_feedback"),
        ("feedback", "self_feedback"),
    ]


def test_suffix_repair_appends_only_to_feedback_valid_prefix() -> None:
    def infer(prompt: str, limit: int) -> str:
        assert "Last valid position: (1,1)" in prompt
        assert "Valid prefix to keep: down, right" in prompt
        return "Suffix: down, right"

    adapter = Seq2SeqRepairerAdapter(
        infer, output_format="suffix_v1", prompt_protocol="suffix_v1"
    )
    result = adapter.repair(
        environment_id=GRID["environment_id"], grid=GRID,
        candidate_plan=["down", "right", "right"],
        feedback_source="executor_feedback",
        verifier_feedback={"first_error_step": 2, "reason": "obstacle_collision"},
    )
    assert result["parse_success"] is True
    assert result["repaired_plan"] == ["down", "right", "down", "right"]

