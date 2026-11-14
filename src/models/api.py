"""Member B's callable interfaces for C's pipeline."""

from src.models.backend import ModelConfig, Seq2SeqBackend
from src.models.output_parser import parse_plan, parse_verification
from src.models.prompts import (
    PROTOCOL_VERSION, build_generation_prompt, build_verification_prompt,
)


class GridModel:
    """Reuse one model instance across calls; load lazily on first inference.

    Both methods return protocol fields plus additive ``metadata``.
    Parse failure is a result, inference/input errors are exceptions.
    No symbolic executor is used to alter model predictions.
    """

    def __init__(self, config=None, *, backend=None):
        self.backend = backend if backend is not None else Seq2SeqBackend(config or ModelConfig())

    def _predict(self, grid, prompt, parser):
        response = self.backend.generate_text(prompt)
        result = parser(response["raw_output"])
        result["metadata"] = {
            **response.get("metadata", {}),
            "environment_id": grid["environment_id"],
            "prompt": prompt,
            "protocol_version": PROTOCOL_VERSION,
            "step_index_base": 0,
            "goal_not_reached_step_convention": "len(plan)",
        }
        return result

    def generate_plan(self, grid):
        return self._predict(grid, build_generation_prompt(grid), parse_plan)

    def verify_plan(self, grid, plan):
        return self._predict(
            grid, build_verification_prompt(grid, plan),
            lambda text: parse_verification(text, plan_length=len(plan)),
        )
