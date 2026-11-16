from pathlib import Path

from src.data.ppnl_io import load_jsonl
from src.executor.executor import execute_plan


def test_every_ood_pair_has_two_independent_reached_mutations():
    path = Path("data/samples/nov15_paired/ood_double_error.jsonl")
    for row in load_jsonl(path):
        ops = row["corruption_provenance"]
        assert len(ops) == 2
        assert ops[0]["logical_position"] != ops[1]["logical_position"]
        original = row["original_plan"]
        original_trace = execute_plan(row["grid"], original)["trace"]
        sequential = list(original)
        independent_labels = []
        for op in ops:
            assert op["independent_plan"] != original
            individual_result = execute_plan(row["grid"], op["independent_plan"])
            assert op["independent_executor_label"] == individual_result
            assert individual_result["success"] is False
            assert individual_result["valid"] is True
            assert individual_result["reason"] == "goal_not_reached"
            independent_labels.append(individual_result)
            assert op["before_plan"] == sequential
            sequential = list(op["after_plan"])
        assert sequential == row["corrupted_plan"]
        assert execute_plan(row["grid"], sequential) == row["executor_label"]
        combined = row["executor_label"]
        assert combined["valid"] is True and combined["success"] is False
        assert combined["executed_steps"] == len(sequential)
        for index, op in enumerate(ops):
            assert op["position"] < combined["executed_steps"]
            assert op["logical_position"] == f"token:{op['position']}"
            assert combined["trace"][op["position"] + 1] != original_trace[op["position"] + 1]
        first_only_trace = independent_labels[0]["trace"]
        second_pos = ops[1]["position"]
        assert combined["trace"][second_pos + 1] != first_only_trace[second_pos + 1]
