from pathlib import Path

from src.data.ppnl_io import load_jsonl
from src.data.build_nov15_paired_conditions import _apply_operation
from src.executor.executor import execute_plan


def test_every_ood_pair_has_two_independent_reached_mutations():
    path = Path("data/samples/nov15_paired/ood_double_error.jsonl")
    for row in load_jsonl(path):
        ops = row["corruption_provenance"]
        assert len(ops) == 2
        assert ops[0]["logical_position"] != ops[1]["logical_position"]
        original = row["original_plan"]
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
            kind = op["corruption_type"]
            assert op["logical_position"] in {
                f"token:{op['position']}", f"gap:{op['position']}", f"stop:{op['position']}"
            }
            if op["logical_position"].startswith("token:"):
                assert op["position"] < len(original)
            else:
                assert op["position"] <= len(original)
            other_plan = _apply_operation(original, ops[1 - index])
            without_this_label = execute_plan(row["grid"], other_plan)
            assert combined["trace"] != without_this_label["trace"] or combined["final_position"] != without_this_label["final_position"]
