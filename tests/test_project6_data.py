import unittest
import tempfile
import json
from pathlib import Path
from src.data.executor import execute_plan
from src.data.project6_data import generate_corruptions, split_by_environment, build_manifest, generate_pilot, validate_resource_limits
from src.data.project6_data import apply_mutation, preflight_pilot_limit, _task_schema

class Project6Tests(unittest.TestCase):
    def setUp(self):
        self.grid={"grid_size":[3,3],"start":[1,0],"goal":[1,2],"obstacles":[[0,1]],"environment_id":"e1"}
    def test_executor_paths_and_failures(self):
        self.assertTrue(execute_plan(self.grid,["right","right"])["success"])
        self.assertTrue(execute_plan(self.grid,["up","down","right","right"])["success"])
        blocked={**self.grid,"obstacles":[[0,0]]}
        self.assertEqual(execute_plan(blocked,["up"])["reason"],"obstacle_collision")
        self.assertEqual(execute_plan(self.grid,["left"])["reason"],"out_of_bounds")
        self.assertEqual(execute_plan(self.grid,["bogus"])["reason"],"invalid_action_token")
        self.assertEqual(execute_plan(self.grid,["right"])["reason"],"goal_not_reached")
        self.assertEqual(execute_plan(self.grid,["right"])["first_error_step"],1)
        self.assertEqual(execute_plan(self.grid,["right","right","right"])["first_error_step"],2)
    def test_double_corruption_and_splits(self):
        rec={"grid":self.grid,"plan":["right","right"]}
        rows=generate_corruptions(rec,"p1",4,ood=True)
        self.assertTrue(rows)
        self.assertTrue(all(len(x["corruption_type"])==2 and not x["executor_label"]["success"] for x in rows))
        for row in rows:
            provenance=row["mutation_provenance"]
            self.assertEqual(len(provenance),2)
            self.assertEqual(len({op["logical_position"] for op in provenance}),2)
            replay=list(row["source_plan"])
            for op in provenance:
                self.assertEqual(op["before_plan"],replay)
                after=apply_mutation(replay,op)
                self.assertEqual(after,op["after_plan"])
                self.assertNotEqual(after,replay)
                replay=after
            self.assertEqual(replay,row["corrupted_plan"])
            self.assertEqual(execute_plan(self.grid,replay),row["executor_label"])
        data=[]
        for i in range(10): data.append({"environment_id":f"e{i}","reason":"x"})
        splits=split_by_environment(data,9); build_manifest(splits,9)
        seen=[{r['environment_id'] for r in x} for x in splits.values()]
        self.assertFalse(any(seen[i]&seen[j] for i in range(4) for j in range(i)))
        self.assertEqual(split_by_environment(data,9),splits)
    def test_eval_adapter_and_pilot_reproducibility(self):
        row={"world":[[0,0,0],[0,2,3],[0,0,0]],"start":[1,0],"goal":[1,2],"target":"right right","plan":"right right"}
        with tempfile.TemporaryDirectory() as d:
            one=Path(d)/"one"; two=Path(d)/"two"
            m1=generate_pilot(one,13,2); m2=generate_pilot(two,13,2)
            self.assertEqual(m1,m2)
            for split in ("train","dev","id_test","ood_test"):
                self.assertEqual((one/f"{split}.jsonl").read_bytes(),(two/f"{split}.jsonl").read_bytes())
                count=len((one/f"{split}.jsonl").read_text().splitlines())
                self.assertEqual(m1["splits"][split]["samples"],count)
            for split in ("train","dev"):
                rows=[__import__('json').loads(line) for line in (one/f"{split}.jsonl").read_text().splitlines()]
                self.assertEqual({r['task'] for r in rows},{"Generation","Verification","Repair"})
            self.assertEqual(m1["goal_not_reached_step_convention"],"len(plan) (0-based)")
            for split in ("train","dev","id_test","ood_test"):
                rows=[json.loads(line) for line in (one/f"{split}.jsonl").read_text().splitlines()]
                for row in rows:
                    self.assertIn("input",row); self.assertIn("output",row)
                    if row["task"]=="Repair":
                        inp=row["input"]; out=row["output"]
                        bad=execute_plan(inp["grid"],inp["corrupted_plan"])
                        good=execute_plan(inp["grid"],out["repaired_plan"])
                        self.assertFalse(bad["success"])
                        self.assertEqual(inp["verifier_feedback"],bad)
                        self.assertTrue(good["success"] and good["valid"])
            id_rows=[json.loads(x) for x in (one/"id_test.jsonl").read_text().splitlines()]
            ood_rows=[json.loads(x) for x in (one/"ood_test.jsonl").read_text().splitlines()]
            self.assertTrue(all(len(r["mutation_provenance"])==1 for r in id_rows if "mutation_provenance" in r))
            self.assertTrue(all(len(r["mutation_provenance"])==2 for r in ood_rows if "mutation_provenance" in r))
    def test_limits_and_manifest(self):
        with self.assertRaisesRegex(ValueError,"No files written"):
            preflight_pilot_limit(12)
        with self.assertRaisesRegex(ValueError,"No files written"):
            validate_resource_limits({"train":[{}]*4001})
        splits={"train":[{"environment_id":"a","task":"Generation"}],"dev":[],"id_test":[],"ood_test":[]}
        m=build_manifest(splits,3)
        self.assertEqual(m["splits"]["train"]["samples"],1)
    def test_task_schema(self):
        feedback={"valid":False,"success":False,"first_error_step":0,"reason":"out_of_bounds"}
        grid={"grid_size":[2,2]}; path=["right"]
        self.assertEqual(set(_task_schema("Generation",grid,path,feedback)),{"input","output"})
        self.assertEqual(set(_task_schema("Verification",grid,path,feedback)["output"]),{"valid","success","first_error_step","reason"})
        self.assertEqual(set(_task_schema("Repair",grid,path,feedback,["down"])["input"]),{"grid","corrupted_plan","verifier_feedback"})

if __name__ == '__main__': unittest.main()
