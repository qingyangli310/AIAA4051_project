"""Legacy A-4 utilities. NOT a formal v1.2 dataset producer.

A-4 used disjoint ID/OOD environments and seed 17; v1.2 requires paired
heldout conditions and seed 42. Do not use this CLI for formal data.
"""
from __future__ import annotations
import random
import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Any

from src.data.executor import execute_plan
from src.data.generate_phase0 import shortest_path, path_to_actions
from src.data.generate_phase0 import make_example
from src.data.ppnl_io import write_json, write_jsonl

CORRUPTIONS = ("action_substitution", "action_deletion", "action_insertion", "premature_stopping")
CAPS = {"train": 4000, "dev": 200, "id_test": 200, "ood_test": 200}


def make_correct_plans(grid: dict[str, Any], seed: int = 0) -> list[dict[str, Any]]:
    rows, cols = grid["grid_size"]
    path = shortest_path(rows, cols, grid.get("obstacles", []), tuple(grid["start"]), tuple(grid["goal"]))
    if path is None: return []
    shortest = path_to_actions(path)
    result = [{"plan": shortest, "plan_type": "shortest"}]
    rng = random.Random(seed)
    for _ in range(100):
        actions = shortest[:]
        if not actions: break
        i = rng.randrange(len(actions)); actions[i:i] = ["left", "right"]
        if execute_plan(grid, actions)["success"]:
            result.append({"plan": actions, "plan_type": "feasible_nonoptimal"}); break
    return result


def corrupt_plan(plan: list[str], kind: str, rng: random.Random) -> list[str]:
    out = list(plan)
    if kind == "premature_stopping":
        return out[:max(0, len(out)-1)]
    if kind == "action_substitution":
        if not out: out = ["invalid"]
        else:
            i = rng.randrange(len(out)); out[i] = rng.choice([a for a in ("up","down","left","right") if a != out[i]])
    elif kind == "action_deletion":
        if out: out.pop(rng.randrange(len(out)))
        else: out.append("invalid")
    elif kind == "action_insertion":
        out.insert(rng.randrange(len(out)+1), rng.choice(("up","down","left","right")))
    else: raise ValueError(kind)
    return out


def apply_mutation(plan: list[str], operation: dict[str, Any]) -> list[str]:
    """Apply a provenance operation at a stable token/gap position."""
    out=list(plan); kind=operation["type"]; pos=operation["position"]
    if kind=="action_substitution": out[pos]=operation["replacement"]
    elif kind=="action_deletion": out.pop(pos)
    elif kind=="action_insertion": out.insert(pos,operation["action"])
    elif kind=="premature_stopping": out=out[:pos]
    else: raise ValueError(kind)
    return out


def _operation(plan: list[str], kind: str, position: int, rng: random.Random) -> dict[str, Any] | None:
    if kind in ("action_substitution","action_deletion"):
        if position>=len(plan): return None
        op={"type":kind,"position":position,"logical_position":f"token:{position}"}
        if kind=="action_substitution": op["replacement"]=rng.choice([a for a in ("up","down","left","right") if a!=plan[position]])
        return op
    if kind=="action_insertion":
        op={"type":kind,"position":position,"logical_position":f"gap:{position}","action":rng.choice(("up","down","left","right"))}
        return op
    if kind=="premature_stopping" and position<len(plan):
        return {"type":kind,"position":position,"logical_position":f"cut:{position}"}
    return None


def generate_corruptions(record: dict[str, Any], source_plan_id: str, seed: int, ood: bool = False) -> list[dict[str, Any]]:
    rng=random.Random(seed); source=list(record["plan"]); result=[]
    count=4
    for idx in range(count):
        accepted=None
        for _ in range(300):
            kinds=rng.sample(CORRUPTIONS,2 if ood else 1)
            used=set(); ops=[]; working=list(source); valid=True
            for kind in kinds:
                choices=list(range(len(working)+1))
                rng.shuffle(choices); op=None
                for pos in choices:
                    candidate_op=_operation(working,kind,pos,rng)
                    if candidate_op and candidate_op["logical_position"] not in used:
                        op=candidate_op; break
                if op is None: valid=False; break
                before=list(working)
                try: after=apply_mutation(before,op)
                except IndexError: valid=False; break
                if after==before: valid=False; break
                op["before_plan"]=before; op["after_plan"]=after
                # Independent replay is the acceptance condition, not type count.
                if apply_mutation(before,op)!=after: valid=False; break
                working=after; used.add(op["logical_position"]); ops.append(op)
            if not valid or len(ops)!=(2 if ood else 1): continue
            if len(used)!=len(ops) or working==source: continue
            label=execute_plan(record["grid"],working)
            if label["success"]: continue
            accepted=(kinds,ops,working,label); break
        if accepted is None: continue
        kinds,ops,corrupted,label=accepted
        result.append({"environment_id":record["grid"]["environment_id"],"source_plan_id":source_plan_id,
          "source_plan":source,"corruption_type":kinds,"corruption_seed":seed+idx,"mutation_provenance":ops,
          "corrupted_plan":corrupted,"executor_label":label,"first_error_step":label["first_error_step"],"reason":label["reason"]})
    return result


def split_by_environment(records: list[dict[str, Any]], seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    envs = sorted({r["environment_id"] for r in records}); random.Random(seed).shuffle(envs)
    splits = {name: [] for name in CAPS}
    # Stable partition ratios; environment assignment never crosses splits.
    n = len(envs); cut1=max(1, round(n*.6)); cut2=max(cut1+1, round(n*.75)); cut3=max(cut2+1, round(n*.9))
    groups = {"train":set(envs[:cut1]), "dev":set(envs[cut1:cut2]), "id_test":set(envs[cut2:cut3]), "ood_test":set(envs[cut3:])}
    for name, ids in groups.items(): splits[name] = [r for r in records if r["environment_id"] in ids]
    for name, rows in splits.items():
        if len(rows) > CAPS[name]: raise ValueError(f"{name} exceeds sample cap {CAPS[name]}")
    return splits


def validate_resource_limits(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Reject over-limit output before any dataset file is opened or written."""
    for name, limit in CAPS.items():
        count=len(splits.get(name, []))
        if count>limit:
            raise ValueError(f"{name} has {count} samples; course limit is {limit}. No files written.")


def preflight_pilot_limit(environments_per_split: int) -> None:
    if environments_per_split < 1: raise ValueError("environments_per_split must be at least 1")
    # Conservative upper bounds: 6 train/dev task rows per environment, at most
    # 18 ID/OOD rows (2 plans x 4 corruption variants x Verification+Repair,
    # plus Generation rows). Check before generation or filesystem writes.
    estimates={"train":6*environments_per_split,"dev":6*environments_per_split,
               "id_test":18*environments_per_split,"ood_test":18*environments_per_split}
    for name,count in estimates.items():
        if count>CAPS[name]:
            raise ValueError(f"requested pilot could produce {count} {name} samples; limit is {CAPS[name]}. No files written.")


def build_manifest(splits: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    ids=[{r["environment_id"] for r in rows} for rows in splits.values()]
    if any(ids[i]&ids[j] for i in range(len(ids)) for j in range(i)): raise ValueError("environment leakage across splits")
    return {"seed":seed,"splits":{name:{"samples":len(rows),"environments":len({r['environment_id'] for r in rows}),
      "task_distribution":dict(Counter(r.get("task", "unspecified") for r in rows)),
      "error_distribution":dict(Counter(r.get("reason", "none") for r in rows)),
      "corruption_distribution":dict(Counter(str(t) for r in rows for t in r.get("corruption_type", [])))} for name,rows in splits.items()},
      "goal_not_reached_step_convention":"len(plan) (0-based)",
      "split_definition":"train/dev include Generation, Verification, Repair; id_test includes valid paths and single-error cases; ood_test emphasizes double-error cases"}


def _task_schema(task: str, grid: dict[str, Any], plan: list[str], label: dict[str, Any], repair_target: list[str] | None=None) -> dict[str, Any]:
    if task=="Generation":
        return {"input":{"grid":grid},"output":{"plan":plan}}
    if task=="Verification":
        return {"input":{"grid":grid,"candidate_plan":plan},"output":{k:label[k] for k in ("valid","success","first_error_step","reason")}}
    if task=="Repair":
        return {"input":{"grid":grid,"corrupted_plan":plan,"verifier_feedback":label},"output":{"repaired_plan":repair_target}}
    raise ValueError(task)


def generate_pilot(output_dir: Path, seed: int = 17, environments_per_split: int = 2) -> dict[str, Any]:
    """Create a small, deterministic P6 pilot; split cohorts have disjoint env IDs."""
    preflight_pilot_limit(environments_per_split)
    rng = random.Random(seed)
    splits = {name: [] for name in CAPS}
    for split_name in CAPS:
        for env_index in range(environments_per_split):
            # Same task family, independently identified environments per split.
            grid = {"grid_size":[5,5], "start":[2,0], "goal":[2,4], "obstacles":[[0,0]],
                    "environment_id":f"pilot_{split_name}_env{env_index:03d}"}
            plans = make_correct_plans(grid, seed + env_index)
            if len(plans) < 2:
                raise RuntimeError(f"could not generate shortest and non-optimal plans for {grid['environment_id']}")
            for pidx, plan_record in enumerate(plans):
                checked = execute_plan(grid, plan_record["plan"])
                if not checked["success"]: raise RuntimeError("correct plan rejected by Executor")
                example = make_example({"grid_size":grid["grid_size"],"grid_shape":"5x5","obstacle_count":1,
                    "obstacles":grid["obstacles"],"environment_id":grid["environment_id"]}, split_name,
                    pidx, tuple(grid["start"]), tuple(grid["goal"]),
                    # Preserve actual target actions while retaining PPNL-compatible fields.
                    _actions_to_path(grid["start"], plan_record["plan"]), False)
                example.update({"plan":plan_record["plan"],"plan_type":plan_record["plan_type"],
                                "executor_label":checked,"label_source":"executor","task":"Generation"})
                example.update(_task_schema("Generation",grid,plan_record["plan"],checked))
                if split_name in ("train", "dev"):
                    splits[split_name].append(dict(example))
                    verify=dict(example); verify.update({"task":"Verification",**_task_schema("Verification",grid,plan_record["plan"],checked)})
                    verify.update({"example_id":f"{example['example_id']}_verification","id":f"{example['id']}_verification"})
                    splits[split_name].append(verify)
                    repairs=generate_corruptions({"grid":grid,"plan":plan_record["plan"]},f"{grid['environment_id']}_p{pidx}",seed+env_index*100+pidx,ood=False)
                    if not repairs: raise RuntimeError("could not create train/dev repair corruption")
                    repair=repairs[0]; feedback=repair["executor_label"]
                    repair_row=dict(example); repair_row.update({"task":"Repair","plan":repair["corrupted_plan"],
                        "target":" ".join(repair["corrupted_plan"]),"executor_label":feedback,"reason":feedback["reason"],
                        "first_error_step":feedback["first_error_step"],"corruption_type":repair["corruption_type"],
                        "source_plan":repair["source_plan"],"mutation_provenance":repair["mutation_provenance"],"repair_target":plan_record["plan"],
                        **_task_schema("Repair",grid,repair["corrupted_plan"],feedback,plan_record["plan"])})
                    repair_row.update({"example_id":f"{example['example_id']}_repair","id":f"{example['id']}_repair"})
                    splits[split_name].append(repair_row)
                if split_name in ("id_test", "ood_test"):
                    corruption_seed=seed + env_index*10 + pidx
                    corruptions=[]
                    for retry in range(100):
                        corruptions=generate_corruptions({"grid":grid,"plan":plan_record["plan"]},
                            f"{grid['environment_id']}_p{pidx}",corruption_seed+retry*1000,
                            ood=(split_name=="ood_test"))
                        if corruptions: break
                    if not corruptions:
                        raise RuntimeError(f"failed to create non-cancelling corruption for {grid['environment_id']} plan {pidx}")
                    for cidx, row in enumerate(corruptions):
                        corrupted_example = dict(example)
                        corrupted_example.update({"example_id":f"{example['example_id']}_corrupt{cidx}",
                            "id":f"{example['id']}_corrupt{cidx}", "plan":row["corrupted_plan"],
                            "target":" ".join(row["corrupted_plan"]),"agent_as_a_point":" ".join(row["corrupted_plan"]),
                            "corruption_type":row["corruption_type"],"corruption_seed":row["corruption_seed"],
                            "source_plan_id":row["source_plan_id"],"executor_label":row["executor_label"],
                            "source_plan":row["source_plan"],
                            "first_error_step":row["first_error_step"],"reason":row["reason"],"label_source":"executor",
                            "mutation_provenance":row["mutation_provenance"]})
                        corrupted_example["task"]="Verification"
                        corrupted_example.update(_task_schema("Verification",grid,row["corrupted_plan"],row["executor_label"]))
                        splits[split_name].append(corrupted_example)
                        repair_example=dict(corrupted_example)
                        repair_example["task"]="Repair"
                        repair_example["repair_target"]=plan_record["plan"]
                        repair_example.update(_task_schema("Repair",grid,row["corrupted_plan"],row["executor_label"],plan_record["plan"]))
                        repair_example["example_id"] += "_repair"
                        repair_example["id"] += "_repair"
                        splits[split_name].append(repair_example)
                    valid_example=dict(example)
                    valid_example["task"]="Generation"
                    splits[split_name].append(valid_example)
    manifest=build_manifest(splits,seed)
    validate_resource_limits(splits)
    output_dir.mkdir(parents=True,exist_ok=True)
    for split, rows in splits.items(): write_jsonl(output_dir/f"{split}.jsonl",rows)
    write_json(output_dir/"manifest.json",manifest)
    return manifest


def _actions_to_path(start: list[int], actions: list[str]) -> list[tuple[int,int]]:
    from src.data.executor import ACTION_DELTAS
    pos=tuple(start); path=[pos]
    for action in actions:
        dr,dc=ACTION_DELTAS[action]; pos=(pos[0]+dr,pos[1]+dc); path.append(pos)
    return path


def main() -> None:
    parser=argparse.ArgumentParser(description="Generate a bounded Project 6 pilot dataset")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--environments-per-split",type=int,default=2)
    parser.add_argument("--legacy-a4", action="store_true", help="explicitly acknowledge historical A-4 protocol")
    args=parser.parse_args()
    if not args.legacy_a4:
        parser.error("historical A-4 only; formal v1.2 generation is deferred to Nov 14-15. Use --legacy-a4 only to reproduce history.")
    print(json.dumps(generate_pilot(args.output_dir,args.seed,args.environments_per_split),indent=2))

if __name__ == "__main__": main()
