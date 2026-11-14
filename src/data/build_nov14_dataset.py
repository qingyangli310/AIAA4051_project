"""Build v1.2 standard 6x6 data, manifests and a bounded single-error pilot."""
from __future__ import annotations
import argparse, hashlib, json, random, subprocess
from collections import deque, Counter
from pathlib import Path
from typing import Any
from src.data.environment import split_environments, base_sample_id
from src.data.load_ppnl import FILES, load_ppnl
from src.data.ppnl_io import write_json, write_jsonl
from src.executor.executor import execute_plan

COMMIT="5fd0376fc50ae9860bc986085165ebac5675fd44"
MAIN_FILES=FILES[:4]
EXPECTED_SHA256={
 '1_train_set_6x6_samples.json':'b20a44269cff617abd5132d3ad209f72fb0ce92f78b4eaefc5aa8a92fac093e2',
 '1dev_set_6x6_samples.json':'aad6f181c4eff903c1f262ae009f2c3ac2d0f787abae4765c51976fd1ee8c574',
 '1_goals_test_seen_6x6_samples.json':'ebdc305d207b0fda3d147b76ddd93fdecdb879dd5bd6acfad90bd242c15a2507',
 '1goals_unseen_6x6_samples.json':'ee841e3adb907116471989331f0e4cffed16038c46592491abdd7bafd34d432e',
 '1_goals_test_unseen_5x5_samples.json':'f34edfe112d248087ae76366c4f4e56cc0e0d31729d0bd981b6c8656344b32cd',
 '1_goals_test_unseen_7x7_samples.json':'db358fd01da87004c2d62551e44cfcb506cb00e88aba29815ebc0fd1c288eb58',
 '1_goals_test_unseen_6x6more_obstacles_samples.json':'8825fee0599751b4da946b9b6a75808bd9aeff8a5595e8db9e833ee38828b16d'}
DETOURS=("up","down","left","right")

def shortest_path(grid:dict[str,Any])->list[str]|None:
    start=tuple(grid['start']);goal=tuple(grid['goal']);rows,cols=grid['rows'],grid['cols'];blocked={tuple(p) for p in grid['obstacles']}
    q=deque([(start,[])]);seen={start};deltas={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
    while q:
        pos,path=q.popleft()
        if pos==goal:return path
        for action,(dr,dc) in deltas.items():
            nxt=(pos[0]+dr,pos[1]+dc)
            if 0<=nxt[0]<rows and 0<=nxt[1]<cols and nxt not in blocked and nxt not in seen:
                seen.add(nxt);q.append((nxt,path+[action]))
    return None

def nonoptimal_plan(grid:dict[str,Any],shortest:list[str])->list[str]|None:
    if not shortest:return None
    reverse={"up":"down","down":"up","left":"right","right":"left"}[shortest[0]]
    candidate=[shortest[0],reverse,*shortest];r=execute_plan(grid,candidate)
    return candidate if r['valid'] and r['success'] and len(candidate)>len(shortest) else None

def single_corruption(grid:dict[str,Any],plan:list[str],kind:str,seed:int)->tuple[list[str],int]|None:
    rng=random.Random(seed); positions=list(range(len(plan)+1));rng.shuffle(positions)
    for pos in positions:
        candidates=[]
        if kind=='action_substitution' and pos<len(plan):
            candidates=[[*(plan[:pos]),a,*plan[pos+1:]] for a in DETOURS if a!=plan[pos]]
        elif kind=='action_deletion' and pos<len(plan): candidates=[plan[:pos]+plan[pos+1:]]
        elif kind=='action_insertion': candidates=[plan[:pos]+[a]+plan[pos:] for a in DETOURS]
        elif kind=='premature_stopping' and pos<len(plan): candidates=[plan[:pos]]
        for candidate in candidates:
            if candidate!=plan and not execute_plan(grid,candidate)['success']: return candidate,pos
    return None

def _record(raw:dict[str,Any],filename:str,index:int)->dict[str,Any]:
    result={k:v for k,v in raw.items() if k not in ('executor_result','original_plan')}
    result['source_file']=filename;result['source_index']=index
    result['base_sample_id']=base_sample_id(COMMIT,filename,index)
    return result

def build(source:Path,processed:Path,pilot:Path,seed:int=42,manifest_dir:Path=Path('data/manifests'))->dict[str,Any]:
    # Verify all seven pinned files and upstream revision before any output is written.
    git_root=source.parent.parent
    actual_commit=subprocess.check_output(['git','-C',str(git_root),'rev-parse','HEAD'],text=True).strip()
    if actual_commit!=COMMIT:raise ValueError(f'PPNL commit mismatch: {actual_commit} != {COMMIT}')
    for filename in FILES:
        if not (source/filename).is_file():raise FileNotFoundError(source/filename)
        digest=hashlib.sha256((source/filename).read_bytes()).hexdigest()
        if digest!=EXPECTED_SHA256[filename]:raise ValueError(f'PPNL checksum mismatch: {filename}')
    full={f:load_ppnl(source/f) for f in FILES}
    excluded=[]
    for f,data in full.items():
        for issue in data['issues']:
            if 'unreachable_source' in issue['error']:
                excluded.append({**issue,'reason':'goal_not_reached_source','included_in_main_splits':False})
            elif f in MAIN_FILES: raise ValueError(f"standard source has non-unreachable conversion issue: {issue}")
    records=[]; shortest_mismatch=0; nonopt=[]
    for f in MAIN_FILES:
        for raw in full[f]['records']:
            row=_record(raw,f,raw['source_index']);g={k:row[k] for k in ('environment_id','rows','cols','start','goal','obstacles')}
            sp=shortest_path(g)
            if sp is None:raise ValueError(f"reachable source has no BFS path {f}:{row['source_index']}")
            check=execute_plan(g,sp)
            if not(check['valid'] and check['success']):raise AssertionError('BFS shortest failed Executor')
            official=raw['plan'];shortest_mismatch+=len(official)!=len(sp)
            row.update({'official_plan':official,'shortest_plan':sp,'shortest_length':len(sp),
                        'official_length':len(official),'official_optimal':len(official)==len(sp),
                        'shortest_executor_label':check})
            np=nonoptimal_plan(g,sp)
            if np is not None:row['nonoptimal_plan']=np;row['nonoptimal_executor_label']=execute_plan(g,np)
            else:row['nonoptimal_unavailable']=True
            records.append(row)
    # Environment identity is layout-only. Each source row remains traceable.
    env_split=split_environments((r['environment_id'] for r in records),seed)
    owner={eid:name for name,ids in env_split.items() for eid in ids}
    splits={name:[] for name in env_split}
    for r in records:splits[owner[r['environment_id']]].append(r)
    for name,rows in splits.items():
        write_jsonl(processed/f'{name}.jsonl',rows)
        write_json(manifest_dir/f'{name}.json',{'seed':seed,'split':name,'environment_ids':env_split[name],
          'environment_count':len(env_split[name]),'sample_count':len(rows),'base_sample_ids':[r['base_sample_id'] for r in rows]})
    write_json(manifest_dir/'excluded_manifest.json',{'source_commit':COMMIT,'count':len(excluded),'records':excluded})
    # Bounded pilot uses up to one eligible row per distinct environment.
    pilot_splits={k:[] for k in ('train','dev','heldout_test')}
    for name,limit in [('train',200),('dev',20),('heldout_test',200)]:
        chosen=[];seen=set()
        for r in splits[name]:
            if r['environment_id'] in seen or r.get('nonoptimal_unavailable'):continue
            chosen.append(r);seen.add(r['environment_id'])
            if len(chosen)>=limit:break
        pilot_splits[name]=chosen
    for name,rows in pilot_splits.items():write_jsonl(pilot/f'{name}.jsonl',rows)
    errors={"train":[],"dev":[]};types=('action_substitution','action_deletion','action_insertion','premature_stopping')
    for split in ('train','dev'):
        for base in pilot_splits[split]:
            grid={k:base[k] for k in ('environment_id','rows','cols','start','goal','obstacles')}
            for kind_idx,kind in enumerate(types):
                seed_i=seed+int(base['base_sample_id'][-8:],16)+kind_idx
                generated=single_corruption(grid,base['shortest_plan'],kind,seed_i)
                if generated is None:raise RuntimeError(f"cannot generate {kind} for {base['base_sample_id']}")
                corrupted,pos=generated;label=execute_plan(grid,corrupted)
                errors[split].append({'environment_id':base['environment_id'],'base_sample_id':base['base_sample_id'],
                  'grid':grid,
                  'source_plan_id':base['base_sample_id']+':shortest','source_file':base['source_file'],'source_index':base['source_index'],
                  'corruption_type':kind,'corruption_position':pos,'corruption_seed':seed_i,
                  'original_plan':base['shortest_plan'],'corrupted_plan':corrupted,'executor_label':label})
    for split in errors:write_jsonl(pilot/f'single_error_{split}.jsonl',errors[split])
    manifest={'seed':seed,'source_commit':COMMIT,'main_files':list(MAIN_FILES),'main_records':len(records),
      'unique_environments':len(owner),'environment_counts':{s:len(v) for s,v in env_split.items()},
      'actual_environment_ratios':{s:len(v)/len(owner) for s,v in env_split.items()},
      'split_samples':{s:len(v) for s,v in splits.items()},'excluded_total_all_seven':len(excluded),
      'excluded_main_four':sum(len(full[f]['issues']) for f in MAIN_FILES),'official_not_shortest':shortest_mismatch,
      'nonoptimal_available':sum('nonoptimal_plan' in r for r in records),
      'pilot_counts':{s:len(v) for s,v in pilot_splits.items()},
      'single_error_counts':{s:dict(Counter(x['corruption_type'] for x in v)) for s,v in errors.items()},
      'ood_double_error_generated':False}
    write_json(pilot/'manifest.json',manifest)
    return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--processed',type=Path,default=Path('data/processed/nov14'));p.add_argument('--pilot',type=Path,default=Path('data/samples/nov14_pilot'));p.add_argument('--manifests',type=Path,default=Path('data/manifests'));p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    print(json.dumps(build(a.source,a.processed,a.pilot,a.seed,a.manifests),indent=2))
if __name__=='__main__':main()
