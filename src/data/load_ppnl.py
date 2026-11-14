"""Strict, traceable PPNL single-goal loader. Never guesses malformed data."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from src.executor.executor import execute_plan
from src.data.environment import environment_id

FILES = (
    '1_train_set_6x6_samples.json', '1dev_set_6x6_samples.json',
    '1_goals_test_seen_6x6_samples.json', '1goals_unseen_6x6_samples.json',
    '1_goals_test_unseen_5x5_samples.json', '1_goals_test_unseen_7x7_samples.json',
    '1_goals_test_unseen_6x6more_obstacles_samples.json',
)
REQUIRED = ('world','nl_description','solution_coordinates','agent_as_a_point','agent_has_direction')

class PPNLFormatError(ValueError):
    pass


def convert_record(record: dict[str, Any], source_file: str, source_index: int) -> dict[str, Any]:
    if not isinstance(record, dict): raise PPNLFormatError('record must be an object')
    missing=[key for key in REQUIRED if key not in record]
    if missing: raise PPNLFormatError(f'missing fields: {missing}')
    world=record['world']
    if not isinstance(world,list) or not world or not all(isinstance(row,list) and row for row in world):
        raise PPNLFormatError('world must be nonempty rectangular matrix')
    cols=len(world[0]); rows=len(world)
    if any(len(row)!=cols for row in world) or any(type(v) is not int or v not in (0,1,2,3) for row in world for v in row):
        raise PPNLFormatError('invalid world shape or cell values')
    markers={v:[[r,c] for r,row in enumerate(world) for c,cell in enumerate(row) if cell==v] for v in (1,2,3)}
    if len(markers[2])!=1 or len(markers[3])!=1: raise PPNLFormatError('expected exactly one start and one goal')
    if not all(isinstance(record[k],str) for k in ('nl_description','agent_as_a_point','agent_has_direction')):
        raise PPNLFormatError('description and action fields must be strings')
    raw=record['agent_as_a_point']
    if raw.strip().lower()=='goal not reachable':
        raise PPNLFormatError('unreachable_source: no action plan; conversion stopped pending team policy, source retained in issues')
    plan=raw.strip().lower().split()
    obstacles=sorted(markers[1])
    grid={'environment_id':environment_id(rows,cols,obstacles), 'rows':rows,'cols':cols,
          'start':markers[2][0],'goal':markers[3][0],'obstacles':obstacles}
    result=execute_plan(grid,plan)
    if not result['success']: raise PPNLFormatError(f'official plan failed executor: {result["reason"]} step={result["first_error_step"]}')
    if record['solution_coordinates'] != result['trace']:
        raise PPNLFormatError('solution_coordinates differs from executable plan trace')
    return {**grid,'plan':plan,'source_file':Path(source_file).name,'source_index':source_index,
            'original_plan':raw,'executor_result':result}


def load_ppnl(path: str | Path) -> dict[str, Any]:
    """Return records AND explicit issues; malformed JSON/top-level type is fatal.

    Every input row is accounted for. Consumers must inspect issues and must not
    treat the returned records alone as a silently cleaned official dataset.
    """
    path=Path(path)
    data=json.loads(path.read_text())
    if not isinstance(data,list): raise PPNLFormatError('PPNL JSON must contain a list')
    records=[]; issues=[]
    for index,row in enumerate(data):
        try: records.append(convert_record(row,path.name,index))
        except PPNLFormatError as error:
            issues.append({'source_file':path.name,'source_index':index,'error':str(error)})
    return {'source_file':path.name,'total':len(data),'records':records,'issues':issues}
