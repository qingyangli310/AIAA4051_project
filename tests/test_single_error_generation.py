import json
from pathlib import Path
from src.data.build_nov14_dataset import single_corruption
from src.executor.executor import execute_plan

def test_all_single_error_types_replay_and_label():
    grid={'environment_id':'e','rows':6,'cols':6,'start':[1,1],'goal':[4,4],'obstacles':[[2,2]]}
    plan=['down','down','down','right','right','right']
    for i,kind in enumerate(('action_substitution','action_deletion','action_insertion','premature_stopping')):
        result=single_corruption(grid,plan,kind,42+i);assert result is not None
        corrupted,pos=result;label=execute_plan(grid,corrupted)
        assert pos>=0 and not label['success']
        assert corrupted!=plan

def test_data_manifests_are_source_complete():
    report=json.loads(Path('results/ppnl_source_audit_20261113.json').read_text())
    assert all(row['exists'] and row['parse_success'] and row['missing_required_fields']==0 for row in report['files'])
    assert sum(row['conversion_issues'] for row in report['files'])==511
