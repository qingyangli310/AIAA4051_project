import json
import os
from pathlib import Path
import pytest
from src.data.load_ppnl import FILES,load_ppnl,convert_record,PPNLFormatError

ROW={'world':[[2,0,3]],'nl_description':'toy','solution_coordinates':[[0,0],[0,1],[0,2]],'agent_as_a_point':'right right ','agent_has_direction':'move forward'}
@pytest.mark.parametrize('filename',FILES)
def test_seven_file_names(tmp_path,filename):
    p=tmp_path/filename;p.write_text(json.dumps([ROW])); data=load_ppnl(p)
    assert data['total']==1 and not data['issues']
    r=data['records'][0];assert r['start']==[0,0] and r['goal']==[0,2] and r['source_file']==filename and r['source_index']==0
    assert r['executor_result']['success']

def test_missing_and_empty(tmp_path):
    with pytest.raises(PPNLFormatError):convert_record({},'x',0)
    p=tmp_path/'empty.json';p.write_text('[]');assert load_ppnl(p)['total']==0
    p.write_text('[{}]');assert len(load_ppnl(p)['issues'])==1
    p.write_text('{}')
    with pytest.raises(PPNLFormatError):load_ppnl(p)

def test_unreachable_explicit_issue(tmp_path):
    p=tmp_path/'x.json';p.write_text(json.dumps([{**ROW,'agent_as_a_point':'Goal not reachable'}]))
    d=load_ppnl(p);assert not d['records'] and 'unreachable_source' in d['issues'][0]['error']

def test_environment_identity_ignores_start_goal_plan(tmp_path):
    a={**ROW,'world':[[2,0,3],[0,0,0]],'solution_coordinates':[[0,0],[0,1],[0,2]],'agent_as_a_point':'right right'}
    b={**ROW,'world':[[3,0,2],[0,0,0]],'solution_coordinates':[[0,2],[0,1],[0,0]],'agent_as_a_point':'left left'}
    x=convert_record(a,'a.json',0);y=convert_record(b,'b.json',0)
    assert x['environment_id']==y['environment_id']

def test_actual_official_files():
    root=Path(os.environ.get('PPNL_DATA_DIR','../ppnl_official/ppnl-spatial-temporal-reasoning/single_goal'))
    if not root.exists():pytest.skip('set PPNL_DATA_DIR to official single_goal directory')
    for name in FILES:
        d=load_ppnl(root/name);assert d['total']>0
        assert len(d['records'])+len(d['issues'])==d['total']
