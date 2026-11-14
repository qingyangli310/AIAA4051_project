from src.data.environment import environment_id,environment_key,base_sample_id

def test_id_ignores_start_goal_and_plan():
    a=environment_id(6,6,[[1,2],[0,3]])
    assert a==environment_id(6,6,[[0,3],[1,2]])
    assert a==environment_id(6,6,[[1,2],[0,3]]) # start/goal/plan are not ID inputs
    assert environment_key(6,6,[[1,2],[0,3]])=='{"cols":6,"obstacles":[[0,3],[1,2]],"rows":6}'
    assert a!=environment_id(6,6,[[1,2]])
    assert a!=environment_id(7,6,[[1,2],[0,3]])

def test_base_id_is_stable_and_traceable():
    assert base_sample_id('commit','file.json',4)==base_sample_id('commit','file.json',4)
    assert base_sample_id('commit','file.json',4)!=base_sample_id('commit','file.json',5)
