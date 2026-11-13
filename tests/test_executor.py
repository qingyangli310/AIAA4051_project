import pytest
from src.executor.executor import execute_plan, Executor
from src.pipeline import run_pipeline
from src.mock_components import MockGenerator, MockVerifier, MockRepairer

GRID={'rows':3,'cols':3,'start':[1,0],'goal':[1,2],'obstacles':[[0,0]],'environment_id':'toy'}
@pytest.mark.parametrize('plan,reason,step',[
    (['up'],'obstacle_collision',0),(['left'],'out_of_bounds',0),
    (['right','jump'],'invalid_action_token',1),(['right'],'goal_not_reached',1),([], 'goal_not_reached',0)])
def test_failures(plan,reason,step):
    result=execute_plan(GRID,plan)
    assert result['reason']==reason and result['first_error_step']==step
    assert not result['success']

def test_success_and_compatibility():
    assert execute_plan(GRID,['right','right'])['success']
    assert execute_plan({**GRID,'grid_size':[3,3]},'right right')['success']
    result=execute_plan(GRID,['right','jump','right'])
    assert result['executed_steps']==1 and result['trace']==[[1,0],[1,1]]
    assert Executor().execute(GRID,[])==execute_plan(GRID,[])

def test_c_pipeline_real_executor():
    g={'environment_id':'toy','rows':3,'cols':3,'start':[0,0],'goal':[2,2],'obstacles':[[0,2]]}
    result=run_pipeline(g,MockGenerator(),MockVerifier(),MockRepairer(),Executor())
    assert result['final_result']['success'] is True
    assert result['final_result']['final_position']==[2,2]


def test_normalization():
    assert execute_plan(GRID, " RIGHT  right ")["success"]
    assert execute_plan(GRID, [" Right ", "RIGHT"])["success"]
