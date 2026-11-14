from src.data.build_nov14_dataset import shortest_path,nonoptimal_plan
from src.executor.executor import execute_plan

def test_bfs_shortest_and_nonoptimal_executor_checked():
    grid={'environment_id':'e','rows':6,'cols':6,'start':[1,1],'goal':[4,4],'obstacles':[[2,2]]}
    sp=shortest_path(grid);result=execute_plan(grid,sp)
    assert result['valid'] and result['success'] and len(sp)==6
    detour=nonoptimal_plan(grid,sp);checked=execute_plan(grid,detour)
    assert checked['valid'] and checked['success'] and len(detour)>len(sp)
