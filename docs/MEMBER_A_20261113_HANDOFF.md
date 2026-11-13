# 成员 A：2026-11-13 交接报告

## 仓库与操作范围

- 团队 origin：`https://github.com/qingyangli310/AIAA4051_project.git`
- 从 main 的 `83e8927`（详细时间表）创建 `feature/a-executor`。无 commit、无 push，main 引用未变。
- 参考仓库 origin：`https://github.com/hermit354/AIAA4051_project.git`，HEAD `2278aa5`；其原有未提交 A-4 文件保持原状。
- 官方 origin：`https://github.com/MohamedAghzal/llms-as-path-planners.git`，HEAD `5fd0376fc50ae9860bc986085165ebac5675fd44`，检查前后工作树均干净，只读。
- 阅读并采用团队 v1.2 Frozen 协议及 11月13日至16日按人时间表。未修改 B/C 模块、模型或配置；未训练、未提交原始 PPNL、未运行14日以后的正式构造/实验。

## 修改和新增文件

- `src/executor/executor.py`：从 A-4 复用真实执行器，兼容协议 `rows/cols` 和历史 `grid_size`；公开 ERROR_TYPES；提供 C Pipeline 使用的 `Executor.execute` adapter。动作按冻结协议去空格、转小写，非法 token 不猜测修正。
- `src/data/executor.py`：保留 A-4 导入路径的兼容层。
- `src/data/load_ppnl.py`：严格字段、矩阵/标记/动作字段类型检查；转换坐标及计划；执行真实 Executor 并核对 solution_coordinates；逐源文件和源索引记录不可转换事项。公开 `FILES`、`convert_record`、`load_ppnl`。
- `src/data/project6_data.py`、`generate_phase0.py`、`ppnl_io.py`：复用已有 A-4 功能和 BFS/I/O 依赖，没有重写。标记为历史方案，历史 CLI 需显式 `--legacy-a4`，不能当作 v1.2 正式数据生成器。
- `tests/test_executor.py`、`test_load_ppnl.py`：新增协议、7文件、异常、空数据、四错误、空路径、0-based、C pipeline adapter 等测试。
- `tests/test_project6_data.py`：保留 A-4 五项 regression；仅移除对参考仓库旧 evaluator.normalize_record 的导入和断言，其余生成、repair、provenance、上限、复现测试保留。
- `scripts/audit_ppnl.py`：七文件的完整核验及少量真实样例导出命令。
- `data/raw/README.md`、README、.gitignore：来源、版本、命令、状态；允许跟踪 raw README，但继续忽略原始数据文件。
- `results/ppnl_source_audit_20261113.json`：文件清单、计数、SHA-256、逐记录异常/不可转换事项。
- `data/samples/ppnl_integration_20261113.jsonl`：7条真实官方转换样例，各来自一个指定文件；包含 environment_id、source_file/index、original_plan 和 executor_result。
- `data/samples/legacy_a4_pilot/*`：旧 A-4 五个文件，逐字节比较与参考工作区原件一致；合计96条，没有重新生成旧Pilot。

## 官方文件核验

7/7文件存在、7/7解析成功，所有37,080条必需字段齐全。

| 文件 | 条数 | 可执行转换 | 显式不可转换 |
|---|---:|---:|---:|
| 1_train_set_6x6_samples.json | 16032 | 15989 | 43 |
| 1dev_set_6x6_samples.json | 2004 | 1999 | 5 |
| 1_goals_test_seen_6x6_samples.json | 2004 | 2001 | 3 |
| 1goals_unseen_6x6_samples.json | 5040 | 5027 | 13 |
| 1_goals_test_unseen_5x5_samples.json | 3750 | 3689 | 61 |
| 1_goals_test_unseen_7x7_samples.json | 3750 | 3750 | 0 |
| 1_goals_test_unseen_6x6more_obstacles_samples.json | 4500 | 4114 | 386 |

合计36,569条官方计划通过执行器且坐标轨迹一致。511条均为官方 `Goal not reachable`，不是 JSON 损坏或字段缺失；不猜测为空动作、不伪造修复、不默默丢弃。其转换停止，文件/索引均在 issues 中保留，调用者必须检查 issues。
本次没有将36,569条转换结果写成数据集；只导出7条联调样例。

## 实际测试

Python 3.12.16 / pytest 8.4.2（隔离CPU环境）：

```bash
uv run --python 3.12 --with pytest==8.4.2 python -m pytest -q
uv run --python 3.12 --with pytest==8.4.2 python -m pytest tests/test_executor.py -v
uv run --python 3.12 --with pytest==8.4.2 python -m pytest tests/test_load_ppnl.py -v
uv run --python 3.12 --with pytest==8.4.2 python -m pytest tests/test_project6_data.py -v
```

全仓33 passed；单独Executor 8 passed、Loader 10 passed、A-4 5 passed。实际官方文件测试已运行（未跳过）；其余10项为团队原有C测试。
首次 pytest 不存在，随后用临时uv环境安装；最终按团队要求固定8.4.2复核，没有升级项目依赖。

真实演示：`1_train_set_6x6_samples.json` source_index=0，起点[1,4]，终点[2,1]，计划 left,left,left,down；结果 valid=true、success=true、executed_steps=4、final_position=[2,1]。
C原有Mock Pipeline不变；另加测试注入真实 `Executor()` 后，mock repair路径经真实执行器成功到达目标。

## 协议差异与未解决事项

1. 当前团队v1.2要求 seed=42、70/10/20 Train/Dev/Held-out、同一Held-out配对ID/OOD。旧A-4是seed17、5x5 toy、互斥ID/OOD；仅作历史回归，不是正式数据。正式划分与配对构造属于14/15日任务，本次未执行。
2. 511条不可达官方样本已由负责人决定在后续阶段单独写入 excluded_manifest，不纳入正常训练及主测试数据。本次只记录决策，保留原 issues，不提前构建 excluded_manifest 或正式数据。
3. A-4 provenance目前只验证顺序重放及不同标签位置。该历史逻辑不等于严格稳定的原始token逻辑位置身份；插入/删除导致索引移动、第二错误可能被截断遮蔽等问题，需在15日双错误正式实现时进一步处理，当前未宣称正式双错误条件已验收。
4. 官方原始获取时间没有可靠日志；本次核验日期为2026-11-13，已提供精确commit、文件校验值及可重新获取命令，没有伪造首次获取日期。
5. C mock对goal_not_reached返回valid=false；A真实执行器依协议Feasibility定义返回valid=true、success=false。未擅自修改C代码。正式接入时请使用success/reason区分未完成与移动违规；负责人已确认 valid 表示是否违反执行规则、success 表示是否到达终点；C mock 的既有行为留待其负责人后续对齐，不在本次修改。
6. 本机执行的是3.12 CPU测试，未验证协议指定的Ubuntu/Python3.11.9/RTX4090训练环境，也未做GPU测试。

## B/C立即可用的接口

```python
import json
from src.executor.executor import execute_plan, Executor
from src.data.load_ppnl import load_ppnl

with open('data/samples/ppnl_integration_20261113.jsonl') as handle:
    sample = json.loads(next(handle))
result = execute_plan(sample, sample['plan'])
# C调用run_pipeline(..., executor=Executor())，无需修改Pipeline实现。
```

交接人工检查：审阅A分支差异和7条样例；后续按14日计划建立正式环境划分，再按15日计划构建配对条件。首次整合阶段未提交；本次最终验收已获得负责人明确授权，按指定 commit 信息提交并正常推送 feature/a-executor，不修改 main。


## 最终验收：后续执行决策记录

以下三项为负责人在11月13日最终验收阶段明确的后续执行规则，仅记录决策，不修改冻结协议 v1.2，不提前执行14/15日任务：

1. **不可达样本**：511条 `Goal not reachable` 单独记录到 `excluded_manifest`，不纳入正常训练及主测试数据。现有逐源索引 issues 保留；正式 excluded_manifest 由后续数据构建阶段产生。
2. **valid / success**：`valid` 表示是否违反执行规则；`success` 表示是否到达终点。`goal_not_reached` 可以 valid=true、success=false。本次已通过的真实 Executor 接口保持不变。
3. **OOD**：要求两次独立、可检测、不可抵消的错误注入，并保留完整 provenance。旧A-4回放证据不等于该正式要求已验收；稳定逻辑位置身份与退化排查在15日正式构建阶段执行。

本次提交范围仅成员A的实现、测试、来源核验元数据、7条真实转换样例、96条历史Pilot和相关说明。完整PPNL原始数据不在提交范围内。

最终提交前复核：`git diff --check` 通过；固定 pytest 8.4.2 的全仓测试结果为 `33 passed in 0.69s`，无跳过。远程目标分支在提交前不存在；远程 main 为 `83e8927c09ae6f779e627ff0bec041a1c218b546`。审核范围内未发现B/C模块修改，历史Pilot逐字节不变；待提交数据仅为7条转换联调样例、96条历史Pilot及核验元数据。
