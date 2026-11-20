# Project 6 结构、接口与运行指南

```text
Project 6
├── configs/                   实验配置
├── requirements-project6.txt Python 模型依赖版本
├── src/
│   ├── data/                  PPNL Loader、环境 ID、冻结数据与 JSONL
│   ├── executor/              确定性路径执行与错误标签
│   ├── models/                Part 1 提示和输出解析
│   ├── part1/                 任务数据、基线与评分
│   ├── verifier/              Verification 模块预留目录
│   ├── repair/                Repair 数据
│   ├── pipeline.py            Generation → Verification → Repair → Execute
│   ├── evaluation/            指标
│   └── smoke_integration.py   联调组件
├── data/
│   ├── manifests/             冻结版本、来源、环境划分与配对
│   ├── processed/nov14/       共享 Train/Dev
│   ├── part1/                 已发布的 Part 1 任务数据
│   └── samples/               Pilot、配对条件与历史样本
├── scripts/                   构建、审计、训练、推理与集成入口
├── tests/                     A/C 和 Part 1 回归测试
├── generation_trace_v9/       B 的独立 Trace 实验
├── generation_models_v5/      v9 复用的模型资产工具
├── artifacts/                 已归档 checkpoint 与复现输入
├── results/                   实验结果与日志
└── docs/                      协议、交接、运行指南和版本索引
```

## 接口与数据流

| 边界 | 输入 | 输出和约定 |
|---|---|---|
| A Loader | PPNL 文件 | `records` 和逐条 `issues`。保留 `source_file`、`source_index`。 |
| A Executor | `execute_plan(grid, plan)`，grid 包含尺寸、起终点与障碍物 | `valid` 表示符合执行规则，`success` 表示执行完且到达目标。返回轨迹、已执行步数、首错位置与原因。 |
| B Generation | 网格 | 保留 `raw_output`、`plan`、`parse_success`。Trace 版本经 `parse_trace(raw_output, grid)` 提取动作。 |
| B Verification | 网格和候选计划 | `valid` 或 `invalid \| step=N \| reason=TYPE`。结构化输出保留解析状态和原始文本。 |
| C Repair | 网格、候选计划、反馈来源与内容 | 完整 `repaired_plan`、原始输出及解析状态。由真实 Executor 评分。 |

坐标为 `[row,col]`。四类标签为 `obstacle_collision`、`out_of_bounds`、`invalid_action_token`、`goal_not_reached`。动作错误为 0-based，未到终点使用 `len(plan)`。Verification 的任务有效性取 Executor 的 `success`，执行可行性保留为 `executor_valid`。C 的 adapter 已实现这层语义转换。解析失败应单独记录，空预测字段不能当作有效的模型诊断。

## 运行

在仓库根目录运行。开发测试可用 Python 3.12 和 pytest 8.4.2。GPU 协议记录 Python 3.11.9、PyTorch 2.9.1 / cu128。模型依赖按 `requirements-project6.txt` 安装，包括 Transformers 4.57.6、Accelerate 1.12.0、Datasets 4.4.2。历史 CUDA 结果实际运行于 PyTorch 2.6.0+cu124，复现时以各自的运行报告为准。

```bash
python scripts/check_environment.py
python scripts/demo_executor_and_leakage.py
python scripts/run_c_day15_16_smoke.py
python scripts/run_part1_symbolic.py --help
python scripts/evaluate_part1_checkpoint.py --help
python scripts/evaluate_generation_repair_cascade.py --help
```

全量冻结审计需要本地固定版本 PPNL 和三个完整 processed split。共享数据审计也要求本地完整 Held-out 文件校验。普通 clone 可以运行演示及支持 clean-clone 的测试。

| 方法 | 运行入口 | 输入 → 输出 / 评价 |
|---|---|---|
| Direct Generation | `evaluate_part1_checkpoint.py --task generation` | 任务 JSONL + Generation checkpoint → 预测、可解析性、Success、Feasibility、Optimality。具体训练与匹配比较见 Part 1 runbook。 |
| Self-feedback | `src/pipeline.py` 的 `run_pipeline(..., feedback_source="self_feedback")`；现有复现入口是 `run_c_day15_16_smoke.py` | 同一候选路径 + 学习式 Verifier 诊断 → Repair → 最终 Executor 结果。当前 smoke 使用占位模型。 |
| Executor-feedback | `evaluate_generation_repair_cascade.py --predictions ... --repair-checkpoint ... --output ... --repair-output-format trace_only` | v9 候选 JSONL + Repair v3 权重；对失败路径提供真实首次错误 → Dev cascade 报告、Repair 成功与 E2E 成功。 |

所有对照固定候选与样本 ID。Verification 记录 Macro-F1、首错步骤与原因准确率，Repair 记录成功率和 E2E 成功率。正式配对从同一 `base_sample_id` 关联 ID/OOD。对一个正式预算配置，Train 三任务合计最多 4,000；各阶段的 Dev 诊断与替换模型应分别记录。

- [Part 1 配置、数据与训练命令](PART1_IMPLEMENTATION_AND_RUNBOOK.md)
- [Repair v3 训练和 Dev cascade](REPAIR_V3_RUNBOOK.md)
- [v9 Trace 训练与复现](../generation_trace_v9/README.md)
