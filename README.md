# AIAA4051 Project 6 — Grid path planning with feedback

研究比较 **Direct Generation、Self-feedback 和 Executor-feedback**。在相同地图和初始候选路径上，观察反馈与 Repair 对最终成功率的影响。Generator 是对照实验的共同输入。模型选择使用 Dev，正式测试使用冻结的 ID/OOD 条件。

| 成员 | 模块 | 主要入口 |
|---|---|---|
| A | PPNL 读取、冻结划分、错误注入、确定性 Executor | `src/data/load_ppnl.py`、`src/executor/executor.py` |
| B | Generation、Verification、提示与解析 | `src/models/`、`src/part1/baselines.py`、`generation_trace_v9/` |
| C | Repair、管线与指标 | `src/repair/`、`src/pipeline.py`、`src/evaluation/metrics.py` |

- [项目结构、接口和运行指南](docs/PROJECT6_UNIFIED_GUIDE.md)
- [实验进度、结果与 checkpoint 索引](docs/PROJECT6_EXPERIMENT_INDEX.md)
- [分支整合报告](docs/PROJECT6_INTEGRATION_REPORT.md)
- [原团队协议](docs/Project6_团队统一协议.md)

## 快速验证

在仓库根目录运行。从学校 GPU 环境运行模型前，请按运行指南检查 Python、PyTorch 和 CUDA 版本。

```bash
uv run --python 3.12 --with pytest==8.4.2 --with pyyaml==6.0.3 python -m pytest -q
python scripts/demo_executor_and_leakage.py
```

统一的真实模型比较入口为 `scripts/run_full_pipeline.py`。它对每个网格只运行
一次 Generator，并让 Direct、Self-feedback、Executor-feedback 和可选的
no-feedback / shuffled-feedback 条件共享同一候选路径。正式 Test 运行必须提供
已锁定的 `configs/final_pipeline.yaml`，防止在 primary model 未确认时消耗冻结测试集。

- [统一 Pipeline 运行与验收](docs/FINAL_PIPELINE_RUNBOOK.md)
- [最终报告内容骨架](report/REPORT_OUTLINE.md)
- [展示材料内容骨架](poster/PRESENTATION_OUTLINE.md)

冻结版本为 `project6-6x6-seed42-20261116`。共享 Train/Dev 分别有 17,510 和 2,517 条基础记录。Part 1 历史训练子集总计 4,000 条（Generation 1,000、Verification 1,500、Repair 1,500）。最终数据预算使用 Generation 2,000、Verification 1,000、Repair 1,000；Dev 使用 100 / 50 / 50。历史数据与最终数据不得混成一次正式实验。冻结 ID/OOD 条件有 162 组配对。

### 正式结果只读审计

从仓库根目录执行：

```bash
python scripts/audit_final_pipeline_results.py
python scripts/analyze_final_pipeline_pairs.py
```

脚本读取冻结 Manifest 与正式 v4 ID/OOD 逐样本记录，将审计和配对统计输出到 stdout。解读见 `report/REPORT_OUTLINE.md` 和 `docs/MEMBER_A_FINAL_AUDIT_HANDOFF.md`。
