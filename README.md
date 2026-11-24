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

冻结版本为 `project6-6x6-seed42-20261116`。共享 Train/Dev 分别有 17,510 和 2,517 条基础记录。Part 1 正式训练子集总计 4,000 条（Generation 1,000、Verification 1,500、Repair 1,500）。冻结 ID/OOD 条件有 162 组配对。

## 使用冻结 Dataset Candidate 运行配对对照

`src.pipeline.run_pipeline` 保留默认的 `candidate_source="generator"`。如要在正式 ID/OOD 对照中复用冻结 candidate，设置 `candidate_source="dataset"` 并传入 `candidate_record`。Dataset 模式不会调用 Generator。当前冻结 ID/OOD manifest 把初始候选存为 `corrupted_plan`；适配器也接受规范字段 `candidate_plan` 和任务包装字段 `input.candidate_plan` / `input.corrupted_plan`。

```python
from src.data.candidate_source import load_candidate_records
from src.pipeline import run_pipeline_comparison_batch

id_rows = load_candidate_records(
    "data/manifests/id_single_error.jsonl", condition="id_single_error"
)
ood_rows = load_candidate_records(
    "data/manifests/ood_double_error.jsonl", condition="ood_double_error"
)
# generator / verifier / repairer / executor 由实验配置构造。
# 比较函数会把每条冻结候选原样用于 direct、self_feedback、executor_feedback。
id_results = run_pipeline_comparison_batch(
    id_rows, generator, verifier, repairer, executor, candidate_source="dataset"
)
```

Loader 输出只包含 `sample_id`、`base_sample_id`、`environment_id`、condition、grid、candidate_plan 和 corruption 标签；`original_plan`、`shortest_plan`、repair target、executor label 等监督字段不会被带入 Pipeline。`run_pipeline_comparison_batch` 不写实验文件；正式运行器仍应把结果写到新的 run 目录并记录数据与配置哈希。

可运行离线示例（合成数据、Mock Repair、无模型/API 调用）：

```bash
python examples/dataset_candidate_mode.py
python -m unittest tests.test_dataset_candidate_mode -v
```

示例和单元测试用于验证接口及公平性，不代表 B/C checkpoint 的真实 ID/OOD 评估结果。正式评估前应在预定协议下运行，不用 Held-out Test 调参。
