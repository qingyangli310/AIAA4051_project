# Dataset Candidate 输入模式实现审计

## 依据的代码与分支

- 仓库：`qingyangli310/AIAA4051_project`
- 新功能基线：`integration/project6-unified`，GitHub 分支头 `1168d45c4484871319d7e4a71f73976c482bfe06`；本次只基于该分支建立功能分支。
- 最新集成文档说明：`docs/PROJECT6_INTEGRATION_REPORT.md`、`docs/PROJECT6_UNIFIED_GUIDE.md`。
- C 的 `feature/c-repair-evaluation` (`6e3c5e9`) 和 `feature/c-repair-pipeline` (`e3927b6`) 已在集成分支的提交祖先中。
- B 的 `feature/b-verifier` (`64c49da`) 没有合入该集成分支；其适配器和接口证据仍单独保留。仓库另有后续 `codex/verification-high-quality-v1` 与 `codex/final-pipeline-integration-v2` 分支，本实现不改写或推送到它们。

## 现有接口观察

集成基线 `src/pipeline.py` 原先每次调用都会执行 `generator.generate(grid)`，随后调用 `verifier.verify(grid=..., plan=...)`；未通过时通过 `build_repair_request` 调用 `repairer.repair(**request)`，最后调用 `executor.execute(grid=..., plan=...)`。C 的 Repair schema 已能消费 `candidate_plan` 与三种 feedback source；因此本次没有修改 Repair 训练数据、模型或训练逻辑。

正式 ID/OOD manifests 的实际字段不是统一的 `candidate_plan`：目前冻结 JSONL 记录保存 `sample_id`、`base_sample_id`、`environment_id`、`grid` 和 `corrupted_plan`，ID/OOD 的区别由文件/实验条件标识，`corruption_type` 对 OOD 是两个错误类型的列表。其他 task-wrapped rows 可能将候选放在 `input.candidate_plan` 或 `input.corrupted_plan`。Loader 将它们映射为同一 `candidate_plan` 字段，未改动源数据。

## 新行为

- `candidate_source="generator"` 是默认值，继续原来的 Generator 行为。
- `candidate_source="dataset"` 要求显式传入 `candidate_record`；读取规范 `candidate_plan` 或冻结格式的 `corrupted_plan`，设置 `generation=None`，不会调用 Generator。
- `run_pipeline_comparison` 和 `run_pipeline_comparison_batch` 支持 direct、self-feedback、executor-feedback 三个条件。一个 dataset candidate 只读取一次；所有条件收到独立副本，防止某个模型组件就地改写后污染其他条件。返回 candidate 的 SHA-256 和 ID 对应关系。
- `load_candidate_records` 只输出允许作为实验输入的标识、grid、candidate 和 corruption 类型。它不传递 `original_plan`、`shortest_plan`、`executor_label` 等目标/标签字段，不写回输入文件。
- 原有 `run_pipeline` 与 `run_pipeline_batch` 调用形式保持兼容；批处理新增 Dataset 输入时强制要求逐行对应的 candidate record。

## B/C 兼容性边界

- Dataset source 解决的是“候选从哪里来”；它不自动把任意 B 模型包装成 Pipeline 接口。
- Pipeline 仍要求 `verifier.verify(grid, plan)` 和 `repairer.repair(environment_id, grid, candidate_plan, feedback_source, verifier_feedback)` 合同。C 的 `build_repair_request` 与 Repair schema 合同一致。
- B 的独立早期 `GridModel` 接口在交接文档中是 `generate_plan(grid)` / `verify_plan(grid, candidate_plan)`，并非统一 Pipeline 的 `generate` / `verify`。本次不改 B 模块；真实运行前，若所选 B adapter 没有统一 wrapper，应在运行配置层提供薄 adapter。该问题不阻塞 candidate loader 和 pipeline mode 的离线验证。
- direct 条件只执行初始候选，不调用 Verifier 或 Repair；self-feedback 由 Verifier 提供反馈；executor-feedback 由 canonical Executor 提供反馈。每种条件均共享同一初始候选与地图。

## 文件变更

- 修改 `src/pipeline.py`：新增 source dispatch、Dataset 模式、配对对照 API 和批量入口。
- 新增 `src/data/candidate_source.py`：将冻结 ID/OOD rows 规范化成安全 candidate 输入。
- 新增 `tests/test_dataset_candidate_mode.py`：测试 source 路由、等候选、ID/OOD 字段、ID映射及只读性。
- 新增 `examples/dataset_candidate_mode.py`：离线合成示例。
- 更新 `README.md`：记录用法和边界。
- 新增本审计报告。

## 验收状态

新增标准库 `unittest` 用例 7/7 通过；原有 `test_pipeline_smoke.py` 与 `test_smoke_integration.py` 的 13 个测试函数逐项调用通过；Python 编译检查、离线合成示例和 `git diff --check` 通过。全仓 pytest 回归没有运行：当前执行环境没有安装 pytest，且访问 PyPI 时 DNS 解析失败，无法安装依赖。因此不能把这次结果表述为全套回归通过。Mock 示例不作为真实模型效果证据；没有启动训练、API 推理或正式 Held-out Test。

## 运行方式

```bash
python -m unittest tests.test_dataset_candidate_mode -v
# 完整历史回归仍使用项目原定 pytest 环境：python -m pytest -q
python examples/dataset_candidate_mode.py
```

本分支不会合并 main，也不会更改冻结 dataset、manifest 或历史 results。
