# 最终 Pipeline 运行与验收

## 当前实现

`scripts/run_full_pipeline.py` 已连接以下真实接口：

1. 本地 seq2seq Generator checkpoint；
2. 严格 Trace 或动作解析；
3. 本地 seq2seq Verifier checkpoint；
4. 本地 seq2seq Repair checkpoint；
5. 项目 canonical deterministic Executor；
6. Success、Feasibility、Optimality、Verification Macro-F1、首错定位、Repair Success 和 End-to-End Success 汇总。

Generator 对每个网格只推理一次。所有实验条件复用同一候选，逐样本结果保存在
`records.jsonl`，汇总与数据哈希保存在 `report.json`。单样本异常会被记录，不会中断批量运行。

## 本地符号验收

该命令只验证数据流、结果结构和指标，不代表语言模型成绩。

```bash
python scripts/run_full_pipeline.py \
  --data data/part1/generation_id_test.jsonl \
  --output results/pipeline_symbolic_smoke \
  --split diagnostic \
  --limit 10 \
  --symbolic
```

## Dev 真实模型联调

所有模型修改、重训和消融决定必须在 Dev 上完成。

```bash
python scripts/run_full_pipeline.py \
  --data data/part1/generation_dev.jsonl \
  --output results/final_pipeline_dev \
  --split dev \
  --generator-checkpoint /path/to/generator \
  --verifier-checkpoint /path/to/verifier \
  --repair-checkpoint /path/to/repair \
  --generator-output-format trace_only \
  --repair-output-format trace_only \
  --conditions direct no_feedback self_feedback executor_feedback \
    shuffled_executor_feedback
```

Shuffled feedback 在相同候选长度的失败样本之间轮换，优先选择首错位置或错误类型
不同的反馈，同时确保错误步骤仍满足格式约束。每条记录保存反馈来源环境、是否真正
轮换以及反馈内容是否发生变化。未真正改变反馈内容的样本不应计入反馈依赖结论。

## 正式 Test 门槛

正式运行前必须编辑 `configs/final_pipeline.yaml`：

- `status` 改为 `locked`；
- 确认唯一 primary model family；
- Generation、Verification、Repair 填写同一 base family；
- 填写三个任务 checkpoint、revision 和 SHA-256；
- 保持训练样本总数不超过 4,000、Dev 不超过 200、每个子集不超过 3 epochs；
- 只根据 Dev 选择模型，不根据 Test 结果继续修改。

随后才能运行：

```bash
python scripts/run_full_pipeline.py \
  --data data/part1/generation_id_test.jsonl \
  --output results/final_pipeline_id \
  --split test \
  --confirm-frozen-test \
  --protocol-config configs/final_pipeline.yaml \
  --generator-checkpoint /path/to/generator \
  --verifier-checkpoint /path/to/verifier \
  --repair-checkpoint /path/to/repair
```

Direct Generation 只需要在 Held-out Test 网格运行一次。ID/OOD 的主要区别是
Verification 和 Repair 接收的单错误或双错误候选；不要把同一批 Direct 结果重复描述成
两个独立实验。

## 冻结 Candidate 输入模式

正式 ID/OOD Verification 与 Repair 对照必须使用冻结数据中的候选，而不是重新调用
Generator。运行器通过 `--candidate-source dataset` 明确启用该模式，并要求声明输入是
单错误 ID 还是双错误 OOD。Loader 只向模型暴露 grid 和 candidate，reference plan 仅以
长度标量参与最终 Optimality 评分。

开发期可先用 Symbolic 组件检查数据流，不消耗语言模型推理：

```bash
python scripts/run_full_pipeline.py \
  --data data/part1/repair_id_test.jsonl \
  --output results/pipeline_dataset_candidate_smoke \
  --split diagnostic \
  --limit 10 \
  --candidate-source dataset \
  --candidate-condition id_single_error \
  --conditions direct no_feedback self_feedback executor_feedback \
    shuffled_executor_feedback \
  --symbolic
```

锁定配置后的正式运行使用相同入口，并分别选择：

- `--candidate-condition id_single_error` 与冻结 ID 文件；
- `--candidate-condition ood_double_error` 与冻结 OOD 文件。

Dataset 模式不会加载或调用 Generator checkpoint，但仍需要真实 Verifier、Repair 和
canonical Executor。每条输出保存 `sample_id`、`base_sample_id`、candidate SHA-256 和
全部条件结果，供 ID/OOD 配对审计。

## 尚需外部资源完成的事项

- 项目经理确认 T5、Flan-T5 或 BART 中的最终 primary family；
- 将三个同 family 的任务 checkpoint 放在可访问位置；
- Repair 训练数据覆盖四类错误，而不只覆盖 obstacle collision；
- 在 GPU 环境完成 Dev 锁模，然后一次性运行冻结 ID/OOD；
- 汇总真实 GPU 时间、峰值显存、吞吐量与 checkpoint 哈希；
- 用正式结果填写报告和展示骨架。

这些事项不能通过 CPU 仓库修改代替，也不能使用 Mock 或 Symbolic 结果填写最终成绩。
