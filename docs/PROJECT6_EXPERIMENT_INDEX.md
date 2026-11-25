# 实验进度与版本索引

| 分类 | 版本 | 证据与适用范围 |
|---|---|---|
| 数据冻结 | A `3d5f188` | [冻结 Manifest](../data/manifests/frozen_dataset_20261116.json)、[审计](../results/data_freeze_audit_20261116.json)、[Train/Dev 交接](MEMBER_A_20261117_DATA_HANDOFF.md)。162 个 ID/OOD 配对，1 条异常。 |
| 正式 Part 1 历史基线 | `93d260d` 实际 CUDA 运行；后续修订至 `7abc640` | [原实验报告](../results/part1_cuda_20261117.md)。匹配 Generation ID 上 T5-small 0/162、BART-base 2/162。后续代码默认值不能当成这次运行的参数。 |
| Dev 诊断 | B v9 `347a10e` | [归档](../artifacts/generation_trace_v9/README.md)、[运行报告](../results/generation_trace_v9/run_report.json)。Flan-T5-base，Train2000 / Dev100，66/100 本地路径诊断，保留最佳 epoch 3 checkpoint 和 Git LFS 指针。v7/v8 原预测与报告保留在同一结果目录。 |
| Dev 诊断 | Repair v3 `dfd42a4` | [结果](../results/repair_v3/README.md)。纯 Generation 66/100，no-feedback Repair 78/100，Executor-feedback 与 shuffled-feedback 都是 80/100。相同结果限制了对具体反馈内容的因果解释。旧 scorer 报告保留，Trace 评分以 `dev_rescored_report.json` 为准。Repair 权重据交接存于团队 CUDA host。 |
| 正式数据候选 | Repair high-quality v2 | [数据说明](../data/repair_high_quality_v2/README.md) 与 `manifest.json`。Train 1000 行来自 500 个不同 base/environment，每个 candidate 保留 no-feedback/executor-feedback 配对；Dev 50 行来自 50 个不同 base/environment。与 Generator v9、Verification high-quality v1 的 base ID 重叠均为 0。v1 checkpoint 不适用于 v2，需重新训练。 |
| Dev 选择 | Repair high-quality v2 Flan-T5-base | [结果](../results/repair_flan_t5_base_high_quality_v2/README.md)。匹配的 2/3 epoch 比较分别为 34/50 与 38/50 Repair Success；选择 3 epoch checkpoint，权重与训练/Dev 数据哈希均已归档。尚未运行冻结 ID/OOD 或完整 pipeline。 |
| 三模型 Dev 联调 | Flan-T5-base integrated v2 | [结果](../results/full_pipeline_dev_flan_v2/README.md)。100/100 行完成、0 异常；Direct 19%，no-feedback 44%，self-feedback 28%，executor-feedback 43%，shuffled-feedback 44%。技术集成完成，但当前结果不支持 feedback-content improvement。 |
| 三模型 Dev 联调 | Flan-T5-base matched-prompt v3 | [结果](../results/full_pipeline_dev_flan_v3_correct_prompts/README.md)。修复 Generator 与 Repair 的 train-inference prompt mismatch 后，100/100 行完成、0 异常；Direct 66%，no-feedback 67%，self-feedback 66%，executor-feedback 67%，shuffled-feedback 67%。三种 Repair 条件在全部 34 个失败候选上输出完全相同，仍不支持 feedback-content improvement。v2 结果已标记为 superseded。 |
| Repair 改进方法 | feedback-localized suffix v4 | [运行说明](REPAIR_V4_RUNBOOK.md)。已完成合规数据与 pipeline 支持：1,000 Train 视图、50 个唯一 Dev 候选、Train/Dev 环境零重叠；no-feedback 与 executor-feedback 使用独立 matched checkpoint，并支持 correct-vs-shuffled focused ablation。当前状态为待 GPU 训练，尚无模型效果数字。 |
| 联调 Pilot | C `6e3c5e9` | [40 ID × 三种条件](MEMBER_C_20261115_16_HANDOFF.md)。使用真实 Executor 与 Mock 模型。 |
| 历史联调 | B `64c49da` | [Day 13](member_b_day13_delivery.md)。8 条往返样例和历史日志，由原分支复现。 |
| 历史 Pilot | A legacy A-4 / Nov14 | `data/samples/legacy_a4_pilot/` 和 `data/samples/nov14_pilot/`。用途和限制见对应 manifest 与交接。 |

v9 权重需要 Git LFS，下载后按归档 README 校验 SHA-256。项目分支保留已发布的配对测试条件。完整 protected processed Held-out 和官方原始 PPNL 需要在授权位置访问。

正式三组反馈对照仍需使用统一候选和配对样本运行，并记录模型 revision、数据哈希、配置、解析状态以及逐样本结果。no-feedback Repair 和 Direct Generation 是两个实验条件，报告时分别命名。v9 的 2,000 条 Generation 数据与原 Part 1 的 1,000 条配置属于不同实验，不能与原 Verification/Repair 预算直接相加。

## Flan-T5 Repair and held-out status

The training entry point now recognizes pinned `flan-t5-small` and
`flan-t5-base` model keys. The reproducible Development comparison and frozen
Repair ID/OOD commands are documented in
[`FLAN_T5_REPAIR_RUNBOOK.md`](FLAN_T5_REPAIR_RUNBOOK.md).

The paired 162-row single-error and double-error datasets have been built and
audited. Historical BART-base Repair ID/OOD summary numbers and symbolic BFS
acceptance exist, but the current T5 Repair improvements and Flan-T5 Generator
v9 have not been evaluated as a locked same-family language-model pipeline on
the frozen ID/OOD conditions. Therefore no Flan-T5 Repair ID/OOD result is
currently available.

The completed Flan-T5-base high-quality Repair Development run selected the
two-epoch checkpoint: 68/90 Repair success, 84/90 parse count, 67/90 exact
match and mean optimality 1.0. The three-epoch comparison reached 65/90 and
showed higher Development loss. These are Repair-only Development results, not
frozen ID/OOD or end-to-end cascade evidence.
