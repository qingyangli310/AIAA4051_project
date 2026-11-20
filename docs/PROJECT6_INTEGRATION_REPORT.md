# Project 6 整合报告

## 分支审计与接入

| 来源 | 审计时提交 | 处理 |
|---|---|---|
| main | `83e8927` | 团队协议、日程及早期集成历史。 |
| A Executor | `3d5f188` | 合入冻结资料、共享 Train/Dev、配对条件和交接审计。 |
| B Verifier | `64c49da` | 保留早期交接、联调样本、原始报告。原代码和专用测试在该提交可复现。 |
| B Generation Trace v9 | `347a10e` | 合入独立代码、Dev 实验、输入和 checkpoint 引用。 |
| C Repair Pipeline | `e3927b6` | 已包含在 main 历史。 |
| C Repair Evaluation | `6e3c5e9` | 由 Part 1 继承的 adapter、指标和 120 条 smoke 日志。 |
| Part 1 | `7abc640` | 由 Repair v3 继承任务数据、CPU/GPU 入口与正式基线。 |
| Repair v3 | `dfd42a4` | 整合基线，保留 Trace 评分、Dev cascade 和相关报告。 |

冻结数据、成员原始分支、实验数据及历史结果保留。整合相对 Repair v3 基线新增的训练输入为 B 已发布的 v9 Train/Dev 归档和 B Day 13 联调样本。已有 ID/OOD 测试制品继续由原 Git 对象引用。根 README 统一入口，整合前的 Repair v3 根说明见 [历史 README](PROJECT6_PRE_INTEGRATION_README.md)。

## 冲突与后续工作

- B Day 13 与 Part 1 同时实现 `src/models/{prompts,output_parser,__init__}.py`，直接接入触发 Git 冲突。当前运行路径使用 Part 1 实现，B Day 13 的 GridModel API、专用脚本与测试仍在原提交。若需接入，先用 adapter 映射 `generate_plan/verify_plan` 到 C 管线的 `generate/verify`，独立验证格式失败和字段保留，再选定共同 prompt/parser。
- B v9 输出 Trace，Part 1 存有传统动作格式和 Trace 解析。Repair v3 cascade 用专门的 Trace 输出选项。以后统一运行器时，应在配置声明输出格式并核验原始预测、解析结果和 A Executor 标签。
- Part 1 的正式预算配置与 v9 Dev 诊断预算不同。冻结协议沿用 Repair v3 上游内容。正式对照前由团队确认共用模型、训练预算分配、Dev 选择和冻结 ID/OOD 比较设置。
- 真实 Self-feedback 三组对照仍待接通并运行。Repair v3 正确反馈和乱序反馈在 Dev 都得到 80/100，待用冻结条件检验反馈内容的贡献。
- Git LFS 指针已继承。当前环境没有 Git LFS 客户端，v9 权重加载未验证。Repair v3 权重由原报告指向团队 CUDA host。GPU 训练、推理和 CUDA 环境未在本次运行。

## 验证

Python 3.12、pytest 8.4.2、PyYAML 6.0.3 下全套整合分支测试 80 项通过。A Loader/Executor、B/Part 1 prompt、parser、baseline，以及 C pipeline、Repair data、metrics 和 smoke 组件导入通过。真实 Executor 演示验证三对环境交集均为 0、ID/OOD 162 对。Part 1 symbolic acceptance 为通过，结果写入临时路径。
