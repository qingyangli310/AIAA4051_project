# Project 6 Final Report Outline

该文件是最终报告的内容骨架。方括号字段必须由冻结实验结果替换，不能填写 Mock、
Symbolic 或仅用于选择模型的 Dev 数字。

## Abstract

[研究问题、方法、冻结 ID/OOD 主要结果和结论]

## 1 Introduction

[网格路径规划问题、Generate Verify Repair 动机、研究问题和贡献]

## 2 Related Work

[PPNL、Chain of Thought、ReAct、Self Refine、Reflexion、神经符号验证、T5 Flan T5 BART]

## 3 Methodology

### 3.1 Grid and Executor

网格坐标使用零基的 `[row, col]`。动作 up/down 分别使行坐标减一/加一，left/right 分别使列坐标减一/加一。确定性 Executor 按顺序执行候选动作。从零开始记录首个非法动作（invalid_action_token）、越界（out_of_bounds）或碰撞（obstacle_collision），并在该步停止。若全部动作合法但最终位置不是目标，则记为 goal_not_reached，首错位置定义为 `len(plan)`。因此 `valid` 表示动作执行合法，`success` 还要求最终位置等于目标。Verification 任务有效性使用 `success`，管线另保留 Executor 的可行性。实现见 `src/executor/executor.py`。

### 3.2 Generation

[输入、输出、解析和 primary checkpoint]

### 3.3 Verification

[valid invalid、首错位置和错误类型]

### 3.4 Repair and Feedback Conditions

[Direct、No feedback、Self feedback、Executor feedback]

### 3.5 Improved Method

[冻结后的改进名称、与基础方法的唯一差异和理论动机]

## 4 Dataset and Experimental Setup

冻结版本 `project6-6x6-seed42-20261116` 采用 Seed 42，以网格尺寸与障碍布局确定环境 ID。Train、Dev、Held-out 按环境隔离，分别有 572、82、163 个环境。正式候选实验从 Held-out 使用 162 组可配对样本，ID 为单处修改，OOD 为两处修改。各对共享基础样本 ID 和完整地图。冻结上游文件的哈希见 `data/manifests/frozen_dataset_20261116.json`，正式任务输入的哈希见 `data/part1/manifest.json`。原 Part 1 训练预算合计 4,000 条，训练子集与冻结基础数据池分别统计。

正式 Frozen Test 使用 `results/final_pipeline_v4_id` 和 `results/final_pipeline_v4_ood`。每个样本固定同一初始候选，比较 direct、no-feedback、self-feedback、executor-feedback 和 shuffled executor-feedback。Dev 用于模型选择与诊断，Supplemental Diagnostics 属于独立实验记录，不进入下述配对统计。[待补充：所选 checkpoint 对应训练配置与硬件的统一溯源表。]

## 5 Results

### 5.1 Two Model Baseline Comparison

[同数据与匹配预算的两个模型 Dev 选择依据]

### 5.2 Frozen ID Results

[Success、Feasibility、Optimality、Macro F1、首错定位、Repair Success、E2E]

### 5.3 Frozen OOD Results

[同上，并报告 ID OOD 差距]

### 5.4 Results by Failure Type

[obstacle、bounds、token、goal not reached]

### 5.5 正式结果的数据质量与配对统计

只读审计核验了两个正式目录各 162 条逐样本记录、配对地图、候选 SHA-256、输入文件哈希与各条件终态，未发现记录级运行异常。ID/OOD 中 self-feedback 的 Verifier 解析失败为 3/7 次。按 no-feedback、self-feedback、executor-feedback、shuffled executor-feedback 顺序，Repair 解析失败分别为 ID 8、15、22、25 次和 OOD 21、28、68、68 次。成功率以全部 162 个样本为分母，解析失败样本保留在记录与统计内。

| Split | A / B | 均成功 | 均失败 | 仅 A 成功 | 仅 B 成功 | A / B 成功率 | B−A（百分点） | 双侧精确 McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| ID | 无反馈 / 执行器反馈 | 69 | 44 | 29 | 20 | 60.49% / 54.94% | −5.56 | 0.25287 |
| ID | 无反馈 / 自反馈 | 39 | 55 | 59 | 9 | 60.49% / 29.63% | −30.86 | 3.91e−10 |
| ID | 执行器反馈 / 打乱反馈 | 72 | 57 | 17 | 16 | 54.94% / 54.32% | −0.62 | 1.0 |
| OOD | 无反馈 / 执行器反馈 | 41 | 63 | 43 | 15 | 51.85% / 34.57% | −17.28 | 0.000307 |
| OOD | 无反馈 / 自反馈 | 19 | 70 | 65 | 8 | 51.85% / 16.67% | −35.19 | 3.23e−12 |
| OOD | 执行器反馈 / 打乱反馈 | 56 | 106 | 0 | 0 | 34.57% / 34.57% | 0.00 | 不适用 |

统计源为各正式目录的 `records.jsonl`，计算见 `scripts/analyze_final_pipeline_pairs.py`。McNemar 基于同一样本的成败差异，只使用两组结果不一致的配对。以上 p 值未作多重比较校正，解释依赖样本对之间独立的假设。结果描述该冻结候选集的表现。自反馈条件包含 Verifier 的路由及解析行为，因此差异无法归因于反馈文本这一单一因素。

## 6 Ablation Study

[同一批样本、匹配配置、真实反馈与打乱反馈或最终选择的 focused ablation]

## 7 Error Analysis

[错误接受、首错定位错误、只修复第一个错误、新碰撞、解析失败和过长路径]

## 8 Repair Case Studies

### 8.1 Representative Successful Repair

[网格、候选、轨迹、反馈、修复、最终轨迹和解释]

### 8.2 Representative Failure

[同一字段，解释失败机制]

## 9 Limitations

实验仅覆盖 Seed 42 的单目标 6×6 网格，跨随机种子、模型规模和网格条件的泛化仍待验证。冻结 OOD 中打乱反馈实际改变内容的样本为 0/162，且两分支没有成败不一致的配对。该条件无法检验反馈内容的因果贡献。ID 中实际改变内容的样本为 125/162，整组比较包含未改变反馈的样本。补充诊断仅作为探索性证据。不同实验的硬件和 checkpoint 溯源需随运行记录核对。[待补充：多种子复验与可实际改变内容的预注册 OOD 干预。]

## 10 Conclusion

[仅陈述冻结结果支持的结论]

## References

[完整引用]
