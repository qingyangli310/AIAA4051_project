# Project 6 团队统一协议

## 文档状态

| 项目 | 内容 |
|---|---|
| 文档版本 | v0.2 |
| 创建日期 | 2026-11-09 |
| 计划冻结日期 | 2026-11-12 |
| 适用成员 | 成员 A、成员 B、成员 C |
| 项目范围 | PPNL 单目标网格路径生成、验证与修复 |
| 当前状态 | 研究与工程协议已冻结，仅 T5/Flan-T5 checkpoint 家族待负责人确认 |

本协议用于统一三个人的数据格式、模型接口、实验规则、评价指标和协作方式。协议冻结后，A、B、C 可以分别开发不同模块，但所有模块必须遵守相同接口。

> 需要项目负责人确认的事项不得自行假设。特别是 T5 与 Flan-T5 checkpoint 不一致的问题，必须在正式训练前确认。

## 一、为什么现在就要统一

如果没有提前统一，可能出现以下问题：

- A、B、C 使用不同的网格坐标格式；
- executor 和 verifier 对错误步骤的编号不同；
- 同一种错误出现多个名称；
- B 的输出无法被 C 的 repair 模块读取；
- train/dev/test 出现同一网格，造成数据泄漏；
- 不同方法使用不同测试样本，无法公平比较；
- 实验结束后才修改指标定义，降低研究可信度；
- 最终代码无法复现报告中的数字。

因此，11月12日前需要冻结基础协议。后续如必须修改，应记录修改原因、日期、影响范围，并重新运行所有受影响实验。

## 二、项目目标和范围

### 2.1 核心研究问题

比较以下三种方法：

1. 直接生成路径；
2. 语言模型自验证后修复；
3. 使用符号执行器反馈后修复。

研究明确的验证反馈能否提高语言模型路径规划的成功率，以及单错误训练能否泛化到双错误计划。

### 2.2 必做范围

- 单目标环境；
- 6×6 网格作为训练和 ID 测试的主要环境；
- 正确路径和可行但非最优路径；
- 单错误训练数据；
- 单错误 ID 测试集；
- 两个独立错误的 OOD 测试集；
- 至少两个 baseline 模型；
- 一个 primary model；
- 一个改进方法；
- 一个消融实验。

### 2.3 暂不纳入主实验

- 多目标环境；
- 依赖 Gurobi 的流程；
- 大量额外随机种子；
- 第二个模型的完整改进和消融；
- 7×7 或高密度 6×6 扩展实验。

这些项目只有在必做实验完成且预算允许时才考虑。

## 三、角色和模块所有权

| 成员 | 主要职责 | 主要模块 | 复核人 |
|---|---|---|---|
| A | 数据、executor、错误生成、数据划分 | `executor/`、`data/` | C |
| B | generation、verification、模型训练 | `models/`、`verifier/` | A |
| C | repair、端到端管线、结果汇总 | `repair/`、`pipeline/`、`evaluation/` | B |

每个关键模块采用：

```text
1 人主写 + 1 人复核
```

模块负责人负责实现和文档；复核人负责测试接口、检查假设和确认结果可复现。

## 四、统一网格数据格式

统一使用 JSON/JSONL，建议格式：

```json
{
  "environment_id": "grid_0001",
  "rows": 6,
  "cols": 6,
  "start": [0, 0],
  "goal": [5, 5],
  "obstacles": [
    [0, 2],
    [1, 2],
    [3, 4]
  ]
}
```

统一规则：

- 坐标格式为 `[row, col]`；
- 左上角为 `[0, 0]`；
- `up` 表示 `row - 1`；
- `down` 表示 `row + 1`；
- `left` 表示 `col - 1`；
- `right` 表示 `col + 1`；
- `start` 和 `goal` 不得位于障碍物；
- 每个网格必须具有唯一且稳定的 `environment_id`；
- 同一网格在所有文件中必须保持相同 ID。

## 五、统一动作和步骤编号

### 5.1 合法动作

只允许：

```text
up
down
left
right
```

程序内部统一表示为字符串列表：

```python
["right", "down", "left"]
```

标准化规则：

- 去除首尾空格；
- 转换成小写；
- 按统一分隔符解析；
- 不属于合法动作集合的 token 标记为 `invalid_action_token`；
- 不静默猜测或替换非法动作。

### 5.2 编号约定

统一为：

```text
网格坐标：0-based
动作步骤：1-based
```

例如：

```text
计划：right, right, down
第一个 right：step=1
第二个 right：step=2
down：step=3
```

如果第二个动作撞到障碍物：

```text
invalid | step=2 | reason=obstacle_collision
```

`goal_not_reached` 定义为动作序列执行结束后的下一检查点，即 `step=len(plan)+1`。

## 六、统一错误类型

至少固定以下四类：

```text
obstacle_collision
out_of_bounds
invalid_action_token
goal_not_reached
```

禁止使用多个名称表示同一错误，例如同时使用 `hit_obstacle`、`collision` 和 `obstacle_error`。

在公共模块中统一定义：

```python
ERROR_TYPES = {
    "obstacle_collision",
    "out_of_bounds",
    "invalid_action_token",
    "goal_not_reached",
}
```

第一次失败后 executor 应停止执行，并记录第一次错误。完整执行轨迹可以保留用于错误分析，但 ground truth 标签以第一次失败为准。

## 七、统一模块接口

### 7.1 Executor 接口

```python
def execute_plan(grid, plan):
    """执行路径并返回确定性的真实结果。"""
```

统一返回：

```json
{
  "valid": false,
  "success": false,
  "first_error_step": 2,
  "reason": "obstacle_collision",
  "final_position": [0, 1],
  "executed_steps": 1,
  "trace": []
}
```

### 7.2 Generation 接口

```python
def generate_plan(grid):
    """根据网格生成候选路径。"""
```

统一返回：

```json
{
  "raw_output": "right, down, down, right",
  "plan": ["right", "down", "down", "right"],
  "parse_success": true
}
```

### 7.3 Verification 接口

```python
def verify_plan(grid, plan):
    """预测路径有效性、第一次错误步骤和错误类型。"""
```

有效路径统一表示为：

```json
{
  "valid": true,
  "first_error_step": null,
  "reason": null
}
```

无效路径统一表示为：

```json
{
  "valid": false,
  "first_error_step": 2,
  "reason": "obstacle_collision"
}
```

模型文本输出固定为：

```text
valid
```

或者：

```text
invalid | step=2 | reason=obstacle_collision
```

### 7.4 Repair 接口

```python
def repair_plan(grid, candidate_plan, feedback):
    """根据网格、候选计划和反馈生成完整修复路径。"""
```

输入：

```json
{
  "environment_id": "grid_0001",
  "grid": {
    "rows": 6,
    "cols": 6,
    "start": [0, 0],
    "goal": [5, 5],
    "obstacles": [[0, 2], [1, 2]]
  },
  "candidate_plan": ["right", "right", "down"],
  "verifier_feedback": {
    "valid": false,
    "first_error_step": 2,
    "reason": "obstacle_collision"
  }
}
```

输出：

```json
{
  "raw_output": "down, right, down, right",
  "repaired_plan": ["down", "right", "down", "right"],
  "parse_success": true
}
```

第一版统一规则：

- Repair 输出完整路径；
- 主 baseline 只允许一轮修复；
- 修复后必须重新交给 executor 检查；
- 无法解析的输出记为 repair failure；
- 不使用人工方式修改模型输出。

### 7.5 端到端接口

```python
def run_pipeline(grid, feedback_source):
    """运行完整 generate-verify-repair 管线。"""
```

`feedback_source` 至少支持：

```text
self
executor
```

## 八、数据构造和划分

### 8.1 划分原则

必须按照 `environment_id` 划分，而不是按照动作序列划分。

```text
train：单错误
dev：单错误
id_test：单错误
ood_test：两个独立错误
```

同一环境的任何路径不得跨越 train/dev/test。

统一保存：

```text
data/manifests/train.json
data/manifests/dev.json
data/manifests/id_test.json
data/manifests/ood_test.json
```

### 8.2 错误构造规则

支持：

- action substitution；
- action deletion；
- action insertion；
- premature stopping。

每个损坏样本必须保存：

```text
environment_id
source_plan_id
corruption_type
corruption_seed
corrupted_plan
executor_label
first_error_step
reason
```

双错误样本需要确认：

- 两个错误由两个独立操作产生；
- 两个错误没有互相抵消；
- 执行器标签经过重新计算；
- 记录两个错误的构造位置，但评价仍以第一次执行失败为主。

### 8.3 随机性

- 项目级随机种子固定为 `42`；
- 数据生成、训练和评价分别记录 seed；
- 随机种子写入配置和实验日志；
- 不根据测试结果选择更有利的 seed。

### 8.4 环境划分和样本预算

环境按照固定种子 `42` 打乱后，采用以下 environment-level 比例：

```text
train：70%
dev：10%
id_test：10%
ood_test：10%
```

如果官方环境数量或格式不允许精确满足比例，应使用最接近比例的整数划分，并在 manifest 中记录实际数量。同一 `environment_id` 只能属于一个 split。

训练样本总预算固定为：

| 任务 | 样本数 |
|---|---:|
| Generation | 1,000 |
| Verification | 1,500 |
| Repair | 1,500 |
| 合计 | 4,000 |

开发集最多使用200个样本；ID 单错误测试集和 OOD 双错误测试集各使用200个样本。如果合格样本不足，则使用全部合格样本并记录实际数量。

Verification 数据中的 `valid` 与 `invalid` 比例固定为 `1:1`。单错误 invalid 数据的四种 corruption 类型尽量均衡：

```text
action_substitution：25%
action_deletion：25%
action_insertion：25%
premature_stopping：25%
```

如果个别类型无法生成足量有效样本，允许采用最接近的比例，但不得为提高测试结果而事后调整。

## 九、模型和训练协议

### 9.1 唯一待外部确认的 checkpoint 家族

课程负责人只需确认第一个 baseline 使用普通 T5 还是 Flan-T5。团队已经固定第二个 baseline 为 BART-base：

```text
Baseline 1：{T5 或 Flan-T5}-small（待负责人确认）
Baseline 2：facebook/bart-base
```

课程材料的链接与模型名称存在 T5/Flan-T5 不一致。正式训练前必须由项目负责人确认准确 checkpoint。

如果负责人确认 Flan-T5，则使用：

```text
google/flan-t5-small
revision=ec9123e078f63797d82dbe0812dc176c7008fc29
```

BART-base 固定为：

```text
facebook/bart-base
revision=aadd2ab0ae0c8268c7c9693540e9904811f36177
```

### 9.2 模型比较规则

- 至少比较两个 baseline 模型；
- 使用相同训练、开发和 ID 测试数据；
- 使用匹配的数据量和训练预算；
- primary model 只能根据开发集结果选择；
- 第二个模型不需要重复完整改进和消融；
- 禁止根据测试集结果选择 primary model。

### 9.3 需要记录的训练参数

```text
checkpoint
tokenizer
max_input_length
max_output_length
batch_size
learning_rate
epochs
seed
decoding_method
beam_size
temperature
GPU 型号
训练时长
峰值显存
```

相同任务的对比实验必须使用相同解码设置。

### 9.4 固定 pilot 配置

```yaml
seed: 42
max_input_length: 512
max_output_length: 128
learning_rate: 5.0e-5
per_device_train_batch_size: 8
per_device_eval_batch_size: 8
gradient_accumulation_steps: 2
num_train_epochs: 3
optimizer: adamw_torch
weight_decay: 0.01
bf16: true
generation_num_beams: 1
do_sample: false
temperature: 1.0
```

如果 Flan-T5-base 或 BART-base 在 pilot 中显存不足，可以降低单设备 batch size，但应增加 `gradient_accumulation_steps`，尽量保持有效 batch size 为16。任何调整必须同时记录在配置和实验日志中。

### 9.5 改进方法和消融

主改进方法固定为：

```text
Structured State Tracking Verification
```

验证器在输出最终诊断前，显式跟踪每一步动作、当前位置和状态。

唯一必做消融固定为：

```text
移除逐步状态跟踪，只保留最终 valid/invalid、step 和 reason 输出。
```

主改进和消融只在 primary model 上运行，并同时评价 ID 单错误和 OOD 双错误测试集。

### 9.6 Repair 训练目标

- 训练目标优先使用官方最短路径；
- 同一个网格存在多条最短路径时，可以使用其中一条确定性选定的路径；
- 评价时，任何由 executor 判定成功的路径都算 repair success；
- 成功但不是最短路径的修复通过 Optimality 单独评价；
- generation、verification 和 repair 使用同一基础 checkpoint 家族，但分别保存任务 checkpoint。

## 十、必须比较的方法

### 10.1 Direct Generation

```text
网格 → 生成路径 → executor 评价
```

### 10.2 Self-Verification and Repair

```text
网格
→ 生成路径
→ 模型验证
→ 模型反馈
→ Repair
→ executor 评价
```

### 10.3 Executor Feedback and Repair

```text
网格
→ 生成路径
→ executor 诊断
→ 正确反馈
→ Repair
→ executor 评价
```

三种方法必须使用：

- 相同测试网格；
- 相同生成模型；
- 相同生成解码配置；
- 相同评价脚本；
- 相同样本数量。

## 十一、评价指标定义

### 11.1 Success Rate

```text
成功到达目标的计划数 ÷ 全部评价计划数
```

### 11.2 Feasibility

统一定义：

```text
没有越界、碰撞或非法动作的计划数 ÷ 全部评价计划数
```

没有违规但未到达终点的计划算 feasible，但不能算 successful，并标记为 `goal_not_reached`。

### 11.3 Optimality

只对成功路径计算：

```text
optimality = 最短路径长度 ÷ 成功预测路径长度
```

最优路径得分为 `1.0`，成功但更长的路径小于 `1.0`。

同时报告：

- 成功路径平均长度；
- 对应最短路径平均长度；
- 最优路径比例。

失败路径不进入平均 Optimality 的分母；失败情况由 Success Rate 单独反映。本公式自协议 v0.2 起冻结。

### 11.4 Verification Macro-F1

对 `valid` 和 `invalid` 两类计算 Macro-F1。

### 11.5 First-error Localization Accuracy

```text
正确预测第一次错误步骤的无效计划数 ÷ 全部无效计划数
```

同时报告：

- 错误类型准确率；
- 步骤和错误类型同时正确的准确率。

### 11.6 Repair Success Rate

```text
修复后成功的原始无效计划数 ÷ 进入 Repair 的原始无效计划数
```

原本已经成功的计划不进入该指标分母。

### 11.7 End-to-End Success Rate

```text
完整管线最终解决的环境数 ÷ 全部评价环境数
```

### 11.8 分组报告

结果至少按以下维度拆分：

- ID 单错误与 OOD 双错误；
- obstacle collision；
- out of bounds；
- invalid action token；
- goal not reached；
- self feedback 与 executor feedback。

## 十二、资源和实验预算

统一遵守：

- generation、verification、repair 的训练样本总计最多 4,000；
- development examples 最多 200；
- ID 单错误测试集最多 200；
- OOD 双错误测试集最多 200；
- 每个适用训练子集最多 3 个 epoch；
- 使用固定 seed；
- 完整项目以 20 GPU-hours 为规划上限；
- 两个 baseline 使用匹配的数据量和训练预算；
- 只有 primary model 运行完整管线、改进和消融。

分别记录：

```text
CPU 数据构建时间
CPU executor 检查时间
GPU 训练时间
GPU 推理时间
```

## 十三、统一实验结果格式

每个样本统一保存为 JSONL：

```json
{
  "experiment_id": "exp_001",
  "environment_id": "grid_0001",
  "split": "ood_test",
  "model": "google/flan-t5-small",
  "seed": 42,
  "feedback_source": "executor",
  "candidate_plan": ["right", "right", "down"],
  "candidate_valid": false,
  "predicted_error_step": 2,
  "predicted_reason": "obstacle_collision",
  "ground_truth_error_step": 2,
  "ground_truth_reason": "obstacle_collision",
  "repaired_plan": ["down", "right", "down"],
  "repair_success": true,
  "final_success": true,
  "predicted_length": 3,
  "optimal_length": 3,
  "runtime_seconds": 0.0
}
```

原始模型输出、解析后输出和 executor 结果都要保存，不能只保存最后的成功或失败。

## 十四、异常和失败处理

统一处理以下情况：

| 情况 | 处理方式 |
|---|---|
| 模型输出为空 | `parse_success=false`，记为失败 |
| 模型输出非法动作 | 保留原始 token，由 executor 标记 `invalid_action_token` |
| 输出超过最大长度 | 记录截断状态，不人工补全 |
| verifier 输出格式错误 | 解析失败，单独统计 |
| Repair 输出无法解析 | 记为 repair failure |
| 推理发生异常 | 保存错误日志，样本不静默删除 |
| GPU OOM | 记录配置，调整后重新运行完整受影响实验 |
| 样本文件损坏 | 停止对应实验并修复，不跳过后继续统计 |

不得手工修改某个测试样本的模型输出以提高成绩。

## 十五、依赖和运行环境

统一环境固定为：

| 项目 | 固定版本或要求 |
|---|---|
| 操作系统 | Ubuntu 22.04 LTS 64-bit |
| Python | 3.11.9 |
| PyTorch | 2.9.1 |
| CUDA wheel | CUDA 12.6，即 `cu126` |
| Transformers | 4.57.6 |
| Accelerate | 1.12.0 |
| Datasets | 4.4.2 |
| SentencePiece | 0.2.1 |
| GPU | 1 × NVIDIA RTX 4090 24 GB |
| NVIDIA Driver | Linux 建议不低于 560.28.03 |
| 默认精度 | BF16；不支持时使用 FP32 |

统一保存：

```text
requirements-project6.txt
check_environment.py
configs/base.yaml
```

正式训练和最终评价统一在同一台 RTX 4090 上完成。CPU 机器可以运行 executor、数据构造、单元测试和小规模 smoke test。

安装 PyTorch：

```bash
python -m pip install torch==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

安装其余依赖：

```bash
python -m pip install -r requirements-project6.txt
python check_environment.py
```

11月12日后不主动升级依赖。必须升级时，由三人共同确认并重新运行所有受影响实验。

## 十六、仓库和 Git 规则

统一结构：

```text
project6/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── docs/
├── src/
│   ├── executor/
│   ├── data/
│   ├── models/
│   ├── verifier/
│   ├── repair/
│   └── evaluation/
├── scripts/
├── tests/
├── results/
├── poster/
├── report/
└── README.md
```

统一分支：

```text
main
develop
feature/a-executor
feature/b-verifier
feature/c-repair-pipeline
```

统一规则：

- 不直接向 `main` 提交；
- `main` 和 `develop` 禁止 force push；
- 每个 Pull Request 至少由一人审核；
- Pull Request 统一采用 Squash and merge；
- 合并前运行相关测试；
- 不提交大型 checkpoint；
- 不提交密钥、访问令牌和本地绝对路径；
- 配置统一写入 `configs/`；
- 实验结果统一写入 `results/`；
- 提交信息说明修改内容和验证方式。

## 十七、模块完成标准

### 17.1 A 的 Executor 最低完成标准

- 正确执行四种合法动作；
- 检测碰撞、越界、非法动作和未到终点；
- 返回第一次错误步骤和错误类型；
- 输出完整执行轨迹；
- 通过成功、碰撞、越界、非法 token 和未到终点测试；
- 相同输入得到相同输出。

### 17.2 B 的模型最低完成标准

- 精确 checkpoint 可以加载；
- generation smoke test 可以运行；
- verification 输出可以解析；
- 保存原始输出和结构化输出；
- 记录输入长度、输出长度、显存和速度；
- 无法解析的输出不会被静默忽略。

### 17.3 C 的 Pipeline 最低完成标准

- 使用 mock 组件跑通 generate-verify-repair；
- 能切换 self 和 executor 两种反馈；
- Repair 输入输出符合统一 schema；
- 修复结果重新交给 executor 检查；
- 保存完整中间结果；
- 可以生成统一格式的结果记录。

## 十八、测试集保护和变更管理

### 18.1 测试集保护

- primary model 只能依据 dev 选择；
- 改进方法和消融配置只能依据 dev 决定；
- 不根据 ID/OOD test 结果修改错误构造、模型或指标；
- test 只用于最终冻结评价；
- 如发现代码错误，记录问题并重新运行全部受影响方法。

### 18.2 协议变更

协议冻结后如需修改，记录：

```text
修改日期
修改人
修改内容
修改原因
受影响模块
需要重新运行的实验
三人确认结果
```

不得只修改某一种方法以提高其测试表现。

## 十九、11月12日会议确认清单

### 19.1 必须确认

- [x] 项目范围已经冻结；
- [x] 网格和坐标格式已经统一；
- [x] 动作格式已经统一；
- [x] step 编号已经统一；
- [x] `goal_not_reached` 的步骤定义已经统一；
- [x] 错误类型已经统一；
- [x] executor/verifier 输出格式已经统一；
- [x] Repair 输入输出格式已经统一；
- [x] environment-level 数据划分已经确认；
- [x] 固定随机种子已经确认；
- [x] 两个 baseline 候选已经确认；
- [ ] 精确 checkpoint 已确认或已向项目负责人提问；
- [x] 三种主要比较方法已经确认；
- [x] Success、Feasibility 和 Optimality 定义已经确认；
- [x] Repair Success Rate 分母已经确认；
- [x] 训练和测试预算已经确认；
- [x] 实验结果格式已经统一；
- [x] Python 和主要依赖版本已经统一；
- [x] Git 分支和审核规则已经统一；
- [x] A、B、C 的模块负责人和复核人已经确认。

### 19.2 必须能够运行

```text
A：executor toy example
B：model inference smoke test
C：mock generate-verify-repair pipeline
```

### 19.3 会议签字区

| 成员 | 是否确认协议 | 未解决问题 |
|---|---|---|
| A |  |  |
| B |  |  |
| C |  |  |

## 二十、唯一阻塞正式训练的外部确认事项

课程负责人需要确认：

```text
Baseline 1 使用普通 T5-small，还是 google/flan-t5-small？
```

得到回复前，可以完成 executor、数据构造、接口实现和模型 smoke test，但不得开始正式 baseline 训练。收到回复后，记录精确模型名称和 revision，并将协议升级为 `v1.0 Frozen`。

Poster 时长与格式、最终提交入口和团队代码提问形式属于后续行政信息，不阻塞当前工程开发；收到课程通知后补充到项目 README 和时间线。
