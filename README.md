# AIAA4051 Project 6：语言模型网格路径规划

> 成员 B 的11月13日接口交付见 [member_b_day13_delivery.md](docs/member_b_day13_delivery.md)：
> Generation/Verification模板与解析器、8条样例、v3模拟输出往返8/8结果。
> 本分支未完成固定模型配置或真实pipeline接入；新增测试待复核，详见交付说明。

本项目研究一个简单但重要的问题：

> 当语言模型生成了一条错误路径，明确告诉它“第几步错了、为什么错了”，能否帮助它把路径修好？

系统处理单目标网格导航任务，完整流程是：

```text
网格输入
  ↓
Generate：生成候选路径
  ↓
Verify：判断路径是否有效，并定位第一次错误
  ↓
Repair：根据错误反馈生成完整的新路径
  ↓
Execute：用确定性执行器检查最终结果
  ↓
Evaluate：统计成功率、修复成功率和路径质量
```

这不是为了用语言模型替代 BFS，而是研究语言模型能否理解并利用结构化错误反馈。

## 当前进度

| 模块 | 负责人 | 当前状态 |
|---|---|---|
| 数据、确定性执行器、数据划分 | A | 待接入真实模块 |
| 路径生成、模型验证器 | B | 待接入真实模块 |
| Repair 接口、端到端管线、评价框架 | C | Mock 原型已完成并通过测试 |

成员 C 当前已经完成：

- `Generate → Verify → Repair → Execute` 端到端管线骨架；
- Repair 输入、输出格式检查；
- 可重复运行的 Mock 演示；
- 0-based 错误步骤编号；
- `goal_not_reached` 的统一编号规则；
- 碰撞、越界、非法动作、未到终点、Repair 失败等自动测试；
- 结果记录 CSV 模板。

当前版本仍是第一阶段原型，不包含正式 Repair 模型训练或最终实验结果。

## 五分钟运行 Mock 演示

Mock 演示不需要 GPU，也不需要下载模型。只要有 Python 3.11 即可。

### 1. 克隆并切换到本分支

```bash
git clone https://github.com/qingyangli310/AIAA4051_project.git
cd AIAA4051_project
git switch feature/c-repair-pipeline
```

### 2. 创建轻量测试环境

```bash
conda create -n project6 python=3.11.9 -y
conda activate project6
python -m pip install pytest==8.4.2
```

### 3. 运行演示

请在仓库根目录执行：

```bash
python scripts/run_pipeline_demo.py
```

预期输出：

```text
Environment: grid_0001
Generated plan: right, right, down
Verification: invalid | step=1 | reason=obstacle_collision
Repaired plan: down, right, down, right
Final executor result: success
```

这里的 `step=1` 表示第二个动作，因为动作编号采用 0-based。

### 4. 运行自动测试

```bash
python -m pytest -q
```

当前预期结果：

```text
10 passed
```

测试覆盖：

- 错误路径可以被修复；
- 正确路径不会进入 Repair；
- 第一个动作编号为 `0`；
- 障碍物碰撞；
- 越界；
- 非法动作，例如 `jump`；
- 动作结束但没有到达终点；
- Repair 后仍失败；
- Repair 输出无法解析；
- `no_feedback` 不向 Repairer 泄漏 verifier 内容；
- 单样本异常被记录且不会中断批量运行。

## 示例是怎样运行的？

示例网格：

```text
S  .  X
.  .  .
.  .  G
```

对应数据：

```json
{
  "environment_id": "grid_0001",
  "rows": 3,
  "cols": 3,
  "start": [0, 0],
  "goal": [2, 2],
  "obstacles": [[0, 2]]
}
```

Mock Generator 输出：

```text
right, right, down
```

第二个 `right` 会从 `[0, 1]` 走向障碍物 `[0, 2]`，因此 Verifier 返回：

```json
{
  "valid": false,
  "first_error_step": 1,
  "reason": "obstacle_collision"
}
```

Mock Repairer 随后给出完整的新路径：

```text
down, right, down, right
```

最后由确定性执行器重新执行，确认修复后的路径可以到达终点。

## 已冻结的接口规则

所有成员的实现必须遵守以下规则。

### 坐标和动作

- 坐标格式：`[row, col]`；
- 左上角：`[0, 0]`；
- 合法动作：`up`、`down`、`left`、`right`；
- 坐标和错误步骤都使用 0-based；
- 不自动猜测或替换非法动作。

### 错误步骤

```text
第一个动作：step=0
第二个动作：step=1
第三个动作：step=2
```

如果所有动作执行完毕后仍未到达终点：

```text
reason = goal_not_reached
first_error_step = len(plan)
```

这个编号表示路径结束位置，不代表额外执行了一个动作。空路径的 `goal_not_reached` 步骤为 `0`。

### 统一错误类型

```text
obstacle_collision
out_of_bounds
invalid_action_token
goal_not_reached
```

### 环境划分与测试条件

| 环境集合或条件 | 内容和用途 |
|---|---|
| Train | Generation、Verification、Repair 三种任务的训练样本 |
| Dev | 三种任务的开发样本，用于模型选择和参数调整 |
| Held-out Test | 与 Train/Dev 环境互斥的保护测试环境池 |
| ID 单错误条件 | 从 Held-out Test 派生的有效路径、可行但非最优路径和单错误样本 |
| OOD 双错误条件 | 在同一批 Held-out Test 基础样本上独立引入两个错误 |

Train、Dev、Held-out Test 必须按 `environment_id` 划分且两两无交集。ID 与 OOD 不是两套环境划分，而是同一批 Held-out Test 上的配对评价条件；每对样本必须共享 `environment_id` 和 `base_sample_id`。

有效样本必须同时包含最短路径和可行但非最优路径。合法绕行路径仍应判定为有效，路径质量由 Optimality 单独评价。

## 主要模块接口

### Generation

输入网格，输出模型原始文本和解析后的动作序列：

```json
{
  "raw_output": "right, down, down, right",
  "plan": ["right", "down", "down", "right"],
  "parse_success": true
}
```

### Verification

有效路径：

```json
{
  "valid": true,
  "first_error_step": null,
  "reason": null
}
```

无效路径：

```json
{
  "valid": false,
  "first_error_step": 1,
  "reason": "obstacle_collision"
}
```

### Repair

Repair 接收网格、候选路径、反馈来源和可选验证反馈：

```json
{
  "environment_id": "grid_0001",
  "grid": {
    "environment_id": "grid_0001",
    "rows": 3,
    "cols": 3,
    "start": [0, 0],
    "goal": [2, 2],
    "obstacles": [[0, 2]]
  },
  "candidate_plan": ["right", "right", "down"],
  "feedback_source": "executor_feedback",
  "verifier_feedback": {
    "valid": false,
    "first_error_step": 1,
    "reason": "obstacle_collision"
  }
}
```

当 `feedback_source` 为 `no_feedback` 时，`verifier_feedback` 必须为 `null`，避免向 Repair 模型泄漏诊断信息。

Repair 必须输出完整的新路径，而不是只输出局部修改：

```json
{
  "raw_output": "down, right, down, right",
  "repaired_plan": ["down", "right", "down", "right"],
  "parse_success": true
}
```

第一版主实验只允许一轮 Repair。修复后的路径必须重新交给确定性执行器检查。

## 正式模型环境

Mock 演示只需要 Python 和 pytest。运行模型训练或正式实验时，使用团队冻结环境：

```text
Python 3.11.9
PyTorch 2.9.1
PyTorch CUDA wheel cu126
Transformers 4.57.6
Seed 42
```

安装命令：

```bash
conda create -n project6 python=3.11.9 -y
conda activate project6

python -m pip install --upgrade pip
python -m pip install torch==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-project6.txt
```

必须先安装固定版本的 PyTorch，再安装 `requirements-project6.txt`，避免 `accelerate` 自动选择其他 Torch 版本。

检查环境：

```bash
python scripts/check_environment.py
```

CPU 机器可以运行 Mock 管线、执行器、数据构造和单元测试。正式模型比较必须在同一台 GPU 机器和同一套冻结环境中完成。

## 仓库结构

```text
.
├── configs/
│   └── base.yaml                 共享数据、模型和训练配置
├── data/
│   └── manifests/                数据划分与构造清单
├── docs/
│   ├── Project6_团队统一协议.md   完整冻结协议
│   ├── Project6_项目说明与三人分工.md
│   ├── Project6_11月10日至12日任务清单.md
│   └── experiment_protocol.md    简明实验协议
├── src/
│   ├── pipeline.py               端到端管线
│   ├── repair_schema.py          Repair 接口检查
│   ├── mock_components.py        第一阶段 Mock 组件和执行逻辑
│   ├── executor/                 A 的真实执行器模块
│   ├── data/                     A 的数据构造模块
│   ├── models/                   B 的生成模型模块
│   ├── verifier/                 B 的验证器模块
│   ├── repair/                   C 的真实 Repair 模型模块
│   └── evaluation/               C 的评价和错误分析模块
├── scripts/
│   ├── check_environment.py      检查冻结环境
│   └── run_pipeline_demo.py      运行 C 的 Mock 演示
├── tests/
│   └── test_pipeline_smoke.py    端到端和协议测试
├── results/
│   └── result_template.csv       统一结果记录格式
├── requirements-project6.txt
└── README.md
```

当前部分目录只包含占位文件，等待 A、B、C 的真实模块逐步接入。

## 三人分工与复核

| 成员 | 主要职责 | 复核人 |
|---|---|---|
| A | 数据、执行器、错误生成、按环境划分数据 | C |
| B | 路径生成、路径验证、模型训练 | A |
| C | Repair、端到端管线、评价和结果汇总 | B |

关键模块采用“一人主写，一人复核”。每个人不仅要展示结果，还要提供可以运行、检查和复现的证据。

## Git 协作规则

推荐分支：

```text
main
develop
feature/a-executor
feature/b-verifier
feature/c-repair-pipeline
```

协作要求：

- 不直接向 `main` 推送；
- 每项逻辑变更使用独立功能分支；
- 测试通过后再创建 Pull Request；
- 至少一名成员复核后才能合并；
- 推荐使用 Squash and merge；
- 不提交模型 checkpoint、密码、令牌、个人绝对路径或大规模生成数据。

## 结果记录

正式实验结果应遵循 `results/result_template.csv`，主要字段包括：

```text
experiment_id
environment_id
split
test_condition
base_sample_id
source_plan_id
corruption_type
corruption_seed
model
model_revision
config_id
seed
feedback_source
raw_generation_output
candidate_plan
candidate_parse_success
candidate_valid
raw_verifier_output
verifier_parse_success
predicted_error_step
predicted_reason
ground_truth_error_step
ground_truth_reason
raw_repair_output
repaired_plan
repair_parse_success
repair_success
final_success
final_error_step
final_reason
feasible
predicted_length
optimal_length
optimality
runtime_seconds
```

计划比较至少三种方法：

1. 直接生成，不进行修复；
2. 模型自验证后修复；
3. 确定性执行器提供反馈后修复。

## 常见问题

### `ModuleNotFoundError: No module named 'src'`

请确认当前目录是仓库根目录，然后运行：

```bash
python scripts/run_pipeline_demo.py
```

### 没有 GPU，能否开发？

可以。Mock 管线、执行器、数据构造和测试均可在 CPU 上完成。只有正式模型训练和最终模型比较要求 GPU。

### 为什么第二个动作显示为 `step=1`？

项目统一采用 0-based 动作编号：第一个动作是 `0`，第二个动作是 `1`。

### 为什么 `goal_not_reached` 使用 `len(plan)`？

因为它表示所有动作已经执行完毕后的路径结束位置，而不是某个实际执行失败的动作。

### 环境检查失败怎么办？

先对照 `scripts/check_environment.py` 输出检查 Python、PyTorch、Transformers 和 CUDA runtime。不要为了临时通过检查而自行更改冻结版本；需要修改版本时，应由团队共同记录并重新运行受影响的实验。

## 详细文档

- `docs/成员C_11月10日至12日工作说明.md`：成员 C 两天内完成的代码、协议、测试和 CUDA 验证记录；
- `docs/Project6_项目说明与三人分工.md`：项目背景、研究问题、完整时间线；
- `docs/Project6_团队统一协议.md`：字段、编号、错误类型、模型和实验规则；
- `docs/Project6_11月10日至12日任务清单.md`：第一阶段三人任务及验收要求；
- `docs/Project6_11月13日至16日修订任务清单.md`：配对 ID/OOD、非最优有效路径和端到端 smoke test 的第二阶段任务；
- `docs/Project6_11月13日至16日按人详细时间表.md`：A、B、C 每日任务、交接、交付物与完成标准；
- `docs/experiment_protocol.md`：当前接口和指标摘要；
- `configs/base.yaml`：程序读取的共享配置。

如果 README 与冻结协议出现冲突，以最新版本的 `docs/Project6_团队统一协议.md` 和 `configs/base.yaml` 为准。
