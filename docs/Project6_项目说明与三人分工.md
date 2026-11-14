# Project 6 项目说明与三人分工

> 让语言模型不仅能够规划网格路径，还能够检查路径哪里出错，并根据错误信息把路径修好。

项目要求来自课程的 Project 6 说明。

## 一、Project 6 在做什么？

想象一个网格地图：

```text
S  .  X  .
.  .  X  .
X  .  .  .
.  .  .  G
```

其中：

- `S`：起点；
- `G`：终点；
- `X`：障碍物；
- `.`：可以经过的位置。

模型需要输出一组动作：

```text
right, down, down, right, down, right
```

程序按照顺序执行这些动作，检查是否成功从起点走到终点。

但语言模型可能犯错，例如：

```text
right, right, down
```

第二个 `right` 撞到了障碍物。

Project 6 不只是判断最后有没有成功，而是要求系统完成三个步骤：

```text
生成路径 → 验证路径 → 修复路径
```

英文叫：

```text
Generate → Verify → Repair
```

## 二、三个核心任务

### 1. Generate：生成路径

输入是一段网格描述，模型输出：

```text
up / down / left / right
```

例如：

```text
输入：起点在左上角，终点在右下角，中间有两个障碍物
输出：right, down, down, right
```

这一步研究模型能不能自己规划出正确路径。

### 2. Verify：检查路径

验证器需要回答三个问题：

1. 这条路径是否有效？
2. 如果无效，第一次在哪一步失败？
3. 失败原因是什么？

有效路径输出：

```text
valid
```

无效路径输出：

```text
invalid | step=1 | reason=obstacle_collision
```

需要识别的主要错误包括：

- 撞到障碍物；
- 走出网格边界；
- 出现非法动作，例如 `jump`；
- 动作结束了但没有到达终点。

### 3. Repair：修复路径

修复模型获得：

```text
网格
原始错误路径
验证反馈
```

例如：

```text
原始路径：
right, right, down

验证反馈：
invalid | step=1 | reason=obstacle_collision
```

修复模型重新输出一条路径：

```text
down, right, down, right
```

最后，再让确定性的程序执行一遍，判断修复是否真正成功。

## 三、这个项目真正研究的问题

项目不是简单地做一个寻路算法。传统算法，例如 BFS，本来就可以找到网格最短路径。

它真正研究的是语言模型的能力：

> 当语言模型生成错误计划后，明确告诉它“哪里错了、为什么错了”，能不能帮助它生成更好的新计划？

因此需要比较三种方法。

### 方法一：直接生成

```text
网格 → 模型 → 路径
```

生成后不修复。

### 方法二：模型自我检查并修复

```text
网格
  ↓
模型生成路径
  ↓
语言模型自己检查
  ↓
语言模型根据自己的反馈修复
```

问题是模型可能连错误原因都判断错。

### 方法三：执行器检查并提供反馈

```text
网格
  ↓
模型生成路径
  ↓
确定性执行器检查
  ↓
把正确的错误位置和原因交给模型
  ↓
模型修复
```

这三种方法的比较能够回答：

- 加入修复是否比直接生成更好？
- 模型自己的反馈是否可靠？
- 精确的执行器反馈能带来多大提升？

## 四、什么是执行器？

执行器可以理解为一个绝对遵守规则的裁判。

它不需要语言模型，而是普通 Python 代码。它记录当前位置，并逐步执行动作：

```python
position = start

for action in plan:
    position = move(position, action)

    if position is outside_grid:
        return out_of_bounds

    if position is obstacle:
        return obstacle_collision
```

执行器负责产生正确答案，也就是 ground truth。

它需要告诉系统：

- 路径是否可执行；
- 第一次错误发生在哪一步；
- 错误类型；
- 最终是否到达终点；
- 路径长度；
- 与最短路径相比是否绕路。

## 五、训练数据从哪里来？

先获得一条正确路径：

```text
right, down, down, right
```

然后自动制造错误。

### 替换动作

```text
正确：right, down, down, right
错误：right, left, down, right
```

### 删除动作

```text
正确：right, down, down, right
错误：right, down, right
```

### 插入动作

```text
正确：right, down, down, right
错误：right, up, down, down, right
```

### 提前停止

```text
正确：right, down, down, right
错误：right, down
```

然后让执行器检查错误计划，自动生成标签：

```text
invalid | step=3 | reason=goal_not_reached
```

## 六、为什么还有单错误和双错误？

训练数据和普通测试主要使用单错误：

```text
right, left, down, right
       ↑ 一处错误
```

更困难的测试使用双错误：

```text
right, left, down, up
       ↑          ↑
```

项目要研究：

> 模型只在单错误数据上训练以后，能不能处理包含两个错误的路径？

这就是：

- ID：单错误测试，与训练数据相似；
- OOD：双错误测试，比训练数据更困难。

## 七、三个人分别做什么？

最简单的理解是：

```text
A：负责裁判和数据
B：负责生成和检查模型
C：负责修复和完整系统
```

## 成员 A：数据和执行器负责人

A 负责建立项目的基础。

主要任务：

- 获取 PPNL 网格数据；
- 实现执行器；
- 检查碰撞、越界、非法动作和未到达终点；
- 找到第一次错误；
- 生成正确路径；
- 自动制造单错误和双错误路径；
- 用执行器核验所有标签；
- 按网格环境划分训练集、开发集和测试集；
- 防止同一个网格同时出现在训练集和测试集；
- 实现部分底层评价指标；
- 确保数据和实验可以复现。

A 最终回答的问题是：

> 我们的数据和正确答案可靠吗？

A 的主要代码可能是：

```text
executor.py
data_builder.py
corruption.py
split_data.py
```

## 成员 B：生成和验证模型负责人

B 负责语言模型的主要训练。

主要任务：

- 确认使用的 checkpoint；
- 比较至少两个模型；
- 当前使用 T5-small 和 BART-base；
- 实现路径生成模型；
- 实现路径验证模型；
- 让验证器输出 valid/invalid；
- 预测第一次错误步骤；
- 预测错误类型；
- 根据开发集选择 primary model；
- 实现一个主要改进方法；
- 完成对应的消融实验；
- 保存训练配置、日志和 checkpoint。

B 最终回答的问题是：

> 模型能不能生成路径，并准确判断路径哪里错了？

B 的主要代码可能是：

```text
train_generator.py
train_verifier.py
generate.py
verify.py
prompts/
model_configs/
```

## 成员 C：修复和端到端系统负责人

C 负责让模型根据反馈修复错误，并将整个系统连接起来。

主要任务：

- 构造 repair 训练数据；
- 训练或提示 repair model；
- 输入网格、错误计划和验证反馈；
- 输出完整的新路径；
- 比较无反馈、自验证反馈和执行器反馈；
- 将 A 的执行器、B 的生成器和验证器连接起来；
- 实现完整的 generate–verify–repair pipeline；
- 测试单错误和双错误计划；
- 统计 Repair Success Rate；
- 统计 End-to-End Success Rate；
- 按错误类型分析结果；
- 研究模型是否只修复第一个错误；
- 检查修复是否引入新错误；
- 整理详细案例、图表和实验结果；
- 主导 Poster 和报告结果部分的整合。

C 最终回答的问题是：

> 有了错误反馈之后，模型能不能真正把路径修好？

C 的主要代码可能是：

```text
repair.py
pipeline.py
end_to_end_eval.py
error_analysis.py
```

## 八、三个人怎样连接起来？

整个合作关系是：

```text
             A
      网格数据和正确答案
             ↓
             B
       生成一条候选路径
             ↓
     B 的验证器进行诊断
        或 A 的执行器诊断
             ↓
             C
       根据反馈修复路径
             ↓
          A 的执行器
       检查最终是否成功
             ↓
             C
      汇总指标和分析结果
```

也可以把三个人理解成：

| 成员 | 类比 | 负责内容 |
|---|---|---|
| A | 出题人＋裁判 | 准备地图、制造错误、给出正确标签 |
| B | 规划者＋诊断者 | 生成路径、判断哪里出错 |
| C | 修理者＋系统集成者 | 修复路径、组合系统、评价最终效果 |

## 九、每个人最后需要证明什么？

### A 需要证明

- 执行器正确；
- 数据没有泄漏；
- 错误标签正确；
- 单错误和双错误数据生成合理；
- 实验可复现。

### B 需要证明

- 两个模型进行了公平比较；
- primary model 只根据开发集选择；
- 验证器能判断有效性；
- 验证器能定位第一次错误；
- 改进方法确实有帮助。

### C 需要证明

- 修复模型能够使用错误反馈；
- executor feedback 是否优于 self-feedback；
- 修复能否提高最终成功率；
- 单错误训练能否泛化到双错误；
- 改进后的系统为什么成功或失败。

## 十、最终实验结果应该长什么样？

最终至少需要一张类似这样的主表：

| 方法 | ID成功率 | OOD成功率 | 验证F1 | 错误定位 | 修复成功率 |
|---|---:|---:|---:|---:|---:|
| 直接生成 | — | — | — | — | — |
| 自验证＋修复 | — | — | — | — | — |
| 执行器反馈＋修复 | — | — | — | — | — |
| 改进方法 | — | — | — | — | — |
| 消融版本 | — | — | — | — | — |

还需要至少两个完整案例：

```text
网格是什么
模型最初生成了什么
第一次错误在哪里
验证器给了什么反馈
模型怎样修复
最后是否成功
```

最简单的团队总结就是：

> A 保证题目和答案正确，B 让模型生成并发现错误，C 让模型根据错误信息完成修复并验证整个方法有没有效果。

---

## 十一、关键时间节点

- **2026-11-10**：正式开始执行项目计划；
- **2026-12-07**：Poster presentation；
- **2026-12-14**：提交最终代码和研究报告；
- **2026-12-15～20**：团队代码提问或项目面试。

项目期间按照约定日期开会检查进度。每次会议都应检查可运行成果，而不只是口头汇报。

## 十二、团队最终目标与交付物

在 12 月 14 日前完成一个可复现的 Project 6 系统：

```text
网格输入
  ↓
生成路径
  ↓
验证路径并定位第一次错误
  ↓
根据模型反馈或执行器反馈修复路径
  ↓
执行器检查最终结果
```

最终需要交付：

- 单目标网格执行器；
- 自动生成正确路径、单错误路径和双错误路径的数据程序；
- 至少两个 baseline 模型的公平比较；
- 一个根据开发集选择的 primary model；
- 直接生成、模型自验证修复、执行器反馈修复三种方法；
- 一个改进方法；
- 一个消融实验；
- 同一批 Held-out Test 环境上按 `base_sample_id` 配对的 ID 单错误与 OOD 双错误测试；
- 完整指标、错误分析和至少两个详细案例；
- 可运行代码、README、实验配置和结果；
- Poster；
- 学术报告；
- 代码提问和团队面试准备材料。

## 十三、协作和复核机制

虽然每个人都有固定负责模块，但不能出现“只有一个人懂某部分”的情况。

每项重要工作至少采用：

```text
1 人主写 + 1 人复核
```

推荐复核关系：

- A 的数据和执行器由 C 复核；
- B 的模型和训练由 A 复核；
- C 的修复和评价由 B 复核；
- 报告、Poster 和最终结果由三个人共同检查。

三个人都必须能够解释：

- executor 如何执行动作并判断第一次错误；
- train/dev/test 如何按照 grid environment 划分；
- 单错误和双错误样本如何生成；
- 为什么选择当前 primary model；
- 三种系统如何进行公平比较；
- Success、Feasibility、Optimality 等指标如何计算；
- 改进方法和消融实验分别说明了什么。

## 十四、时间线和阶段验收目标

### 11月10日—11月12日：环境与实验协议确定

**阶段目标：** 三个人都能运行仓库，并冻结基础实验定义。

成员 A：

- 下载并检查 PPNL；
- 建立代码仓库结构；
- 开始实现 executor；
- 给每个网格分配唯一的 `environment_id`。

成员 B：

- 确认可使用的模型 checkpoint；
- 运行 small 模型推理 smoke test；
- 确认生成和验证任务的输入输出格式。

成员 C：

- 设计端到端管线；
- 设计 repair 数据格式；
- 冻结主要评价指标定义；
- 准备结果记录模板。

**11月12日会议验收：**

- 三个人都能运行项目；
- executor 能运行一个简单样例；
- 确认三个人的配置均使用 `google-t5/t5-small`；
- 确认错误步骤从 0 还是从 1 开始编号；
- 确认 Success、Feasibility 和 Optimality 的计算方法；
- 确认仓库分支、代码提交和审核规则。

### 11月13日—11月16日：完成数据管线

**阶段目标：** 得到可靠且可复现的 Train、Dev、Held-out Test 环境划分，以及在同一 Held-out Test 上配对的 ID 单错误与 OOD 双错误条件。

成员 A：

- 完成 executor；
- 完成 Train/Dev/Held-out Test 的 environment-level 数据划分；
- 生成最短路径、可行但非最优路径和单错误样本；
- 在同一批 Held-out Test 基础样本上生成配对的 ID 单错误与 OOD 双错误条件；
- 保存 `environment_id`、`base_sample_id`、`source_plan_id`、corruption 位置、类型和 seed；
- 使用 executor 自动核验全部标签。

成员 B：

- 建立 generation 和 verification 数据格式；
- 跑通小规模训练；
- 测试模型输出解析程序及解析失败处理；
- 记录 tokenizer 截断、checkpoint revision、配置和随机种子；
- 取得项目负责人对普通 T5-small 与 Flan-T5 选择的书面确认。

成员 C：

- 建立 repair 数据格式；
- 实现基础评价指标；
- 冻结 `optimality = optimal_length / predicted_length`，且只对成功路径计算；
- 检查不同错误类型的数据数量；
- 跑通 `no_feedback`、`self_feedback`、`executor_feedback` 三个条件；
- 对解析失败和单样本异常做隔离，确保批量运行不中断；
- 使用真实或明确标记的 mock 输出完成 20～50 条端到端 smoke test。

**11月16日会议验收：**

- Train/Dev/Held-out Test 环境两两无交集；
- ID/OOD 使用相同的 Held-out Test `environment_id` 集合并由 `base_sample_id` 配对；
- 数据同时包含最短有效路径和可行但非最优有效路径；
- 单错误和双错误数据可以重复生成，且双错误没有抵消或完全退化；
- executor 单元测试全部通过；
- 完成至少 20～50 条样本的端到端 smoke test；
- 输出数据规模、错误类型统计表、标签核验报告、至少一个成功案例和一个失败案例；
- Mock 与真实模型结果明确区分，全部结果可追溯到配置、模型 revision 和原始样本。

### 11月17日—11月19日：完成两个 baseline

**阶段目标：** 使用相同数据和预算公平比较至少两个模型。

成员 A：

- 检查两个模型使用的数据完全一致；
- 检查评价脚本；
- 记录运行时间、CPU 时间和 GPU 资源消耗。

成员 B：

- 训练或提示两个 baseline；
- 完成 generation 和 verification 初步结果；
- 保存统一格式的配置和日志。

成员 C：

- 完成基础 repair baseline；
- 建立统一结果表；
- 收集典型失败案例。

**11月19日会议验收：**

- 两个 baseline 都完成最小规模实验；
- generation、verification 和 repair 都有初步结果；
- 找出输入截断、输出格式、显存或速度问题；
- 明确不能根据测试集表现选择模型。

### 11月20日—11月23日：选择主模型并完成完整管线

**阶段目标：** 根据开发集选择 primary model，并跑通三种方法。

必须比较：

1. 直接生成；
2. 自验证＋修复；
3. 执行器反馈＋修复。

成员 A：

- 检查开发集选择过程；
- 固定测试集；
- 检查是否存在测试信息泄漏。

成员 B：

- 根据开发集选择 primary model；
- 完成 verifier；
- 输出 `valid/invalid`、`step` 和 `reason`。

成员 C：

- 组装完整 generate–verify–repair 管线；
- 完成 self-feedback 和 executor-feedback 两种修复；
- 保存每一步的中间输出和执行轨迹。

**11月23日会议验收：**

- primary model 已冻结；
- 三种方法都能完整运行；
- 结果表包含 ID 单错误开发集结果；
- 测试集和主要实验配置被冻结。

### 11月24日—11月26日：实现改进方法

建议使用结构化状态跟踪验证：

```text
initial position=(0,0)
step=0 action=right position=(0,1) status=ok
step=1 action=down position=(1,1) status=obstacle_collision
```

成员 A：

- 检查结构化状态转移是否与 executor 一致；
- 提供标准执行轨迹。

成员 B：

- 实现结构化状态跟踪 prompt 或训练格式；
- 使用 primary model 运行改进实验。

成员 C：

- 将改进后的验证器接入 repair 管线；
- 分析修复成功率和端到端成功率的变化。

**11月26日会议验收：**

- 改进方法可以完整运行；
- 改进方法与 baseline 使用相同数据；
- 获得初步对比结果；
- 冻结消融实验设计。

### 11月27日—11月30日：消融和 OOD 双错误实验

推荐消融：

```text
完整方法：结构化状态跟踪
消融版本：去掉状态跟踪，只输出最终诊断
```

成员 A：

- 检查两个错误是否真正独立；
- 核验第一次错误的 ground truth。

成员 B：

- 完成改进方法和消融实验；
- 输出 verification Macro-F1 和第一次错误定位准确率。

成员 C：

- 完成 ID 与 OOD 端到端评价；
- 分析模型只修复第一处错误的情况；
- 收集成功与失败案例。

**11月30日会议验收：**

- 单错误 ID 结果完整；
- 双错误 OOD 结果完整；
- 改进和消融实验完成；
- 不再增加新的主要方法；
- 开始冻结最终实验。

### 12月1日—12月3日：最终实验和结果冻结

**阶段目标：** 生成最终可用于报告和 Poster 的结果。

三个人共同完成：

- 使用固定随机种子重新运行最终实验；
- 检查所有方法使用相同测试集；
- 检查模型、数据量和训练预算一致性；
- 整理总结果表；
- 按错误类型拆分结果；
- 整理至少两个详细案例；
- 记录典型失败模式。

**12月3日会议验收：**

- 所有核心数字冻结；
- 图表和表格完成；
- 不再根据测试结果调整方法；
- Poster 和报告开始正式撰写。

### 12月4日—12月6日：Poster 和报告初稿

成员 A 撰写：

- 数据集；
- executor；
- 数据构建；
- 实验可复现性。

成员 B 撰写：

- baseline；
- 模型和训练设置；
- 验证方法；
- 改进与消融。

成员 C 撰写：

- repair 管线；
- 评价指标；
- 实验结果；
- 错误分析；
- Poster 内容整合。

**12月6日会议验收：**

- Poster 完整初稿；
- 报告主要章节初稿；
- 三个人逐页检查；
- 所有数字与代码输出一致；
- 准备 3～5 分钟讲解和问答。

### 12月7日：Poster Presentation

推荐讲解分工：

- A：研究问题、数据和 executor；
- B：模型、验证和改进方法；
- C：修复、实验结果和结论。

三个人都必须能够回答整个系统的问题，而不仅是自己负责的模块。

### 12月8日—12月10日：代码和报告整合

**阶段目标：** 将“可以运行的研究代码”整理成“其他人可以复现的提交代码”。

需要完成：

```text
README
requirements/environment
数据构建命令
训练命令
评价命令
配置文件
随机种子
结果目录
预期输出
```

**12月10日会议验收：**

- 从干净环境按照 README 可以运行；
- 删除硬编码绝对路径；
- 所有结果可以追溯到配置和日志；
- 报告正文完整；
- 引用、表格和图片编号正确。

### 12月11日—12月13日：最终检查

12月13日进行一次模拟提交和模拟答辩。

检查清单：

- 代码可以运行；
- 文件路径全部使用相对路径；
- checkpoint 名称准确；
- train/dev/test 没有环境泄漏；
- 所有指标定义明确；
- 报告数字与程序输出一致；
- Poster 数字与报告一致；
- Git 仓库没有大型无关文件；
- 三个人都能解释任意核心模块；
- 准备提交文件的备份压缩包。

### 12月14日：提交全部材料

建议在当天中午前完成最终提交，并由三个人分别下载和检查一次提交内容。

### 12月15日—20日：代码提问和团队面试

每个人都需要准备回答：

- 你具体实现了什么？
- executor 如何判断第一次错误？
- 为什么按照 grid environment 划分数据？
- 为什么需要测试双错误计划？
- self-verification 和 executor feedback 有什么区别？
- primary model 为什么这样选择？
- 改进方法具体改变了什么？
- 消融实验说明了什么？
- 哪些结果支持最终结论？
- 系统失败的主要原因是什么？

## 十五、依赖和运行环境

### 15.1 统一环境结论

三个人统一使用以下主环境：

| 项目 | 固定版本或要求 |
|---|---|
| 操作系统 | Ubuntu 22.04 LTS 64-bit |
| Python | 3.11.9 |
| PyTorch | 2.9.1 |
| PyTorch CUDA wheel | CUDA 12.8，即 `cu128` |
| Transformers | 4.57.6 |
| Accelerate | 1.12.0 |
| Datasets | 4.4.2 |
| SentencePiece | 0.2.1 |
| Hugging Face Hub | 0.36.0 |
| GPU | 1 × NVIDIA RTX 4090 24 GB |
| NVIDIA Driver | Linux 建议不低于 570.26；学校 HPC 镜像须实际验证 |
| 随机种子 | 42 |
| 默认精度 | BF16；不支持时使用 FP32 |

采用这套组合的原则是优先保证稳定性和复现性，不在项目中途追随最新版升级。除非当前组合无法运行，否则三个人不得自行更换 Python、PyTorch、Transformers 或 CUDA wheel。

### 15.2 模型 checkpoint 和 revision

推荐将下面两个模型作为必做 baseline：

| 用途 | Hugging Face 模型 | 固定 revision |
|---|---|---|
| Baseline 1 | `google-t5/t5-small` | `df1b051c49625cf57a3d0d8d3863ed4d13564fe4` |
| Baseline 2 | `facebook/bart-base` | `aadd2ab0ae0c8268c7c9693540e9904811f36177` |

如果之后项目负责人要求切换为 Flan-T5-small，则固定为：

| 用途 | Hugging Face 模型 | 固定 revision |
|---|---|---|
| 可选模型 | `google/flan-t5-small` | `ec9123e078f63797d82dbe0812dc176c7008fc29` |

加载模型时必须显式提供 revision：

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google-t5/t5-small"
MODEL_REVISION = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
)
```

不要只写 `revision="main"`，因为远程仓库的 `main` 将来可能发生变化。

### 15.3 安装方式

推荐使用 Conda 创建统一环境：

```bash
conda create -n project6 python=3.11.9 -y
conda activate project6
python -m pip install --upgrade pip
```

单独安装官方 CUDA 12.8 版 PyTorch：

```bash
python -m pip install torch==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

再安装项目其余依赖：

```bash
python -m pip install -r requirements-project6.txt
```

PyTorch wheel 已经包含对应 CUDA runtime，通常不需要每个人另外安装完整 CUDA Toolkit；GPU 机器必须安装兼容的 NVIDIA Driver。

2026 年 11 月 14 日，团队为匹配学校 HPC 镜像仓，将 PyTorch 2.9.1 的
CUDA wheel 从 `cu126` 同步为 `cu128`。验收环境时必须保存
`torch.__version__` 和 `torch.version.cuda`，不能只依据 `nvidia-smi`
报告的驱动兼容版本。

### 15.4 固定 requirements

`requirements-project6.txt` 固定为：

```text
transformers==4.57.6
accelerate==1.12.0
datasets==4.4.2
evaluate==0.4.6
huggingface-hub==0.36.0
sentencepiece==0.2.1
safetensors==0.6.2
numpy==2.2.6
pandas==2.3.3
scikit-learn==1.7.2
PyYAML==6.0.3
tqdm==4.67.1
pytest==8.4.2
```

本项目不需要 `torchvision` 或 `torchaudio`，因此不安装它们，减少无关依赖和版本冲突。

### 15.5 运行设备和显存规则

正式训练和最终实验统一使用：

```text
GPU：NVIDIA RTX 4090
显存：24 GB
GPU 数量：1
```

统一训练原则：

- 优先使用 BF16；
- 不使用 TF32 作为不同实验之间的隐藏差异；
- 如果显存不足，优先减小 batch size，并使用 gradient accumulation；
- 不允许某个模型使用明显更多的训练样本、epoch 或 GPU 时间；
- 两个 baseline 使用相同的数据划分、最大输入长度和训练预算；
- 每次正式运行记录峰值显存、运行时间和 GPU 型号。

建议初始设置：

```text
max_input_length = 512
max_target_length = 128
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
num_train_epochs <= 3
seed = 42
bf16 = true
```

以上 batch size 是 pilot 起点，不是不可修改的研究参数。如果 T5-small、未来切换的 Flan-T5-small 或 BART-base 显存不足，可以降低 batch size，但必须通过 gradient accumulation 尽量保持相同的有效 batch size，并把变更记录到配置文件。

### 15.6 统一环境检查

三个人在开始工作前都要运行：

```bash
python --version
python -m pip freeze
nvidia-smi
python check_environment.py
```

期望看到：

```text
Python: 3.11.9
PyTorch: 2.9.1
Transformers: 4.57.6
CUDA runtime: 12.6
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
GPU memory: approximately 24 GB
```

如果三个人在不同机器上开发，可以在 CPU 上运行 executor、数据构造、单元测试和小规模 smoke test；正式模型对比和最终评价必须在同一台 RTX 4090 上完成。

### 15.7 每次实验必须保存的环境证据

每次正式实验至少保存：

```text
experiment_id
git_commit
python_version
torch_version
transformers_version
cuda_runtime
nvidia_driver
gpu_name
peak_gpu_memory_mb
model_name
model_revision
seed
config_path
start_time
end_time
```

最终提交前保存：

```bash
python -m pip freeze > results/environment/pip-freeze.txt
nvidia-smi > results/environment/nvidia-smi.txt
git rev-parse HEAD > results/environment/git-commit.txt
python check_environment.py > results/environment/environment-check.txt
```

### 15.8 版本变更规则

- 11月12日确认环境后冻结版本；
- 任何依赖升级都必须由三个人共同确认；
- 升级后必须重新运行 smoke test；
- 如果升级会影响模型输出或数值结果，需要重新运行全部对比实验；
- 最终报告必须写明实际使用的版本，而不是只写“使用 PyTorch 和 Transformers”。

## 十六、推荐仓库结构

```text
project6/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
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

## 十七、会议固定检查模板

每次会议不要只汇报“做了多少”，每个人都需要回答四个问题：

1. 上次会议以后完成了什么？
2. 有什么可以运行、检查或复现的证据？
3. 当前阻塞问题是什么？
4. 下次会议前具体交付什么？

最重要的阶段目标可以压缩为：

```text
11月16日：数据和 executor 完成
11月23日：三种基础管线完成
11月30日：改进、消融、OOD 实验完成
12月3日：最终结果冻结
12月6日：Poster 和报告初稿完成
12月7日：Poster 汇报
12月10日：代码和报告整合完成
12月13日：最终检查
12月14日：提交
12月15～20日：代码提问和团队面试
```
