# 成员 B：11月13日接口与联调交付

整理日期：2026-11-14。分支：`feature/b-verifier`。
本次交付 Generation/Verification 输入模板、输出解析器、模型接口、测试、8条联调样例及结果。
不是正式模型训练或最终评价，也不代表11月13日全部任务已验收。

## 已有证据

- `results/day13/20261114T022018_469933Z/roundtrip_report.json`：实现版本
  `member-b-day13-v3-bare-action-tokens`，8/8样例往返通过。
- 同目录的 `generation_examples.jsonl`、`verification_examples.jsonl`：各8条，
  保存输入、目标和来源，均标记 `usage=integration_only`。
- `results/day13/legacy_v2_pytest_sanitized.txt`：修订前版本78项测试通过的历史日志，
  仅隐去个人绝对路径。它不证明v3新增测试或与团队仓库合并后的测试通过。
- 本次整理没有运行测试、下载模型或进行推理。最新版全套回归待用户执行并补充日志。

报告使用 A 的标签模拟模型输出（`model_loaded=false`、`executor_rerun=false`）。
提交时仅将结果文件换行统一为LF，JSON内容未改写；源样例SHA-256与报告一致，
并通过.gitattributes固定样例换行，避免跨平台检出影响来源校验。
8/8代表模板、接口与解析器往返一致，不代表模型准确率100%，也不证明 A 的 executor 已重新执行。
样例覆盖成功1条、碰撞3条、越界2条、非法动作1条、未到终点1条。

## 输入与输出规则

公共网格字段为 `environment_id`、`rows`、`cols`、`start`、`goal`、`obstacles`。
坐标为0-based `[row, col]`。up/down改变row，left/right改变col。

Generation：`GridModel.generate_plan(grid)`。
返回 `raw_output`、`plan`、`parse_success`、`metadata`。
文本为英文逗号分隔的动作，合法动作仅up/down/left/right。
每项去首尾空白、转小写；保留未知单个token，如move_up、turn-left、123、teleport。
不删除非法动作，不把down.偷偷修为down，合法性留给executor判定。
空项、项内部空白、不可打印字符、括号/引号/反引号包装属于格式失败；
失败时plan为空列表且parse_success=false，raw_output始终保留。

Verification：`GridModel.verify_plan(grid, plan)`。
输出文本为 `valid` 或 `invalid | step=N | reason=TYPE`。
返回核心字段 `valid`、`first_error_step`、`reason`，并保留raw_output、parse_success、metadata。
有效输出的错误字段为null，解析失败的三个核心判断字段全部为null。
四种错误固定为obstacle_collision、out_of_bounds、invalid_action_token、goal_not_reached。
动作错误的步骤为 `0 <= step < len(plan)`；goal_not_reached为 `step=len(plan)`。
GridModel调用解析器时检查步骤范围，但不纠正模型在范围内作出的错误预测。

必须区分 A 与 B 的 valid：

```python
verifier_valid = executor_label["success"]
```

A.valid表示移动合法，A.success表示任务完成。B.valid表示路径合法且到达终点。
sample_005中A.valid=true、success=false，对应B的
`invalid | step=1 | reason=goal_not_reached`。转换保留完整executor_label。

## 数据与文件

输入源为 `data/samples/member_A_executor_integration_samples_20261113.jsonl`，
保留A提供的sample_id、environment_id、source_file、source_index、correct_plan、
candidate_plan和executor_label。Generation目标用correct_plan；Verification输入只含
网格和candidate_plan，标签来自executor_label，不向prompt添加正确路径或真实诊断。

来源文件名中的train/dev/test不作为本项目split，多个样例共享environment_id。
5×5和7×7样例只用于接口兼容测试；这8条不能直接作为正式训练或测试划分。
损坏样例文件的停止策略仍需按团队协议第14节完善：当前诊断脚本记录坏行并继续收集问题，
最终以失败状态退出，但会写部分样例文件；失败产物不能用于训练或评价。

- `src/models/prompts.py`：统一文本模板与网格检查。
- `src/models/output_parser.py`：动作和验证输出解析。
- `src/models/api.py`：供调用方复用的GridModel。
- `src/models/backend.py`：延迟加载模型；离线mock不需要Torch或Transformers。
- `src/models/day13_contract.py`：源样例到两类文本记录的转换。
- `scripts/member_b_day13.py`：离线往返报告。
- `scripts/model_interface_smoke.py`：可选真实模型smoke入口，非本次执行结果。
- `scripts/prepare_day13_model_config.py`：先前的T5家族配置校验工具；尚不支持BART，
  不作为全组配置权威。本次未提交空白确认表或变更模型配置。

## 运行与复核

在仓库根目录、团队约定的Python环境中执行：

```bash
python -m scripts.member_b_day13
python -m pytest tests/test_output_parser.py tests/test_verification_parser.py tests/test_model_api.py tests/test_day13_roundtrip.py tests/test_day13_model_config.py -v
python -m pytest tests -v
```

第一条只依赖Python标准库，每次生成新的 `results/day13/<UTC时间戳>/` 目录。
预期往返结果8/8。测试命令不加载真实模型，最后一条还包含仓库已有C的测试。
依赖版本遵循 `requirements-project6.txt`，不要为本次接口检查升级团队环境。

## 交给 C

先检查parse_success，再读取plan或valid。不能把valid=null当成普通invalid。
记录原始输出和解析状态；不偷偷用executor反馈替换模型反馈。
分别复用Generation和Verification模型实例，后续各自加载任务checkpoint。

当前main的pipeline调用generator.generate和verifier.verify，而本模块提供
generate_plan和verify_plan。后续接入需要方法适配及解析失败分支；不能直接声称已接入真实pipeline。
A负责复核B模块，C需确认附加字段、反馈映射和失败分支。此次仅提交功能分支，尚无团队签字。

## 模型配置与后续工作

本次不改变 `configs/base.yaml` 或团队冻结协议。
团队v1.2已规定T5-small和BART-base及其revision；如以后依据Dev选择或变更模型，
必须保留模型/版本记录并遵守协议，不能根据Held-out Test调参。
本地后端的历史默认值仍是main和64个新token，不符合正式pilot固定版本与128输出长度要求；
正式推理/训练前需对齐配置，不能把这些默认值当成冻结实验配置。
固定checkpoint加载、新版全套测试、正式数据划分、训练与推理、统一实验日志及C接入仍待完成。
