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

[坐标、动作、四类错误、首错定义和 canonical Executor]

### 3.2 Generation

[输入、输出、解析和 primary checkpoint]

### 3.3 Verification

[valid invalid、首错位置和错误类型]

### 3.4 Repair and Feedback Conditions

[Direct、No feedback、Self feedback、Executor feedback]

### 3.5 Improved Method

[冻结后的改进名称、与基础方法的唯一差异和理论动机]

## 4 Dataset and Experimental Setup

[环境级划分、2000 500 1500 或最终预算、Dev 预算、162 对 ID/OOD、seed、epochs、GPU 和哈希]

## 5 Results

### 5.1 Two Model Baseline Comparison

[同数据与匹配预算的两个模型 Dev 选择依据]

### 5.2 Frozen ID Results

[Success、Feasibility、Optimality、Macro F1、首错定位、Repair Success、E2E]

### 5.3 Frozen OOD Results

[同上，并报告 ID OOD 差距]

### 5.4 Results by Failure Type

[obstacle、bounds、token、goal not reached]

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

[单 seed、单目标、模型规模、反馈依赖证据和硬件差异]

## 10 Conclusion

[仅陈述冻结结果支持的结论]

## References

[完整引用]
