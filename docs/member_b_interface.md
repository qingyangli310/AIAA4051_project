# 成员 B 接口入口

最新接口、解析规则、运行命令和交付限制见 [11月13日交付说明](member_b_day13_delivery.md)。

```python
from src.models.api import GridModel

# 用已明确配置的backend/model实例提供下面两个方法：
# model.generate_plan(grid)
# model.verify_plan(grid, candidate_plan)
```

协议实现版本为 `member-b-day13-v3-bare-action-tokens`。
本分支包含模拟输出联调证据，尚未完成固定模型配置和真实pipeline接入。
实际运行参数以团队 `configs/base.yaml` 和冻结协议为准。
