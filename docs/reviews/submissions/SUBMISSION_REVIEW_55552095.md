# Submission Review 55552095

## 提交信息

- ref：`55552095`
- 文件：`output/candidate_crossscale_delta_public60_40.csv`
- SHA256：`0ece5d0b108d34ca065d83a63d5eb02f53d18c03f33b75b379b741abd9957a5d`
- 行数：647,896
- NaN/Inf：0
- Kaggle状态：`COMPLETE`
- Public：**0.145**
- Private：尚未显示

## 方案

```text
公开LB0.142参考 60%
原v3 20%
Cross-Scale Delta 20%
```

Cross-Scale Delta使用：

```text
v3 Market[::2] - v2 Market
v3 Flow[::2] - v2 Flow
v3 Order[::2] - v2 Order
```

## 离线结果

固定原v3与差分模型50/50：

- Proxy：0.15338；
- Middle：0.14981；
- Late：0.16888。

测试候选与当前Public0.145候选相关性为0.99105，变化幅度明显。

## Public结果

| 提交 | Public |
|---|---:|
| 独立Stack 55538309 | 0.145 |
| Cross-Scale Delta 55552095 | **0.145** |

离线三折提升和较低测试相关性没有转化为Public提升。

## 复盘结论

1. Cross-Scale Delta在历史滚动折上有效，但仍存在CV到Public迁移失败。
2. 单纯增加v3-v2差分信息不能突破当前0.145。
3. 该架构不再继续做checkpoint或邻近权重搜索。
4. 当前Public最佳仍为`55538309`的0.145。
5. 距离Top10约0.153仍差0.008。

## 决策

- 停止Cross-Scale Delta架构族；
- 保留独立Stack 0.145作为回退方案；
- 下一步必须寻找能提升Public强度的新训练目标或新监督方式，而非继续扩展输入流；
- 未经明确批准不再提交。
