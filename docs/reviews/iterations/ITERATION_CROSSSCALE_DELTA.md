# Cross-Scale Delta独立模型

## 目标

在独立Stack Public0.145基础上，训练新的信息表征，而不是继续调整已有模型权重。

## 新信息表征

输入v2和v3网格，并新增三个跨尺度差分流：

```text
v3 Market[::2] - v2 Market
v3 Flow[::2] - v2 Flow
v3 Order[::2] - v2 Order
```

模型共九路流：v2三路、v3三路、跨尺度差分三路。d_model64、两层Transformer、effective batch1024。

## 三折结果

固定原v3 50% + Cross-Scale Delta 50%，差分checkpoint 5/6/7：

| Fold | 原v3 | 混合后 |
|---|---:|---:|
| Proxy | 0.14752 | **0.15338** |
| Middle | 0.14825 | **0.14981** |
| Late | 0.15995 | **0.16888** |

混合后的月均和最差月均在三折均优于原v3，满足独立模型离线闸门。

## 全量推理

全量训练因运行时间达到上限于第10轮停止，但固定checkpoint 5/6/7已完整保存并完成测试推理。

生成候选：

`output/candidate_crossscale_delta_public60_40.csv`

测试相关性：

- 差分模型与原v3：0.82833；
- 差分模型与公开参考：0.78255；
- 候选与Public0.145方案：0.99105。

该候选比上一版独立Stack有明显更大变化。

## 与独立Stack组合审计

将差分模型加入LGB/RealMLP/v3/Joint/Multi-Resolution独立Stack后，三折最佳粗网格组合相对当前独立Stack的提升约0.0004，Middle没有同步提升。因此不自动替换当前Stack。

## 决策

- 不自动提交；
- 保留差分候选，等待用户明确批准后再测试Public；
- 若提交，目的应是验证跨尺度差分是否能转化为Public增益，而非宣称已达到Top10；
- 继续方案应优先改善差分模型的Middle稳定性，而不是扫描近邻权重。
