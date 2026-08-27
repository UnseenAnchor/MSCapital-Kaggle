# 迭代复盘：target round4——Late Gate 失败

## 假设

公开 RealMLP 方案将 target round 到 4 位小数，可能过滤微小标签噪声。当前 v3 MultiStream 仅加入 `TARGET_ROUND=4`，模型、输入、采样、损失均不变。

## 结果

| checkpoint | centered cosine |
|---:|---:|
| ep2 | 0.14361 |
| ep4 | 0.14923 |
| ep5 | 0.15429 |
| baseline v3 ens(4,5,6) | **0.15995** |

## 决策

- Late gate 失败，停止剩余轮次；不扩展三折、不提交；
- target round4 冻结；
- 与 RQ target auxiliary 一起表明，公开方案的有效性主要不来自简单 target 处理，可能来自其完整 factor 特征和不同训练管线。
