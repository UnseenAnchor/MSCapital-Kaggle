# 迭代复盘：随机成对排序辅助损失——Late Gate 失败

日期：2026-08-30。

## 假设

Cosine 目标优化整体相关性；新增随机样本对排序辅助损失，试图改善横截面方向：

```text
loss = cosine + SmoothL1 + 0.05 * softplus(-Δprediction_z * Δtarget_z)
```

输入、采样、模型结构均不变；每个 batch 随机打乱配对，固定 `PAIR_LAMBDA=0.05`。

## 结果

| Checkpoint | centered cosine |
|---:|---:|
| ep4 | 0.14640 |
| ep5 | 0.14607 |
| ep6 | 0.14444 |
| baseline v3 ens(4,5,6) | **0.15995** |

即使单点最佳也远低于 baseline，固定 checkpoint ensemble 不可能达到 Late gate。

## 决策

- **Late gate 失败，立即止损**；不扩展三折、不生成候选、不提交；
- 成对排序目标冻结；保留 `PAIR_LAMBDA` 开关用于复现实验；
- 当前没有新的可提交候选，最佳 Public 保持 0.146。
