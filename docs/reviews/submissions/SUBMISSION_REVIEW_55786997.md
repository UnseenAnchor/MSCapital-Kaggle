# Submission Review 55786997

## 提交信息

- ref：`55786997`
- 文件：`output/candidate_coral_dual_event_public60_40.csv`
- SHA256：`04b8c7d76b4f575c1342edba0d2b1d6822f1c6cc53505bcb6b0c72e1378c78cd`
- 行数：647,896；ID 对齐；无 NaN/Inf；centered norm=1
- Kaggle 状态：`COMPLETE`
- Public：**0.146**
- Private：尚未显示

## 方案

```text
公开参考 60% + 自研 Stack 40%
```

自研 Stack 延续 Dual-Event 配方：

- 原 Event256：0.10；
- supervised SSL Event256：0.20；
- 原五成员权重保留；
- v3 MultiStream 成员替换为 CORAL 全量模型；
- CORAL 训练时用 train/test 无标签 batch 的 latent 协方差对齐，`CORAL_LAMBDA=0.5`。

## 离线依据

CORAL Late 探针本身为 `0.159510`，低于 baseline `0.159950`（-0.000440），因此这是一次用户明确批准的高风险全量校准尝试，而非通过离线门槛的候选。

全量模型使用 v3 grid 400/120、D64/2层、有效 batch1024、12 epochs；ep4/5/6 完成 test 推理。

## Public 结果

| 提交 | 方案 | Public |
|---|---|---:|
| 55601441 | 原 Event256 Stack | **0.146** |
| 55713688 | Dual-Event Stack | **0.146** |
| **55786997** | **CORAL 全量 + Dual-Event Stack** | **0.146** |

## 决策

- 55786997 与最佳持平，无回退，但没有可见提升；
- CORAL 方向冻结，不再提交相邻权重或重跑 CORAL；
- 主基线仍为 `55601441`，Public **0.146**；
- 本轮已完成用户要求的全量训练、候选校验和 Kaggle 提交。
