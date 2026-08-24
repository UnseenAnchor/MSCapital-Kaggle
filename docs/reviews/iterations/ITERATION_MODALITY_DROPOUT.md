# 迭代复盘：v3 Modality-Dropout——Late Gate 失败

日期：2026-08-25。

## 假设

现有 MultiStream v3 只在卷积内部使用 Dropout，可能过度依赖某一条输入流。训练时固定按 batch 采样：55% 完整 market+tx+order，15% 屏蔽 market，15% 屏蔽 tx，15% 屏蔽 order；验证始终使用完整三流。

配置：grid_v3 400/120，D64/2层，有效 batch1024，LR 6e-4，70% cosine +30% SmoothL1，12 epochs，checkpoint 4/5/6。

## Late 结果

| 版本 | Global |
|---|---:|
| v3 baseline ens(4,5,6) | **0.159950** |
| modality-dropout ens(4,5,6) | 0.153951 |
| 增量 | **-0.005999** |

其他参考：dropout ep5 单点 0.151039，ep3/4/5/6 四点平均 0.154218；均低于基线。

## 决策

- **Late gate 失败，立即止损**；不跑 Proxy/Middle/FULL，不生成候选，不提交；
- modality dropout 与当前三流训练的完整信息需求冲突，冻结；
- 代码开关保留但默认关闭。
