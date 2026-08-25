# 迭代复盘：Latent CORAL-lite——Late Gate 未通过

日期：2026-08-28。

## 假设

已有域适应只改变采样或最终权重；本轮在 supervised v3 grid 训练时加入 representation-level train/test alignment。每个有标签 train batch 配对一个无标签 test batch，对模型进入 head 前的 latent representation 加 CORAL 协方差损失：

```text
loss = supervised_loss + 0.5 * mean((Cov(z_train) - Cov(z_test))²)
```

不使用 test 标签、不使用伪标签、不改验证/测试输入。

配置：grid_v3 400/120，D64/2层，有效 batch1024，LR 6e-4，70% cosine +30% SmoothL1，12 epochs，ep4/5/6。

## Late 结果

| 版本 | Global | Month mean | Worst month |
|---|---:|---:|---:|
| v3 baseline ens(4,5,6) | **0.159950** | — | — |
| CORAL-lite ens(4,5,6) | 0.159510 | 0.148191 | 0.121916 |
| 增量 | **-0.000440** | — | — |

单点 ep6 达到 0.157128，但 checkpoint ensemble 仍回退。训练额外成本约为普通训练两倍，未见收益。

## 决策

- Late gate 未通过，不扩展 Proxy/Middle/FULL；
- CORAL-lite 默认关闭，域对齐监督路线暂时冻结；
- 当前仍无新候选，最佳 Public 保持 0.146。
