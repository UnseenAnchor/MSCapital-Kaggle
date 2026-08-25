# 迭代复盘：Joint v3 + Test-Domain SSL——Proxy 止损

日期：2026-08-27。

## 假设

在 v3 grid 的三条基础流上使用 train+test masked SSL，再迁移到更强的 `MODEL_VARIANT=joint`；joint 融合流随机初始化，基础 market/tx/order encoder 加载 SSL 权重。

固定配置：grid_v3 400/120、D64/2层、有效 batch1024、LR 6e-4、70% cosine +30% SmoothL1、12 epochs、ep4/5/6。

## Late 探针

Late 单模 ens(4,5,6)：

- Joint baseline：0.163029
- Joint + SSL：0.165264
- 增量：+0.002234

直接替换 Joint 成员后 global 上升，但月均/最差月下降；在保留旧 Joint、额外加入新 Joint-SSL 5% 时，Late 三项小幅全正向。

## Proxy Gate

Proxy 使用同一 SSL 初始化和同一训练口径：

- Joint baseline ens(4,5,6)：0.149576
- Joint + SSL 单轨最佳 checkpoint：0.140784
- 单模明显回退，未达到 +0.002 门槛；
- 不继续 Middle，不生成候选，不提交 Kaggle。

## 决策

- **Proxy 止损**：Late 局部提升不能跨折迁移；
- Joint SSL 方向冻结；
- 本轮没有可提交候选，最佳 Public 仍为 0.146。
