# 迭代复盘：RQ target auxiliary——Late Gate 失败

## 背景

公开 Kaggle RealMLP Notebook 使用递归残差 KMeans（RQ）对 target 编码，并以分类头辅助回归。该目标构造此前未在本项目验证，因此在当前 microstructure Top128 RealMLP 上复现：3 层、每层 3 类、仅 train 子集拟合 codebook，`LAMBDA_RQ=0.1`。

## 结果

| 指标 | RQ Late EMA | baseline v3 |
|---|---:|---:|
| global cosine | 0.13596 | **0.15995** |
| month mean | 0.13304 | — |
| month min | 0.11492 | — |

单 epoch 最佳 global 约 `0.14037`，仍大幅落后 baseline。

## 决策

- Late gate 失败，立即停止；不扩展三折、不生成 test candidate；
- RQ target auxiliary 冻结；
- 公开 Notebook 的提升不能简单归因于 RQ 目标，可能来自其完整 factor 特征与不同特征筛选流程。
