# 迭代复盘：月份均衡采样——Late Gate 失败

日期：2026-08-29。

## 假设

训练样本按月份数量不均衡，早期月份可能压制近期状态。启用现有 `MONTH_BALANCED` 采样开关，对 train(month<62) 按月份逆频率采样；验证仍使用完整 Late（month 62–70）。配置保持 v3 grid 400/120、D64/2层、有效 batch1024、LR 6e-4、70% cosine +30% SmoothL1、12 epochs。

## 结果

| 版本 | Late 最佳单点 global |
|---|---:|
| v3 baseline ens(4,5,6) | **0.159950** |
| month-balanced ep2 | 0.14360 |
| month-balanced ep4 | 0.15286 |
| month-balanced ep12 | 0.11092 |

最佳单点为 ep2 `0.14360`，仍大幅低于 baseline；checkpoint ensemble 无需继续评估即可判定失败。

## 决策

- **Late gate 失败，立即止损**；不扩展 Proxy/Middle/FULL，不提交；
- 现有月份均衡采样开关默认保持关闭；
- 采样权重类分布适配已与 density-ratio、recency 一并冻结。
