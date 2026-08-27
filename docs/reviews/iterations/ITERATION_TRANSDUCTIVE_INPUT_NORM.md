# 迭代复盘：无标签域适配归一化——Late Gate 失败

## 假设

CORAL 直接对齐 latent covariance 失败后，测试更简单的输入级无标签适配：训练标签仍只取 month<62，但 normalization 统计量使用全部 train covariates（Late 区间作为无标签域），模拟真实 train+test 统计量标准化。

## 结果

| checkpoint | centered cosine |
|---:|---:|
| ep2 | 0.14516 |
| ep4 | 0.14997 |
| ep5 | 0.15192 |
| ep6 | 0.14820 |
| baseline v3 ens(4,5,6) | **0.15995** |

## 决策

- Late gate 失败，停止剩余训练；不扩展三折、不提交；
- 输入级 transductive normalization 冻结；
- CORAL、输入归一化、SSL 域适配共同表明：当前 shift 不是简单的边际分布漂移可修复。
