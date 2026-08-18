# 目标多样性扩展：Cosine-MultiRes、Recency止损、Joint-Cosine与Pairs Stack

## 1. Cosine目标扩展到Multi-Resolution

`LAMBDA_COS=1.0`重训Multi-Resolution，与原Multi-Resolution 50/50：

- Proxy：0.15007 → 0.15194（混合后）；
- 四路目标多样性（v3 + Cosine-v3 + MultiRes + Cosine-MultiRes各25%）：
  - Proxy：**0.15509**、Middle：**0.15145**、Late：**0.17022**；
  - 三折全局、月均、最差月全部超过原v3。

## 2. Recency采样止损

v3 + `RECENCY_HALFLIFE=18`（训练采样按时间近因加权）：

- Proxy最高仅0.12757，较原v3下降约0.020；
- 近因采样在Proxy上大幅失败，不训练Middle/Late，不生成候选；
- 该杠杆关闭。

## 3. Joint-Cosine

Joint模型改用纯Cosine目标：

| Fold | 原Joint | Cosine-Joint | 50/50混合 |
|---|---:|---:|---:|
| Proxy | 0.14958 | 0.15307 | **0.15439** |
| Middle | 0.14934 | 0.14394 | — |
| Late | 0.16303 | 0.16161 | — |

Proxy提升明显，Middle/Late单独不及原Joint，但与原Joint混合后用于Stack。

## 4. Pairs Stack（最终组合）

固定权重不变（LGB20/Real15/v3pair15/Jointpair35/MultiRes15），其中v3pair与Jointpair为原模型+Cosine版50/50：

| Fold | 原独立Stack | Pairs Stack |
|---|---:|---:|
| Proxy | 0.15865 | **0.15989** |
| Middle | 0.15603 | 0.15566 |
| Late | 0.17169 | **0.17259** |

三折平均约+0.0006，Proxy/Late提高、Middle微降。

## 5. 测试候选

已生成，未提交：

`output/candidate_pairs_stack_public60_40.csv`

SHA256：`218f096b5455692485aa4aad2d72d06911af3f523d6c18d5da2deda303321735`

- Cosine-v3与原v3测试相关性：0.9063；
- Cosine-Joint与原Joint测试相关性：0.9211；
- 新候选与Public0.145方案相关性：**0.99936**。

## 决策

- 候选相关性过高，离线增益低于Public显示精度，不自动提交；
- 目标多样性（同一输入、不同损失）是当前唯一稳定有效的离线增益来源，但在Stack中被LGB/RealMLP/Joint高相关性稀释；
- 回退方案仍为`55538309`（Public 0.145）；
- 若用户批准，可用Pairs Stack做一次Public校准提交。
