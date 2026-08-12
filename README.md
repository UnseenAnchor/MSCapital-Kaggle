# MSCapital – Real Financial Market Forecasting

Kaggle 比赛实战记录：用真实市场微观结构数据（盘口/委托流/成交）预测未来价格变化。

- 比赛页: https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting
- 指标: 余弦相似度（预测 vs 真实目标）
- 截止: 2026-10-10 00:00 (北京时间) | 每日 5 次提交 | 个人赛

## 环境

- RTX 4070 12GB / 10 核 CPU / 64GB RAM
- Python 3.9 + LightGBM + PyTorch 2.1 (cu121)

## 数据

| 表 | 行数 | 内容 |
|---|---|---|
| train/market | 2.22亿 | 10 分钟盘口+聚合成交（13 列） |
| train/order | 1.70亿 | 1 分钟委托流（价格/数量/方向/增撤单） |
| train/transaction | 1.04亿 | 1 分钟逐笔成交（价格/数量/主动方向） |
| train/label | 126万 | sample_id → target（未来收益） |
| test 三表 | ~3.1亿 | 647,896 条预测 |

## 管线

```
data/*.feather → feat_market.py / feat_order_tx.py / feat_v2.py → features/*.parquet
→ train_v2.py (LightGBM, 时序验证) → output/submission_v2.csv → submit.py (kaggle API)
```

## 提交记录

| # | 日期 | 描述 | Val cosine | Public | 备注 |
|---|---|---|---|---|---|
| 1 | 2026-08-10 | lgb baseline 56 特征 (v1) | 0.1214 | 0.104 | 基础盘口+订单+成交聚合 |
| 2 | 2026-08-10 | lgb v2 93 特征 (多窗口变化率) | 0.1360 | 0.116 | v1+v2 特征合并 |
| 3 | 2026-08-11 | GRU 三路序列（GPU，15 epoch） | 0.1358 raw | - | RTX 4070，缓存序列训练 |
| 4 | 2026-08-11 | LGB+GRU 融合 40%/60% | 0.1483/0.1489 | 0.122/0.123 | 融合显著提升，提交 ref 55419933/55419935/55419936 |
| 5 | 2026-08-11 | LGB 50% + GRU 30% + Transformer 20% | **0.1503** | 待提交 | 三模型候选 `submission_blend_three_50_30_20.csv` |
| 6 | 2026-08-11 | LGB 30% + GRU 30% + Transformer 10% + Hybrid 30% | **0.1548** | 0.128 | ref 55444499；Public 第76/107名 |
| 7 | 2026-08-12 | Unit ensemble：LGB10+GRU20+Hybrid20+MultiStream-v2/v3各25 | late约0.166 | **0.136** | ref 55450450；Public 第42/107名，提升34名 |
| 8 | 2026-08-12 | Public-0.136配方90% + 434维Micro-v3 LGB 10% | late 0.16969 | **0.137** | ref 55458320；Public 第41/108名 |

## 关键经验

1. **验证必须防泄漏**：全量模型（含验证月）在验证集上 cosine 高达 0.30，真实无泄漏仅 0.136。按月份时序切分（0-61 训练 / 62-70 验证）。
2. **多窗口价格变化率**（10s/30s/60s/120s/300s 的 mid 变化率）是当前最有效特征族，Val 0.1214 → 0.1360。
3. 余弦指标关注方向一致性 → 特征越贴近"未来方向"越重要。
4. 四模型融合 Public 从 0.125 提升到 0.128，方向有效，但 Val 0.1548 与 Public 0.128 相差约 0.027，单一验证区间仍有过拟合风险。
5. 下一轮必须先做滚动月份验证，并拆分组件评估 Hybrid 的真实贡献，再决定是否上传。
6. 高分辨率 v2/v3 MultiStream + unit-normalized ensemble 将 Public 0.128 提升到 0.136、排名76→42，证明结构与尺度修复有效。
7. late 离线约0.166而Public仅0.136，仍有约0.030泛化差距；下一步不能继续按late单折调权，必须做多折OOF和成员归因。
8. 434维增强LGB（旧92 + 342维microstructure）在early/middle/late三折分别为0.13091/0.13701/0.14638；与旧LGB单位融合后三折均继续提升。
9. v2 MultiStream多seed在middle由单seed约0.140–0.145提升到0.14622；但未充分收敛的late seed会拖累融合，成员必须按OOF质量筛选，不能机械平均。
10. OOF稳健区间为旧LGB10% + 增强LGB30% + v2 60%；考虑Public泛化差距，下一候选只对已验证Public配方加入10%增强LGB。
11. Micro-v3候选将Public 0.136提升到0.137，但与旧候选相关系数0.99846，增量上限有限；下一阶段应开发低相关强模型，不再做相邻权重扫点。
12. 采用无拆月的LB代理CV（month 0–44训练、45–70验证）后，增强LGB得分0.13664，与Public 0.137高度接近，比late单折更有解释力。
13. Top128 RealMLP（batch1024、16 epochs、8-member、train-only筛选）与增强LGB相关性约0.87–0.89；LGB60%+RealMLP40%在proxy/middle/late均提升，且提高最差月份。
14. Top320 RealMLP代理CV仅0.13005，低于Top128，证明tabular神经网络同样需要控制特征维度。

## 下一步

- [x] 深度学习序列模型（GRU 三路编码，GPU 训练）
- [x] LGB + GRU 模型融合（Val 0.136 → 0.149）
- [x] Hybrid（GRU 序列 + 92 个 LGB 特征，Val 0.141）及四模型融合（Val 0.155）
- [x] Transformer 序列模型（Val 0.1285，作为多样性模型参与融合）
- [x] 高分辨率 MultiStream CNN–Transformer（v2 200/60 + v3 400/120）
- [x] 修复序列时间方向与 unit-normalized 融合
- [x] 滚动 LGB：配置、轮数、seed、Top-K、target clipping 审计
- [x] Top10 技术路线审计：见 `TOP10_STRATEGY.md`
- [x] MultiStream v2 多 seed middle/late OOF与弱成员淘汰
- [x] 增强 microprice/signed amount/new-cancel pressure 多窗口特征后的 LGB
- [x] 三折增强LGB与两折融合权重稳定性验证
- [ ] MultiStream v3 多 seed OOF（仅在成本/收益合理时继续）
- [x] 增强特征 RealMLP：Top128、batch1024、16 epochs、8-member与多折验证
- [x] 无拆月80万/46万LB代理CV及train-only特征筛选
- [ ] v3 MultiStream或其他低相关强单模，继续缩小Top10差距
