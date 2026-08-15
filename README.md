# MSCapital – Real Financial Market Forecasting

Kaggle 比赛实战记录：用真实市场微观结构数据（盘口/委托流/成交）预测未来价格变化。

- 比赛页: https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting
- 指标: 余弦相似度（预测 vs 真实目标）
- 截止: 2026-10-10 00:00 (北京时间) | 每日 5 次提交 | 个人赛
- 复盘索引: [`docs/reviews/README.md`](docs/reviews/README.md)

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
| 9 | 2026-08-12 | Public-0.137配方95% + ProxyCV RealMLP-v4 5% | late 0.16977 | **0.138** | ref 55459787；Public 第41/108名，领先0.137分组 |
| 10 | 2026-08-13 | Public-0.138配方80% + v3 effective-batch1024 20% | late 0.17087 | **0.140** | ref 55469774；进入0.140分组 |
| 11 | 2026-08-13 | v3成员替换为seed42 80% + seed13 20% | late 0.17101 | **0.140** | ref 55482231；未跨三位小数阈值 |
| 12 | 2026-08-13 | Public-0.138基础80% + 原v3 8% + 时间对齐联合流v3 12% | late 0.17192 | **0.140** | ref 55487734；Public无可见增量 |
| 13 | 2026-08-14 | 自研Public-0.140候选40% + 公开LB0.142参考60% | 端点/相关性估计0.1442–0.1452 | **0.144** | ref 55496148；第31/115名，提升约9名 |
| 14 | 2026-08-14 | 纯v9_big事件计数通道模型，诊断提交 | Proxy 0.14591 | **0.129** | ref 55500749；诊断CV–LB偏差，不影响最佳成绩 |
| 15 | 2026-08-14 | 公开LB0.142中的纯ens5组件，诊断提交 | 公开冻结组件 | **0.136** | ref 55500948；确认MultiStream强度块 |
| 16 | 2026-08-14 | 公开LB0.142中的纯v10 RealMLP组件，诊断提交 | 作者两折均值0.13994 | **0.129** | ref 55500950；确认低相关多样性块 |
| 17 | 2026-08-15 | 纯原始事件序列Event Transformer，诊断提交 | Proxy 0.1459 | **0.103** | ref 55517939；独立信号存在但最优融合仅预计+0.000017 |

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
15. RealMLP 5%候选将Public 0.137提升到0.138；代理CV成功识别了late单折几乎看不到的测试状态增益，后续应提升RealMLP单模而非扫相邻融合权重。
16. RealMLP seed13只轻微改善middle但损害late，seed77更弱；多seed不是无条件有效，二者均未进入全量候选。
17. 新v3 MultiStream采用micro-batch128×梯度累积8=effective batch1024、12 epochs，在proxy/middle/late的LGB40+v3 60融合分别达到0.15419/0.15299/0.16575以上，并显著提高最差月份。
18. 新v3与当前Public候选测试相关性0.885，具备比Micro-LGB更强的独立增量；下一候选保守加入20%。
19. v3 20%候选将Public从0.138提升到0.140，三套CV再次正确预测Public方向；下一步应继续增强独立单模，不提交25%/30%相邻权重扫点。
20. v3 seed13通过严格逐级闸门；固定4/5/6 checkpoint、seed42 80%+seed13 20%在proxy/middle/late均同时提高全局、月均和最差月。
21. 多seed只替换Public-0.140候选中的v3成员，late重建0.170865→0.171012；候选变化很小但归因清晰。
22. 多seedPublic仍显示0.140，说明+0.0001级离线增益不足以支持提交；保留多seed作内部降方差，但停止相邻seed权重实验。
23. RAM整批索引、GPU端归一化与physical256×accum4将端到端吞吐从约1700提升到约4040 samples/s（2.36倍），v3训练每轮降至约5分钟。
24. 时间对齐联合流在Proxy/Middle/Late和完整late重建均显著提升，但Public仍为0.140；说明现有历史CV仍不能完全代表隐藏测试期，停止该架构的相邻权重搜索。
25. Train/Test对抗验证AUC 0.795，确认协变量漂移明显；密度比重采样和domain门控均损害测试相似组，已淘汰。
26. 公开LB0.142 slim pack的6成员公式可逐行复现（最大误差1.1e-16）；其预测与我们的Public-0.140候选相关性0.901，40/60单位融合理论Public约0.1442–0.1452。
27. 40/60公开锚点融合Public实测0.144，验证低相关分母收益；不扫30/70或50/50等相邻权重，下一步需寻找第三个低相关强锚点。
28. v9_big诊断单模Proxy 0.14591但Public仅0.129；与最佳候选相关性0.8875仍不足以补偿强度差，理论最优融合仅增约0.000023，停止该架构族。
29. 公开LB0.142拆解为ens5 Public 0.136与v10 Public 0.129；二者相关性0.7526带来显著分母收益，但相对当前0.144的三端点重优化仅预计+0.000008，不提交。
30. 以嵌套时间切分训练Residual-LGB、Residual-RealMLP和Residual CNN–Transformer；序列残差在三折均为正，但Late完整候选仅+0.000026且最差月下降，现有特征空间残差基本耗尽。
31. 原始Event Transformer保留逐笔顺序、连续时间、事件间隔和增撤单；全历史直接目标在三折2%权重均不下降，测试与0.144候选逐行重算相关性0.7044，纯模型诊断已获Public 0.103。
32. Event Public 0.103虽高于正边际阈值，但与当前0.144的唯一最优融合仅预计+0.000017；不提交融合，早期0.6695相关性口径已由逐行重算纠正为0.7044。
33. 全量审计结论：当前0.144的有效来源是公开LB0.142参考60% + 已实测Public0.140自研锚点40%；低相关弱模型均不能替代自研锚点，后续只优化v3自研锚点，不再做诊断连投或相邻权重搜索。详见`docs/reviews/iterations/ITERATION_FULL_RESULTS_AUDIT_0144.md`。
34. Top10门槛0.153要求新的自研锚点约达到0.151–0.153；当前所有已测模型均不达标。下一候选必须先通过自研锚点强度闸门，不达标不生成提交文件。
35. v3 d_model96/3层容量增强Proxy ensemble仅0.14086，低于原v3 0.14752；简单堆模型容量不是Top10路线，已在Proxy止损。
36. v2+v3 Multi-Resolution混合三折达到0.15348/0.15079/0.16679；原v3+Multi-Resolution 50/50在五个测试相似分组全部提高，最像测试五分位0.14020→0.14393，当前保留为最强自研锚点候选。
37. Multi-Resolution全量锚点与旧自研锚点相关性0.99894，旧Public派生候选仍占80%；下一步必须替换旧锚点主体，不能继续做20%槽位微调。
38. 经用户批准提交ref 55537873，Public仍为0.144；Multi-Resolution三折提升未转化为Public提升，下一候选必须是真正独立的自研锚点。

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
- [x] v3 MultiStream effective-batch1024、12 epochs及proxy/middle/late共同验证
- [x] RealMLP多seed审计与弱seed淘汰
- [x] 新v3候选Public验证：0.138 → 0.140
- [x] v3第二seed逐级Proxy/Middle/Late验证与全量训练
- [x] 多seed候选Public验证：仍为0.140，停止seed权重微调
- [x] Train/Test对抗验证与domain weighting审计
- [x] 公开LB0.142包来源、公式及逐行复现审计
- [ ] 公开0.142×自研0.140融合候选；获用户批准前不提交
