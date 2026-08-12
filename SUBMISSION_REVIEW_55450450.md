# 提交复盘：55450450

## 结果

- 日期：2026-08-12
- 文件：`candidate_top10_multiscale_conservative.csv`
- Public：**0.136**
- 排名：**42 / 107**
- 上一最佳：0.128，76 / 107
- 提升：+0.008（相对 +6.25%），排名提升34名
- Top10 门槛：0.152，当前差距0.016

## 候选构成

所有成员先去均值并单位范数化：

- LGB v2：10%
- GRU：20%
- Hybrid：20%
- MultiStream v2（200/60）：25%
- MultiStream v3（400/120）：25%

## 得到支持的假设

1. 高分辨率时间网格有效，64/32 压缩确实限制了模型上限。
2. 相对价格、microprice、盘口斜率、signed flow、增撤单压力比绝对原始序列更适合非平稳市场。
3. MultiStream CNN–Transformer 与 tabular/GRU 信号具有互补性。
4. Unit normalization 修复了旧融合中模型尺度不同导致的权重失真。

## 不能得出的结论

1. 不能确认 v2 与 v3 哪个 Public 更好，因为没有单独提交。
2. 不能确认旧 GRU/Hybrid 是否仍值得保留。
3. 不能把 late 验证0.166视为真实泛化能力；Public只有0.136。
4. 不能继续使用late单折搜索融合权重，否则会加剧过拟合。

## 风险与下一步门槛

下一轮必须先完成：

1. v2/v3 至少两个seed的 middle/late OOF；
2. 每个成员的OOF相关性、逐月cosine和最差月份；
3. 使用OOF拟合非负、带正则的融合权重；
4. 增强LGB/RealMLP的OFI、EWM、microprice、signed amount特征；
5. 新候选必须在多折均值和最差折均优于当前候选，才申请下一次上传。
