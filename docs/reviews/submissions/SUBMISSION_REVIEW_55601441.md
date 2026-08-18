# Submission Review 55601441

## 提交信息

- ref：`55601441`
- 文件：`output/candidate_event256_stack_public60_40.csv`
- SHA256：`f6669b38ba11a0fb2a869c0c03b9a08c18f662c461cf838bd5cc8688b404fb91`
- 行数：647,896；无NaN/Inf；格式校验通过
- Kaggle状态：`COMPLETE`
- Public：**0.146**
- Private：尚未显示

## 方案

```text
公开LB0.142参考 60% + 自研Stack 40%
```

自研Stack（Event256权重12%，其余等比缩放）：

```text
LGB 17.6% + RealMLP 13.2% + v3 13.2% + Joint 30.8% + MultiRes 13.2% + Event256 12%
```

Event256 = 全长度（256事件）原始 order/transaction 事件Transformer（修复原32事件截断）。

## Public结果

| 提交 | Public |
|---|---:|
| 独立Stack 55538309 | 0.145 |
| Pairs Stack 55590785 | 0.145 |
| **Event256 Stack 55601441** | **0.146** |

## 复盘结论

1. 完整事件流（256 vs 32）是自独立Stack以来首个真正"全新信息源"，被Public验证为有效。
2. 三折OOF平均+0.0015，Public +0.001，迁移率约2/3，方向一致。
3. 找到了之前所有"目标多样性/多尺度/差分"路线失败的根因：它们复用同一聚合输入。只有扩展**输入信息量**才能突破。
4. 后续方向：继续沿"扩展原始事件信息"（更长序列、双流grid+event联合）而非再组合聚合特征。

## 决策

- 更新最佳方案为`55601441`（Public 0.146）；
- 继续探索Event序列的进一步强化（512长度、更多通道、grid双流融合）。