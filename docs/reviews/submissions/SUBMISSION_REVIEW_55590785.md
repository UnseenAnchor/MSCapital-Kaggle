# Submission Review 55590785

## 提交信息

- ref：`55590785`
- 文件：`output/candidate_pairs_stack_public60_40.csv`
- SHA256：`218f096b5455692485aa4aad2d72d06911af3f523d6c18d5da2deda303321735`
- 行数：647,896
- NaN/Inf：0
- Kaggle状态：`COMPLETE`
- Public：**0.145**
- Private：尚未显示

## 方案

```text
公开LB0.142参考 60%
Pairs Stack 40%
```

Pairs Stack：

```text
LGB 20% + RealMLP 15% + v3pair 15% + Jointpair 35% + Multi-Resolution 15%
```

其中v3pair和Jointpair为原模型与纯Cosine目标模型的50/50混合。

## Public结果

| 提交 | Public |
|---|---:|
| 独立Stack 55538309 | 0.145 |
| Pairs Stack 55590785 | **0.145** |

三折离线平均+0.0006的目标多样性增量未转化为Public提升。

## 复盘结论

1. 目标多样性（同输入不同损失）在OOF稳定有效，但在Public上再次被高相关性稀释。
2. 连续三次结构不同的候选（Multi-Resolution旧主体、Cross-Scale Delta、Pairs Stack）均停在0.145，说明当前Stack框架已达Public上限。
3. 自研锚点隐藏测试强度估计仍约0.133–0.135，距离Top10所需0.152差距约0.017–0.019，靠模型组合无法弥合。
4. 唯一被Public验证的增益仍是：独立Stack替换旧Public派生主体（0.144→0.145）。

## 决策

- 冻结Pairs Stack，不再提交该框架近邻；
- 回退方案保持`55538309`；
- 后续只有出现全新信息来源（而非新组合）时才值得再提交；
- 未经明确批准不再上传。
