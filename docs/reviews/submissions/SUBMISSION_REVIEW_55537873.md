# Submission Review 55537873

## 提交信息

- ref：`55537873`
- 文件：`output/candidate_top10_multires_self_anchor.csv`
- SHA256：`ef649237fbf9ef69f010e9f56a31ccf37baf8598155537e77bc233e0e2ac129d`
- 行数：647,896
- 格式：`sample_id,prediction`
- NaN/Inf：0
- 提交描述：`top10 trial: fixed public60 self-anchor40 multires-v3`
- Kaggle状态：`COMPLETE`
- Public：**0.144**
- Private：尚未显示

## 提交前方案

```text
公开LB0.142参考 60%
新自研锚点 40%
```

新自研锚点由旧自研主体和Multi-Resolution改进槽位构成；Multi-Resolution固定checkpoint 5/6/7，并与原v3 50/50混合。

## 离线预期

Multi-Resolution与原v3的三折混合结果：

| Fold | 原v3 | 混合后 |
|---|---:|---:|
| Proxy | 0.14752 | 0.15348 |
| Middle | 0.14825 | 0.15079 |
| Late | 0.15995 | 0.16679 |

测试相似状态五分组中，50/50混合全部优于原v3，最像测试组由0.14020提高至0.14393。

## Public结果

当前最佳提交`55496148`也是Public **0.144**。本次提交没有产生可见提升：

```text
55496148: 0.144
55537873: 0.144
```

## 复盘结论

1. Multi-Resolution的三折Proxy/Middle/Late提升没有转化为Public提升。
2. 全量新自研锚点与旧自研锚点相关性为0.99894，说明实际提交中的独立新信号权重太小。
3. 离线提升主要来自历史CV局部状态，仍存在CV→Public迁移偏差。
4. 固定公开60%后，新候选没有突破当前0.144；距离Top10约0.153仍差0.009。
5. 这次提交只消耗一次额度，证明“在旧Public派生主体上增加20%新模型”不是Top10路线。

## 决策

- 不再提交当前Multi-Resolution候选的任何近邻权重；
- 冻结当前Public0.144方案作为回退方案；
- 下一次提交必须来自真正独立的自研锚点，且自研部分不再含80%的旧Public派生主体；
- 未得到明确批准前不再上传。

## 文件状态

本次候选和全量Multi-Resolution checkpoint保留，用于后续独立自研锚点实验；不作为新的最佳提交文件。
