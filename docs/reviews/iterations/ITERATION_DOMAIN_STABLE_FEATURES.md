# 迭代复盘：域稳定特征筛选——Late 止损

日期：2026-08-23。

## 假设

RealMLP v4 的 128 个特征长期沿用 Proxy target-only 排名，可能包含大量 train/test 漂移特征。使用已有无标签 adversarial domain gain 做固定惩罚：

```text
stable_score = target_gain / sqrt(1 + domain_gain / median(domain_gain))
```

- target gain：`proxy_lgb_trainonly_importance.csv`，只来自训练标签；
- domain gain：已有 train-vs-test classifier，未使用 target；
- 不扫描惩罚系数；仍固定 TOPN=128；
- 先 Late（train <62, validate 62–70），通过后才扩展。

## 结果

域稳定排名训练 16 epochs，Late EMA：

- global：0.127187
- month mean：0.126465
- worst month：0.107226

将该 RealMLP 作为原 Stack 的 RealMLP 成员替换，结果如下：

| Event 成员 | RealMLP 权重 | Global 变化 | 月均变化 | 最差月变化 |
|---|---:|---:|---:|---:|
| 原 Event256 | 13.2% | -0.000620 | -0.000362 | -0.000500 |
| 原 Event256 | 10.0% | -0.000449 | -0.000252 | -0.000367 |
| 原 Event256 | 5.0% | -0.000205 | -0.000106 | +0.000277 |
| SSL Event | 13.2% | -0.000621 | -0.000357 | -0.000500 |
| SSL Event | 10.0% | -0.000449 | -0.000248 | -0.000367 |
| SSL Event | 5.0% | -0.000205 | -0.000104 | +0.000049 |

候选 RealMLP 与旧 RealMLP OOF 相关性 0.8991；不是纯重复，但个体质量不足以抵消替换损失。

## 决策

- **Late gate 失败，止损。** 不扩展 Proxy/Middle/FULL，不生成候选，不提交 Kaggle；
- 保留排名生成器作为可复现实验资产，但不作为当前模型；
- 下一方向切换到 test-domain SSL 的强主干 v3 grid，而不是继续筛选同一 RealMLP 特征族。

工程验证：训练脚本编译通过；本轮未修改测试护栏。
