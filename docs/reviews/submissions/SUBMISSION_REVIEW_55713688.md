# Submission Review 55713688

## 提交信息

- ref：`55713688`
- 文件：`output/candidate_dual_event_public60_40_e10_s20.csv`
- SHA256：`9d91dd27849cf144d77bf66a03abb4ad91fd7d1313682e4841999f31586b15b0`
- 行数：647,896；ID 0..647895 唯一对齐；无 NaN/Inf；centered norm=1
- Kaggle状态：`COMPLETE`
- Public：**0.146**
- Private：尚未显示

## 方案

```text
公开 LB0.142 参考 60% + Dual-Event 自研 Stack 40%
```

自研 Stack 同时保留：

- 原 Event256，参数权重 0.10；
- supervised test-domain SSL-Event，参数权重 0.20；
- 原五成员权重不变：LGB .176 / RealMLP .132 / v3 .132 / Joint .308 / MultiRes .132；
- 全部成员中心化并 unit-normalize，再整体归一化权重。

没有使用 pseudo-label OOF 或 pseudo-label checkpoint。

## 离线门槛

相对原 Event256 w20 的当前 Public 0.146 基线：

| Fold | Global | Month mean | Worst month |
|---|---:|---:|---:|
| Proxy | +0.001343 | +0.001586 | +0.003467 |
| Middle | +0.001345 | +0.001288 | +0.001769 |
| Late | +0.001101 | +0.001375 | +0.000750 |

- 折级 9/9 指标正向；
- LOO held-out global +0.001362 / +0.001123 / +0.000893；
- 36/45 月正向，Late 8/9；
- 与 ref 55601441 相关性 0.99884837。

完整输入哈希与审计：`CANDIDATE_DUAL_EVENT_E10_S20_MANIFEST.json`。

## Public 结果

| 提交 | 方案 | Public |
|---|---|---:|
| 55601441 | 原 Event256 Stack | **0.146** |
| 55666656 | SSL-Event 替换版 | **0.146** |
| **55713688** | **原 Event256 + SSL-Event** | **0.146** |

## 复盘结论

1. 双事件保留策略三折稳定且 Public 无回退，说明互补信号存在，但增量未达到 0.001 显示精度。
2. Public60 稀释后预估提升仅 +0.0004~0.0006；最终显示 0.146 与预期风险一致。
3. 不提交相邻权重。继续扫描 e/s 只会放大选择偏差，且没有足够期望收益。
4. 最佳 Public 仍为 0.146；55713688 可作为结构更稳健的并列方案，但不替代 55601441 的已验证基线地位。

## 决策

- 保留 `55601441` 为主基线，记录 `55713688` 为并列 0.146；
- Dual-Event 后置组合方向停止；
- 当日剩余提交额度：2。
