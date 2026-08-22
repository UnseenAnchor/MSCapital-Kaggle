# 迭代复盘：伪标签自训练（SSL + Pseudo-Label）——三折 +0.0225，历史性突破

日期：2026-08-21。前情：test 域 SSL（ITERATION_EVENT_SSL_TESTDOMAIN）单模三折 +0.0069 但 Public 0.146 持平。

## 动机

SSL 只用 test 的**输入**结构（无监督），Public 未迁移。伪标签用 test 的**输入→输出映射**：
用当前最强 ensemble（Public 0.146 候选 w20）对 64.8 万 test 打伪标签，监督微调模型在 test 域的输出映射，攻击迁移差距的最深层。

## 方法（A/B，唯一变量 = 是否加 PL 阶段）

- 基础：train+test SSL 预训练 → 监督微调（EPOCHS=12, ckpt 6/9/12）→ **伪标签微调（test 64.8万 + ensemble 伪标签，PL_EPOCHS=12, lr=2e-4, ckpt pl6/9/12）**
- 评估：三折验证集（proxy 45-70 / middle 51-70 / late 62-70），ens(6,9,12) unit 平均，同口径
- PL_ONLY 模式复用已有 ep12 checkpoint，A/B 干净
- 新增：`src/train_event_ssl.py` PL 阶段（PL_EPOCHS/PL_TARGET/PL_LR/PL_ONLY/FULL+PL）、`predict_event_ssl_tt.py` CHECKPOINT_SUFFIX、`make_candidate_event_ssl.py` EVENT_SSL_CSV/USE_PL_OOF

## 结果：三折单模 ens(6,9,12) global cosine

| Fold | SSL(train+test) 监督 | +伪标签微调 | 增量 |
|---|---:|---:|---:|
| Proxy | 0.13656 | **0.15614** | **+0.01958** |
| Middle | 0.13733 | **0.15993** | **+0.02260** |
| Late | 0.14719 | **0.17257** | **+0.02538** |
| 平均 | | | **+0.02252** |

- 三折一致 +0.02 级提升，项目历史首次；最差月全部大幅改善（proxy 0.1246 / middle 0.1239 / late 0.1260 vs 监督 0.0977/0.1065/0.1177）
- 三个 PL checkpoint（pl6/9/12）单模型全部大幅高于监督（如 late: 0.1697/0.1732/0.1740 vs 监督 0.1439/0.1375/0.1340）——非单点噪声

### 融入 Stack（event 权重 12/15/20%，PL 版 vs 原版 event256）

| Fold | w=12% | w=15% | w=20% |
|---|---:|---:|---:|
| Proxy | +0.00059 | +0.00081 | +0.00120 |
| Middle | +0.00037 | +0.00052 | +0.00082 |
| Late | +0.00062 | +0.00085 | +0.00128 |
| 平均 | +0.00053 | +0.00073 | **+0.00110** |

三折全正向，w20 最优（stack 权重上限内的净增量）。

## 泄漏审查

- PL 训练数据 = test 流 + ensemble 对 test 的预测；验证 = train 各 fold（mo 45-70 等），无重叠；ensemble 预测不包含 train 验证标签 → 无直接泄漏；
- A/B 唯一变量是 PL 阶段；三折独立验证一致性 +0.02 → 非过拟合单折；
- 残余风险：伪标签含 ensemble 偏差，模型可能部分学到偏差；但验证集（真标签）大幅提升说明学到的是有效映射。

## 提交

- **2026-08-21 ref 55683040**：`output/candidate_event_ssl_tt_public60_40_w20.csv`（PL 版，SHA256 `94457caf9514b015...`），经用户批准提交；
- 前次 55666656（SSL 版 w20）Public 0.146 持平；
- 55683040 Public 评分待回填。

## 决策矩阵

| Public 结果 | 后续 |
|---|---|
| > 0.146 | 新基线；固化 PL 管线；考虑调 event 权重/多轮 PL |
| = 0.146 | 单模强度未转化为 Public；保留研究，评估是否多轮 PL/伪标签迭代 |
| < 0.146 | 伪标签有害，冻结 |

## 资产

- 代码：train_event_ssl.py（PL 全流程）/ predict_event_ssl_tt.py / make_candidate_event_ssl.py（均已入库）
- 产物：`output/event_ssl_tt_{proxy,middle,late,full}_pl*.pt`、`{fold}_pl_oof.npz`、`submission_event_ssl_tt_full_pl_unit.csv`（gitignore）
- 训练总量：SSL 预训练 ×4 fold + 监督 ×4 + PL ×4，~10 GPU 小时
