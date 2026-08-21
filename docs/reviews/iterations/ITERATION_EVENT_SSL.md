# 迭代复盘：Event256 SSL 自监督预训练——正信号但未过门槛，停止

日期：2026-08-20。分支：`opt/event-ssl`。执行依据：`docs/optimization/EXECUTION_PLAN_EVENT_SSL.md`（已 ignore）。

## 目标

自监督预训练（本项目从未测过的唯一剩余信息杠杆，见 ITERATION_POSTHOC_SWEEP_146）：
在 125 万无标签事件流上学习订单流结构，再微调监督任务，检验能否突破 Event256 基线。

## 方法（Proxy fold MVP）

- 输入：`features/event_cache_v2`（256 长度 order/transaction 事件序列），训练月 <45（797,486 样本）；
- SSL 预训练 8 epochs：15% 随机遮盖有效 token → 特征重建(Huber) + side 回归(MSE,0.3) + 时间间隔回归(Huber,0.2)；
- 编码器与监督 EventEncoder 同构（conv×2 + Transformer×2 + pos），51/51 权重 key 直接迁移；
- 监督微调与 Event256 基线完全一致：D64/NL2/EPOCHS12/LAMBDA0.8/ckpt 6,9,12/direct target；
- 评估：ens(6,9,12) unit 平均，与基线同口径。
- 新增代码：`src/pretrain_event_ssl.py`、`src/train_event_ssl.py`、`src/mscapital/run.py`、`src/mscapital/experiment.py`、`configs/event_ssl_proxy.yaml`。

## 结果（Proxy fold）

### 单模 ens(6,9,12) global cosine

| 版本 | global | 月均 | 最差月 |
|---|---:|---:|---:|
| Event256 基线 | 0.13113 | 0.12765 | 0.09700 |
| **Event256 + SSL** | **0.13300** | **0.12890** | **0.09770** |
| 增量 | **+0.00187** | +0.00125 | +0.00070 |

### 融入独立 Stack（event 12%，其余等比，proxy fold，OOF 实测）

| 组合 | global | 月均 | 最差月 |
|---|---:|---:|---:|
| 5 成员 base（无 event） | 0.15865 | 0.15503 | 0.13639 |
| + event256 原版 12% | 0.16035 | 0.15641 | 0.13754 |
| + event_ssl 12% | 0.16069 | 0.15674 | 0.13787 |
| SSL 净增量 vs 原版 | +0.00034 | +0.00033 | +0.00032 |

（5-member base 0.15865 与 ITERATION_EVENT256_FULL_SEQUENCE 文档一致，验证口径正确。）

## 判定

| 门槛（方案） | 要求 | 实测 | 判定 |
|---|---:|---:|---|
| Proxy 单模提升 | ≥ +0.003 | +0.00187 | ✗ |
| Stack 净增量（vs 含 event256 的 0.16026） | ≥ +0.0015 | +0.00034 | ✗ |

**结论：Proxy 未过门槛 → 按方案决策矩阵记录负结果，停止 SSL，不跑 Middle/Late。**

## 分析

1. SSL 产生的是**真实但不足**的正信号：单模三指标（global/月均/最差月）全部正向，Stack 净增量三指标也全部正向，方向与"扩展输入信息量"主线一致；
2. 但 +0.00034 的 Stack 净增量落在项目历史噪声区间（±0.0002–0.0005 不迁移 Public，见 README 经验 22/24/39/43）；
3. 与 ITERATION_POSTHOC_SWEEP_146 预判一致：SSL 受 D96/fusion/市场饱和强负信号制约，属高成本不确定赌注——本次验证确认其增益不足以支撑提交门槛；
4. SSL 单模相关性与原版 0.7953，多样性保留（与原版非高度共线），但强度提升不足。

## 资产沉淀（保留，未来可复用）

- `src/pretrain_event_ssl.py` / `src/train_event_ssl.py`：SSL 预训练 + 微调全流程（可换任务/加长预训练/换 Fold 复跑）；
- `src/mscapital/`：统一入口 `python -m mscapital.run --config ...`、实验 manifest 记录、splits/metrics/artifacts 模块；
- `tests/`：19 个数据契约测试（Event 缓存 256/时间切分/OOF 对齐/提交文件），防 32 截断复发；
- `configs/event_ssl_proxy.yaml`、`configs/event256_baseline.yaml`；
- 训练产物：`output/event_ssl_proxy_*.pt`、`output/event_ssl_proxy_oof.npz`（均 gitignore）。

## 决策矩阵落位

| 结果 | 后续动作 |
|---|---|
| Proxy 未过门槛 | ✅ 记录负结果，停止 SSL |
| Proxy 过、其他 Fold 失败 | 未触发 |
| 三折过、Public 不提升 | 未触发 |
| 三折过且 Public 提升 | 未触发 |

## 建议

1. 维持 `55601441`（Public 0.146）为最终方案，无新候选；
2. 若未来重启 SSL：优先加长预训练（8→16 epochs）+ 时间间隔任务加重，或换 GPT-style 自回归任务，且必须先过同一 Proxy 门槛；
3. 测试护栏与工程收口（阶段 1/4）已交付并入库，无论 SSL 结果如何均生效。
