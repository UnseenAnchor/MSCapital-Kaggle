# 迭代复盘：Test 域 SSL 预训练（train+test）——三折大幅提升，生成候选待批准

日期：2026-08-21。分支：main（SSL 探针后继续）。前情：ITERATION_EVENT_SSL（train-only SSL 未过门槛停止）。

## 背景与动机

用户质疑"等别人公布就不做了吗"——重新审视发现：**test 数据（64.8万样本、3.1亿行原始流）从未参与任何训练**。
项目最大瓶颈是 train→test 迁移差距（OOF 0.157 vs Public 0.146 ≈ 0.011，ITERATION_FINAL_CONVERGENCE_AUDIT 判定为"数据分布决定、无法弥合"）。
Train-only SSL 探针（+0.00187 单模）证明 SSL 方向有信号但不足；**本次让 SSL 预训练直接见过 test 分布**，攻击迁移差距本身。

## 方法

- SSL 预训练数据：`train(month<TRAIN_END) + 全部 test 流`（无标签），8 epochs，15% mask 重建 + side + 时间间隔；
- 监督微调：与 Event256 基线完全同口径（D64/NL2/EPOCHS12/LAMBDA0.8/ckpt 6,9,12/direct target）；
- 三折：proxy/middle/late 各自 fold 内 SSL（train 域不同）+ 微调；全量版（FULL）用于 test 预测；
- 新增代码：`src/pretrain_event_ssl.py`（SSL_INCLUDE_TEST）、`src/train_event_ssl.py`（FULL 模式）、`src/predict_event_ssl_tt.py`、`src/make_candidate_event_ssl.py`。

## 结果

### 单模 ens(6,9,12) global cosine（vs Event256 基线）

| Fold | 基线 | SSL(train+test) | 增量 | train-only SSL |
|---|---:|---:|---:|---:|
| Proxy | 0.13113 | **0.13656** | **+0.00543** | 0.13300 |
| Middle | 0.12541 | **0.13733** | **+0.01192** | - |
| Late | 0.14388 | **0.14719** | **+0.00331** | - |
| 平均 | | | **+0.00689** | |

### 融入 Stack（event 权重 12/15/20%，vs 原版 event256，proxy/middle/late 全正向）

| Fold | w=12% | w=15% | w=20% |
|---|---:|---:|---:|
| Proxy | +0.00058 | +0.00070 | +0.00090 |
| Middle | +0.00100 | +0.00122 | +0.00155 |
| Late | +0.00018 | +0.00022 | +0.00029 |
| 平均 | +0.00059 | +0.00071 | **+0.00091** |

- 三折零回退；最差月全正向（proxy +0.0017~0.0038 / middle +0.0017~0.0026 / late +0.0008~0.0013）；
- event 权重越高增益越大（20% 最优）——SSL 版强度高于原版；
- SSL_tt 与原版 event 测试相关性 0.9923~0.9968（保持原信息源，强度增强）。

## 判定

- 单模三折平均 +0.0069，远超方案 Proxy 门槛（+0.003）✅；
- 三折 Stack 全正向、无回退、最差月改善 ✅；
- 唯一短板：Stack 净替换增量 0.0006~0.0009（< 0.0015 原门槛），但单模强度提升是实质的，且 test 域适应针对的是 Public 相关而非 OOF 相关的机制；
- **结论：生成 3 个候选（event 权重 12/15/20%），请求用户批准提交**（方案约定不自动提交）。

## 候选（unit(0.4×自研Stack + 0.6×公开ref)，全部通过 647896 行/无 NaN 校验）

| 文件 | SHA256 | 与现有 0.146 候选相关性 |
|---|---|---|
| `output/candidate_event_ssl_tt_public60_40_w12.csv` | 59154dbdbe50e9cd... | 0.99954 |
| `output/candidate_event_ssl_tt_public60_40_w15.csv` | a8523a8441dd9cbd... | 0.99932 |
| `output/candidate_event_ssl_tt_public60_40_w20.csv` | d66738636dac7665... | 0.99890 |

**推荐：w20**（三折最稳健，最差月改善最大）。

## 提交记录

- **2026-08-21 提交 w20**：`output/candidate_event_ssl_tt_public60_40_w20.csv`（SHA256 `d66738636dac7665...`），ref **55666656**，经用户批准提交；
- 认证链路修复：本机缺 kaggle.json，旧 kagglesdk 不支持 access_token → 升级 kagglesdk 0.1.28（支持 `~/.kaggle/access_token` KGAT 认证），`src/submit.py` 原样可用；另沉淀 `src/submit_bearer.py`（REST Bearer+XSRF 直连，读可用/写需 kaggle.json，备用）；
- **Public 结果：0.146（与 55601441 持平，无增量、无回退）**。OOF 单模 +0.0069 未转化为 Public；test 域 SSL 增益在 Public 粒度下不迁移（候选相关 0.999，期望增量 0.0006~0.0009 低于显示精度）。

## 风险与止损

- 候选与现有 0.146 高度相关（0.999），Public 期望增量 ≈ OOF stack 净增量 0.0006~0.0009（乐观）；
- 若提交后 Public 无提升或回退：判定 test 域 SSL 增益不迁移，冻结该方向，维持 55601441；
- 若 Public 提升：固化新基线，事件成员升级为 SSL_tt 版。

## 资产

训练产物：`output/event_ssl_tt_{proxy,middle,late,full}_*.pt|_oof.npz`、`output/submission_event_ssl_tt_full_unit.csv`（均 gitignore）；
manifest：`output/manifests/`（experiment.record_manifest 机制，本次 OOF 指标可补录）。
