# 迭代复盘：v3 Grid Test-Domain SSL——Proxy 门槛未通过

日期：2026-08-24。

## 目标

将已在 Event256 验证过的 train+test masked SSL 迁移到更强的 MultiStream v3 grid 主干，测试是否能同时获得 test 域适应与更强的原始 grid 表示。

严格使用 v3 基线口径：

- `grid_v3`：market 400 / tx 120 / order 120；
- d_model 64，Transformer 2 层；
- SSL：train(month<45)+全部 test，无标签，8 epochs，15% token mask；
- 监督：有效 batch 1024（512×2），12 epochs，LR 6e-4，70% cosine + 30% SmoothL1；
- checkpoint 对齐既有 recipe：ep4/5/6。

## SSL 资产

- `src/pretrain_grid_ssl.py`：对 market/tx/order 三个 Stream 做 masked reconstruction；保存与 `Stream.c/t/pos` 同构的 encoder 权重；
- `src/train_multistream_grid.py`：新增 `SSL_INIT_PREFIX` 加载逻辑与 OOF 保存；
- Proxy SSL loss：0.6422 → 0.6268；
- 训练监督时三路 encoder 均成功迁移（market 50/50，tx/order 38/50 keys）。

## Proxy 结果

| 版本 | Global | Month mean | Worst month |
|---|---:|---:|---:|
| v3 baseline ens(4,5,6) | 0.14752 | 0.14376 | 0.12158 |
| v3 + test-domain SSL ens(4,5,6) | **0.14931** | **0.14659** | **0.12803** |
| 增量 | **+0.00179** | +0.00283 | +0.00645 |

单模增量低于预注册 Proxy 门槛 **+0.003**，因此不进入三折。

将新 v3 成员替换进现有 Event Stack 的 Proxy 结果：

- 原 Event256：global +0.000599，月均 +0.000689，最差月 +0.001147；
- SSL Event：global +0.000630，月均 +0.000714，最差月 +0.000722；
- Stack global 增量低于 +0.0015 提交门槛。

## 决策

- **Proxy gate 失败，止损。** 不训练 Middle/Late，不生成候选，不提交 Kaggle；
- v3 主干 SSL 代码与结果保留，未来若改变预训练任务可复用；
- 本次没有使用伪标签，也没有 test 标签泄漏；
- 既有 Event256 test-domain SSL 仍作为并列 0.146 资产，但新增 v3 SSL 不升级当前 Stack。
