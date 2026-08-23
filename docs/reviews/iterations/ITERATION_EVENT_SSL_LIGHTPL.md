# 迭代复盘：轻量伪标签（冻结 Backbone，仅 Head）——Late Gate 未通过

日期：2026-08-21。承接重型 PL：三折 OOF 单模 +0.0225，但 Public 0.145（较最佳 -0.001），判定为逼近 teacher、损失多样性的过拟合。

## 假设

用极低强度 PL 保留 SSL 模型多样性：

- 冻结 `tx` / `o` encoder、cross transformer 和 type embedding；仅训练 `head`（41,217 参数）；
- `PL_LR=1e-5`，test 伪标签训练 2 epochs；
- teacher 必须精确对应 Public 0.146 的 Kaggle ref **55666656**；
- 先做 Late fold，Stack global 不得低于无 PL 的 SSL 版本，否则止损。

## Teacher 资产修复

重型 PL 实验曾覆盖可变路径 `candidate_event_ssl_tt_public60_40_w20.csv`。本轮从 Kaggle API 下载原提交：

- `output/teacher_submission_55666656.csv`
- SHA256 `d66738636dac7665d1fc2c4993939907174d8e6d4b0cf619ae3ea9007cfe6411`
- 与原复盘记录 `d66738636dac7665...` 完全一致。

代码默认 PL teacher 改为该稳定文件，避免后续候选生成覆盖训练标签。

## 实验 1：只从 ep12 继续

| 模型 | Late global |
|---|---:|
| 监督 ens(ep6,9,12) 参考 | 0.147194 |
| ep12 + light PL epoch1 | 0.141806 |
| ep12 + light PL epoch2 | 0.141844 |
| light PL ens(1,2) | 0.141930（-0.005264） |

失败原因：仅适配 ep12，丢失原 ep6/9/12 checkpoint ensemble 多样性。

## 实验 2：多样性保留修正

分别从监督 checkpoint ep6、ep9、ep12 启动相同轻量 PL，再 ensemble 三条独立轨迹。

单轨迹 ens(PL epoch1,2)：

| 起点 | Late global | vs 监督 ens |
|---|---:|---:|
| ep6 | 0.142963 | -0.004232 |
| ep9 | 0.138886 | -0.008308 |
| ep12 | 0.141930 | -0.005264 |

与候选生成一致的 Late w20 Stack 口径：

| event 成员 | global | month mean | worst month | vs SSL global |
|---|---:|---:|---:|---:|
| SSL 无 PL | **0.174998** | **0.164752** | 0.142644 | — |
| light PL 三轨迹 epoch1 | 0.173978 | 0.164203 | **0.142690** | **-0.001020** |
| light PL 三轨迹 epoch2 | 0.173653 | 0.163989 | 0.142661 | -0.001345 |
| light PL 六模型全平均 | 0.173804 | 0.164090 | 0.142675 | -0.001194 |

最佳版本仅让最差月 +0.000046，但 global -0.001020、month mean -0.000549，明确未过门槛。

## 决策

- **止损：不扩展 Proxy/Middle/FULL，不生成候选，不提交 Kaggle。**
- 轻量 PL 仍然降低 Stack 主指标；冻结 backbone 无法消除 teacher 偏差与多样性损失。
- test 伪标签方向（重型和轻型）全部关闭。最佳仍为 ref 55601441 / 55666656，Public 0.146。

## 工程改动

`src/train_event_ssl.py` 新增：

- `PL_HEAD_ONLY`：只训练 head，并固定 frozen backbone 为 eval 模式；
- `PL_RESUME_EPOCH`：分别从 ep6/9/12 恢复；
- `PL_TAG`：隔离 checkpoint/OOF 名称，避免覆盖重型 PL；
- `PL_SAVE_EPOCHS`：支持短 PL 实验（如 1,2）；
- PL 参数与保存 epoch 一致性检查。

验证：`py_compile` 通过，pytest **19/19** 通过。
