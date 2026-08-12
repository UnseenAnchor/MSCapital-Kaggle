# MSCapital Top10 技术路线审计（2026-08-12）

## 结论

最有可能冲击 Top10 的路线不是继续单独调 LGB，也不是普通 GRU，而是：

1. **多尺度高分辨率微观结构网格**：Market 200/400 格；Order/Transaction 60/120 格。
2. **流内 CNN + Transformer**：CNN 捕获局部冲击/订单流形态，Transformer 捕获长程依赖。
3. **三路独立编码 + Cross-stream attention**：盘口、委托、成交先分开，再交互。
4. **强 tabular 分支**：LGB 多 seed + RealMLP batch ensemble，使用完整微观结构因子。
5. **直接优化 cosine，推理去均值并单位范数化**。
6. **多尺度、多容量、多 seed 模型 unit ensemble**，而不是按原始预测数值直接加权。

## 证据

- 当前项目最佳 Public：0.128，第76/107。
- 当前 Top10 Public 约 0.152–0.155。
- Kaggle 公开包 `yangq369/kaggle-lb0142-upload` 明确记录 LB 0.142：
  - 5 个 CNN–Transformer 成员（200/60 与 400/120 两类网格）；
  - RealMLP 8-member tabular ensemble；
  - `0.6 * mean(unit(ens5)) + 0.4 * unit(v10)`。
- 当前本地严格 late 验证：
  - LGB：0.1360；旧 GRU：0.1353；旧 Hybrid：0.1410；
  - 旧四模型融合：0.1548；
  - 新 MultiStream v2 单 checkpoint：raw 0.1480；checkpoint ensemble centered 0.1566；
  - 新 MultiStream v3 checkpoint ensemble centered 0.1558；
  - unit-normalized 多模型融合最高约 0.166–0.169（仅 late 折，不能直接当泛化结论）。
- middle 折（month<51 训练，51–60验证）新 MultiStream：0.1386，对比该折 LGB 约 0.129，提升不是单折偶然。

## 历史代码关键问题

1. 旧 cache 用 `arr[::-1]` 产生“最近→最旧”序列，GRU 最终状态聚焦旧信息。
2. 旧序列仅 64/32 步，压缩掉了大量事件形态；高分公开方案使用 200/400 与 60/120。
3. 旧深度输入大量使用绝对价格和全局 z-score；更稳的是相对 mid、microprice、相对 spread、OFI、signed flow、增撤单压力。
4. 旧融合按 raw prediction 直接加权，不同模型尺度相差数十万倍，权重没有真实含义。
5. 旧 LGB 特征缺少：microprice、二档相对 spread、盘口 slope、realized volatility、OFI、指数时间权重、完整窗口 signed amount/action pressure。
6. 单一 late 验证被过度用于融合调权，存在过拟合风险。

## 已实施代码

- `src/build_grid_v2.py`：高分辨率相对微观结构网格，支持 v2/v3、BUILD_ONLY。
- `src/train_multistream_grid.py`：可配置尺度/容量的 MultiStream CNN–Transformer。
- `src/predict_multistream_grid.py`：多 checkpoint test 推理并 unit ensemble。
- `src/make_unit_ensemble.py`：尺度不变的单位范数融合。
- `src/train_gru_chrono.py`：修复旧 GRU 时间方向。
- `src/train_realmlp.py`：RealMLP batch ensemble 原型。
- `src/rolling_lgb*.py`：滚动 LGB 稳健性、seed、特征筛选、target clipping 审计。

## 当前候选（禁止自动提交）

- `output/candidate_top10_multiscale_conservative.csv`
- 权重：LGB 10%、旧 GRU 20%、旧 Hybrid 20%、MultiStream v2 25%、MultiStream v3 25%。
- 所有成员先去均值并单位范数化。
- 文件完整：647,896 行，无 NaN。

## 提交前门槛

必须满足并由用户确认：

1. 至少 late + middle 两个时间折支持新结构；已满足。
2. 候选不使用公开 LB0.142 的预测值本身，只参考公开代码思想；已满足。
3. 运行最终文件完整性检查；已满足。
4. 明确此次提交要回答的唯一问题：多尺度 unit ensemble 是否能将 Public 从 0.128 推高。
5. 用户明确批准后才能上传。

## 后续优先级

1. 对 v2/v3 各再训练 1–2 个 seed，降低单 seed 方差。
2. 把公开 LGB baseline 中的 OFI/EWM/microprice/signed-amount/pressure 因子并入本地 tabular 特征。
3. 用这些增强因子重新训练 LGB 多 seed和 RealMLP。
4. 用三折 OOF 预测拟合非负融合权重，替代单 late 折网格搜索。
5. 最终保留两个归纳偏置不同的 Final Submission，防 Private shake-up。
