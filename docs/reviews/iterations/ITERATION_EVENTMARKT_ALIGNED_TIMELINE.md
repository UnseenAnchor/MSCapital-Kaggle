# 迭代复盘：事件对齐合并时间线（eventmkt）——负结果

## 立项背景

用户批准测试最后一条待验证分支：把 market 快照与 order/tx 事件按时间戳对齐为单一序列，
让模型看到每笔委托/成交发生时的订单簿状态（order-flow→price 微观结构动态）。

## 实现

- 新增 `src/build_event_market.py`：对 event_cache_v2 的 256 个事件槽位做 asof-join，
  匹配每个事件时刻"最近且不晚于该时刻"的订单簿快照，把事件价格/量改成 book-relative 表达：
  - order(10ch)：price_mid_rel, vol_depth_rel, side, action, spread, imb1, micro, sec/60, inter, pos
  - tx(9ch)：price_mid_rel, vol_depth_rel, side, spread, imb1, micro, sec/60, inter, pos
- 掩码感知标准化，float16 mmap 写入 grid_v2；
- 新增 `src/train_event_market.py`：双流 Stream（ordermkt/txmkt）→ cross → head。

## 结果（proxy OOF）

| 模型 | proxy(ens/best) | 与v3相关 | 与event256相关 | 栈+12% |
|---|---:|---:|---:|---:|
| event256（raw绝对价格） | 0.1318 | 0.660 | 1.0 | **0.1603** |
| eventmkt（book相对） | 0.1320(ep7) | 0.725 | 0.694 | 0.1597 |

## 结论

1. book-relative 重表达**丢掉了绝对价格信息**——event256 的增益部分来自 raw 绝对价位（含资产/点位regime信息）；
2. 注入订单簿上下文后与 market/v3 相关性升高（0.660→0.725），**多样性下降**；
3. 最终栈融合贡献不如 event256（0.1597 < 0.1603）。

事件对齐合并时间线分支关闭。该实验也再次印证：event256 的价值在于"原始绝对序列"与网格/流程的**正交性**，任何朝 market 信息靠拢的改造都会吃掉多样性。

## 全部输入信息方向现状（已穷尽）

- market@200 全分辨率（v3已用）、order/tx 60s窗口逐事件144维（event256 已用）；
- 512序列、D96容量、多种子：无效；
- raw事件注入强模型（fusion）：无效；
- 事件对齐合并时间线（eventmkt）：无效。

最终方案维持 `55601441`（Public **0.146**）。