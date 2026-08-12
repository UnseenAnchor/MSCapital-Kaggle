# 提交复盘：55459787

## 结果

- 日期：2026-08-12
- 文件：`candidate_proxycv_realmlp5.csv`
- Public：**0.138**
- 排名：**41 / 108**
- 上一最佳：0.137，第41 / 108
- 显示提升：+0.001
- Top10门槛：0.152，当前显示差距0.014

## 构成

- 95% Public-0.137候选
- 5% RealMLP-v4 checkpoint ensemble
- RealMLP配置：Top128、8成员、batch1024、16 epochs、checkpoints 6/9/11
- 所有成员去均值并单位范数化

## 验证证据

固定LGB60 + RealMLP40在proxy/middle/late三个验证均提高，并改善最差月份。最终上传权重根据现有完整Public配方的late重建保守收缩到5%。

- Proxy baseline：0.13664；加入RealMLP最高约0.14025
- Middle baseline：0.13701；固定checkpoint融合0.13960
- Late baseline：0.14638；固定checkpoint融合0.14894
- 完整Public配方late：0.16969；加入5% RealMLP：0.16977

## 得到支持的假设

1. 不拆月的month 0–44 / 45–70代理CV能补充late单折，并对Public方向有解释力。
2. RealMLP与现有候选相关性约0.794，提供真实独立信号。
3. train-only特征筛选、MSE主导损失和checkpoint ensemble比cosine主导的长训练更稳健。
4. 即使late提升很小，跨26个月代理CV上的稳定边际增益仍可能转化为Public提升。

## 不能得出的结论

1. Kaggle仅显示三位小数，无法知道真实提升是否完整达到0.001。
2. 不能据此认为RealMLP权重10%或更高会继续提升；late已在10%开始回落。
3. 不能继续提交5%附近的相邻权重，候选与旧预测相关性0.99953，信息价值很低。

## 下一阶段

优先提高低相关单模强度：

1. 为RealMLP增加不同seed或稳定的特征子集成员，但必须在proxy/middle/late共同验证；
2. 做v3 MultiStream的proxy/middle验证，判断其是否覆盖不同市场状态；
3. 保留代理CV作为排行榜方向门槛，同时继续监控最差月份；
4. 不提交相邻融合权重，等待实质性单模提升。
