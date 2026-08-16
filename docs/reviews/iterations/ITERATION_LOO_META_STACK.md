# Leave-One-Fold Meta-Stack审计

## 目标

在已被Public验证为0.145的独立Stack上，测试是否能通过跨折留一法学习更好的固定组合：每次用两折拟合Ridge Meta-Stack，再在第三折验证。

## 输入

- LGB；
- RealMLP；
- v3；
- Joint；
- Multi-Resolution。

没有使用旧Public派生预测，也没有使用Public参考预测。

## 结果

固定独立Stack基线：

- Proxy：0.15865；
- Middle：0.15603；
- Late：0.17169。

各alpha下最优留一折结果：

| Held-out fold | 基线 | LOO Meta最优 | 变化 |
|---|---:|---:|---:|
| Proxy | 0.15865 | 0.15899 | +0.00034 |
| Middle | 0.15603 | 0.15587 | -0.00016 |
| Late | 0.17169 | 0.17230 | +0.00061 |

## 决策

- 未达到+0.001闸门；
- Middle出现下降；
- 不生成测试候选；
- 不提交；
- 停止继续搜索已有Stack的固定或Meta权重。

下一步必须训练新的独立信息模型，不能继续重排已有预测。
