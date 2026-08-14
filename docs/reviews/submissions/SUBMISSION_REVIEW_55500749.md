# 提交复盘：55500749

## 结果

- 日期：2026-08-14
- 文件：`submission_v9big_full_unit.csv`
- 类型：纯单模诊断提交
- Public：**0.129**
- 当前最佳：0.144（不变）
- SHA256：`253bae96083cd53c2d9ea47aafa0ce81fb4e93cf841c6dda0215a2ef5ad923d3`

## 模型

- v2 Market 200点、Transaction/Order 60点；
- 12/8/10通道，增加Market和Transaction事件计数；
- CNN96 + d_model96 + 3层Transformer；
- physical batch256 × accumulation4 = effective batch1024；
- 全部1,257,637条训练样本训练12轮；
- 固定checkpoint 4/5/6中心化单位ensemble。

## 提交前证据

- Proxy ensemble：0.14591；
- Middle ensemble：0.14596；
- Late ensemble：0.15849；
- 与当前Public-0.144候选测试相关性：0.88753；
- 与公开包v9_big相关性：0.75336；
- checkpoint之间相关性：0.798–0.872。

该模型作为性能候选未通过Late最差月闸门，但因每日提交额度未用满，经用户明确批准作为诊断提交。

## 复盘价值

1. 量化出Proxy 0.14591对应Public仅0.129，CV–LB偏差约0.017。
2. 证明低相关性本身不够；第三锚点还必须有足够的Public单模强度。
3. 单模0.129略高于对当前候选的正边际阈值约0.1278，但优势太小。
4. 根据两个Public端点和相关性，理论最优权重仅约3.9%，预计增益约0.000023。
5. 不利舍入情形预计增益仅约0.000001，无法支持第二次融合提交。

## 决策

- 不生成或提交v9_big融合候选；
- 不扫2.5%/5%/7.5%权重；
- 停止事件计数v9_big架构族；
- 后续诊断提交应优先选择能够区分整个模型方向、且结果可改变决策的单模。
