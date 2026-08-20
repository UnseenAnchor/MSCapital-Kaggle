# ITERATION: Kaggle 论坛/Discussion 内容侦察（结果：不可获取 + 无实质公开讨论）

日期：2026-08-19
目标：搜索 MSCapital 赛题 Kaggle 论坛中他人高分方案/讨论，寻找值得深挖的方向以突破 0.146。

## 结论（先给答案）
**MSCapital 的论坛/Discussion 内容在本环境全部手段下均不可获取，且搜索引擎证据显示该赛题没有实质性的公开高分讨论（搜索结果全部指向其他时间序列赛题：Jane Street / Optiver / g-research / ubiquant，而非本赛）。**

这一方向视为**穷尽/死路**，不对 0.146 突破提供任何新信息杠杆。

## 已尝试的全部获取手段（逐一否决）
1. **DiscussionService gRPC-gateway API（正确路径）**：`/api/i/discussions.DiscussionsService/GetForum`、`GetTopicListByForumId`、`GetForumTopicById` 等（从 app.js 5MB bundle 中精确提取的完整方法→HTTP 路径映射）。用 `~/.kaggle/access_token`（KGAT token）调用。
   - 结果：**403 `Permission 'forums.get' was denied`**。token 可读 kernels/leaderboard（HTTP 200），但**缺少 forum/discussion 读取权限**。
   - 早先错误路径 `/api/i/discussions/...` 返回 404；正确路径返回空 body 400（schema 不符）或 403（无权限）。
2. **`/api/v1/competitions/{slug}/discussions`、`comments` 等公共 v1 端点**：全部 404。
3. **渲染后的 SPA 讨论页 HTML**（fetched discussion_733271.html / discussion_index.html）：纯客户端渲染 shell，无 `__NEXT_DATA__` 服务端数据，body 为空 `<div id=root>`。
4. **Wayback Machine**：无该讨论页快照存档。
5. **kagglesdk Python 包**：无 DiscussionService 模块被安装；对应端点 404。
6. **搜索引擎提取**（web_search 多角度 query + domainFilter kaggle.com）：返回结果均为**其他赛题**（Jane Street / Optiver / g-research / ubiquant），未检索到本赛题的实质讨论/高分 writeup。强烈暗示本赛属**合成/虚构赛题**，无真实公开社区讨论沉淀。

## 可用且更新的信息（已抓到）
- **当前 Leaderboard 快照**（`output/kaggle_search/leaderboard.json`、`LEADERBOARD_SNAPSHOT_0826.txt`）：
  - Top1 0.162（红烧肉）；0.157×2；0.156×3；0.155×2；0.154；**Top10 门槛 0.153**（MAIDANG/Xman/跟我的Opus说去吧）。
  - 当前最佳提交 `55601441` Public **0.146**，差距 **+0.007**。
- **全部公开 Kernel 已复审**（`kernels_score_top.json`）：仍是无超出 LB 0.142 的公开 notebook（yangq369/submit-lb142 9票、yunsuxiaozi/rfmf-realmlp 12票、dc5e9647.../lgb-baseline 34票、sweetyheehee transformer-baseline 14票、各 EDA）。**无新增高分公开方案。**

## 对突破 0.146 的含义
论坛/公开方案方向无新信息。根据此前全扫（后置优化/CV/重排/校准/TTA/meta 均负或噪声），唯一剩余真实信息杠杆仍是 **自监督预训练（SSL）**：对 125 万无标签 order/tx/market 流做掩码重构预训练再微调。属多小时高成本不确定工程，待用户立项决策。

## 关键文件
- `output/kaggle_search/app.js`（解出的 DiscussionService 方法→HTTP 路径映射）
- `output/kaggle_search/leaderboard.json` + `LEADERBOARD_SNAPSHOT_0826.txt`
- `output/kaggle_search/kernels_score_top.json`
