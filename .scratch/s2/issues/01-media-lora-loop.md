# 01: 图像/截图数据→多模态决策LoRA→dev500同条件评测闭环

ID: s2:01
Milestone: S2
Category: enhancement
Status: ready-for-agent
Execution: claimed
Owner: main-session
Worktree: /Users/bamboo/Githubs/Lite-OmniJev
Blocked by: None

## What to build

按.scratch/s2/spec.md在一个闭环内完成：图像/截图池冻结（媒体sha校验+预留holdout/calibration family）→多模态训练扩展（pixel_values过冻结vision tower，LoRA仍仅decoder）→smoke→图像与截图主训练→dev500同条件双遍评测+文本回归+反事实翻转→S2机器门报告。

## Acceptance criteria

- [x] 图像池冻结：候选=允许集排除dev500 family后全部样本，双遍字节一致，媒体逐文件sha校验通过，ledger+报告绑定SHA；截图池冻结同理。（图像：候选2738→dev剔除515行→train 1389/calib 623/holdout 211，train SHA 96ed2bef…；截图：扩展扫描1500 episode→1100候选→dev剔除500→train 420/calib 120/holdout 60，train SHA 70468d92…；截图预留由110/330调整为60/120——600余候选下原比例只留160训练样本，正式holdout仍由s0:05走官方gold，依据记录于票内）
- [x] 多模态训练扩展：LoopConfig支持image/screenshot/mixed（联合单LoRA为产品契约，"旧模态≤1pp"门的实现载体），vision/audio tower冻结+eval，官方processor，媒体sha预检+多树解析，LoRA仅decoder投影；账本校验支持多ledger并集；新增公共行为回归+静态通过。（115项回归、全仓ruff/format、mypy strict；fable子代理实现+主会话集成）
- [ ] smoke：图像与截图各300步真实GPU，训练子集自测acc与loss下降记录（工程信号，非门）。
- [ ] 主训练：图像与截图各samples-seen=min(10000,5×池)，50% midway adapter对照，SwanLab真实指标。
- [x] dev500同条件双遍：image与screenshot精确重复，字段与S0基线runner一致；文本回归评测（text dev500，S2 adapter）≥98.6%；反事实翻转双遍确定性，flip率实测。
- [x] 门槛报告：eval.gate原值判定原样记录（image≥80.6%、screenshot≥63.2%、text_regression≤1pp、flip≥.80）；未达标如实保留，结构性不可达按ADR0013格式另录。
- [ ] lint/type/build+新增回归；STATUS更新；commit带Lore trailers。

## Evidence

Evidence: 恢复修复bbdc041（65项相关回归通过）；reports/s2-counterfactual-rescore-001.json证明旧cf统计错误，ADR0014结构性结论失效；v2修正版GPU双遍评测运行中，原质量门保持，未关票。历史规格与证据：规格.scratch/s2/spec.md；复用s1:01机制与ADR0013原则。基线：OneJev4 image 83.6%/screenshot 66.2%，E4B 74.4%/51.4%（s0-baseline-summary.md）。

## Out of scope

音频/视频（S3）、E2B/量化（S6）、封存holdout构建（s0:05）、新数据源下载、门槛静默调整。

恢复证据（2026-10-15）：reports/s2-stage-acceptance-v2-001.json（FAILED，仅图像78.8%<80.6%；其余门通过），三cohort报告逐记录双遍一致。勾选“生成评测/判定报告”不代表阶段通过，Execution保持claimed。E01/E02探针未达扩量条件停止；E03小批量方案见reports/explore/。
