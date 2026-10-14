# S0真实基线汇总

2026-10-15：s0:07完成，S0尚缺真实人工封存/gate（s0:05），S1未启动。每支持组合500题×2，30000记录/15000精确重复对；六模型JevBench231各两遍另有独立报告。

| cohort | E4B | E2B | OneJev4 | OneJev27 | Jev-Omni | Valen |
|---|---:|---:|---:|---:|---:|---:|
| text | 93.6% | 88.4% | 96.6% | 98.2% | 93.8% | 35.8% |
| image | 74.4% | 69.8% | 83.6% | 87.4% | 78.6% | 66.4% |
| screenshot | 51.4% | 26.0% | 66.2% | 81.2% | 74.0% | 73.0% |
| audio | 63.4% | 44.2% | unsupported | unsupported | 38.0% | unsupported |
| video | 55.6% | 49.8% | 86.2% | 94.2% | 69.8% | 54.0% |
| crossmodal | 51.2% | 50.4% | unsupported | unsupported | 59.2% | unsupported |

截图使用v2；旧网格负例造成坐标捷径，旧成绩全部superseded。其余五cohort证据复用；跨模态复用image/audio组件，不虚增独立family。结果仅覆盖各cohort首primitive切片，不代表完整产品任务分布。

真实token、warm前向/预处理p50/p95/max、总耗时及SHA见s0-development-compute-budgets-001.json。共享GPU质量评测时延仅诊断预算；Valen未计预处理保持null，不能冒称发布p95。准确率低值原样保留，不据此调集/分母。

入口：s0-baseline-expansion-acceptance-001.json绑定证据SHA；s0-development-baselines-table-001.json绑定每模型报告，原始记录在忽略的artifacts和远端持久盘；真实费用未知null。GPU作业已全部结束，显存使用0MiB，已通知用户关卡；agent未关机。

已复用集中源码review及34项相关测试/全仓静态/build；GUI修复另7项定向测试及Linux双构建。最后收口只做30份SHA/分母/同cohort源hash一致性检查，不重复模型/浏览器/全仓审查。

人工封存：外部隔离身份、6000题/12000真人原始判断/逐类κ≥.7仍缺。ADR0011迁移交付截止S4前仅proposal；当前仍阻断S0出口，不能开始S1。
