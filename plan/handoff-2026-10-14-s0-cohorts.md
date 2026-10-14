# S0接续入口（尚未全部验收）

2026-10-14。先读AGENTS.md、plan/STATUS.md、plan/DELIVERY-PLAN.md和.scratch/s0/issues/07-expand-baselines.md。STATUS为唯一当前调度快照；本入口取代此前已过时的PID/队列说明，不覆盖原始报告。

- P0、教师MMAU1000/MVBench固定9任务1800双轮、100题HF/native/gateway四概率严格<.001、真实Chromium及缓存实测均完成，证据索引见STATUS；禁止重复消耗GPU。
- 六模型公开JevBench231双轮完整；24/24非截图开发支持组合完整，截图v2 E4B51.4%双轮已核（教师截图v2也已核74.0%，Valen截图v2也已核73.0%，OneJev4截图v2亦完成，E2B截图26.0%也完成，当前29/30支持对），reports/s0-public-baselines-table-001.json和s0-development-baselines-table-001.json。各cohort500，仅首primitive切片；不支持音频/组合的6对明确unsupported，不填零。
- 截图v1负例网格泄漏：无图/无指令坐标规则497/500。全部旧截图成绩superseded，失败/完成报告保留。v2随机合法坐标修复7tests/ruff/mypy通过；Linux两遍构建字节相同、保留原500family（498原题/2题因几何换同family步骤），冻结旧规则19/500；不按GPU分数选集。
- 当前GPU仅补测截图六模型：/root/autodl-tmp/s0-screenshot-v2-final-001 HEAD7b4f382。学生queue343888执行四模型（E4B已exit0，OneJev4也已exit0，E2B也已exit0，原queue343888已停止未来派发，仅27B继续），artifacts/screenshot-baselines-v2-001；教师→Valen queue343927（教师已exit0/281.014秒，当前Valen child344135），artifacts/{teacher-screenshot-v2-001,valen-screenshot-v2-001}。27B改为独立提前并行queue344403/child344408，artifacts/screenshot-27b-parallel-v2-001，原2400秒已按实测交接watchdog344615，同一27B child344408继续，总上限4500秒；parent344403暂时暂停、模型终态后恢复采集exit，见reports/s0-screenshot-27b-deadline-handoff-001.json。原E2B终态后已核三模型exit0并结束343888；watcher遇zombie nvidia-smi安全暂停后已收口，收据reports/s0-screenshot-parallel-handoff-001.json，原27B没有启动；不重启原队列。其余job1800秒；共享GPU只做质量，不冒称隔离发布时延。接手核实际进程/exit/完整报告，不能仅因观察超时重启。
- 数据入口configs/s0-development-cohorts-h20.json已绑定截图v2。历史dev统一排除union位于/root/autodl-tmp/s0-screenshot-v2-final-001/artifacts/dev-exclusion-history-union-v2-001/dev-exclusion.jsonl，SHA3110f9740d9e7dd9c39023b106d977e927b3446f5dfa82e168f729c127aeda1e；旧/新题与媒体别名都保留。正式训练须查family及媒体SHA，并预留从未训练的S5校准family；不能事后删训练账本制造隔离。
- 收齐截图当前版本30支持对后，更新矩阵并运行python3 .scratch/s0/summarize-development.py产生真实token/耗时诊断预算汇总（读取已有sidecar，无模型执行），完成07。检查必要变更/失败即可，集中源码review已完成，不再全检/逐模型review。
- s0:05仍外部blocked：真实隔离身份、6000封存/12000真人原始判断/逐类κ≥.7缺失。ADR0011仅proposal，无明确批准仍S0硬门；不得代签、读封存、把性能授权当真人豁免。08依赖完成才认领。S1未启动。
- 用户授权整个方案性能/资源估值按实测自主校准，记录证据和新标准，限额优化即推进（ADR0010）；准确率/正确性/许可/真人要求保留。5b/探索关闭，不push/付费/自动开关卡/生产部署。

SSH root@region-42.seetacloud.com:48926，mux /tmp/omnijev-ssh-root-final；凭据不入仓。精确环境/模型/媒体路径见STATUS，冻结教师/IEEE树dfdfd935不改。所有GPU作业结束后确认报告/原始记录已持久化，通知用户关卡；不自动关机。

本轮停点：完整goal active，07 claimed，真实截图GPU仍运行。不是s0-complete交接；最终通过后另写用户要求的完整S0出口文档。
