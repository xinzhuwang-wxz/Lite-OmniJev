> 最新覆盖：截图v1坐标捷径497/500，旧截图成绩superseded；43477/343447已结束。v2双构建exit0待核，入口plan/STATUS.md；只重跑六截图模型，其余24支持对已完成不重跑。

# S0接续入口（尚未完成）

2026-10-14更新。先读AGENTS.md、plan/STATUS.md、plan/DELIVERY-PLAN.md和.scratch/s0/issues/07-expand-baselines.md。STATUS是最新调度事实，保留全部历史失败，不重跑有效证据。

- 六cohort各500真实构建及双遍确定性完成，配置configs/s0-development-cohorts-h20.json；crossmodal复用组件family，未来训练用union排除，不能假称新增独立family。
- 六公开模型JevBench231双遍完成；reports/s0-public-baselines-table-001.json。教师MMAU/MVBench、数值/网关/真实浏览器已完成，复用证据。
- 开发基线通过表见reports/s0-development-baselines-table-001.json，包含23个已核验支持对/30个必需支持对；剩余GPU矩阵实际运行。旧scheduler391342已在当前E4B视频正常exit0后终止，kernel退出码见reports/s0-development-scheduler-handoff-001.json；不要重启。其余video/crossmodal接续/root/autodl-tmp/s0-development-final-001/run-development-queue-fla-002.py（892905），输出artifacts/development-baselines-fla-002。
- Valen适配已整合18b5b45，实际CPU四支持类别通过；开发queue796345在/root/autodl-tmp/s0-valen-development-final-002，artifacts/valen-development-001，text已实际passed，其余按原torch fallback运行。教师六cohort在/root/autodl-tmp/s0-development-final-001/artifacts/teacher-development-001；截图原等待控制器已结束，新queue60167运行run-screenshot-queue-002.py、artifacts/screenshot-baselines-002，自动等待旧队列完成标记和教师finished再运行四模型；先核实际状态而非历史PID。
- 用户授权整个方案性能/资源估值用实测自主校准（ADR0010）；S0缓存已收口，停止配置搜索。正确性/质量/许可/真人证据不豁免。FLA单候选质量重复通过、内部warm诊断2.505倍，可在显式后续profile使用；已完成作业不改，后续派发显式profile，primary环境不修改。
- ADR0011人工封存截止S0→S4前仅proposal，待用户明确答复；无答复仍S0硬门，不替用户批准。继续独立技术工作。
- 全部S0依赖完成才执行08一次集成验收及正式s0-complete交接；S1未开始。

SSH root@region-42.seetacloud.com:48926，mux /tmp/omnijev-ssh-root-final，凭据不入仓；精确环境/模型/媒体路径见STATUS。冻结教师/IEEE树dfdfd935不可改。GPU用完记资源通知用户关卡；不自动操作。

goal active；5b/探索关闭，不push、不付费、不自动开关卡、不生产部署。

统一dev排除：reports/s0-development-bundle-001.json，远端/root/autodl-tmp/s0-valen-development-final-001/artifacts/dev-bundle-001/dev-exclusion.jsonl。3000family节点/2500原始family/2000组件，媒体SHA连接跨模态复用；后续train须同时排除family与媒体别名。calibration/holdout仍pending。

教师计时控制已按实测调整：watchdog48305/run-teacher-deadline-handoff-001.py，总模型上限5400秒；原timer675329暂停，同一模型675330仍运行。完成认model.exit与deadline-handoff-001.json的kernel退出码以及完整report；原shell控制退出137保留，不能误判模型失败或重启。

Valen截图73.2%/500×2已核；视频实测1.40秒/题，原queue796345暂停，同一video模型由watchdog227700/run-video-deadline-handoff-001.py监管，上限1800秒，完成认video.exit及video-deadline-handoff-001.json。不重启原queue。

最新终态：Valen四支持类别已全部完成，video.exit=0/kernel0/finished齐全，796345与227700均已结束，不重启。E2B视频也完成49.8%。当前27B video child684037由watchdog765745按实测3000秒监管，原892905暂停未来派发，模型终态后自动恢复scheduler继续crossmodal；以STATUS最新条目与实际进程为准。

教师六类实际开发GPU全完成：reports/s0-teacher-development-complete-001.json，6000记录/3000对、model.exit0/kernel0。原timer退出137为已记录控制替换，不是模型失败；675330/48305均已结束，禁止重启。截图等待控制60167无GPU记录已终止，新43477/run-screenshot-queue-003.py、artifacts/screenshot-baselines-003提前并行前三模型（free≥24GiB），27B截图仅在free≥65GiB后开始。
首次正式训练前必须绑定global dev-exclusion和真实holdout分配账本，给未来校准预留未训练family/确定性分池规则；不能将已训练family事后挪为calibration。
