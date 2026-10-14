# S0接续入口（尚未完成）

2026-10-14更新。先读AGENTS.md、plan/STATUS.md、plan/DELIVERY-PLAN.md和.scratch/s0/issues/07-expand-baselines.md。STATUS是最新调度事实，保留全部历史失败，不重跑有效证据。

- 六cohort各500真实构建及双遍确定性完成，配置configs/s0-development-cohorts-h20.json；crossmodal复用组件family，未来训练用union排除，不能假称新增独立family。
- 六公开模型JevBench231双遍完成；reports/s0-public-baselines-table-001.json。教师MMAU/MVBench、数值/网关/真实浏览器已完成，复用证据。
- 开发基线当前9支持对已核验；reports/s0-development-baselines-table-001.json。远端旧队列/root/autodl-tmp/s0-development-baselines-002/scheduler391342仅捕获五cohort，实际状态核queue.jsonl/exit/report，不能凭旧PID文件重启。截图、教师与Valen开发仍须实际调度/收口。
- Valen开发适配子切片/tmp/lite-public-baselines-001尚待主会话整合；真实head/LoRA/visual公共评测已通过。不能用Qwen裸基座代Valen。
- 用户授权整个方案性能/资源估值用实测自主校准（ADR0010）；S0缓存已收口，停止配置搜索。正确性/质量/许可/真人证据不豁免。FLA单候选质量重复通过、内部warm诊断2.505倍，可在显式后续profile使用；不修改运行队列/primary环境。
- ADR0011人工封存截止S0→S4前仅proposal，待用户明确答复；无答复仍S0硬门，不替用户批准。继续独立技术工作。
- 全部S0依赖完成才执行08一次集成验收及正式s0-complete交接；S1未开始。

SSH root@region-42.seetacloud.com:48926，mux /tmp/omnijev-ssh-root-final，凭据不入仓；精确环境/模型/媒体路径见STATUS。冻结教师/IEEE树dfdfd935不可改。GPU用完记资源通知用户关卡；不自动操作。

goal active；5b/探索关闭，不push、不付费、不自动开关卡、不生产部署。
