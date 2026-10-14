# 08: 集成全部真实S0证据并决定阶段出口

ID: s0:08
Milestone: S0
Category: enhancement
Status: ready-for-agent
Execution: done
Owner: none
Worktree: none
Blocked by: p0:04, 01, 02, 03, 04, 07, 10, 11, 13, 14

## What to build
最终S0集成票；剩余数据与全部基线统一由07闭环交付，硬依赖完成后才可认领，以数据→推理→评测→服务的完整链决定是否进入S1。

## Acceptance criteria
- [x] 每cohort500开发、核心及Valen同条件基线表、各模态token/时延预算均有完整可复现证据；缺失或不可用的必需技术能力保持blocked；1k/cohort现成gold封存及自动gate按ADR0012在S4前交付，05独立阻断S4/FINAL。
- [x] 同一冻结代码/数据/配置版本汇总全部S0门槛：MMAU63.10±2pp、ADR-0006固定九任务MVBench真实baseline、重复完全一致、100题严格<1e-3、ADR0010真实prefix cache-on/off冷暖基线及概率一致性（原≤2仅诊断目标）、基线完整；新增真人双标/κ按用户明确决定ADR0012删除；现成gold来源/隔离和真实S4门保留。
- [x] 基线同机双轮输入/原始概率/预测完全一致且来源哈希可复现；跨机器/三运行时/端侧兼容性在S6实际发布验收，不额外成为S0前置（与07已冻结口径一致）。
- [x] 真实网关/当前已实现媒体浏览器闭环、错误路径、模型/配置版本完整；独立里程碑review后保存报告、资源清单及handoff。
- [x] S0全部通过后写 `plan/handoff-日期-s0-complete.md`：逐票/逐门槛证据索引、实际代码/配置/数据/模型hash、环境与产物绝对路径、GPU/进程/断点和资源账、S1依赖与首个可执行闭环、失败恢复及不可触碰边界、可直接复制的新session启动词；新session可从文档独立恢复，不依赖聊天记忆。用户2026-10-14明确要求。
- [x] 任何fail/缺失都不能pass；全部通过才由主会话关闭S0并展开S1，额外预算/方向改变由用户决定。

## Evidence
Evidence: reports/s0-stage-acceptance-001.json；plan/handoff-2026-10-15-s0-complete.md；ADR0010/ADR0012。
尚未执行；本票仅冻结验收。所有实跑证据必须包含命令/退出码、代码版本、数据与配置哈希、环境、耗时、成本；失败也保留。

## Out of scope
封存内容暴露、S1先跑后补证、生产发布、自动跳过失败模型/模态。

## Integration result
2026-10-15：依据用户明确ADR0012，删除新增真人双标/κ/外部人工账号，05改现成可靠gold/自动gate工程票，S4/FINAL仍依赖真实数据质量/隔离和模型评测。S0全部实际技术证据已通过，6数值机器门实跑exit0，52核心/阶段门回归通过；公开/开发基线、数值/浏览器、缓存和集中review按SHA复用，不消耗新GPU。

正式交接plan/handoff-2026-10-15-s0-complete.md已完成；S1本session未启动。旧票据/人工延期解释留在Git历史及被取代ADR，不作为当前阻塞。
