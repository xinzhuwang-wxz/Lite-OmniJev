# 13: 构建首批文本500开发题和训练排除ledger

ID: s0:13
Milestone: S0
Category: enhancement
Status: ready-for-agent
Execution: done
Owner: none
Worktree: none
Blocked by: s0-data:01, 03

## What to build
按text-development-spec.md复用已准入CLINC公开train，构建真实500dev与14750训练余集、可验证family allocation和脱敏dev ledger。不读封存，不启动GPU。

## Acceptance criteria
- [x] 固定原始输入SHA/revision/许可/15250行；原文件不修改；合法Choice原句/候选/target完整保留。
- [x] 按固定种子/family哈希分配500dev，剩余训练全部排除dev family；记录旧全量train已被此分配取代，未来训练入口须核ledger。
- [x] 脱敏ledger绑定逐item源hash/来源/许可，输出无人工κ虚报；明确只覆盖intent Choice，其他primitive/整体文本分布未由本票验收。
- [x] 两遍真实LinuxCPU构建字节相同、精确数量与family隔离通过，留环境/命令/退出码/耗时/代码/数据/配置hash。
- [x] 公共行为拒绝回归、ruff/mypy/build通过；不把构建当模型质量或训练准入。

## Evidence
Evidence: reports/s0-text-development-build-001.json; reports/s0-text-development-build-002.json; reports/s0-text-development-acceptance-001.json
待执行；规格.scratch/s0/text-development-spec.md。与已有s0-data:01不同：原票是全量来源准入，本票只做开发池分配及训练排除。属于s0:12独立已知text来源切片，不等待其他五类来源规划或教师GPU。

## Out of scope
新下载、读validation/test/holdout、模型推理/训练、伪造人标、替代完整六类开发基线。

2026-10-14父会话真实Linux源码f8edd50在/root/autodl-tmp/s0-text-dev-001跑两遍，均exit0/约2.3sec。500dev/14750余train，dev144intent/余train151intent，family交集0；四输出字节相同。dev SHA8c7809e8b07673904b71bc4676a49a5e31abb22ee7126f2dc09e8fde45b4e070，余train SHA12a42005478ff1cd2f618fbe4951ced1af878b9e115a9a055c57d9d79578cb11。独立父校验全15250原句/问题/候选/target只改pool/version且源文件SHA未变；不因覆盖144而按模型结果改选题。原全量训练版本superseded_for_training，未来训练入口必须绑定新allocation ledger/hash；新余集本票不自动training_admitted。
10本票回归/54相关回归、ruff/302format/83源mypy/build通过。没有GPU或封存读取，Score/Noul/其他五类dev/模型准确率仍待对应闭环，不扩大本票验收。接续plan/handoff-2026-10-14-s0-cohorts.md。
