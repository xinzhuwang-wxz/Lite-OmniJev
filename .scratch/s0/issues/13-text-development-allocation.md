# 13: 构建首批文本500开发题和训练排除ledger

ID: s0:13
Milestone: S0
Category: enhancement
Status: ready-for-agent
Execution: claimed
Owner: main-session-gpu-transition
Worktree: /Users/bamboo/Githubs/Lite-OmniJev
Blocked by: s0-data:01, 03

## What to build
按text-development-spec.md复用已准入CLINC公开train，构建真实500dev与14750训练余集、可验证family allocation和脱敏dev ledger。不读封存，不启动GPU。

## Acceptance criteria
- [ ] 固定原始输入SHA/revision/许可/15250行；原文件不修改；合法Choice原句/候选/target完整保留。
- [ ] 按固定种子/family哈希分配500dev，剩余训练全部排除dev family；记录旧全量train已被此分配取代，未来训练入口须核ledger。
- [ ] 脱敏ledger绑定逐item源hash/来源/许可，输出无人工κ虚报；明确只覆盖intent Choice，其他primitive/整体文本分布未由本票验收。
- [ ] 两遍真实LinuxCPU构建字节相同、精确数量与family隔离通过，留环境/命令/退出码/耗时/代码/数据/配置hash。
- [ ] 公共行为拒绝回归、ruff/mypy/build通过；不把构建当模型质量或训练准入。

## Evidence
待执行；规格.scratch/s0/text-development-spec.md。与已有s0-data:01不同：原票是全量来源准入，本票只做开发池分配及训练排除。属于s0:12独立已知text来源切片，不等待其他五类来源规划或教师GPU。

## Out of scope
新下载、读validation/test/holdout、模型推理/训练、伪造人标、替代完整六类开发基线。
