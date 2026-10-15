# 01: 全模态有界训练→完整质量与封存评测闭环

ID: s4:01
Milestone: S4
Category: enhancement
Status: ready-for-agent
Execution: claimed
Owner: main-session
Worktree: /Users/bamboo/Githubs/Lite-OmniJev
Blocked by: s3:01, s0:05

## What to build
按.scratch/s4/spec.md与iteration-policy.md，完成实际训练/开发筛选/冻结候选/同条件教师与外榜/私有holdout/阶段门。数据gate工程复用s0:05，不重新按来源或探针拆票。

## Acceptance criteria
- [ ] s0:05实际验收有效；训练新组合/变体严格来自train，来源/历史曝光和组件别名可追溯。
- [ ] 每轮冻结假设、预算、数据/配置/adapter、dev切片与退出条件；监控和中间检查点实际产生，小批量无收益停止，有收益才放大。
- [ ] 六cohort及保留来源切片的最强允许教师同条件比率达到.97；原cohort不被补充来源总分掩盖。
- [ ] 私有完整封存/匹配dev差≤.03，完整度、来源、模型/模板/token/输入哈希及失败拒绝证据齐全。
- [ ] Decision Index、JevBench真实可复现测量均不低于同尺寸基线，缺测不算通过。
- [ ] configs/gates.json S4全部门通过，资源/成本/创新结果有实证并独立集成复核；不以CPU/小样本工程报告代替阶段验收。

## Evidence
尚未启动S4训练，未通过S4质量门。已有S3及S4公开诊断只用于形成实验假设；见plan/STATUS.md。新实验在本票内记录，质量门保持。

## Out of scope
S5校准正式验收、S6/E2B正式验收、5b、付费/租卡开关、生产部署及大预算自动扩量。
