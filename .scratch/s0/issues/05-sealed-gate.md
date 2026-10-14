# 05: 交付现成gold封存数据与自动评测gate

ID: s0:05
Milestone: S4
Category: enhancement
Status: needs-triage
Execution: open
Owner: none
Worktree: none
Blocked by: p0:04

## What to build
按ADR0012从已准备/可取得的许可明确来源选用已有gold或程序真值，交付独立holdout及自动汇总gate。保留ID供历史依赖；不再是需用户真人/账号的外部阻塞。当前不提前细拆S4；S1入训前必须预留pool/family，实际构建与gate在S4放量前完成。

## Acceptance criteria
- [ ] 六cohort各1000独立holdout、各500dev来源/许可/revision/标签规则/hash可追溯；不得把现有dev/公开已调参题兼作封存，数量不足如实报告。
- [ ] 真值为已有可靠数据集gold、公开benchmark gold或可验证程序真值；拒绝缺失/非法标签和把教师输出冒充评测gold。新增真人/双标/κ均不要求，κ为not_applicable。
- [ ] train/dev/calibration/holdout按原始family及媒体别名去重，历史dev排除完整；S1前预留未训练family，禁止事后删训练记录伪造隔离。
- [ ] 自动gate独立作业读取封存内容，开发agent仅收数量/hash/来源/质量与模型指标汇总；实际权限/日志/错误拒绝测试通过，不依赖用户提供外部人工账号，不把root同机chmod当外部安全隔离证明。
- [ ] 绑定真实数据、固定配置/种子/代码与运行环境，S4/5/6评测汇总可复现；模拟canary不代实际数据验收。S4放量及FINAL依赖本票。

## Evidence
现有四池family机制、历史dev union与来源manifest可复用。旧v2人工双标canary仅历史工程证据，当前不是首发必走合同；新的实际gold gate尚未实现，不宣称通过。详docs/adr/0012-existing-gold-without-new-human-labeling.md与docs/runbooks/sealed-gate.md。

## Out of scope
新增真人标注/招聘/付费外包、要求用户提供真人或外部gate账号、模型标签冒充gold、开发agent读取封存载荷、提前S4模型评测。
