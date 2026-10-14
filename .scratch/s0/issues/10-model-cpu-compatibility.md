# 10: 冻结必需基线的官方CPU输入与读出能力

ID: s0:10
Milestone: S0
Category: enhancement
Status: ready-for-agent
Execution: claimed
Owner: main-session-gpu-transition
Worktree: /Users/bamboo/Githubs/Lite-OmniJev
Blocked by: p0:04, 01, 03

## What to build
使用现有七模型资产形成五核心及额外Valen依赖的处理器/模板/答案读出支持矩阵；不加载权重，不推理。

## Acceptance criteria
- [ ] 固定revision/许可/文件hash证据，来源与训练数据权利分离；读取已锁元数据与正常safe配置，不执行上游module/加载pickle。
- [ ] 逐模型核对官方processor/tokenizer、chat template、思考关闭语义、2/3/4/20答案和U的合法单token映射；教师专用head沿用已审计规范，不强套学生模板。
- [ ] 对声明支持的输入类别做固定公开CPU探针，记录完整官方输入摘要/映射；不支持与未探测分别标记，不能臆称音频/视频支持。
- [ ] 输出版本/代码/输入hash/依赖/耗时/退出码和support matrix，明确weights_loaded=false/gpu_forward=false/quality_admitted=false；阻断项留具体缺口与后续票建议。
- [ ] 公共行为定向测试与ruff/mypy通过，实际Linux官方CPU探针退出码留证；CPU不能代真实forward。

## Evidence
待执行；规格见 `.scratch/s0/baseline-expansion-spec.md`。固定种子、源码/配置/数据hash、命令/退出码/耗时/成本必须留证；缺项不能pass。

## Out of scope
租卡/开关卡/付费/GPU推理、S1训练、暗改门槛、封存内容暴露、假分数/假人工一致性。

## Comments
2026-10-14主会话由s0:07展开：01/03技术前置已实测，02/04仍是S0总出口门槛但不阻塞该CPU独立闭环；用户已授权按原方案自主技术安排。与现有票概念核对后新增，不复制s0-data:01或05/06的人类/身份工作。
