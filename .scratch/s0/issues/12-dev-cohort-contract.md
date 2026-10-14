# 12: 收口开发集与封存集cohort和来源合同

ID: s0:12
Milestone: S0
Category: enhancement
Status: ready-for-agent
Execution: claimed
Owner: main-session-gpu-transition
Worktree: /Users/bamboo/Githubs/Lite-OmniJev
Blocked by: p0:04, 01, 03

## What to build
核对原HTML、数据资产清单、已实现gate和来源准入，冻结每模态500开发集的可执行构建协议及封存隔离交接边界；不读取任何封存内容。

## Acceptance criteria
- [ ] 区分物理text/image/audio/video与text/image/screenshot/audio/video/crossmodal评测类别；明确原文/现有四模态gate/六类别清单的映射、计数和未解差异，截图和跨模态不能被隐式省略，不暗改500/1000/κ门槛。
- [ ] 逐源记录revision/许可/标签来源/实际资产与训练分布关联，复用CLINC准入；每个500dev类别有独立family候选、配比与确定性采样/去污染/训练排除规则，缺源保留blocked。
- [ ] 原标签/程序真值优先；不把100 parity变体或下载报告当独立质量数据，不把单标签冒充人类双标。公开基准、研究数据与商用训练分开。
- [ ] 明确dev/calibration/holdout的版本与family allocation ledger，封存只由独立身份处理；只收数量/哈希/κ等汇总，外部身份/人工资源不由agent自证。
- [ ] 产出能直接实施的首批独立CPU构建规格/依赖票及门槛，未知实验不提前乱拆；若必须改变人工数量/范围或原门槛，留证据与待决策，不自动改实现。

## Evidence
待执行；规格见 `.scratch/s0/baseline-expansion-spec.md`。固定种子、源码/配置/数据hash、命令/退出码/耗时/成本必须留证；缺项不能pass。

## Out of scope
租卡/开关卡/付费/GPU推理、S1训练、暗改门槛、封存内容暴露、假分数/假人工一致性。

## Comments
2026-10-14主会话由s0:07展开：01/03技术前置已实测，02/04仍是S0总出口门槛但不阻塞该CPU独立闭环；用户已授权按原方案自主技术安排。与现有票概念核对后新增，不复制s0-data:01或05/06的人类/身份工作。
