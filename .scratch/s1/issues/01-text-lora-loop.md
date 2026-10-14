# 01: 文本数据→决策LoRA→dev500同条件评测闭环

ID: s1:01
Milestone: S1
Category: enhancement
Status: ready-for-agent
Execution: claimed
Owner: main-session
Worktree: /Users/bamboo/Githubs/Lite-OmniJev
Blocked by: None

## What to build

按.scratch/s1/spec.md在一个闭环内完成：文本池冻结分配（训练前预留未训练holdout/calibration family）→答案token/模板探针→2k/300步smoke→30k samples-seen主训练（10k/30k对照点）→dev500同条件双遍评测→S1机器门报告。模型/来源/探针/普通修复都在票内清单，不拆小票。

## Acceptance criteria

- [x] 池分配冻结：10750 s1-train/1000 holdout-reserve/3000 calibration-reserve，确定性协议（seed42，family级），双遍字节一致，与dev500及历史排除family零交集，ledger+报告绑定输入/输出SHA；训练入口校验并拒绝dev/预留family。（远端双遍run-a/run-b字节一致，报告artifacts/s1-text-pools-v1-001/run-*/report.json passed；train 75206358…、calib 6d387816…、holdout 6bba594d…、ledger ce49ae4b…；preflight账本校验14750行/2000样本verified）
- [x] 答案token探针：A..T+U在真实generation boundary单token，thinking关闭，20选项最大情形覆盖，探针证据保存后才开始训练。（smoke preflight ready：2000样本全部经prepare_examples边界校验，max_prompt_tokens=188，blockers=[]；主训练preflight含10750样本同校验）
- [x] R0 smoke：2k样本/300步真实GPU训练，训练样本自测acc≥.98，loss下降，adapter+完整provenance报告。（reports/s1-smoke-run-001.json completed/300步/861s，loss窗口均值0.455→0.0169，adapter SHA58a58946…，峰值18.4GB，SwanLab online真实上传；reports/s1-smoke-selfacc-001.json passed，自测acc=.998≥.98，ECE15=.0027，267.9s）
- [ ] R1主训练：10750池30k samples-seen真实GPU完成，10k样本点与30k终点各保存adapter，SwanLab真实指标（ADR0003），checkpoint可恢复。
- [ ] dev500同条件评测：10k/30k两adapter各双遍，500×2精确重复，字段含原始/归一概率、pU、confidence、预测，协议同S0基线（spec评测节）。
- [ ] 门槛报告：`eval.gate`机器可读，smoke/.98、onejev −3pp、base +5pp、ECE ≤.06、30k−10k +2pp全部实测原样记录；未达标项保持未达标并按ADR0010式记录调整依据，不暗降。
- [ ] lint/type/build通过+新增公共行为回归；STATUS更新；本地commit带Lore trailers。

## Evidence

Evidence: 待执行；规格.scratch/s1/spec.md。S0证据复用reports/s0-stage-acceptance-001.json，不重跑S0。GPU为既有授权资源（用户2026-10-15要求按计划速速推进）。

## Out of scope

其他模态、E2B、量化、生产服务、封存holdout实际构建（s0:05）、教师打标、新数据源下载、质量门槛静默调整。
