# S0 CPU评测适配接续（阶段0尚未完成）

用户要求S0全绿后写正式`plan/handoff-日期-s0-complete.md`及可复制启动词；该要求已冻结在s0:08。本文仅当前业务闭环交接，不能当作S0完成证据。

## 本闭环

s0:11完成。固定fstandhartinger/jevbench@7625ecb1b075cfe4d09d34141d353325825eea1d八文件；231题139Choice/18Score/74Noul，全量无过滤。data/jevbench.py做可逆标签/输入适配，eval/jevbench.py保留官方argmax、macro-family、Brier/ECE及36对paraphrase计分；scripts/s0_jevbench.py提供validate/build/score。pU旁报，不送入官方scorer。

父会话两遍build的SHA均11223d982606c0d26eadee412d4f1fbadfe29f1eeae8d9d75df8d446d803c75d。reports/jevbench-adapter-validation-001.json、build-001/002.json、oracle-001.json有命令、代码/配置/数据hash、环境及退出码。oracle为公开expected测试，非模型质量；GPU未使用。7定向测试、ruff/format、82源mypy、uv build通过。教师195场景口径未证实，不能混同官方36组一致性。

## 当前并行工作

教师resume005仍在/root/autodl-tmp/gpu-reference-fp32-complete-001，冻结HEADdfdfd935；教师PID169823，快照4785/5600、Rl，未终止。先通过SSH mux /tmp/omnijev-ssh-root-final核PID、exit及records行数；活作业不得重启。精确启动/断点见STATUS和上份scheduled-cache handoff。不触碰冻结树源码或封存内容。

s0:10真实LinuxCPU报告reports/s0-baseline-cpu-linux-001.json：3passed，OneJev4/27 thinking证明partial，Valen metadata路径blocked。真实Valen在artifacts/quarantine/Valen-Team__Valen-Preview-0923/30096aaabb2689614c83a79f64f38c865c2ce1c1；正在/tmp/lite-baseline-cpu-fix-15ea9ab修复显式路径与官方语法证据，勿重下模型。

ADR0009澄清六评测cohort与四物理输入；独立切片/tmp/lite-sealed-cohorts-v2仅修改eval/sealed_gate.py、scripts/sealed_gate.py、tests/test_sealed_gate.py。父会话整合验证后更新runbook/票/STATUS。真实独立身份和两人12000判断仍外部条件。

缓存冷比仍失败：async2.3164/2.3335，其余失败留档。教师占卡，禁止重复失败配置、改阈值或静默切引擎。TF32x3仅评估，无已准入实现。

## 接手指令

在/Users/bamboo/Githubs/Lite-OmniJev先读AGENTS.md、plan/STATUS.md、docs/02-project.md和当前claimed票。核教师真实作业及两个CPU切片工作树，保留已有改动；已结束代理由主会话接管。整合定向验证并续完教师第二遍，继续S0未过门槛。S0全绿后执行s0:08独立集成验收，写正式s0-complete交接和新session启动词；未全绿不能进入S1或标goal complete。不push、不付费、不自动开关卡、不读封存payload。
