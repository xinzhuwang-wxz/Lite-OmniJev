# Lite-OmniJev S0 技术交接（真人门未通过）

2026-10-15，Asia/Shanghai。用户要求先收尾阶段0、剩余训练在新session无缝继续。本文件交付全部已完成技术证据和唯一外部停点；**不是S0全部通过，不得直接启动S1**。完整原生goal仍是P0/S0–S6/FINAL，未缩小范围。

## 先执行

1. 在 `/Users/bamboo/Githubs/Lite-OmniJev` 读AGENTS.md、plan/STATUS.md、docs/02-project.md和docs/adr/0010-measured-performance-targets.md。
2. `git status --short`、`git log -1`、`uv run python -m scripts.loop validate`。主树最后应无未提交修改；35票、frontier/claimed为空，s0:05外部blocked，s0:08依赖05而open。不要重启历史GPU队列。
3. 检查用户是否已明确答复ADR0011或交付真实外部gate/真人证明；目前没有。先解决这项阶段入口，不因“继续”或性能调整授权替用户批准。
4. 若当前卡仍开，先只读确认 `nvidia-smi`；截至交接显存0MiB、没有计算进程，已通知用户可关卡。agent未自动关机。

## 已完成与证据

| 闭环 | 权威入口 |
|---|---|
| P0真实10步/CPU准备 | reports/p0-engineering-acceptance.md；7模型70文件121.09GB、29来源与锁定离线runtime |
| 教师技术/head安全转换 | reports/teacher-technical-admission-gpu-001.json；仅教师专用head，学生仍主干冻结+决策LoRA |
| 教师效果复现 | reports/s0-teacher-complete-acceptance-001.json；5600记录/2800精确对，MMAU两次620/1000=62%，MVBench ADR0006固定9任务两次976/1800=54.222%，exit0 |
| 数值/网关/浏览器 | reports/s0-full100-gpu-acceptance-001.json；FP32 IEEE HF/native100和gateway100四概率严格<.001；真实Chromium153六类×三primitive及错误路径。FP32仅数值参考，不当生产精度性能 |
| 缓存实测 | reports/s0-cache-measured-baseline-acceptance-001.json；真实native/gateway on/off冷暖及命中KV288/媒体265、四概率最大差2.365e-7。冷比native3.93/gateway5.02，原≤2未达仅诊断目标，ADR0010允许实测基线出口；停止性能配置搜索 |
| 六模型公开基线 | reports/s0-public-baselines-table-001.json；JevBench231各双轮，E4B76.1905%、E2B63.6364%、OneJev4 75.7576%、27B83.1169%、教师85.7143%、Valen47.1861%。教师195口径未证实，独立协议不冒称复现 |
| 六开发cohort/完整矩阵 | reports/s0-baseline-expansion-acceptance-001.json及reports/s0-baseline-summary.md；各500、30支持组合×两遍=30000记录/15000精确对，6明确unsupported留空。仅各来源首primitive切片，不冒称完整产品分布 |
| token/耗时与来源 | reports/s0-development-compute-budgets-001.json和s0-development-baselines-table-001.json；绑定每报告/原始sidecar SHA、代码版本、数据hash、模型revision及许可、命令/环境。共享GPU质量耗时仅诊断预算；Valen未计预处理保持null，实际费用未知null |
| review/定向验证 | reports/s0-technical-code-review-001.md；唯一P2（Valen模板TypeError省略关闭thinking）已关闭，历史真实记录均显式false。34相关tests/全仓ruff+格式/95源码mypy/build通过；GUIv2另7相关tests/ruff/mypy、Linux双构建通过 |

最后收口验证：30报告均passed/exit0/500分母、每cohort全模型数据SHA一致；30sidecar完整1000记录且SHA一致，双遍输入/原概率/预测按各runner字段精确。数据不变的有效证据直接复用，不为新session/Git HEAD不同重复耗卡。

截图v1发现网格标签捷径：无图无指令坐标规则497/500=99.4%，旧分数全部superseded并保留。v2确定性随机合法坐标、500原family保留（498原题/2同family几何重选）、双构建字节相同；冻结旧规则19/500=3.8%。只重测截图六模型，最终E4B51.4%、E2B26.0%、OneJev4 66.2%、27B81.2%、教师74.0%、Valen73.0%。不得拿旧GUI分数作训练基线。

## 精确资产、数据与环境

SSH：`ssh -p 48926 root@region-42.seetacloud.com`，已有mux `/tmp/omnijev-ssh-root-final`。凭据不入仓；若连接身份/端口变更才需要用户提供。资产根 `/root/autodl-tmp/lite-omnijev/artifacts`。H20 97871MiB，cgroup150GiB/16CPU；持久盘800G/已用414G/余387G。

- 主推理Python：`/root/autodl-tmp/envs/serve-vllm026-offline-check/bin/python`，TF5.13/Torch2.11+cu130/vLLM.26。
- 教师Python：`/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python`，TF5.17/Torch2.13+cu130。
- CPU：`/root/autodl-tmp/envs/root-main-offline-check/bin/python`。Python3.11，离线锁和安装证据见before-gpu runbook；不要在原环境全局升级。
- 冻结教师/IEEE参考树：`/root/autodl-tmp/gpu-reference-fp32-complete-001`，HEADdfdfd935，不改。
- 最后截图运行树：`/root/autodl-tmp/s0-screenshot-v2-final-001`，HEAD7b4f382；精确执行源码/配置hash在每模型报告。控制器为未跟踪操作脚本，不是模型代码；报告绑定实际命令与来源，不把主树文档提交冒充运行版本。
- E4B：`models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2`；其余模型确切路径/revision/逐权重SHA见configs/project-model-assets.json及每份模型报告，禁止猜新revision或换基座。
- OneJev4/27后续质量作业显式使用官方FLA0.5.2 optional overlay，wheel SHA5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761；运行profile记录PYTHONPATH/TRITON_CACHE_DIR。primary环境未全局修改。其公开231双轮质量候选证据s0-qwen-fla-candidate-result-001.json；warm约2.505倍仅共享诊断。
- Valen是官方Qwen3.5-2B+安全delta/head/LoRA/视觉428tensor，不是裸Qwen。delta路径`valen/Valen-Preview-0923/30096aaabb2689614c83a79f64f38c865c2ce1c1/weights.safetensors`，SHA27b8f6cd4874a356af8a5d6a3676d2a9a03e4b3301d066d13645f673f72f4468；禁止重跑pickle或不安全上游代码。

六数据绝对路径在 `configs/s0-development-cohorts-h20.json`，六dev SHA在s0-baseline-expansion-acceptance-001.json：text CLINC、image AOK、screenshot GUIv2、audio Clotho原一致Noul、video CLEVRER train exist Noul/16端点帧、crossmodal image+audio合取Noul。训练来源的许可/split隔离记录不得省略。

**当前训练排除入口**：`/root/autodl-tmp/s0-screenshot-v2-final-001/artifacts/dev-exclusion-history-union-v2-001/dev-exclusion.jsonl`，SHA3110f9740d9e7dd9c39023b106d977e927b3446f5dfa82e168f729c127aeda1e。3000family节点/2500原始family/2000连接组件；旧/新GUI题及媒体别名都保留。跨模态组件复用同池image/audio，不虚增独立family。任何后续train/calibration/holdout须查family和媒体SHA；S5每模态3–5k校准从未训练family选择，训练前预留，不能事后删训练账本制造隔离。

## 作业终态与失败恢复

- 最后27B：`artifacts/screenshot-27b-parallel-v2-001`，模型耗时2507.207秒，1000记录、report passed、model/parent/kernel exit0、finished。原2400秒估计按实测改4500，保留同一child344408；watchdog344615及parent344403均结束。收据reports/s0-screenshot-27b-deadline-handoff-001.json，不能看到历史STOP记录就重启。
- 原343888三学生任务exit0后停止未来派发，原27B从未启动。watcher遇已退出nvidia-smi子进程安全暂停，核实仅zombie后已收口；错误日志/SHA保留，reports/s0-screenshot-parallel-handoff-001.json。禁止恢复旧队列造成重复27B。
- 原始GPU记录同时在远端各报告指明的持久路径及本地忽略的`artifacts/s0-*-gpu-*`。当前截图本地`artifacts/s0-screenshot-v2-gpu-001`，教师位于其teacher子目录；有效模型报告绑定本地原始文件SHA。
- 缺报告/哈希不符先核原持久路径，不重跑全套；不得覆盖失败/历史原始报告。没有活动计算、没有需恢复训练checkpoint；S1尚无产物。
- 汇总可离线复现：在主树执行`python3 .scratch/s0/summarize-development.py`（已实际跑通），只读已冻结sidecar；不执行模型。

## 唯一未解除的S0门与下一步

s0:05仍缺真实外部gate身份、六类各1000封存/共12000真人原始判断/逐类κ≥.7及签认。工程canary证明的是拒绝/脱敏/κ算法，不是真人验收。开发agent禁止读封存载荷、代签或用模型标签制造真人证据。

ADR0011是**待用户决定**：数量/κ/隔离全部保留，仅将交付截止S0移至S4放量前。已有待答问题，尚未明确批准。若用户提供真实证明则按05验收；若明确批准迁移则落实ADR与阶段依赖，05继续作为S4/FINAL硬门，S0报告明确该缺口被迁移。两条路均不得伪称人标已通过。

解除后才认领08完成一次阶段出口，复用已有源码review/数值浏览器/基线证据，仅查真正缺口。S0全部成立再写正式`s0-complete`交接、按plan/DELIVERY-PLAN.md展开一个S1文本数据→LoRA→同条件评测闭环。原S1质量门：2k/300步≥98%、文本≥OneJev4−3pp且较E4B+5pp、ECE≤.06、30k较10k+2pp；不能用性能授权暗降。不得在当前停点提前启动训练。

整个方案的**预估性能/资源**按用户授权ADR0010实测自主校准，记录原值/证据/新标准/边界，限额优化后推进；质量、正确性、许可、隔离、真人要求保留。5b/探索默认关闭，不换基座/引擎，不push/付费/自动开关卡/生产部署。

## 可直接复制的新session启动词

> 在/Users/bamboo/Githubs/Lite-OmniJev继续原完整goal。先读AGENTS.md、plan/STATUS.md、plan/handoff-2026-10-15-s0-technical-complete.md和docs/02-project.md，验证票据/实际进程。P0和S0技术闭环全部完成，30/30支持开发基线及六公开模型双轮齐全，GPU已空闲；不要重跑已完成模型或恢复旧队列。s0:05真实真人/gate仍外部blocked，ADR0011仅proposal，请结合我本次的明确决定落实阶段入口；未解除不能启动S1。解除后只做一次S0出口，复用冻结证据，再按一个业务闭环推进S1。方案性能/资源估值按实测自主校准，质量/隔离不暗降，不细拆或过度验证，5b/探索不开，开关卡/付费/生产由我决定。

2026-10-15终态补充：连续三轮确认同一外部真人/gate缺口、无可执行frontier或活动作业后，原生goal标blocked；完整目标未完成。用户提供资源或明确ADR0011阶段决定后再恢复，恢复后重新做阻塞审计，不沿用旧计数。技术报告/基线无需重跑。
