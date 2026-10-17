# Lite-OmniJev S0 完成交接

2026-10-15，Asia/Shanghai。**S0按用户批准的ADR0010/ADR0012已通过，s0:08 done，S1尚未开始。** 当前session只完成S0与交接，剩余训练由新session继续。完整目标仍为P0、S0–S6、API/单卡/E2B端侧发布及FINAL，未声称全项目完成。

## 先执行

1. 在 `/Users/bamboo/Githubs/Lite-OmniJev` 读AGENTS.md、plan/STATUS.md、docs/02-project.md、本文件和docs/adr/0012-existing-gold-without-new-human-labeling.md。
2. `git status --short`、`git log -1`、`uv run python -m scripts.loop validate`。S0全部技术/出口票done；s0:05已移至S4工程票、needs-triage/open，不再是人类阻塞；FINAL仍等待后续阶段。不要恢复历史GPU队列。
3. 复用reports/s0-stage-acceptance-001.json：实际S0六指标门全通过，绑定来源报告SHA。人工双标/κ已按用户明确指令删除（not_applicable），不是填成已通过。ADR0011仅延期已被ADR0012取代；禁止到S4再要求用户找真人。
4. 进入S1前展开一个文本数据→决策LoRA→同条件评测业务闭环，先核已有数据、pool/family分配和环境；只跑必要小规模探针，再按原质量门推进。不要重做S0基线、缓存或浏览器验收。
5. GPU最后核实无计算进程/0MiB，已通知用户可关卡，未自动关机；新session若机器关卡，不自动开卡，先完成可CPU工作。费用未知null。

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

## 数据真值与后续阶段入口

用户最新决定：“我没有什么人标注的能力，这部分删了要不，就用能找到的数据”。ADR0012删除新增两真人、12000判断、κ和用户外部gate账号前置，改用许可明确的现成可靠gold/公开benchmark gold/可验证程序真值。不付费标注，不拿教师标签冒充最终评测真值，不把已有dev兼作holdout。

train/dev/calibration/holdout整family及媒体隔离、来源/revision/hash、独立holdout和自动gate汇总仍保留。已下载数据可直接复用其可靠真值，但独立测试分池/映射/去重仍是工程任务；不是要求用户新增标注。S1正式入训前冻结分配并预留未训练holdout/calibration family，S4前05完成六cohort各1000实际gold holdout及自动gate，FINAL直接依赖05。旧v2双标canary仅历史工具，不是首发必走接口，新的gold gate由05交付，不假称已经实现。

S0实际门：MMAU delta .011≤.02、精确重复1、HF/native最大四概率差3.229375648383659e-5<.001、完整基线1、MVBench独立baseline核验1、真实prefix-cache测量核验1。机器命令：`uv run python -m eval.gate reports/s0-stage-metrics-001.json reports/s0-stage-thresholds-001.json`（已真实exit0）。技术报告旧字段“S0未完成/人标待决定”是当时历史，当前由s0-stage-acceptance-001.json及ADR0012覆盖，原测量值未改。

S1首个可执行闭环：核CLINC dev500/余train14750及已有文本来源、预留未训练池→按当前基座/官方模板/答案token关闭thinking→2k/300步小规模LoRA→固定dev500同条件评测→10k/30k扩量对照。原门保持：smoke≥98%、文本≥OneJev4−3pp（当前96.6%基线）且较E4B+5pp（当前93.6%）、ECE≤.06、30k较10k+2pp。不要暗降质量数值，不把不同选项/数据协议成绩互比。S1规格/票在新session按一个闭环展开，不提前拆成来源/探针小票。

完整goal工具当前仍保留此前人标阻塞时的历史blocked状态（当前工具没有resume操作）；本次用户决定已经解除本地阶段阻塞，不能据历史状态继续要求真人，也不能把整个goal标complete。新session按原完整目标继续，按当前票据/ADR恢复。

## 可直接复制的新session启动词

> 在/Users/bamboo/Githubs/Lite-OmniJev继续原完整goal。先读AGENTS.md、plan/STATUS.md、plan/handoff-2026-10-15-s0-complete.md和docs/02-project.md。S0已按批准ADR0010/ADR0012通过，30/30支持开发基线及六公开模型双轮齐全，所有GPU作业已结束；禁止重跑S0或恢复旧队列。用户已明确删除新增真人双标/κ/外部人工账号，后续使用许可明确的现成可靠gold和自动gate，数据/质量/隔离保留；不要再请求人标能力。现在按一个文本数据→LoRA→同条件评测业务闭环展开并推进S1，先预留未训练holdout/calibration family；S4前05完成实际现成gold封存及自动gate。方案预估性能/资源按实测自主校准，质量不暗降，不细拆或过度验证，5b/探索不开，开关卡/付费/生产由我决定。
