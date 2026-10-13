# STATUS

更新时间：2026-10-14。当前 S0；P0 工程冒烟完成，S0–S6/FINAL 未完成。

## 当前用户指令与资源
- 全项目必需模型、数据和运行时先下载、校验并完成 CPU 准备，再通知用户开卡；当前无卡，禁止 GPU 作业。
- SSH root@region-42.seetacloud.com:48926，ControlPath=/tmp/omnijev-ssh-%C。
- 远端项目 /root/autodl-tmp/lite-omnijev；800GiB 数据盘（750GiB 付费）。不自行开关卡或购买服务。
- 下载用 source infra/bootstrap/autodl-env.sh、source /etc/network_turbo（静默），HF_ENDPOINT=https://huggingface.co；禁止输出凭据。

## 当前票
- bootstrap:01–08、p0:01–07 done；证据 reports/p0-engineering-acceptance.md。
- offline-prep:01 claimed：七项模型约121GB；PID1827 flock 队列仍运行，67文件约37.1GB校验，零报告失败。以实际进程和最新报告为准。
- offline-prep:02 claimed：数据来源、原料与许可准备；Valen20文件约6.36GB全部哈希通过，仅来源隔离，不准入训练。
- offline-prep:03 claimed：运行时准备；已有环境、FFmpeg及量化/端侧源码归档校验完成，CPU构建/剩余依赖仍需完成。
- offline-prep:04 pending：依赖前三票；下载、校验、CPU准备与缺口审计全部完成才能通知开卡。
- s0:01 claimed：教师CPU头安全转换及修复已合入，独立FIFO修复复核关闭；完整教师推理/复现未验收。
- s0-data:01 done：CLINC train15250条语义保真、官方33token通过；权威证据 clinc-admission-corrected-001.json。
- s0:03 claimed offline-runtime：实际Gemma4 CPU媒体处理器通过；严格HF/vLLM概率门槛仍failed，不能关票。
- s0:05 blocked：canary工具已验证，真实独立封存身份缺失。
- s0:06 blocked：需要真实独立双人标注，不能伪造κ。
- delivery:01 未完成；S0全部通过前禁止进入S1。

## 当前无卡产出与下载
- reports/project-model-download-snapshot-001.json 是 running 快照，不作完成证据；旧11060/1408下载进程已终止，不再使用。
- reports/valen-source-download-001.json：20文件全哈希通过；源资产在 artifacts/data/quarantine/valen-general-100k/。
- 文本来源17文件30,666,690B本地核对固定上游哈希，已同步远端并逐个SHA256复核；reports/stage1-text-source-verification.json。
- JevEmbed 600k候选行/Open-Jev原料仅来源审计，未证明独立样本族或训练准入。
- Clotho官方原料3.14GB；scripts.prepare_clotho实现许可白名单提取：1659 CC0/CCBY，排除332未知/NC/其他许可。
- Clotho首次任务因辅助脚本导入路径错误退出；已修正PYTHONPATH；第二次遇HTTP IncompleteRead，改用curl有界重试，shell PID7933（后续以ps核实）。报告 clotho-aqa-source-download-001.json；最终音频报告 clotho-aqa-audio-preparation-001.json 尚待下载完成。
- Clotho未知来源/许可不准入；原始压缩包隔离，只有白名单音频提取至source-audit并保留归属。
- reports/cpu-media-probe-offline-runtime.json：真实TF5.19 Gemma4Processor图像/截图/16kHz WAV/3帧视频输出与答案边界通过；无模型forward，不代表概率或质量验收。
- 量化/端侧源码归档约389MB已校验：llm-compressor、llama.cpp、Ollama、LiteRT-LM；源码到位不等于构建完成。
- teacher head四张量FP32隔离转换通过；父进程输出nofollow/nonblock/regular/size/timeout修复已合入并独立复核；完整教师模型仍待下载和GPU验证。

## 并行所有权
- offline_runtime_dependencies：/tmp/lite-omnijev-offline-runtime；CPU媒体切片已交付，root整合。
- speech_assets：/tmp/lite-omnijev-speech-assets；TTS模型/语音许可及CPU canary，仍执行。
- whole_project_datasets：/tmp/lite-omnijev-project-datasets；ScienceQA/ChartQA/DocVQA/A-OKVQA来源与媒体准备，仍执行。
- teacher_head_conversion：/tmp/lite-omnijev-s0-teacher；下一片为Valen嵌套PT的官方规格与隔离转换，仍执行。
- root独占主目录、票据、STATUS与合入；子代理显式workdir，只写指定worktree，不改全局计划。

## 下一步
1. 完成模型、Clotho和剩余许可明确数据的下载/哈希；保留失败证据，不因running报告就认定进程存活。
2. 整合语音与阶段2来源切片，完成CPU运行时构建/依赖缓存、Valen安全转换；未来生成的LoRA/量化产物单列待生成。
3. 离线集成票逐项验证全项目外部依赖；缺口写blocked，不能以下载清单代替实际到位。
4. 下载/CPU准备全部满足才通知用户开GPU，安排教师/媒体/概率任务批次减少卡时；不改变原门槛。

## 待我决策
- 当前不需要用户开卡。正式改概率门槛需充分实测证据与ADR，现有门槛保持failed，D004搁置。
- 真实封存隔离身份/人工标注资源后续仍需用户资源；先推进不依赖这些资源的离线工作。
- 付费、生产、删除、方向/范围和预算改变仍由用户决定；账号/许可条款不擅自接受。

## 验证与卫生
- Clotho+教师修复加入后本地226测试、Ruff/format/mypy/build通过；新CPU媒体切片整合后的可选依赖类型导入问题已修复；229测试、Ruff/format、mypy38源文件、wheel/sdist和29票规划校验全通过。
- 不push、不创建远端issue/PR、不轮询gh；只本地提交和SSH同步。
- 保留模型、数据缓存与P0约561MiB完整备份；只清理自己创建且过期资源。
- 运行循环 plan/LOOP.md；交接 plan/handoff-2026-10-14-offline-prep.md。
