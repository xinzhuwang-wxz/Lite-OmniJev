# STATUS

更新时间：2026-10-14。当前S0；P0工程冒烟完成，S0–S6/FINAL未完成。终局未完成，原生状态以get_goal为准；当前已证实外部资源阻塞。

## 当前目标与决策
- 用户已租实例，要求开卡前完成所有可做的模型/数据/依赖/CPU管线准备；未获开卡授权，保持无GPU作业。
- 原方案、阶段顺序和门槛不变；5b/探索关闭，不换基座、不改引擎、不放宽阈值。
- ADR-0004：AudioCaps首发省略，以Clotho/TTS补题量；ADR-0005：12k视频QA用CC0 CLEVRER，5k图题由允许教师在CC0/BY媒体生成，DocVQA仅研究隔离；NextQA/VGGSound/受限LLaVA仅候选，不下载1.28TB全量LLaVA。
- ADR-0006用户已批准：仅将S0 MVBench发布53.10±2复现替换为固定9任务1800行独立baseline；MMAU63.10±2、S3≥90%、S4≥97%及其他门槛不变。
- DocVQA、NDK r28b SDK协议已具体批准，CommonVoice/HLE/模型访问已解决，不重复索要凭据/条款。
- 用户要求当前会话直接推进，heartbeat已删除，禁止重建。不push/远端issue/PR/自动租卡开关卡/付费/生产部署。

## 资源与版本
- 远端 `root@region-42.seetacloud.com:48926`，原树 `/root/autodl-tmp/lite-omnijev`；用户已恢复SSH，主会话mux `/tmp/omnijev-ssh-root-final`，凭据不保存。
- 最新带完整Git历史的隔离源码快照：`/root/autodl-tmp/before-gpu-release-4366197`，目录名是初始快照名，新隔离源码 `/root/autodl-tmp/before-gpu-release-32fb559` 实际代码HEAD655a838；旧目录HEAD3dbbe0d；后续每次核对真实HEAD，不推定目录名=版本。
- 本地bundle `artifacts/before-gpu-final-3dbbe0d.bundle`，远端同名在 `/root/autodl-tmp/`；未覆盖用户原树/未改远端分支。
- 无卡cgroup2GiB RAM/0.5CPU，重作业锁 `/tmp/omnijev-cpu-heavy.lock`、数值线程1；不得按宿主机free调度。最新远端剩余430,532,730,880B；保留150GB checkpoint及18.16GB模型临时空间。
- vLLM0.26仍primary，0.31仅准备/诊断环境，不静默切换。026/031两套完整wheelhouse/fresh安装已实际通过CPU导入。
- Mac24GiB内存，仅小型公开fixture/metadata，不搬大模型/原始包。只清理自己创建且无用的资源，不删权重缓存/用户容器。

## 票据状态
- bootstrap:01–08、p0:01–07 done，仅工程闭环，见 `reports/p0-engineering-acceptance.md`。
- offline-prep:01 done：7模型70文件121,094,724,397B，固定revision/许可/逐文件哈希通过；`reports/project-model-offline-acceptance-001.json`。
- offline-prep:02 done：29来源/批准排除/未来生成/外部gate汇总；`reports/project-data-offline-acceptance-001.json`。原料准备不等于训练准入。
- offline-prep:03 done：必需运行时外部依赖统一通过，`reports/project-runtime-offline-acceptance-001.json`；新版teacher/job入口空env fresh no-index重建224.9秒，mandatory imports/probes与pip check退出0。
- offline-prep:04 blocked（无活跃owner），依赖01/02/03/s0:09；全项目 `ready_to_open_gpu=false`。native processor CPU验证及Linux真实网关视频资源验证尚未通过，不能隐入GPU-only。
- offline-prep:05/06 done：pytest锁修复、真实Linux教师音频CPU适配。
- s0:09 done：官方HF100题×2输入相同；4原始family变体不是100独立质量样本。`reports/s0-prepare-integrated-acceptance-002.json`，仅CPU输入。
- s0:01 blocked（无活跃owner）：安全head/loader/teacher forward/benchmark runner代码与CPU控制已合入；全13文件23,955,723,321B已Linux校验；GPU真实概率/复现门槛pending，不关整票。
- s0:03 blocked（无活跃owner）：canonical媒体/原生chat/collector/网关/版本/失败路径代码已整合；历史HF/vLLM maxTV .01129/.02292仍failed，严格<.001，D004未批准/未应用；GPU真实概率和浏览器模型验收pending。
- s0:05/06 blocked：独立封存gate身份及两真人6000题/12000判断、κ≥.7资源缺口。不得agent自标/合成κ。
- S0全绿前禁止S1；delivery:01未完成，原生goal不complete。

## 已完成外部准备
- GUI1679train episodes/25809 PNG/19,499,790,993B全量下载/CRC/SHA/decode通过exit0；4652.928秒，作业已结束，不重复78分钟解码。逐文件manifest在远端，摘要 `reports/gui-odyssey-full-decode-cpu-001.json`。
- 固定公开视频1606媒体/1800行/27997唯一PNG/28800帧引用全部SHA/尺寸/头部检查通过；`reports/public-video-protocol-freeze-001.json`，GPU教师质量pending。
- DocVQA完整tar EOF/12767成员/引用缺失0、32图decode；MMAU1000音频全decode；CommonVoice、Clotho、CLEVRER、STAR补齐、Perception/FunQA/STA、ScreenSpot、阶段2图表/图像原料均有CPU证据。来源/许可/研究隔离及确切检查边界统一见29来源汇总，不将代表decode称全量。
- RouterBench与Valen/Jev-Omni head安全转换/隔离/独立审查已完成；真实恶意pickle主机读取与网络拒绝证据已重验。禁止Jev API/JEV-27B权重或标签；封存内容不读。
- SwanLab仅真实训练指标；Kokoro CPU4 WAV、llama.cpp CPU build、Ollama官方版本、quant独立no-index安装、LiteRT私有loader/Androidarm64 CLI/NDK交叉编译均已有证据，见runtime汇总。
- Android端侧首发以既定CLI/E2B离线运行交付为当前依赖范围；AAR仅未来API集成缓存，Gradle transitive/App build没有完成且不标done；依据 `reports/litert-android-delivery-scope-audit-20261014.json`，真机/模型运行仍S6门槛。

## 本轮代码与验证
- 共享helper：HLS/间接引用拒绝、SHA/路径边界、max8媒体/累计40MP先header后decode；取消请求仍占用执行槽直到CPU线程结束，关机drain。
- 原生chat：完整requested IDs+U、精确expanded IDs/usage、raw概率校验；计入集合外sampled token，集合内重复logprob须相同，唯一token原始质量不得>1。
- TF5.17视频422已定位并修复：官方仅输出pixel_values_videos/video_position_ids，5.19额外framecount不强制/不伪造；native检查仍从冻结帧indices严格核对video_frame_counts和timestamps，未降低等价门槛。
- 教师runner固定MMAU1000/九任务1800分母、两遍/断点/源码与模型hash绑定；真实Mac官方3音频+1视频processor通过，TF5.17与5.19教师样本摘要一致。CPU/fake-forward不证明教师分数。
- prefix cache已真实CPU冻结协议/媒体前缀，native/gateway runner已整合；mock ratio失败照留，不冒充≤2性能通过。
- 最新主树实际全量：560 passed/1 skipped，44.24秒；skip仅Linux root隔离，远端正确reader单项1pass/2.30秒。Ruff/format263文件/mypy73+infra2/build/32票validate均通过；`reports/before-gpu-cpu-validation-005.json`；004历史证据保留。
- 实际Chromium153官方CPU处理器+网关+明确fake backend：文本/图像/截图/音频/8/16视频×3primitive、坏路径422隐私、未配置503通过；`reports/browser-s0-cpu-004.json`及截图/trace。不当作真实模型或时延通过。
- 独立只读复核概率/版本/帧数三个delta：11pass/无新增可证问题；前述取消/资源预算复核9pass。
- SSH恢复后前端399包fresh offline install downloaded0/build exit0；组件LICENSE文本/hash已缓存（Animate为MIT+Commons Clause），`reports/offline-runtime-remote-recovery-20261014.json`。
- teacher/job环境真实PEFT0.21.2/accelerate1.15/pytest9.1.1/pip check0，新增统一入口按manifest校验exact overlay wheels/--no-deps防Torch被升级。新fresh作业见下项。

## 已证实缺口与活跃作业
- Linux全suite退出137，不能称通过；轻量52pass/1未配reader skip，正确隔离reader补验1pass。初始错路径exit4及错data reader失败保留，不伪装产品问题。
- Linux官方视频修复前422根因已解决；修复后有效8/16batch仍137/零completed，memory.events oom/oom_kill=0，未知killer，不称kernel OOM/CPU不支持。`reports/linux-video-processor-compat-001.json`。
- native vLLM CPU正常ModelConfig子进程SIGKILL；同官方类进程内诊断到creating_processor后137，零cases；没有修改生产引擎。`reports/s0-native-vllm-processor-summary-001.json`。
- Docker辅助CPU尝试与恢复SSH后的官方下载中转均下载截断/TLS/OCI503，未完整SHA、不安装/不开engine/零processor；`reports/s0-docker-processor-001.json`及relay002。已停止同路径重试，无自建活跃container，不动用户container/daemon。
- 新版teacher/job fresh作业已退出0，env `/root/autodl-tmp/envs/before-gpu-teacher-job-final-001`，报告 `reports/before-gpu-teacher-job-fresh-001.json`；无活跃fresh安装作业，不重复启动。
- primary026对应TF5.13/tokenizers0.22.2已在独立Mac target完成100题两遍；HTTP/media kwargs/答案映射/input IDs均相同。20video缺可选count且video tensor多前导batch维，字节SHA相同但shape差异明确记录；`reports/primary-transformers513-cpu-full100-comparison-001.json`。不称完整跨版本tensor等价/GPU通过。
- 新官方CPU通路已下载并双端核验完整ARM vLLM0.31 CPUwheel68,640,121B/SHA8a9f…f9cb；官方GHCR ARM base拉取180秒超时，未形成image，无container/install/processor，已停止；`reports/s0-official-arm-cpu-alternate-003.json`。
- primary026 LinuxHF完整100题首次通过exit0/94.092秒；第二遍到90/100后137，未通过Linux重跑；native官方类诊断creating_processor137/零cases。`reports/primary026-linux-cpu-acceptance-001.json`，Mac/Linux60case有编码/张量摘要差异照留，不冒充跨机器模型等价。
- 原生网关向HTTP后端发包前释放已完成摘要的大HF媒体Tensor，保留精确input IDs/canonical bytes/summary绑定；真实weakref先红后绿、94网关/client测试与独立review通过。此改进不解决2GiB全部资源门槛；新两视频batch137，无完成标记。
- FFmpeg stdin消费真实复现并修复：probe/decode子进程DEVNULL，实时q输入视频先红后绿，26媒体测试通过；不让SSH管线下一行命令被吞。
- 所有监督句柄/子代理作业已结束；无活跃CPU/GPU/container作业。最终同步receipt随本批报告保存，不能重复启动失败配置。

## 待用户/外部条件
- 当前不要开卡；真实native CPU processor与Linux视频在2GiB条件下仍无通过证据。用户已决定“继续保持当前实例”，不调整资源、不自动开卡/付费。更大无卡资源或其他解决路径需新的外部条件；不反复询问同一决定。
- 真实gate身份、双人标注、S6真机及后续运营观测须安排；不阻塞其余本地准备，也不伪造终局门槛。
- 未来LoRA/校准/量化/派生训练集是待生成产物，不是必需外部下载遗漏；S4放量预算须届时按实测重估。

## 下一步
1. 当前可执行分支已收口，原始下载/环境/CPU测试不重复；需外部条件改变才恢复native/视频/重跑的资源受限CPU验收。
2. 统一开卡判定 `reports/project-before-gpu-readiness-001.json` 仍false；模型/数据/运行时01–03 done，04blocked；不静默把CPU门槛移到开卡后。
3. 当前plan/handoff仅CPU检查点；用户要求真正可开卡时写专用开卡交接，届时固定HEAD/资产与配置hash/环境/命令/预算/断点/失败恢复/关卡清单。S0/GPU/终局门槛仍待验。
