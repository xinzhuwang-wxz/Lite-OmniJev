# STATUS

## 当前会话接续（2026-10-14，有卡；本节优先于历史资源记录）
- 用户已开卡并授权按原方案自主技术决策/调度，减少GPU时间；不换基座/引擎、不降低阈值，不代开关卡/付费/生产部署/push。5b/探索关闭。
- H20/97871MiB、driver580.65.06、CUDA13.0；cgroup150GiB/16CPU，盘余约420GB。SSH socket `/tmp/omnijev-ssh-root-final`；凭据不入仓库。
- 冻结执行树 `/root/autodl-tmp/gpu-reference-fp32-complete-001` HEADdfdfd935；网关验收runner树 `/root/autodl-tmp/gpu-gateway-contract-001` HEAD0ca8ff1。原用户树/764ee500树保留。不要从目录名推断版本。
- s0:03 done；正式FP32/IEEE HF100两遍精确/162.734秒、native100两遍精确/176.527秒；raw/joint/conditional/pU最大差分别0.0000322938/0.0000322830/0.0000167780/0.0000322830，全部严格<0.001。新汇总 `reports/s0-full100-gpu-acceptance-001.json`，独立verifier通过；25定向回归通过。
- 网关正式100两遍精确/123.138秒，conditional及pU与native最大差0；真实Chromium153六类×三primitive/坏媒体422通过/31.353秒，artifacts/browser/s0-live-fp32-ieee-001含截图trace。置信度validator已按既定chance-adjusted公式修复，API未改；14回归通过。
- primary026/TF5.13/Torch2.11，audio extra已完整锁定/pip check0。视频官方归一化参数及同Linuxprocessor/HF100逐字节输入等价已通过。FP32仅数值参考；BF16训练不变，最终服务精度/性能另验。
- 原BF16两遍raw漂移0.091862、HF/native差0.136941失败仍留；FP32仅官方config副本四个dtype字段，TRITON_F32_DEFAULT=ieee和TF32关闭解决参考误差，不更改原权重/模板/阈值。
- 验收native PID94918/18001与gateway PID130178/18000已发送TERM；本机转发55174→18000暂保留，服务不再保证可用。实时/proc+health绑定在artifacts/s0-parity-full100-20261014/live-process-binding-001.json，无签名remote attestation，不伪称独立签名证明。两服务仅开发数值验收，非生产。
- 教师公开text/audio/video两遍真实forward通过/44.564秒；原tensor格式→safehead的GPU功能参考47.418秒exit0：四tensorhash、5counts边界及三类两遍logits/probs精确；31回归通过，s0:01原README声明/安全GPU链已获独立verifier PASS，已关闭；s0:02主会话认领效果复现。严禁正常进程加载pickle或执行上游module。
- 教师benchmark原pilot900秒退出124，保留1397条；MMAU首遍1000题620正确/62.0%，在63.1±2pp内；MVBench首遍397题仅部分，不作完整baseline。两遍完整质量门槛仍pending。
- 教师resume004安全停于3812/5600条/exit1；MMAU两遍1000对预测/概率精确一致、均62.0%，单benchmark通过（reports/teacher-mmau-two-pass-complete-001.json），教师整体未准入。异步短测结束后启动resume005：监督169821/timeout169822/教师169823，实时Rl，reports/teacher-benchmark-progress-005.json快照4236/5600、GPU仅教师24794MiB，最多3600秒；同DF冻结源码和records显式续跑，无自有服务占卡。
- 原生子代理：parity_verifier/teacher_reference均完成，临时树已清理；cache_protocol_review已交只读建议；cache_assessment已整合显式新policy（7949365）；34pass/1显式livefixture skip、ruff/mypy通过，架构只读review CLEAR。父会话统一票/STATUS/整合。
- 原生get_goal已确认active；当前S0；完整教师/缓存/独立gate/人工一致性未全绿，S1尚不启动，goal终局未complete。无卡时期失败均历史证据，当前不重复资源失败路径。

下一步（本轮优先）：
- 01/03已关票；02教师resume005继续MVBench第二遍。MMAU两遍均620/1000且完全一致，MVBench第一遍976/1800；第二遍未完成，不关闭02。先核实际PID/exit，活作业不重启。
- 04仍claimed/性能failed：旧顺序eager/graph失败保留；新scheduled eager冷比2.3469/2.3250，graph1024并发10为2.584/2.551、并发2为3.190/3.188、并发3为3.045/3.047。实际KV覆盖/并发与四概率均通过；未做失败配置gateway/off；两server已停、GPU交还教师。见reports/s0-scheduled-native-eager-canary-001.json及s0-scheduled-native-graph-canaries-001.json。
- CPU候选已完成：Linux官方100题300次准备精确/40imagecache命中；失败CPUjob清理独立复核APPROVE。新网关并发adapter含实际ASGI wall/work分离、upstream输入/raw/API逐qid绑定、重复case拒绝及摘要JSON roundtrip；独立review APPROVE（reports/s0-scheduled-gateway-review-001.json）；48相关pass、78源mypy/lint/282format/build/32票validate通过，仅CPU准入。
- async eager官方候选已测：冷比2.3164/2.3335仍fail、四概率max2.36e-7、并发9/KV288，启用日志/真实cmdline与env绑定；已停止server169427、不扩展gateway/off，见reports/s0-scheduled-native-async-canary-001.json。下一TF32x3仅官方安装源评估可用、尚未GPU准备（reports/s0-tf32x3-source-assessment-001.json含实际源码hash）；会改变部分数值实现，不能当IEEE相同，若做必须隔离编译缓存并重验完整100概率/重复，收益假设偏弱，先保持教师占卡。
- s0:07已认领剩余基线/开发数据规格：移除02/04对纯CPU规划的人工串行依赖（08硬门槛不变），01/03实测充分。模型/协议事实已收集；10的CPU CLI切片正在独立树实现，父会话发现metadata应对expectedhash、partial退出码和Valen实际adapter目录等边界缺口，已要求先修并回归，不整合未通过版本；OneJev4/27、E2B和额外Valen仍缺基线forward，JevBench231/literal含null148（147非空groups）与教师195口径未证实，dev500不由100 parity fixture替代；规划见.scratch/s0/baseline-expansion-spec.md。
- s0:10/11已认领，12为可执行frontier；35票validate通过。JevBench实际八文件/231行审计已完成，纠正easy group为JSON null而非easy字符串；官方36original对只用于paraphrase consistency，不能当195场景准确率；139Choice/18离散Score/74Noul都可适配，gold发布方声明非Jev API，但hard模型作者不充真人双标。见reports/jevbench-public-protocol-audit-001.json，下一实现合法adapter/官方计分语义。
- 用户新要求：S0全绿后产出完整 `plan/handoff-日期-s0-complete.md` 及可复制的新session启动词；已加入s0:08验收，不把中途handoff当作S0完成。
- 补充真实Linux官方processor+ASGI网关回放联调：已发现并修复测量器写死q0..q9的阻断，现保留真实cache-probe-*及顺序绑定；15回归先红后绿/mypy/ruff通过。官方CPU回放9.896秒exit0、8测量/44题/50回放调用、JSON摘要重载通过，reports/s0-scheduled-gateway-official-cpu-replay-001.json，明确无GPUforward/性能/真实KV证据。
- 接续：plan/handoff-2026-10-14-scheduled-cache.md。05/06独立gate/真人资源仍外部条件；S0全绿前不启动S1，终局goal active。

## 当前目标与决策
- 用户已完成开卡，原无卡准备收口；现在复用现有模型、数据、环境及P0产物推进真实GPU闭环，不代开关卡或改租赁。
- 第十一节逐项阶段检查点见plan/optimization-checkpoints.md：到建议阶段评审收益/预算并记录决定，探索/5b仍默认关闭；当前未启动可选实验。
- 原方案、阶段顺序和门槛不变；5b/探索关闭，不换基座、不改引擎、不放宽阈值。
- ADR-0004：AudioCaps首发省略，以Clotho/TTS补题量；ADR-0005：12k视频QA用CC0 CLEVRER，5k图题由允许教师在CC0/BY媒体生成，DocVQA仅研究隔离；NextQA/VGGSound/受限LLaVA仅候选，不下载1.28TB全量LLaVA。
- ADR-0006用户已批准：仅将S0 MVBench发布53.10±2复现替换为固定9任务1800行独立baseline；MMAU63.10±2、S3≥90%、S4≥97%及其他门槛不变。
- DocVQA、NDK r28b SDK协议已具体批准，CommonVoice/HLE/模型访问已解决，不重复索要凭据/条款。
- 用户要求当前会话直接推进，heartbeat已删除，禁止重建。不push/远端issue/PR/自动租卡开关卡/付费/生产部署。

## 资源与版本
- 远端 `root@region-42.seetacloud.com:48926`，原树 `/root/autodl-tmp/lite-omnijev`；用户已恢复SSH，主会话mux `/tmp/omnijev-ssh-root-final`，凭据不保存。
- 最新带完整Git历史的隔离源码快照：`/root/autodl-tmp/before-gpu-release-4366197`，目录名是初始快照名，新隔离源码 `/root/autodl-tmp/before-gpu-release-32fb559` 此前冻结HEAD36e51f8（执行代码655a838）；旧目录HEAD3dbbe0d；后续每次核对真实HEAD，不推定目录名=版本。
- 最终CPU检查点bundle `artifacts/before-gpu-checkpoint-final.bundle`：冻结HEAD36e51f8、完整Git历史，已同步隔离源码32fb559目录并远端37报告hash/32票validate通过；`reports/before-gpu-source-sync-001.json`。本地后续仅此receipt/调度元数据，不覆盖原树或改远端分支。
- 无卡cgroup2GiB RAM/0.5CPU，重作业锁 `/tmp/omnijev-cpu-heavy.lock`、数值线程1；不得按宿主机free调度。最新远端只读核查剩余420,558,426,112B；保留150GB checkpoint及18.16GB模型临时空间。
- vLLM0.26仍primary，0.31仅准备/诊断环境，不静默切换。026/031两套完整wheelhouse/fresh安装已实际通过CPU导入。
- Mac24GiB内存，仅小型公开fixture/metadata，不搬大模型/原始包。只清理自己创建且无用的资源，不删权重缓存/用户容器。

## 票据状态
- bootstrap:01–08、p0:01–07 done，仅工程闭环，见 `reports/p0-engineering-acceptance.md`。
- offline-prep:01 done：7模型70文件121,094,724,397B，固定revision/许可/逐文件哈希通过；`reports/project-model-offline-acceptance-001.json`。
- offline-prep:02 done：29来源/批准排除/未来生成/外部gate汇总；`reports/project-data-offline-acceptance-001.json`。原料准备不等于训练准入。
- offline-prep:03 done：必需运行时外部依赖统一通过，`reports/project-runtime-offline-acceptance-001.json`；新版teacher/job入口空env fresh no-index重建224.9秒，mandatory imports/probes与pip check退出0。
- offline-prep:04 done：当前无卡可自主完成准备已验收，`reports/project-before-gpu-readiness-002.json` ready_to_switch_to_gpu_mode=true；资源受限CPU检查显式列为切换后首批前置，失败证据不改写，S0门槛未通过。
- offline-prep:05/06 done：pytest锁修复、真实Linux教师音频CPU适配。
- s0:09 done：官方HF100题×2输入相同；4原始family变体不是100独立质量样本。`reports/s0-prepare-integrated-acceptance-002.json`，仅CPU输入。
- s0:01 done：13文件23,955,723,321B/原README声明/隔离转换/真实GPU原值功能参考/安全拒绝回归/独立复核全部通过，仅技术准入。s0:02 claimed：完整效果复现待两遍5600条，当前2610条checkpoint/resume004运行。
- s0:03 done：真实FP32/IEEE正式100、网关100和模型浏览器及独立复核通过；历史BF16失败保留，门槛不变。
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
- 无卡收尾时作业均结束；当前有卡活跃作业以顶部接续节为准，禁止重复启动。

## 待用户/外部条件
- 用户已完成切换；2GiB历史资源阻塞已按首批补验通过，不再要求切换或重跑无卡路径。独立gate/真人标注仍需外部条件。
- 真实gate身份、双人标注、S6真机及后续运营观测须安排；对应阶段门槛仍blocked，不伪造终局证据。
- 数据许可/来源训练准入逐源执行，研究隔离照留；未来LoRA/校准/量化/阶段派生训练集待生成，S4放量预算届时按实测重估。

## 无卡阶段收尾与下一步
1. 已审核32票无frontier/claim，三个离线总账及子回执逐SHA验证、远端7个必要路径存在/源码干净/无作业，资源与失败边界写入readiness002。当前无卡可自主执行剩余工作为空；不重复原始下载/环境安装/同2GiB失败配置。
2. 专用接续文档 `plan/handoff-2026-10-14-gpu-transition.md`：精确环境/缓存/fixture、首批CPU资源补验、GPU小smoke、S0门槛、预算/断点/失败恢复与关卡通知；可复制新session启动词已写好。
3. 最新源码同步用 `reports/before-gpu-transition-sync-001.json` 与远端同名侧车核对HEAD/完整Git bundle；此前source-sync001仅历史CPU检查点。后加receipt仅元数据，执行代码仍655a838。
4. 用户切换后按交接先读资源，补同Linux HF100两遍/native正常六case与full100/网关视频8/16，再真实模型与S0。历史maxTV失败及D004未批准不变，S0全绿前禁止S1。
5. 无卡交接时goal曾blocked；当前用户已恢复，工具确认active。完整目标未完成，不重建heartbeat。
