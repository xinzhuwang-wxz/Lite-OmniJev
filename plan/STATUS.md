# STATUS

更新时间：2026-10-14。当前 S0；P0 工程冒烟完成，S0–S6/FINAL 未完成。

## 用户指令与资源
- 整个项目必需外部模型、数据、依赖下载并哈希/CPU验收后，才通知开卡；当前无卡，禁止 GPU 作业。
- SSH root@region-42.seetacloud.com:48926；ControlPath=/tmp/omnijev-ssh-%C；远端 /root/autodl-tmp/lite-omnijev。
- 数据盘800GiB（750GiB付费），最近实际剩余约510GiB（548,087,324,672B）。无卡cgroup只有2GiB RAM/0.5核；证据 reports/no-gpu-resource-limits.json，不能按宿主机free/os.cpu_count调度。
- 网络：静默source /etc/network_turbo，HF_ENDPOINT=https://huggingface.co；凭据不输出、不进Git。
- 重型CPU作业用 /tmp/omnijev-cpu-heavy.lock 串行；网络继续。llama.cpp原暂停树已恢复并完成CPU构建，不再使用旧暂停PID判断当前状态。

## 票据与门槛
- bootstrap:01–08、p0:01–07 done，reports/p0-engineering-acceptance.md；工程冒烟不代表质量达标。
- offline-prep:01 done：7模型70必需文件121,094,724,397B全部哈希，与固定manifest逐项相符；已预留150GB checkpoint/adapter与18.16GB模型临时空间。reports/project-model-offline-acceptance-001.json；只关闭模型下载，不代表全项目开卡就绪。
- offline-prep:02 claimed：全阶段来源原料与许可准备；下载和媒体验证继续，账号/许可/精确协议缺口不能标齐备。
- offline-prep:03 claimed：运行时/依赖闭包；已有环境与源码归档不能替代可执行文件、完整依赖和CPU验证。
- offline-prep:05 done：仅pytest现有开发依赖修复，锁9.1.1，主树273测试无skip通过；没有升级远端GPU环境。
- offline-prep:04 pending：前三票及全部外部缺口解除前，开卡准备验收不通过。
- s0:01 claimed：Valen安全转换428张量已通过；完整教师推理/复现未验收。
- s0-data:01 done：CLINC train15250条语义保真、官方33token通过；clinc-admission-corrected-001.json。
- s0:03 claimed：真实Gemma4 CPU媒体处理器通过；HF/vLLM概率一致性maxTV .01129/.02292，严格<.001仍failed，原门槛未改。
- s0:05/06 blocked：真实独立封存身份/双人标注资源缺失；工具/合成真值不代替真实身份或κ。
- S0全通过前禁止S1；delivery:01未完成，D004 FP32 LoRA补丁搁置。

## 下载实际状态（以PID/文件/最终退出报告为准）
- 模型全部完成，最终退出0/零失败；reports/project-model-download-001.json。
- AndroidControl官方22对象49,930,349,992B与Mind2Web49文件13,577,379,722B均完成哈希，最终报告002；仅研究隔离，不准入商业训练。
- Clotho003外部6次续传已耗尽，379,707,764B partial保留；新004先验证1MiB Range返回206及精确Content-Range，再启动4并发8MiB有界片段，每片3次/180秒，拼接后必须官方MD5才提升。reports/clotho-range-recovery-004.json，目前running；完成后锁内1659 CC0/CCBY音频提取仍待办，不能算就绪。
- MMAU/STAR/STA/FunQA已哈希。Perception11,199,611,859B下载exit0；整包安全成员/CRC、4任务各200引用、15代表视频全流解码通过；video代理继续同锁FunQA/STA集成验收。最终证据待整合，不拿抽样解码冒称全视频解码。
- Valen来源20文件约6.36GB全部哈希通过，隔离原料不准入训练；reports/valen-source-download-001.json。
- 文本来源17文件30,666,690B本地上游/远端SHA通过；JevEmbed600k/Open-Jev113k仍是候选原料，不是样本族隔离/训练准入证据。
- ScienceQA3parquet626MB、ChartQA875MB、AOKVQA已知可商用COCO3832图约660MB完成；reports/stage2-remote-completion-001.json。剩余已知许可研究专用图继续准备，未知许可不进池。
- Decision Index retry002终态82/83完成，5,978文件5,384,593,281B；唯一RAGTruth旧503失败通过独立固定git fetch/checkout恢复，9文件36,621,334B SHA，HEAD精确匹配c103204b9ce28d6bbad859304bf30de72b8ed8fe，reports/ragtruth-source-recovery-001.json；旧失败保留，联合证据不冒称完整私有榜可重建。RouterBench pickle仍未加载，安全转换待办。HLE274,276,147B官方SHA通过，仅评测。
- JevBench正确项目fstandhartinger/jevbench固定7625ec...，231公开决策/8文件清单及远端SHA通过；literal group148不等于教师195，精确聚合协议仍缺，不下载隐藏题/预测。

## 已完成的CPU证据
- reports/cpu-media-probe-offline-runtime.json：真实TF5.19图像/截图/16kHz WAV/3帧视频及答案边界通过，无模型forward。
- Valen010源PT仅在chroot/dropuid/seccomp沙箱加载，官方safetensors写出428张量289,932,536B，SHA27b8f6cd4874a356af8a5d6a3676d2a9a03e4b3301d066d13645f673f72f4468；张量逐项相等。父进程诊断FIFO/symlink/大小边界独立复核关闭；并非完整教师验收。
- Kokoro canonical量化ONNX87.09MB/双voice已下载，专用Python3.11.17+Misaki G2P实际2文本×2voice→4份有限24kHz WAV通过；reports/speech-kokoro-text-g2p-cpu-canary.json。旧q8f16失败保留，不充当通过证据。
- reports/public-eval-media-validation-001.json：Mac全1000音频完整解码/有限值/帧数通过；Linux32代表样本FFmpeg完整解码通过；STAR全425member安全/CRC+32视频全流解码通过。
- STAR ActionSequence仅188/200，缺12个引用；video代理已从官方Charades固定S3版本补齐12段，35Range请求36,286,885B、完整解码通过；3条官方end越界，裁剪/钳位协议待冻结，不暗排样本。禁止拿缺失猜教师2786题排除。MMAU69份六声道、66份超30秒，裁剪/单声道协议待冻结，不暗排样本。
- 远端曾exit137但oom/oom_kill=0，原因未证实；公开媒体获批复制1.94GB到Mac验证，无卡准备不以开GPU解决。

## 并行所有权
- offline_runtime_dependencies：/tmp/lite-omnijev-offline-runtime；d695652/fdafbc2已整合：Ollama官方SHA/解包/版本通过；llama.cpp原树锁内编译rc0、4二进制hash通过；quant完整85wheel/2.7GiB及no-index独立安装/导入通过，无serve-next路径；GCC13缓存44deb/111MiB已到位，未改宿主机。LiteRT v0.9.0官方Linux二进制28,710,976B已哈希，经私有Noble loader依赖解析/--help/CPU缺模型边界验证通过；未改宿主机包。Android NDK协议/移动构建仍未就绪，不因此开GPU。
- whole_project_datasets：/tmp/lite-omnijev-project-datasets；fac39dd/e1e1a1f已整合：阶段2剩余19860张COCO已知许可图3,344,218,752B完成，0 missing/bad-size，受限池隔离，总阶段2 5.506GB；GUI官方all_annot+4split共3.96MB哈希通过，1169个app_split:test仅评测候选，截图92.64GB multipart超20GB内部切片配额，媒体仍缺，需选择性恢复或明确磁盘计划；DocVQA官方RRC登录/Download条款仍缺。
- video_eval_sources：/tmp/lite-omnijev-video-eval；Perception004外部有界续传；Charades缺12视频选择性恢复已验收，bae5338合入58588c6，原3条窗口越界保留；CPU恢复切片20定向测试/Ruff/mypy/build通过。
- decision_index_assets：/tmp/lite-omnijev-decision-index；e99910a已合入8d8f164、47914bd合入d2e7512；retry队列17457补非门禁网络失败来源，≤10GB，不读pickle、不绕过HLE。
- mdc_transport_fix：MDC闭环已整合并独立复核，过期worktree已清理；同一executor的独立非阻断offline-prep:05已交付合入746c534，主树273测试全部通过，不升级远端GPU环境。
- teacher_head_conversion：阶段3审计7267a5e已合入e549ca1，35官方小原料36.2MB远端校验；CLEVRER官方CC0确认，继续同worktree挂官方25.49GB来源队列（root<=27GB），不接受条款/读取隐藏答案。
- speech_assets已交付；所有切片主会话统一整合/关票。root独占主树、票据、STATUS；子代理不得写这些文件。

## 待我决策
- 用户新授权通过本人浏览器代办申请：root已实际打开AudioCaps三页申请表、RRC注册页和Android官方页；AudioCaps缺真实姓名/机构/验证邮箱，RRC还需地址国家/新密码/邮件验证。已集中请求缺项；新密码仅用户在浏览器设置。未提交申请、未接受新协议。授权清单 docs/runbooks/required-authorizations.md：当前3项（DocVQA/AudioCaps/Android），MVBench5类仅选中任务才触发；商业权利另核，不能把研究许可当商用许可。
- 当前不需要开卡。用户已明确确认CommonVoice官方MDC与HLE均已授权/登录；HLE真实浏览器已显示“You have been granted access”。登录授权与远端下载接入分开核验，不重新要求接受已完成条款，不用镜像绕过。
- CommonVoice所选544,267,165B官方English CC0资产：cmu5nqn1h00vwmi07b4dbk085；用户提供的MDC key已仅存远端 /root/.config/lite-omnijev/mdc.key 0600，不回显/不进Git/日志。下载器DNS pin已合入faaf0f8并独立复核通过；资产直连禁代理，实际MDC标准curl POST已返回200并完成544,267,165B下载、官方SHA通过；urllib403原因未定，不能误判为条款未授权。用户指定本地同SHA副本已按明确指令删除，reports/commonvoice-user-file-move-001.json；其他本地文件未删。
- 门槛改动、付费/开关卡、生产、删数据、方向/预算改变仍用户决定；真实隔离身份/人工资源后续需安排。其余下载继续。

## 下一步与验证
1. 完成在途数据、失败项续传和完整CPU运行时依赖；逐文件哈希和退出报告入账。最近汇总 reports/offline-prep-progress-20261014.json（03:54本地），ready_to_open_gpu=false，不能替代各最终报告。
2. MDC/HLE授权下载均已真实完成并哈希核验；reports/mdc-commonvoice-download-002.json、reports/hle-source-download-001.json。凭据只放私密文件，API请求用官方curl流程；不公开签名URL/密钥、不准入训练。
3. CLEVRER/GUI原未知状态已由官方证据解除，执行新批准的限定镜像；LLaVA全量1.28TB仍禁止，5k所选原媒体权利需逐项确认；AudioCaps原音频需官方表单。
4. 完成全项目offline-prep:04审计，未来LoRA/量化/校准产物仅列待生成；全齐备前不开GPU。
- 最新主树pytest9.1.1下282测试全部通过（27.49秒，专用媒体runtime）；此前273测试无skip、Ruff/format167文件、mypy44源文件、wheel/sdist通过（显式专用媒体runtime）；MDC独立安全32测试通过/1 torch测试按范围排除；新增STAR恢复20定向测试、169文件格式、mypy45/build通过。
- 不push、不建远端issue/PR、不轮询gh；只本地提交/SSH同步，保留缓存与约561MiB P0备份。
- 循环见plan/LOOP.md；native goal当前工具返回blocked（旧状态），用户已重新要求继续无卡准备，仍按原终局推进；工具不提供agent恢复接口，不能虚称active/complete。
