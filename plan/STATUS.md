# STATUS

更新时间：2026-10-14。当前 S0；P0 工程冒烟完成，S0–S6/FINAL 未完成。

## 用户指令与资源
- 整个项目必需外部模型、数据、依赖下载并哈希/CPU验收后，才通知开卡；当前无卡，禁止 GPU 作业。
- SSH root@region-42.seetacloud.com:48926；ControlPath=/tmp/omnijev-ssh-%C；远端 /root/autodl-tmp/lite-omnijev。
- 数据盘800GiB（750GiB付费），最近实际剩余637GiB。无卡cgroup只有2GiB RAM/0.5核；证据 reports/no-gpu-resource-limits.json，不能按宿主机free/os.cpu_count调度。
- 网络：静默source /etc/network_turbo，HF_ENDPOINT=https://huggingface.co；凭据不输出、不进Git。
- 重型CPU作业用 /tmp/omnijev-cpu-heavy.lock 串行；网络继续。llama.cpp编译树8791/8962/8963/8965/8968/12081/14088/14089/14090暂停保留，先找官方预编译。

## 票据与门槛
- bootstrap:01–08、p0:01–07 done，reports/p0-engineering-acceptance.md；工程冒烟不代表质量达标。
- offline-prep:01 claimed：7模型70必需文件121,094,724,397B；68文件47,350,012,013B已校验，剩余两个大权重在传；不得关票。
- offline-prep:02 claimed：全阶段来源原料与许可准备；下载和媒体验证继续，账号/许可/精确协议缺口不能标齐备。
- offline-prep:03 claimed：运行时/依赖闭包；已有环境与源码归档不能替代可执行文件、完整依赖和CPU验证。
- offline-prep:04 pending：前三票及全部外部缺口解除前，开卡准备验收不通过。
- s0:01 claimed：Valen安全转换428张量已通过；完整教师推理/复现未验收。
- s0-data:01 done：CLINC train15250条语义保真、官方33token通过；clinc-admission-corrected-001.json。
- s0:03 claimed：真实Gemma4 CPU媒体处理器通过；HF/vLLM概率一致性maxTV .01129/.02292，严格<.001仍failed，原门槛未改。
- s0:05/06 blocked：真实独立封存身份/双人标注资源缺失；工具/合成真值不代替真实身份或κ。
- S0全通过前禁止S1；delivery:01未完成，D004 FP32 LoRA补丁搁置。

## 下载实际状态（以PID/文件/最终退出报告为准）
- 模型 flock1827/1828仍活；project-model-download-001.json running且零失败。另有约22.5GB和18.6GB partial，不计入已校验数。
- AndroidControl官方GCS22对象49.93GB固定generation/MD5/CRC32C；9047/9048仍活，3文件2,345,618,122B校验。仅研究，商业训练/发布不准入。
- Multimodal Mind2Web49文件13.58GB；10668/10669仍活，13文件2,634,066,033B校验。001曾误校验LFS指针，失败报告保留；002已区分原文件SHA与指针GitSHA。
- Clotho3,140,677,494B归档尚未完成；5小文件全校验。002因curl内部重试丢掉本次新增内容而失败，保留123,907,028B partial；003改外部6次有界续传、每次独立curl且MD5通过才提升，PID17404/17408已确认活。完成后锁内提取1659 CC0/CCBY音频，排除332 NC/未知/其他许可。
- MMAU mini1,213,407,227B和STAR727,887,646B哈希通过；STA1,085,038,855B和FunQA1,350,969,357B最新实际sha OK/exit0。
- FunQA→Perception旧队列已退出；Perception5866MiB后curl18/sha失败，video代理已保留6,151,711,684B partial并启动PID17494续传；不能使用旧running快照。
- Valen来源20文件约6.36GB全部哈希通过，隔离原料不准入训练；reports/valen-source-download-001.json。
- 文本来源17文件30,666,690B本地上游/远端SHA通过；JevEmbed600k/Open-Jev113k仍是候选原料，不是样本族隔离/训练准入证据。
- ScienceQA3parquet626MB、ChartQA875MB、AOKVQA已知可商用COCO3832图约660MB完成；reports/stage2-remote-completion-001.json。剩余已知许可研究专用图继续准备，未知许可不进池。
- Decision Index公开0.3已镜像5,861文件2,388,936,304B，21固定Git+11HTTP来源；HF内容和6个失败Git/HTTP来源继续补齐；HLE受控，RouterBench只隔离原pickle未加载；不冒称完整套件可重建。
- JevBench正确项目fstandhartinger/jevbench固定7625ec...，231公开决策/8文件清单及远端SHA通过；literal group148不等于教师195，精确聚合协议仍缺，不下载隐藏题/预测。

## 已完成的CPU证据
- reports/cpu-media-probe-offline-runtime.json：真实TF5.19图像/截图/16kHz WAV/3帧视频及答案边界通过，无模型forward。
- Valen010源PT仅在chroot/dropuid/seccomp沙箱加载，官方safetensors写出428张量289,932,536B，SHA27b8f6cd4874a356af8a5d6a3676d2a9a03e4b3301d066d13645f673f72f4468；张量逐项相等。父进程诊断FIFO/symlink/大小边界独立复核关闭；并非完整教师验收。
- Kokoro canonical量化ONNX87.09MB/双voice已下载，专用Python3.11.17+Misaki G2P实际2文本×2voice→4份有限24kHz WAV通过；reports/speech-kokoro-text-g2p-cpu-canary.json。旧q8f16失败保留，不充当通过证据。
- reports/public-eval-media-validation-001.json：Mac全1000音频完整解码/有限值/帧数通过；Linux32代表样本FFmpeg完整解码通过；STAR全425member安全/CRC+32视频全流解码通过。
- STAR ActionSequence仅188/200，缺12个引用；video代理从官方Charades范围请求补齐，禁止拿缺失猜教师2786题排除。MMAU69份六声道、66份超30秒，裁剪/单声道协议待冻结，不暗排样本。
- 远端曾exit137但oom/oom_kill=0，原因未证实；公开媒体获批复制1.94GB到Mac验证，无卡准备不以开GPU解决。

## 并行所有权
- offline_runtime_dependencies：/tmp/lite-omnijev-offline-runtime；官方llama.cpp预编译、Ollama1.427GB、quant独立完整依赖≤20GB、LiteRT/JDK准备；13直接轮子不算完整闭包。
- whole_project_datasets：/tmp/lite-omnijev-project-datasets；阶段2剩余明确许可原料，≤10GB，复用已有图。
- video_eval_sources：/tmp/lite-omnijev-video-eval；Perception续传、Charades缺12视频选择性恢复≤500MB；4801edd已合入f44588c。
- decision_index_assets：/tmp/lite-omnijev-decision-index；e99910a已合入8d8f164；补非门禁网络失败来源，≤10GB，不读pickle、不绕过HLE。
- mdc_transport_fix：/tmp/lite-omnijev-mdc-transport；专属下载器DNS pin/TLS原host测试，API凭据隔离；root整合后由独立security reviewer复核。
- speech_assets、teacher_head_conversion已交付；所有切片主会话统一整合/关票。root独占主树、票据、STATUS；子代理不得写这些文件。

## 待我决策
- 当前不需要开卡。已向用户一次发出CommonVoice官方MDC与HLE登录/条款授权请求；不替用户接受，不用镜像绕过。
- CommonVoice所选544,267,165B官方English CC0资产：cmu5nqn1h00vwmi07b4dbk085；凭据待合法授权后仅放远端 /root/.config/lite-omnijev/mdc.key 0600，不发聊天/Git/日志。下载器DNS pin已合入faaf0f8并独立复核通过；资产直连禁代理，实际MDC访问仍待授权验证。
- 门槛改动、付费/开关卡、生产、删数据、方向/预算改变仍用户决定；真实隔离身份/人工资源后续需安排。其余下载继续。

## 下一步与验证
1. 完成在途模型/数据、失败项续传和完整CPU运行时依赖；逐文件哈希和退出报告入账。
2. MDC原4项及DNS pin发现全部关闭，reports/mdc-download-security-closure.json；授权后安全接入，不对外声称凭据下载已实跑。
3. 完成全项目offline-prep:04审计，未来LoRA/量化/校准产物仅列待生成；全齐备前不开GPU。
- 最近完整批次271测试无skip、Ruff/format167文件、mypy44源文件、wheel/sdist通过（显式专用媒体runtime）；随后MDC pin变更16定向测试/Ruff/mypy通过，独立安全32测试通过/1 torch测试按范围排除。
- 不push、不建远端issue/PR、不轮询gh；只本地提交/SSH同步，保留缓存与约561MiB P0备份。
- 循环见plan/LOOP.md；native goal当前工具返回blocked（旧状态），用户已重新要求继续无卡准备，仍按原终局推进；工具不提供agent恢复接口，不能虚称active/complete。
