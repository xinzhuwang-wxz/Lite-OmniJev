# STATUS

更新时间：2026-10-14。当前S0；P0工程冒烟完成，S0–S6/FINAL未完成。

## 本轮目标与已确认决策
- 用户已批准ADR-0006：只替换S0 MVBench发布53.10±2复现项为固定9任务/1800行真实独立baseline；MMAU±2、S3≥90%、S4≥97%及其他门槛不变。
- 先完成首发阶段0–6必需外部模型、数据、依赖下载/哈希/无卡验收，再通知开卡；当前禁止GPU作业。
- 用户要求当前会话直接推进，不要定时任务；heartbeat `lite-omnijev` 已删除，禁止重建。
- 用户已具体接受DocVQA-SP研究下载条款、NDK r28b引用2019 SDK协议及适用2026修订；不重复索要相同确认。
- ADR-0004：AudioCaps首发省略，用Clotho/TTS补足原音频题量；不再等待申请。
- ADR-0005：12k视频QA用CC0 CLEVRER，原LLaVA 5k由允许教师在CC0/BY媒体上生成；DocVQA原图只隔离研究，商用文档题用自有/许可明确来源；题量、family隔离和质量门槛不减。
- NextQA/VGGSound/受限LLaVA原媒体为未来候选，不作为首发外部下载前置；1.28TB全量LLaVA禁止。

## 资源与执行边界
- SSH `root@region-42.seetacloud.com:48926`，mux `/tmp/omnijev-ssh-%C`；远端 `/root/autodl-tmp/lite-omnijev`。
- 800GiB盘本轮快照剩余482,313,883,648B；保留150GB checkpoint/adapter及18.16GB模型临时预算，不新增付费。
- 无卡cgroup仅2GiB RAM/0.5核；重CPU用 `/tmp/omnijev-cpu-heavy.lock` 串行，网络锁外。不能按宿主机free/cpu调度。
- 静默source `/etc/network_turbo`、HF_ENDPOINT=https://huggingface.co；密钥、私密URL不输出、不进Git/argv/日志。
- Mac本轮约13GiB空闲/24GiB RAM，不承接大原始包；仅复制约32MB processor metadata和约5MB选定公开媒体做CPU编译。只清理自己创建且无用资源，不删用户其它文件。
- 不push、不建远端issue/PR、不轮询gh；只本地Lore提交/SSH同步。循环见 `plan/LOOP.md`。

## 票据与产品质量门槛
- bootstrap:01–08、p0:01–07 done；reports/p0-engineering-acceptance.md仅工程冒烟。
- offline-prep:01 done：7模型70必需文件121,094,724,397B固定revision/逐文件哈希相符；reports/project-model-offline-acceptance-001.json。
- offline-prep:02 claimed（root-data-integration）：GUI 1679完整train episodes/25809 PNG下载、CRC/size/逐文件hash终态通过；DocVQA完整归档及32代表图CPU验收通过；GUI全量PNG解码作业在途，不能以partial关闭。
- offline-prep:03 claimed：root/工具依赖闭包、NDK/Chromium在途；已有imports不代替离线fresh安装。
- offline-prep:04 pending：依赖01/02/03及新增s0:09；全部必需外部准备和可做CPU主链代码/测试通过前，ready_to_open_gpu=false。GPU-only不能用来掩盖尚未写的适配代码。
- offline-prep:05 done：锁pytest9.1.1修复；offline-prep:06 done：教师音频实际Linux CPU适配通过。
- s0:01 claimed（teacher-cpu-recovery）：安全权重转换通过，正补项目持有的本地输入/processor适配与无卡预检；教师forward/严格复现未通过。
- s0:03 claimed：HF/vLLM maxTV .01129/.02292，严格<.001仍failed；D004 FP32补丁未应用。只读审计确认现parity/网关仍纯文本，新增s0:09为共享多模态输入CPU前置；官方0.26/0.31 chat指定token原生协议已核验并补ADR-0002，适配代码/HTTP负测仍待写。
- s0:09 claimed（multimodal-cpu-prep）：独立managed worktree `/Users/bamboo/.codex/worktrees/s0-multimodal-cpu/Lite-OmniJev`，拥有train/multimodal.py、scripts/s0_prepare.py及对应测试/runbook/公开fixture。真实100题、20×5类别、20video中10×8/10×16帧，4 source families变体不是100独立质量样本，family排除清单待整合。
- s0:05/06 blocked：真实独立封存身份/双人标注资源未安排；不能用agent自标或合成κ代替。
- S0全绿前禁止S1；delivery:01未完成。本轮get_goal已确认同目标status=active，继续沿用；此前blocked快照已过时，终局仍未完成。

## 已完成外部准备与证据
- AndroidControl22对象49,930,349,992B，Mind2Web49文件13,577,379,722B：官方/固定清单哈希通过，均仅研究隔离；source-download-002报告。
- CommonVoice544,267,165B：官方MDC下载/SHA、8709成员安全检查、32代表音频完整EOF解码通过；reports/commonvoice-cpu-prep.json。8703物理音频不等于5898独立clip，准入仍需族去重。用户指定本地同SHA副本已按指令删除。
- HLE274,276,147B：授权完成、官方LFS SHA通过，仅评测；reports/hle-source-download-001.json，不读隐藏题/答案。
- Clotho3,140,677,494B：官方MD5/SHA、1991成员安全检查、1659 CC0/BY音频提取/hash、32代表完整EOF解码通过；reports/clotho-aqa-audio-preparation-002.json。2个名称差异仅固定alias，旧001失败保留。
- CLEVRER10官方对象25,489,948,056B：HTTPS/尺寸/SHA快照/全部ZIP CRC，9代表三split完整视频EOF解码通过；reports/clevrer-media-cpu.json。上游无SHA不称官方哈希匹配，不称全20k视频decode。
- MMAU：全部1000音频Mac完整decode及Linux32代表EOF；STAR：全部CRC/32代表decode，官方补齐12缺媒体后600/600公开引用；reports/star-missing-source-recovery-001.json。
- Perception/FunQA/STA：三包1200/1200公开引用、全ZIP安全CRC、27代表完整decode通过；reports/public-eval-archives-validation-002.json。
- ScreenSpot原作者6文件597,662,567B、610图全decode；reports/screenspot-source-audit.json，仅隔离评测，不证明第三方截图商业再分发权。
- 阶段2：ScienceQA3parquet626MB、ChartQA875MB、COCO3832许可明确图660MB、19860已知受限研究图3.344GB全部准备，未知许可不进池；CPU reader已通过：ChartQA全CRC、ChartQA/两COCO池共96代表图完整decode、ScienceQA3split只读footer；reports/stage2-media-cpu-validation-001.json。
- DecisionIndex公开0.3：5978文件5.384GB；旧RAGTruth唯一失败通过固定gitHEAD恢复9文件36.6MB，联合证据不冒称完整私有榜。RouterBench两固定pickle已chroot/dropuid/seccomp转换36497/36511行JSONL，源值摘要与输出摘要相等；真实恶意pickle Linux/root10测试通过、独立安全review已关闭，be41a35；reports/routerbench-cpu.json及routerbench-security-review.json。
- 文本17文件30.67MB已hash；JevEmbed/OpenJev仅候选原料，非训练准入。JevBench231公开决策/8文件已hash；148literal groups与教师195精确协议仍缺。
- Valen20来源6.36GB；chroot/dropuid/seccomp转换428张量289,932,536B并逐张量相等、独立安全复核通过；reports/valen-source-download-001.json及teacher-head报告。
- Jev-Omni固定head安全转换4张量3,964,424B、相等及独立安全复核通过；reports/teacher-head-conversion.json，非完整teacherforward。
- Kokoro量化ONNX/双voice/G2P：CPU实际2文本×2voice生成4份有限24kHz WAV；reports/speech-kokoro-text-g2p-cpu-canary.json。
- Ollama0.35官方SHA/版本，llama.cpp0.6实际CPU build/4二进制hash，quant85wheels独立no-index安装/import，LiteRT0.9私有loader实际--help/CPU边界通过；未改host系统包。详见reports/runtime-cpu-builds.json。
- 教师音频新adapter实际Linux6声道48k→30秒mono16k/480k帧通过，8测试无skip，输入未改/临时清理；reports/teacher-audio-cpu.json。官方loader保持原样，forward集成仍待阶段验收。

## 并行所有权与当前关键路径
- root独占主树/票据/STATUS/整合；子代理各worktree，只交限定切片，不覆盖他人。
- GUI：2082154已合入119c9d2，旧下载PID35376结束；最终25809文件/19,499,790,993B/1679train episodes、无无效复用。10,565,313B逐文件manifest在远端持久盘，SHA c025093e6fec56716f7efe8c26b42782fdf8f55e59141f8e0f2df9780804a66f；摘要reports/gui-odyssey-download-acceptance-001.json。root新增scripts.validate_gui_media，只读全量PNG decode监督作业PID42729/独立snapshot artifacts/gui-full-decode-cpu-001/，报告未落盘前pending。Linux公共CLI6 passed/0 skipped；本机无Pillow的两项skip不作decode证据。
- offline_runtime_dependencies：`/tmp/lite-omnijev-offline-runtime`；d040acd已合入a61098e：工具独立no-index CPU canary、前端322包offline安装、NDKr28b官方SHA/交叉编译及LiteRT Android arm64 CLI hash/ELF已通过。root完整锁闭包/servev031/Chromium/前端实际build继续；root selected-noheavy+借serve-next torch的canary仅partial，pytest需按主树9.1.1锁补齐，不能当完整fresh通过。
- DocVQA：两源文件8,581,945,667B已下载/hash；完整tar/gzip读至EOF，12767安全成员、train/val引用缺失0、32代表图完整decode，exit0。test载荷未读，仍研究隔离、不训练准入；reports/docvqa-source-download-001.json。c232402已合入964014a补metadata边界报告。
- mdc_transport_fix：`/tmp/lite-omnijev-routerbench-cpu`；RouterBench真实隔离转换、Linux恶意fixture及独立安全review全部完成，5347b2b合入be41a35。
- video_eval_sources：`/tmp/lite-omnijev-video-eval`；公开teacher verification仅提取协议/汇总metadata，不记录题/预测。精确14任务/2786IDs未公开，用户已批准ADR-0006改为固定9任务1800行真实baseline；此代理继续全行CPU协议冻结/16帧RGB hash验收，GPU两遍未做。
- teacher-cpu-recovery：`/tmp/lite-omnijev-s0-teacher`；93667e9暂未合入，专属teacher_inference/preflight及测试/runbook/报告。root Python实际30 passed，无torch/numpy skip。safehead强制SHA锚/spec/finite，音频PCM→numpy；Linux137 killer未证实保留失败。独立review其余安全边界无问题，官方17帧round+set会少采样，保持上游公式并补明确拒绝诊断/边界测试。正用共享Mac小型metadata复测text/image/audio，并补可在CPU写/测的安全forward入口；真实24GB GPU forward未运行。
- Mac共享媒体target：上述managed worktree `artifacts/mac-media-libs`，multimodal-cpu-prep唯一写入后只读共享，固定Pillow11.3.0/torchvision0.29.0 wheels/hash/license；复用root torch2.14.1/transformers5.19/numpy2.4.6，没有改root包或复制主干。s0:09 Linux两次loading processor即137/0 cases，不伪称Linuxpass；Mac先前100×2通过后发现fps元数据缺口，已修真实fps/indices并重冻结8/16帧两遍，最终证据待收口。

## 待我决策与后续资源
- 本轮已具体确认的DocVQA/Android协议与首发来源无需再确认；AudioCaps无需办理。CommonVoice/HLE/模型访问均已完成，不再索要key/token。
- ADR-0006已批准：无需为现有九任务另办Moments/SSv2/TVQA/Matterport/NTU授权，只有明确扩范围才触发；禁止把候选升级为当前必办。
- 真实独立gate身份、双人标注、阶段6真机和后续生产观测仍需安排；reports/existing-human-annotation-evidence.json未找到可直接满足四模态配对Cohenκ的现成来源。现行契约两真人6000题/12000判断，不能把众包计数或模型自标冒充；不阻塞其他公开下载。
- 门槛/方向/大预算改变、付费/开关卡/生产/删数据仍用户决定。当前不要开卡。

## 下一步与验证
1. 收口GUI全量decode、wheelhouse完整fresh离线安装、浏览器/前端build和教师输入/视频全行CPU协议；NDK已有通过证据。用户最新确认已租实例，保持无卡，把可做下载/管线测试尽量做完再通知开卡。
2. root阶段2CPU验证已用已有media-cpu Pillow环境完成exit0；未改共享环境，非全量图decode/标签准入。
3. RouterBench安全review、剩余外部协议/依赖闭包审计后再执行offline-prep:04；未来LoRA/量化/校准模型是待生成产物，不是下载缺口。
4. 本轮完整317 passed/3 skip（Linux root隔离集成1及本机root缺Pillow2；分别有独立Linux集成与GUI CLI6/0skip证据），Ruff/format206文件/mypy50/build/32票validate通过；reports/offline-integration-validation-20261014-003.json。新增代码后再跑相关检查；旧报告不覆盖。
5. 只有全部必需外部准备/可做CPU管线验收真正通过才通知开GPU；严格质量门槛及S0人工/身份阻塞另列。reports/no-gpu-readiness.json仅P0历史范围，不可作为本轮开卡凭据；runbook已纠正。无GPU作业、无开关卡/付费/远端写入。
