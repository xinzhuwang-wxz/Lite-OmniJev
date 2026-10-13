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
- 800GiB盘最近剩余525,057,953,792B；保留150GB checkpoint/adapter及18.16GB模型临时预算，不新增付费。
- 无卡cgroup仅2GiB RAM/0.5核；重CPU用 `/tmp/omnijev-cpu-heavy.lock` 串行，网络锁外。不能按宿主机free/cpu调度。
- 静默source `/etc/network_turbo`、HF_ENDPOINT=https://huggingface.co；密钥、私密URL不输出、不进Git/argv/日志。
- Mac仅约11GiB空闲，不承接大原始包。只清理自己创建且无用资源，不删用户其它文件。
- 不push、不建远端issue/PR、不轮询gh；只本地Lore提交/SSH同步。循环见 `plan/LOOP.md`。

## 票据与产品质量门槛
- bootstrap:01–08、p0:01–07 done；reports/p0-engineering-acceptance.md仅工程冒烟。
- offline-prep:01 done：7模型70必需文件121,094,724,397B固定revision/逐文件哈希相符；reports/project-model-offline-acceptance-001.json。
- offline-prep:02 claimed：GUI/DocVQA和安全原料转换在途；不能以partial代替最终报告。
- offline-prep:03 claimed：root/工具依赖闭包、NDK/Chromium在途；已有imports不代替离线fresh安装。
- offline-prep:04 pending：01/02/03及全部必需外部准备通过前，ready_to_open_gpu=false。
- offline-prep:05 done：锁pytest9.1.1修复；offline-prep:06 done：教师音频实际Linux CPU适配通过。
- s0:01 claimed：安全权重转换通过，教师forward/严格复现未通过。
- s0:03 claimed：HF/vLLM maxTV .01129/.02292，严格<.001仍failed；用户质疑门槛但未批准变更，D004 FP32补丁未应用。
- s0:05/06 blocked：真实独立封存身份/双人标注资源未安排；不能用agent自标或合成κ代替。
- S0全绿前禁止S1；delivery:01未完成。native goal旧状态blocked，工具无agent恢复接口，不虚称active/complete。

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
- DecisionIndex公开0.3：5978文件5.384GB；旧RAGTruth唯一失败通过固定gitHEAD恢复9文件36.6MB，联合证据不冒称完整私有榜。RouterBench未直接反序列化，隔离转换进行中。
- 文本17文件30.67MB已hash；JevEmbed/OpenJev仅候选原料，非训练准入。JevBench231公开决策/8文件已hash；148literal groups与教师195精确协议仍缺。
- Valen20来源6.36GB；chroot/dropuid/seccomp转换428张量289,932,536B并逐张量相等、独立安全复核通过；reports/valen-source-download-001.json及teacher-head报告。
- Jev-Omni固定head安全转换4张量3,964,424B、相等及独立安全复核通过；reports/teacher-head-conversion.json，非完整teacherforward。
- Kokoro量化ONNX/双voice/G2P：CPU实际2文本×2voice生成4份有限24kHz WAV；reports/speech-kokoro-text-g2p-cpu-canary.json。
- Ollama0.35官方SHA/版本，llama.cpp0.6实际CPU build/4二进制hash，quant85wheels独立no-index安装/import，LiteRT0.9私有loader实际--help/CPU边界通过；未改host系统包。详见reports/runtime-cpu-builds.json。
- 教师音频新adapter实际Linux6声道48k→30秒mono16k/480k帧通过，8测试无skip，输入未改/临时清理；reports/teacher-audio-cpu.json。官方loader保持原样，forward集成仍待阶段验收。

## 并行所有权与当前关键路径
- root独占主树/票据/STATUS/整合；子代理各worktree，只交限定切片，不覆盖他人。
- whole_project_datasets：`/tmp/lite-omnijev-project-datasets`；GUI官方train1679完整episodes/25,809 PNG/19,499,790,993B选择性Range恢复；a36c793已合入29cd742，8并发2MiB批比32MiB少过读。PID30008继续，早先root物理观察11,355图/8.536GB未代表终态；恢复复用CRC边界与最终全文件校验待补。
- offline_runtime_dependencies：`/tmp/lite-omnijev-offline-runtime`；d040acd已合入a61098e：工具独立no-index CPU canary、前端322包offline安装、NDKr28b官方SHA/交叉编译及LiteRT Android arm64 CLI hash/ELF已通过。root完整锁闭包/servev031/Chromium/前端实际build继续；root selected-noheavy+借serve-next torch的canary仅partial，pytest需按主树9.1.1锁补齐，不能当完整fresh通过。
- remaining_authorizations：`/tmp/lite-omnijev-authorizations`；DocVQA-SP官方Annotations1,713,866B已hash，Images8,580,231,801B正在下载，总8.582GB<30GB；严格TLS链/hostname验证，不关闭校验，私密URL仅0600/STDIN。
- mdc_transport_fix：`/tmp/lite-omnijev-routerbench-cpu`；RouterBench两固定pickle仅在chroot/dropuid/seccomp限资源沙箱转换，先恶意fixture，再真源，交付后独立安全review。
- video_eval_sources：`/tmp/lite-omnijev-video-eval`；公开teacher verification仅提取协议/汇总metadata，不记录题/预测。精确14任务/2786IDs未公开，用户已批准ADR-0006改为固定9任务1800行真实baseline；此代理继续全行CPU协议冻结/16帧RGB hash验收，GPU两遍未做。
- teacher_head_conversion：来源调整切片已完成整合；无未完成写入。

## 待我决策与后续资源
- 本轮已具体确认的DocVQA/Android协议与首发来源无需再确认；AudioCaps无需办理。CommonVoice/HLE/模型访问均已完成，不再索要key/token。
- ADR-0006已批准：无需为现有九任务另办Moments/SSv2/TVQA/Matterport/NTU授权，只有明确扩范围才触发；禁止把候选升级为当前必办。
- 真实独立gate身份、双人标注、阶段6真机和后续生产观测仍需安排；reports/existing-human-annotation-evidence.json未找到可直接满足四模态配对Cohenκ的现成来源。现行契约两真人6000题/12000判断，不能把众包计数或模型自标冒充；不阻塞其他公开下载。
- 门槛/方向/大预算改变、付费/开关卡/生产/删数据仍用户决定。当前不要开卡。

## 下一步与验证
1. 继续GUI/DocVQA、wheelhouse/NDK/浏览器二进制；下载exit0→固定hash→有界CPU reader或fresh离线安装，不把running当done。
2. root阶段2CPU验证已用已有media-cpu Pillow环境完成exit0；未改共享环境，非全量图decode/标签准入。
3. RouterBench安全review、剩余外部协议/依赖闭包审计后再执行offline-prep:04；未来LoRA/量化/校准模型是待生成产物，不是下载缺口。
4. 当前主树最新301测试全部通过（7.17秒），core47/GUI5定向通过，Clotho定向18通过（2.98秒）；Ruff/format198文件/mypy48源文件通过，最新wheel/sdist通过；reports/offline-integration-validation-20261014-002.json。新一批整合后再验证相关检查。
5. 只有全部必需外部准备/CPU验收真正通过才通知开GPU；严格质量门槛及S0人工/身份阻塞另列，不能暗改或混成全部产品完成。
