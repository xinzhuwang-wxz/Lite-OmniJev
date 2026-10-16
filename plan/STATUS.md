# STATUS

更新：2026-10-16。**S3已通过；S4数据/gate工程已通过，S4模型质量未通过。原生goal仍active，产品未交付。**

## 当前票与下一步
- **本轮最新覆盖**：用户明确主线与探索继续相辅相成，未要求暂停探索；HF开源是最终去向，模型出来前暂不做HF，已有CPU模板留存不继续扩展/上传。原“默认不公开发布”仍适用于无关部署，最终HF已有用户意向、需真实门槛和账号目标闭合。
- native事件条件课程b44f94d CPU27.75秒/isolated，60新family/官方原题gold30yes30no+60replay；排除540families，camera-transition proxy仅抽样metadata，不当official executor/新标签/可见性证明。root launch14197已completed10/175.758秒，init f2db，adapterf5feb5ab5b1f56092b46b1d0f3891bfb36b3fec82d28f96c2db85e08e615d0b8。DEV50 79063已33/50持平、Brier.437525vs父.447486改善/ECE.142716vs.104194更差、coverage.18vs.12；原输入hash50相同。video150 session9982已terminal0：113vs父116（temporal73vs74/motion12vs13/existence28vs29），停止该配方/不跑500。CPU配对091bf7a：关系22正确13持平，absence8从5→4且yesFP3→4，present14改善，motion28概率更差；保留f2db。当前GPU空闲；continuous_main_pool在中断后已followup恢复016全history核对，gradient准备同60原视频video-only10step/120draw两遍的有界监督强度对照，等016实际pin再CPU审计，不用stale015。rawrun+named报告reports/s4-native-event-condition-*，source旧core3fe0ab2/archive9132c2d…。
- 连续池step10完整video150已115/150vs父116，120.749秒，没有采用收益；不跑step20全150/全500。两个连续段训练/DEV都终态，停止第3段覆盖；状态保留015精确240draw。发布CPUregistry2a4539d已完成：18配置/15可路由，默认unfitted，可信local bootstrap注入/liveprocess/files/template/token绑定，截图可信路由/真实S5fit与接线pending。root29新tests与57affected tests过，38既有licensedmedia fixture skip不算真实验收，ruff/mypy通过。HF模板8e98f68 CPU仅准备，当前按用户要求搁置。
- 当前票s4:01 claimed/main-session，规格.scratch/s4/spec.md，策略iteration-policy.md；S0数据gate与S1–S3通过，S4质量不通过，S5/S6/FINAL pending。
- 当前工作parent f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3：artifacts/s4-temporal-coverage-train-001/adapter。video150116/150、原cross10074/100、old50 audio38/image40/screenshot50/text49；fullvideo500354/500=70.8%/Brier.381444/ECE.061116/coverage.164、wall365.744秒。fullimage500410/500=82%（S3同分）/Brier.321490/ECE.150568/coverage.68、207.675秒。视频门91.374%、图像门84.778%未达；未晋级。
- 原CE0回退4b8e422f…video150109/cross73/old39,40,50,49，S3回退1bebae31…；所有候选/失败/副本保留。阶段接续完整handoff仍plan/handoff-2026-10-15-s3-complete.md。
- 最新已完整曝光015：artifacts/s4-pilot-exposure-015，reportdebb448fab4cfe4affe3e578c53f6bc98f79ac2f89f32a44bdfe4c8ed50b9e5e，29.510秒/isolated/69known/unknown0。连续池同一1200rows仅实际240draw，step0→10→20/fullAdamW+RNG恢复；rawrun可变，两个runtime snapshots永久SHA绑定。014直接bindings全部保留；failed001收据字段问题经CPUrecovered002闭合（83240d5d…），不重跑训练。native新增更新后必须用016，正式gate前刷新gold准入绑定。
- 用户09:18给剩24h：保守至10-17 09:15，至少03:15起留6h导出/验收/修复；非平台核验，不授权续租。.scratch/s4/wallclock-budget.md。单次通常10steps/240秒，dev50/150≤180秒，每25题观察；有效才扩大、不盲跑100step/全500。

## 已停止或保留的路线

|路线|实测结论与证据入口|
|---|---|
|新family coverage|f2db改善CE0；第二段250ef full500356但概率/跨模态更差，第三段5cff DEV37但video150108vs116，停止扩段。reports/s4-temporal-coverage-*、reports/explore/s4-temporal-coverage-full500-diagnostic-001|
|连续固定池|step10 dfcf DEV34/Brier.424180、video150115；step20 bf59 DEV33/Brier.476134；停止第3段。新core235ad68/archivee036f037…；reports/s4-continuous-main-pool-*|
|pair/KL/Brier|pair与CE预测相同且显存更高；referenceKL无稳定保留收益；Brier仅微改善matchedCE、仍差于CE0。reports/explore/s4-{claim-pair-pilot,reference-retention,temporal-brier-pilot}-001|
|采样/课程/极性|motion16与uniform16同30/50、概率更差；旧时序课程video150110vs109但概率更差；极性32vscontrol36、停止。reports/explore/s4-{temporal-motion-diagnostic,temporal-curriculum-pilot,video-polarity-results}-001|
|collision顺序程序gold|DEV33持平、概率更差；TRAIN12 CE下降27.9%但泛化无收益。原生DEV任务覆盖不同；reports/explore/s4-temporal-order-contrast-results-001、s4-order-train12-001|
|RL/奖励目标|train/decision_reward.py仅CPU原型；onehot expectedreward等价qgold缩放CE、无新监督，不冒称RLCD/创新。reports/explore/decision-training-innovation-next-001；未来有新真值/目标证据可再试|
|原生MoE路由/融合|BF16原生路由概率差超.001、argmax同不能代过，停止当前配置；reports/s4-native-routing-comparison-100-001。全部原模型/cache保留|

创新仍按ADR0015/0017自主进行，主线与探索互相促进，不以成熟度排除、不为凑清单重复失败方案；采用/SOTA必须实测证明。新诊断：DEV22原生in/out关系、15event-motion、13start/end，残余错聚集关系与事件不存在；不支持窄时间间隔主导，像素遮挡未知。reports/explore/temporal-observability-001。

## 外部评测与后续准备
- JevBench CPU入口ba5ca19准备通过；最终候选231题GPU待冻模，不与教师195直接混比。remote s4-public-code-ba5ca19。
- DecisionIndex1530文件/5.90GB、cl100k离线cache、8Gitroots/923材料真实provenance bundle/派生source已完成；不可让materialized目录继承父repo revision，不执行rawpickle。reports/decision-index-provenance-install-001、decision-index-tiktoken-offline-001。
- 最新独立manifest005：artifacts/s4-index-evaluation-admission-005/manifest.json SHA7b527a6774b512322bbc8bbdb6e4b375271d1bb4fc05390069dcf2fe3256f4d2，27/44eval-only、17未准入、readyfalse，未完整rebuild。12实际输入+6publisher本地文件+固定metadata projection核验，原21/files保留。禁止TRAIN/再分发，GPQA canary/在线例题边界保留、WinoGrande CC-BY版本不明不标4.0。reports/decision-index-evaluation-admission-005.json；历史001–004不改。
- S5温度CPU核心129ff82、gatewayregistry2a4539d；18组配置，当前15可路由/截图可信入口pending；默认unfitted、local bootstrap/liveprocess/files/template/token绑定，真实private导出/fit/激活未做。runbooks/calibration.md、gateway-calibration.md。root57tests过/38existingmediafixtures skip，ruff/mypy过；CPU不代S5。
- 三档发布目录CPU模板f0eb151检查inventory_incomplete/0invalid、37成员归档读回exit0，不是发布包。原文不强制唯一Android/APK/双移动OS，但实际目标设备离线模态/内存/时延必须验。reports/delivery/edge-scope-and-bundle-001。
- HF开源模板8e98f68只CPU准备，按用户最新要求现在搁置；模型完成再做HF。不创建仓库/上传/访问token。OpenAI Decisions继续暂缓。

## 已验收数据与入口
- .scratch/s4/selected-gold-manifest-001.json SHA e073a601…：六cohort各3000cal/1000hold，原dev各500；image/audio/crossmodal额外dev各500单列。旧AOK623/211、video900/300保留为来源诊断及历史曝光，不重复计入主配额。
- reports/s4-gold-data-admission-001.json evidenceSHA9cf11cea…：来源代码128绑定、source链154工件、CLINC原始Parquet全链重放、Clotho保留120完整字段重放、13500条整理记录仅URI变化、19973媒体路径SHA验证均通过。
- 历史34报告基线保留；最新015覆盖69known、unknown0；native新增后待016。所有曝光只认实际报告，不把准备池容量算已训练。
- gold准入绑定的远端总清单（新增TRAIN组合之后仍保留其原始绑定）：artifacts/s4-gold-inventory-001/crossmodal-dev-002/inventory-manifest.json SHA6eb3224906c96ad04042b019f41acfcc533789b8e6b5105c1a1c490a20b53769。144938当前/历史观测零跨池；16963图像引用全定位、12463唯一图像RGB实解码通过，缓存observed-image-rgb-cache-002.json SHA45ab0b7a…。
- GUI3000/1000保留原178；audio为Clotho1400/500+speech1600/500，原Clotho120保留；video为CLEVRER validation3000/1000；image为VQA原题3000/1000；crossmodal同池组件AND3000/1000。crossmodal校准/封存依赖组610/206，不声称IID或有效N。
- speech-gold-001/allocation.jsonl原始speaker/book/text/media别名、quarantined-reserves/exclusions.json SHA8901483c…继续保留。新增训练组合必须只用train并维护组件账本，不能挪预留数据。
- 来源/池/规范化精确规则见.scratch/s4/{image-gold-policy,audio-gold-policy,crossmodal-gold-policy}.md。真实构建失败及修正报告均保留。

## 私有gate工程与复现
- scripts/private_gold_job.py：固定inventory-audit，独立普通数字UID、永久lease、输入只读、私有日志、结束撤权。真实Linux live/reused UID/保留目录拒绝及13项错误/日志/timeout探针通过；job003绑定最新6eb32249清单。仅ordinary-identity/root-visible边界，不防root、不强制网络/namespace。
- work-root固定 /var/tmp/omnijev-gold-jobs-001；62041/42、62051/52、62061、621xx、622xx、62301/2视为已用，后继从新的空闲UID开始并让launcher检查，不能复用lease。原job001权限claim已由s4-private-job-review-correction-001纠正。
- scripts/private_stage_eval.py最终代码02c30c4；受信任root-visible固定subprocess，模型/tokenizer/adapter/数据/code/配置全绑定，逐题输出和日志0600。TERM/HUP回收worker并留失败receipt；stage核验真实选集与固定11份证据，不能自声明admitted。file URI在显式根内校验后仅内存规范化，共享resolver不放宽。
- 模型入口56测试在本机/租机通过，独立review两项阻断已修复。真实H20 engineering dev50双遍逐题输入/答案映射/概率完全一致，50记录0差异；总job wall227.643秒，含校验/加载，不是服务延迟。reports/s4-private-stage-{engineering-001,engineering-002,repeatability-001,validation-001}.json。
- 入口当前固定E4B原始决策评测；S5校准、S6量化/E2B及三运行时在对应里程碑扩展并正式验收。工程quality_status始终not_evaluated；不能填作S4质量门。
- 真实spec：artifacts/s4-gold-inventory-001/private-stage-engineering-input-001.json SHAb0cecb6a…；源码快照/root/autodl-tmp/s4-stage-eval-code-02c30c4，archive SHA72bdf4e8…本地/远端一致。runbook docs/runbooks/sealed-gate.md已更新，旧双标canary只保留兼容说明。

## 当前模型与已知质量
- 冻结E4B revision ee0ef6023621cff504d758262d4e04895a5af4a2；S3 adapter SHA1bebae31309f799395d01499fb9deec5e44622599d0c082473ea8b5a014e5755。
- S3五模态dev500双遍：image82.0%、screenshot96.4%、text99.4%、audio80.0%、video16帧63.6%。原S3四门通过；16帧优于8帧60.8%。图像1120为质量配置，不是最终低成本服务默认。
- 原crossmodal固定50：候选66%/基座48%；初版34%是JSON yes/no排序恢复错误，权威recovered002/paired002与correction001，旧失败不覆盖。
- 补充来源固定50配对GPU：候选80%/ECE.1673、基座66%/ECE.2546；逐题prepared输入相同，candidate-only14/base-only7，合计86.6秒。s4-crossmodal-matched-paired-001。工程入口另一个随机seed42切片82%/ECE.0854仅工程证据，不与前切片直接比较。
- S4原视频最强已测教师OneJev27B94.2%，同原dev500 SHA79ed30…与baseline固定16帧；97%比率目标91.374%。S3的Jev-Omni69.8%对照不能替代它。S4仍需最强允许教师同来源/同输入比率≥.97、封存/开发差≤.03、Decision Index/JevBench不低于同尺寸基线。小切片与数据gate不替代完整门槛。
- train.trainer/s1_dev_eval已支持image+audio并传官方mm_token_type_ids；63回归与真实CPU processor10例逐张量一致。已执行首轮联合模态训练，见本STATUS首段。
- S3 HF/vLLM FP32合并副本数值对照已通过，不是低成本服务/校准/最终浏览器验收；原模型/adapter保持，派生缓存不擅删。

## 环境与授权
- 远端/root/autodl-tmp/s1-text-lora-001，主checkout仍869ee95；后续实跑使用具名不可变源码快照，勿盲覆盖远端主树。SSH mux /tmp/omnijev-ssh-root-final，端口48926；训练Python /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python；H20约95GiB。
- 用户指定：常规实现/测试Sol6.1，简单执行/监控Luna6，主会话负责分析/统筹；原生子代理独立worktree。所有凭据在项目外，不写入Git/日志。
- 用户授权有界创新/RL与实测资源调整；先小批量、有效再扩大，持续监控与早停。冻结基座/无新分类头/不换引擎、质量门不降、5b不启用。大预算/付费/租卡开关/生产部署仍须用户决定；不push。
- S5、S6和FINAL尚未完成；终局包含真实API/单卡/端侧/浏览器/性能与无未解决问题。快与低成本须同条件实测，未知价格不编造成本，无同条件Jev实测不声称超越。
- 用户2026-10-16明确暂缓OpenAI Decisions对照；保留资料，不请求凭据、不接入、不付费。继续S4主线。GPU真正用完再通知用户关卡，不代执行关机。
