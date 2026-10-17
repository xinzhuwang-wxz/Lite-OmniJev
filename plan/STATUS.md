# STATUS

更新:2026-10-17 02:20。**S3已通过;S4按ADR0023带实测局限收口;S5进行中:fit007(e465版)完成+holdout完成,候选已换D3P-150,FIT009重校准运行中。原生goal仍active,产品未交付。**

## 当前票与下一步
## 当前票与下一步
- **17:10 终局(续跑窗口)**:**交付候选升级为 D3P-600**(adapter 767be09b,教师蒸馏600题,video500 75.6% 全集保持+6):FIT011七温度(fitted,video T=2.23)+封存holdout 6000题(s5-holdout-eval-008,零掉分,ECE text .0014/screenshot .0108/image .0202 过门,audio .0310/video .0321 近门,crossmodal .0528 披露)——封存video 72.6%(+0.6 vs e465,+0.9 vs D3P-150)。发布目录lite-omnijev-release-001已更新(model卡/release notes含D3P-600封存实测)。级联范式:oracle GO(90.7%)+质量门(静态近完美/事件谓词全缺口)→RL形式化器=下一窗口旗舰(奖励穿离散执行器,预注册门齐)。训练资源纪律:混合稀释双证伪(COMBOFULL 118/SAS-full 116),纵深(单family+教师分布)是唯一正向配方;teacher-1500止步113/1500(25h投影,600缓存已足)。eval-005/006失败根因=DEV参照候选身份链缺失,已修(五模态留任补跑+runtime绑定同候选)。

- **20:12实时**：S5首次真实校准fit007已进入inference（text起，18000题，UID65301/2，job SHA af4b7b48…，producer fa83acc，GPU~49%）。此前fit003–006排障链四根因全修并留证：8个老池sealed行文件0644→0600（真实边界泄漏，probe正确拒绝）、/root/.ssh 0755→0700、沙箱代码补serve（producer修订fa83acce6f4b0345a2694ec883a81647a6279f8d，已部署/root/autodl-tmp/s5-calibration-code-fa83acc）、gui-gold-002 119张缺失校准媒体从quarantine源sha256验证恢复（reports/s5-gui-media-restore-001.json，119/119零失败）。UID 648xx/649xx/650xx/651xx(诊断)/652xx/653xx已消耗不复用。
- 外榜收口：候选e465 JevBench231 76.62%/macro74.93%（两遍exit0）；父模型同协议基线76.19%/macro77.39%（reports/s4-parent-jevbench-001.json）。"不低于同尺寸基线"headline门通过（+0.43pp），macro−2.46pp如实记录（≈97%来自ambiguous家族n=7）；候选Brier/ECE/paraphrase全优，父ordinalMAE更优。S4原教师比率门未达，按ADR0023保留。
- 当前票s4:01 claimed/main-session。用户2026-10-16明确接受实测质量局限下推进，ADR0023已记录：保留原gate失败，不假称原目标通过；完整实测、有界优化后可closed_with_measured_quality_limitations。正确性、数值、来源、隔离不放宽。S5/S6/FINAL尚未完成。
- **完整课程已完成**：官方程序四分层CE，100step/1200实际draw=600原TRAIN视频family+600exact回放；不是全部8299条视频或原方案大规模全量。f2db parent/rank16/LR2e-5/16frames/1120不变，全AdamW/RNG连续恢复，检查点每5步，DEV在10/20/40/60/80/100监控。90min用户新窗口独立保留，旧20min文件不改。controller session78190 exit0/course_completed；reports/s4-program-stratified-full-controller-001.json。
- **当前最有潜力候选e46550143b28f8798484515f0abe21996fe92ad241625a4cf6da15dc093277fd**：远端artifacts/s4-program-stratified-pool-train-001/adapter-invocation-step-0100；training report421b950611a8e8db3fa1b0ae75c960790ae56c29a80a218976c171fbdf67ae97。待完整旧模态/跨模态复核后正式采用；f2db/CE0/S3仍保留，不继承历史分数。
- 完整video500真实372/500=74.4%（parent354=70.8%，+3.6pp），Brier.367875改善、ECE.079020及coverage.06变差，366.635秒exit0，原91.374%目标未达；完整image500414/500=82.8%（parent410=82.0%），Brier.315845/ECE.150904/coverage.682，原84.778%未达；完整audio500399/500=79.8%，Brier.323714/ECE.129549/coverage.124，227.215秒exit0。证据reports/s4-program-stratified-{video,image,audio}500-step0100.json。
- **已完成全DEV/当前GPU**：text500修复后真实498/500=99.6%/ECE.004504/coverage1.0/87.646秒exit0；originalcrossmodal500真实355/500=71.0%/ECE.251960/Brier.505223/coverage.146/377.117秒exit0，置信度明显待校准；supplementalcross500真实413/500=82.6%/ECE.144088/coverage.49/351.878秒exit0；screenshot500已真实485/500=97.0%/ECE.027021/Brier.054754/coverage1.0/260.272秒exit0；text500第一次failed exit1/14.753秒：纯文本错误传image_max_soft_tokens，gradient负责最小CLI修复和CPU测试，失败记录保留、随后新路径重试；generic runner v2 9b005a4f…；不重复已完成评测。完整DEV50 step10034/50/Brier.419482；此前20step30/50没有因3题噪声停整轮，完整video500随后验证增益。不宣称能力饱和或SOTA。
- 完整训练曝光028已CPU实际56.472秒/99known/unknown0/isolated闭合，600video+600replay完整计入；inventory manifest807003d0dfc3443f69f3d992566c62d60e7e7824db2655e1347e8bb9d546d6b8。S5 fresh UID inventory实际19.056秒exit0/isolated/boundarytrue，receipt e14f7dd48d07bffe779ea1998e0e17bdb044a70438255d930173aa5241aa6ea0；UID64101/64102永久保留，后续不得复用。
- **S5准备票s5:01 open依赖s4:01；下一步**：真实private calibration producer原daeaa7f已review；新增snapshot/progress fe58854正独立delta review；实际Linux边界6项(normal/timeout/TERM/parentkill/competition/stale recovery)全部通过5.626秒，ACL精确恢复、无残留UID进程，reports/s5-calibration-linux-boundary-001.json。bounded所有者仅metadata构建实际7组18000校准JobSpec（11组无数据明确unfitted），私有行/原始logits/logs不供agent读取。实际fit001已真实失败：session66909 exit1、processed0/18000、elapsed11.197秒，在state/worker创建前snapshot setup拒绝。固定metadata诊断核原model元数据owner/ancestor和privateadapter0600不满足publicreadonly；无私有行/日志读取。UID64201/2无残留进程、ACL还原dcad12c3…、recoveryjournal清理；保留失败，不复用UID。rootowned同盘baseview已真实ready /root/autodl-tmp/s5-calibration-base-view-001（15992595884bytes只读大权重hardlink+32198319metadata独立复制root0444，原件不改）；privateadapter snapshot/source09ba已修，118tests与独立review通过。9cc3f3a private-media snapshot fallback已整合部署（原权限不改，仅声明cal媒体复制）；focused Linux private_snapshot实测exit0/passed/.830秒，UID64401/2永久保留，ACL精确恢复。新增safe progress与真实runtime_recipe的fe58854已部署，archive4b1ffd3829599d65cd30fc9418e1846591922002655f7d0cc533519afc305ca1；新JobSpec002预检14.024秒exit0/SHA4b345752a5bada15426845c305f206500694ad1191ee00e8c862e42b0ed1aa77；旧001保留；新fit使用新UID64801/2（须核unused）、新spec及最新实际曝光inventory proof；workroot /var/tmp/omnijev-calibration-jobs-001保留旧lease；需完整保留复核及匹配推理精度；BF16校准不能无验证套用FP32服务。calibration timeout准备14400秒是上界，root仍按实际进度/租期控制，不无监控盲跑。
- **E2B实际准备**：first10 gold bootstrap session34777已terminalexit0/paused10/138.941秒/120draw/adaptere49accdf…，source /root/autodl-tmp/s6-e2b-readiness-code-001/archive6a79a3c4…，复用已曝光1200TRAIN pool，独立E2B无E4Badapter。此前真实CPU23.005秒全9文件hash/33tokens/thinkingoff/205actualweightshape/metaPEFT/6processor/完整1200账本通过；准入仅精确增加冻结E2B。该作业是端侧候选准备，不代S6验收/不冒称蒸馏。text500真实442/500=88.4%/ECE.098971/coverage.888；固定video50基座25/50→候选27/50，100同张量forward实际69.710秒exit0；非完整E2B质量通过。continuous所有者正在029实际曝光及freshUIDinventory002，cal新spec必须绑定它；decision所有者准备完整legal+U真teacher缓存/跨模型蒸馏，不伪SoftTarget/U=0，未启动teacher或新训练。
- **holdout入口**：4d7639a起始CPU101tests完整6000合成；cc8ffaa两项可选路径身份阻断已09ba/8e0修复且review通过，T必须绑定真实rawreceipt/runtime，DEV实际base/recipe匹配缺证则pending；视频imagebudget只按两套源码真实CPU同张量proof11547bde…接受，不假定输入变化；实际holdout未运行，不宣称S4关闭。
- **当前GPU**：已冻结E4B e465的JevBench231真实作业session23323/source s4-public-code-ba5ca19/timeout420s，结束优先calfit修复003；cal新immutable /root/autodl-tmp/s5-calibration-code-19bb739，archive51779140327ce3883e09fc2587474512abdc9949d0593c67c28d8a4ef21f0a0f，producer883a9640…，需最新029及freshproof。
- user授权创新/探索继续与主线相辅相成；模型完成前不做HF；OpenAI Decisions暂缓。r64/条件排序等失败路线停止，保留reports/explore，不为了清单重跑。新分层覆盖是已测增益，不宣称颠覆性创新。
- 用户09:18给剩24h：保守至10-17 09:15，03:15起至少6h导出/验收/修复；非平台核验、不续租/开关卡。.scratch/s4/wallclock-budget.md。唯一接续入口本STATUS；最近完整handoff plan/handoff-2026-10-15-s3-complete.md。

- **🟢READY(19:00)**:开卡前准备**全部完成**——RL形式化器包(GRPO G=4+KL锚0.05+SFT混0.3+硬闸门内嵌:每20视频出F1,连续2次不升/掉>0.05脚本自杀留证)+一日计划(reports/explore/day1-plan-and-monitoring-001.md,H0资产抽查→冒烟→RL主训练→分时课程两段(stage1≥123才许stage2)→胜者重校准→封存→HF)+HF包+全部脚本CPU自测绿(commit a541499已合并)。**等用户开卡**:开卡后SSH恢复,按一日清单H0起步,main挂Monitor执行。

- **⚠租卡计划确认(18:20)**:用户将在准备就绪后开**1日卡**。目标=快速增量迭代收敛最优(RL形式化器+SAS→D3P顺序课程,预注册门判定)→胜者重校准+封存→**配套产品完成(HF发布按社区惯例,准备中;上传需用户HF token)**。CPU准备已全面展开(HF包/RL脚本/课程数据),开卡即纯执行。01日小时级清单见docs/release/one-day-plan.md(fable协作产出)。

- **⛔租卡阻塞点(17:35,18:35更正)**:卡片为**关机非销毁**(用户澄清,AutoDL磁盘保留,SSH地址开卡后恢复)——/root/autodl-tmp全部产物原样,**开卡即用,无需任何重建**;原provision脚本计划取消。续租后第一天执行清单=RL场景形式化器训练(fable CPU准备中:循环设计/奖励评估集/脚本,目标四事件谓词F1+0.2且级联≥124)+SAS→D3P顺序课程+D3P-600封存汇总收尾。不开卡=当前D3P-600+七温度交付态维持不变。**待用户决策:是否续租。**

## 已停止或保留的路线

|路线|实测结论与证据入口|
|---|---|
|新family coverage|f2db改善CE0；第二段250ef full500356但概率/跨模态更差，第三段5cff DEV37但video150108vs116，停止扩段。reports/s4-temporal-coverage-*、reports/explore/s4-temporal-coverage-full500-diagnostic-001|
|原连续固定池|step10 dfcf DEV34/Brier.424180、video150115；step20 bf59 DEV33/Brier.476134；停止第3段。新core235ad68/archivee036f037…；reports/s4-continuous-main-pool-*|
|pair/KL/Brier|pair与CE预测相同且显存更高；referenceKL无稳定保留收益；Brier仅微改善matchedCE、仍差于CE0。reports/explore/s4-{claim-pair-pilot,reference-retention,temporal-brier-pilot}-001|
|采样/课程/极性|motion16与uniform16同30/50、概率更差；旧时序课程video150110vs109但概率更差；极性32vscontrol36、停止。reports/explore/s4-{temporal-motion-diagnostic,temporal-curriculum-pilot,video-polarity-results}-001|
|collision顺序程序gold|DEV33持平、概率更差；TRAIN12 CE下降27.9%但泛化无收益。原生DEV任务覆盖不同；reports/explore/s4-temporal-order-contrast-results-001、s4-order-train12-001|
|RL/奖励目标|train/decision_reward.py仅CPU原型；onehot expectedreward等价qgold缩放CE、无新监督，不冒称RLCD/创新。reports/explore/decision-training-innovation-next-001；未来有新真值/目标证据可再试|
|原生MoE路由/融合|BF16原生路由概率差超.001、argmax同不能代过，停止当前配置；reports/s4-native-routing-comparison-100-001。全部原模型/cache保留|
|原生关系课程/视频强化/首matched30pairs|nativeDEV33且video150113vs116；video-onlyDEV30vs33；首matchedDEV32vs33且Brier/ECE差，均不采用/不扩500。完整证据reports/s4-{native-event-condition,video-specialist,video-text-matched}-*。新固定课程是有限连续优化窗口，不重启旧失败作业|

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
- 历史34报告基线保留；最新020覆盖80known、unknown0；matched累计240draw与旧continuous240分别核算。所有曝光只认实际报告，不把准备池容量算已训练。
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
- 用户授权有界创新/RL与实测资源调整；先小批量、有效再扩大，持续监控与早停。冻结基座/无新分类头/不换引擎、质量推进按ADR0023透明记录、正确性不放宽、5b不启用。大预算/付费/租卡开关/生产部署仍须用户决定；不push。
- S5、S6和FINAL尚未完成；终局包含真实API/单卡/端侧/浏览器/性能与无未解决问题。快与低成本须同条件实测，未知价格不编造成本，无同条件Jev实测不声称超越。
- 用户2026-10-16明确暂缓OpenAI Decisions对照；保留资料，不请求凭据、不接入、不付费。继续S4主线。GPU真正用完再通知用户关卡，不代执行关机。
