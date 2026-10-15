# STATUS

更新：2026-10-16。**S3已通过；S4数据/gate工程已通过，S4模型质量未通过。原生goal仍active，产品未交付。**

## 当前票与下一步
- s0:05 done：reports/s4-gold-gate-engineering-acceptance-001.json。数据准入、私有数据审计、固定私有模型入口、失败拒绝和真实GPU双遍工程验证齐全；不代表S4/5/6质量通过。
- s4:01 claimed main-session，主树 /Users/bamboo/Githubs/Lite-OmniJev；规格.scratch/s4/spec.md，规则iteration-policy.md。继续有界训练→固定dev筛选→完整dev→冻结候选→私有holdout/外榜→S4机器门。
- Sol6.1构建器已合入87c766c，真实200对TRAIN组合构建通过：reports/s4-crossmodal-train-build-001.json；train SHA d3b0be2a…，新完整inventory SHA244a5c4f…，100/50/50 AND规则，原始train图像/音频各200唯一组件。源与媒体SHA及全库存隔离均通过；主树22针对测试通过。
- 首个S4候选30步完成：300曝光，六cohort各50；训练wall319.167秒，CUDA peak36.94GB，online SwanLab无错误。adapter SHA6bb86b09…，reports/s4-joint-pilot-train-001.json。首轮原crossmodal固定50从33→35，补充固定50保持40；两个selected ID SHA与旧基线一致，已通过预设跨模态筛选触发条件；不是S4门通过。
- 五旧cohort筛选已完成并通过预设警戒：image40→40/50、screenshot50→50/50、text49→49/50、audio38→37/50、video99→99/150。reports/s4-pilot-screening-001.json qualified_for_expansion=true仅筛选资格，不是正式质量通过；音频小切片少1题需继续关注。
- 第二轮新视频课程已完成30步/300曝光，wall350.082秒、CUDA peak36.845GB、step15检查点保留；adapter14496e28…（reports/s4-video-pilot-train-001.json）。video99→113/150，分组existence34/39、motion11/14、temporal68/97；audio37→39/50。
- 第二轮原crossmodal35→33/50，违反预设保留门，reports/s4-video-pilot-retention-001.json qualified_for_full_dev=false；未跑其余不必要筛选，不能推广。第一轮6bb86b09…仍为保留候选。当前GPU进程已实查为空，未关卡。
- 有界融合pilot-003已结束，两比例均未全过：alpha.5原cross36/video107/audio39，但补充cross39<40；alpha.25两cross36/40，但video101<102。都不推广，不新增比例，不运行完整dev；报告两份s4-fusion*-screening、050-retention及crossmodal报告；025在video失败后未启其余保留评测。alpha.5/.25 CPU最大残差.000656/.000496均通过数值门；未证明任何S4质量门。首轮仍保留。
- 原方案A4明确列有按模态路由多LoRA，ADR0015允许按实测干扰启动；已只读确认固定vLLM0.26 Gemma4ForConditionalGeneration声明SupportsLoRA，实际多LoRA运行/数值/成本仍未验证，不能据此称部署就绪。当前先检验B3输入预算。
- 新瓶颈假设B3：官方加载的Gemma4 video_processor默认70soft tokens/帧，CPU固定6题×16帧70/140/280输入约1173/2229/4373（reports/s4-video-budget-cpu-001.json）。140仍在4096内；现已按reports/explore/s4-b3-video140-001.md启动唯一video140对照，首轮adapter、同dev150/16帧不变，≥102才继续，600秒timeout。当前session11581，远端/root/autodl-tmp/probe-video140.py，source f592776，输出reports/s4-video140-probe(-execution)-001.json。第一次启动因引用模块局部transformers属性失败、未启GPU；已改直接导入，日志001保留，当前日志002。
- 两轮实际runtime canonical data SHA与准备JSONL不同，已逐ID完整Sample等价核验并追加historical TRAIN；reports/s4-pilot-exposure-reconciliation-001.json真实通过，38份已知update-bearing报告、追加2份canonical工件、全库存隔离通过。派生库存artifacts/s4-pilot-exposure-001/inventory-manifest.json；原gold11证据尚需在正式S4封存前刷新当前准入绑定。
- 首轮训练+7筛选总jobwall665.642秒（reports/s4-pilot-resource-001.json），不是总租赁账单/服务时延；价格未知cost=null。第二轮损失/计时/曝光/SwanLab online真实记录齐全。
- JevBench adapter入口ba5ca19已合入，主树26tests及真实CPU preflight通过；候选231题GPU仍待最终候选冻结。源/root/autodl-tmp/s4-public-code-ba5ca19。
- Decision Index真实映射c5c8d52已执行：1530文件/5,899,934,817bytes，固定tar/Router/HLE/RAGTruth hash核验；metadata在artifacts/s4-index-inputs-001。安全审查发现umask扩大权限，e13d65e已修复并16tests通过。真实私有materialize12.34秒通过，全树700/600核验，reports/s4-index-materialization-001.json；work artifacts/s4-index-work-001；ready仍false，未执行rebuild/未加载pickle。
- Decision Index offline准备入口aaf2fc6已合入，16tests通过；只准备独立Layout/安全解包/RouterBench JSONL seam，ready=false，实际映射和materialize已完成，评价许可/git provenance/tiktoken/动态闭包/完整rebuild与verify仍未完成。固定tar实际SHA127198…已确认旧config漏首位1，修正证据reports/decision-index-metadata-correction-001.json，不变更任何数据字节或许可。
- 即时接续以本STATUS/当前票为准；最近完整里程碑handoff仍plan/handoff-2026-10-15-s3-complete.md。不为每个小切片另写handoff。

## 已验收数据与入口
- .scratch/s4/selected-gold-manifest-001.json SHA e073a601…：六cohort各3000cal/1000hold，原dev各500；image/audio/crossmodal额外dev各500单列。旧AOK623/211、video900/300保留为来源诊断及历史曝光，不重复计入主配额。
- reports/s4-gold-data-admission-001.json evidenceSHA9cf11cea…：来源代码128绑定、source链154工件、CLINC原始Parquet全链重放、Clotho保留120完整字段重放、13500条整理记录仅URI变化、19973媒体路径SHA验证均通过。
- 已知34份实际更新训练报告、18份输入和7个旧dev版本全部纳入当前库存（s4-known-exposure-reconciliation-001）。不假称覆盖未知外部作业。
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
- 暂无需要用户操作的阻塞；继续独立工作。GPU真正用完再通知用户关卡，不代执行关机。
