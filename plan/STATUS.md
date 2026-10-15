# STATUS

更新：2026-10-16。**S3已通过；S4数据/gate工程已通过，S4模型质量未通过。原生goal仍active，产品未交付。**

## 当前票与下一步
- s0:05 done；s4:01 claimed main-session，规格.scratch/s4/spec.md与iteration-policy.md。S4质量未通过，S5/S6/FINAL未完成。
- 当前工作候选coverage adapter0deef8cfd3d94f4dd3b42cfdbe995c825c388230721df7e6716e5963cb9dc818，artifacts/s4-crossmodal-retention-train-001/adapter。真实30步/360draw/449.90秒，峰值37.14GB；reports/s4-crossmodal-retention-training-001。
- 已测：原video500为337/500=67.4%（S3 63.6%，+3.8pp）；原cross100为72（首轮73，纠正2/退化3，输入逐题一致）；补充cross50为41（首轮40）；audio39(+2)、image40(=)、screenshot50(=)、text49(=)，均50题。reports/s4-coverage-video-full-001、s4-cross-batch100-original-paired-001、s4-cross-batch50-supplemental-coverage-001、s4-coverage-old50-001。小测支持继续探索，不是S4验收。
- 用户授权ADR0022：小筛选一题差异不机械否决；机时优先，默认短批次、单遍、每25题进度，通常180秒单作业；完整验收在候选冻结后。先前fullcross队列76057已取消（31.28秒/18记录留存），不重启；87250/5929/63186/50489均终态。当前GPU作业见下一条。
- 成对监督准备190dbf5已真实完成40.94秒，600扩展及120draw两次全库存isolated；artifacts/s4-claim-pair-pilot-001，data72c9bbf0…/ledger c2376c1b…。训练接入3fe0ab2已137项测试，默认0保持旧配置/checkpoint；固定两arm各10步/240秒，checkpoint5，设计reports/explore/s4-claim-pair-pilot-001.md。
- CE0会话69827终态0，10步/120draw完成、191.15秒，逐步数值有限；PID139745/timeout139744已结束，step5检查点2c25d988…已保存；CPU预检ready28.07秒。源码/root/autodl-tmp/s4-claim-pair-code-001（3fe0ab2，archive9132c2d552af09aa4569c33df3b370fc61abe5c61eb34fa2e235b795a28f2378）；launcher /root/autodl-tmp/launch-claim-pair-ce0-001.py，输出artifacts/s4-claim-pair-ce0-train-001/run.json、reports/s4-claim-pair-ce0-training-001.json。核验现有会话，勿重启。
- CE0原cross100已完成73/100（起点72）；40062终态0，reports/s4-claim-pair-ce0-original100-001，尚无稳定收益结论。pair01 CPU预检ready44.31秒；91212当前GPU PID184812/timeout184811，已9/10步正常，step5检查点7ea8b450…，peak56.94GB，逐步pair_count=2。
- pair01终态后9510自动并行CPU曝光审计--claim-pair（artifacts/s4-pilot-exposure-004）与GPU原cross100（artifacts/s4-claim-pair-pair01-original100-001），只在训练10步completed才评估，180秒；launcher/root/autodl-tmp/launch-claim-pair-pair01-001.py。曝光helper b757213已14项测试；此队列正在等终态，勿重复启动。
- pair CPU输入已validated33.60秒：sidecar963813e9…、config00f1b141…、receipt972d9414…，40输入/20pairs通过，原CE配置不变。下一比较脚本.scratch/s4/compare-claim-pair-arms.py已准备：严格核同init/data/配置（仅pair字段不同）、同100逐题输入，并比较joint NLL/Brier/ECE与配对纠正退化；待两arm评测终态执行。
- 最新已通过曝光inventory artifacts/s4-pilot-exposure-003（42已知update-bearing报告），reports/s4-pilot-exposure-reconciliation-003；新pairs数据尚需隔离/实际训练后再核曝光。正式S4封存前刷新gold准入绑定，不能复用旧绑定冒充新曝光。
- 公开dev监控9268d9a已87项测试，真实old50部署验证；source /root/autodl-tmp/s4-monitored-eval-code-001（74e96e2，archive b8c5b3184ea2d09e6bb1184c7065c27fd6d354022dcf0e37169e6e390f1d6516）。参考首轮6bb86b09…、S3 1bebae31…及所有失败候选保留。alpha.5融合与旧模态完整重复评测暂停。

## 最近实验结论（完整证据保留）
- 第二轮video CE：30步/350.08秒，adapter14496e28…；video113/150、audio39/50，但原cross33/50<35，不推广。
- 有效delta融合alpha.5和.25：分别补充cross39<40、video101<102，不推广、不扫比例；s4-fusion*-screening-001。
- B3视频70→140 soft tokens/frame：99→96/150且jobwall120.66→168.11秒，停止；s4-video140-probe-execution-001。
- A4原生多LoRA：8题小试过，100题输入IDs全部匹配但23题概率差超.001（raw最大.05721、conditional .05813），argmax0变化不替代数值门。停止当前BF16路由配置，不改引擎/阈值；s4-native-routing-comparison-100-001及diagnostic-100-001。
- CE+reference KL .5：TRAIN reference300draw/200.16秒；139正确replay启用，video150纯CE。文本mapping预检bug44214af修复、复用sidecar b135b862…；30步439.06秒，adapter8e5c0e45…，video113/150、原cross33/50，停止不扫λ。s4-reference-retention-training/screening/diagnostic-001。与第二轮cross50预测完全相同：较首轮纠正2/退化4，no→yes10→14；由此启动当前覆盖试验。所有上一轮67829/13789/1399/32634作业已终止。

## 外部基准准备
- JevBench：adapter入口ba5ca19及真实CPU preflight通过；候选231题GPU待最终候选冻结。远端/root/autodl-tmp/s4-public-code-ba5ca19；禁止将公开231题当教师195题协议。
- Decision Index：真实1530文件/5.90GB映射+私有materialize通过，work artifacts/s4-index-work-001；ready=false。Forecast36动态引用齐全，latest-llm未被固定recipe引用。cl100k缓存已下载SHA/离线加载验证，artifacts/decision-index-tiktoken-001（重建设置TIKTOKEN_CACHE_DIR），reports/decision-index-tiktoken-offline-001。
- 仍待：评价许可、Git来源修复真实执行、完整rebuild/verify。Git修复helper b5eeea3已集成，27项CPU测试通过；真实bundle生成通过（8原Git root、923材料），artifacts/s4-index-provenance-bundle-001；derived SHA fc9e5c84…/map3aa79d8f…，已安装独立派生source artifacts/s4-index-derived-source-001，8源lookup及未知父repo拒绝真实验证通过；未rebuild。报告decision-index-provenance-install-001。许可证据并行调查中。materialized7目录会错误继承父repo869ee95，1目录缺失；原8真实Git roots/pins均验证，详见reports/explore/decision-index-closure-001.md。不得以父Git revision伪记数据来源，不运行raw pickle。原料许可不因下载成功而准入。
- OpenAI Decisions对照按用户2026-10-16明确要求暂缓，不接入、不请求凭据、不付费。
- 即时接续以本STATUS/当前票为准；最近完整里程碑handoff仍plan/handoff-2026-10-15-s3-complete.md。小切片不另写handoff。

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
- 用户2026-10-16明确暂缓OpenAI Decisions对照；保留资料，不请求凭据、不接入、不付费。继续S4主线。GPU真正用完再通知用户关卡，不代执行关机。
