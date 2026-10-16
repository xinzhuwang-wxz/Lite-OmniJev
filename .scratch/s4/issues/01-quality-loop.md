# 01: 全模态有界训练→完整质量与封存评测闭环

ID: s4:01
Milestone: S4
Category: enhancement
Status: ready-for-agent
Execution: claimed
Owner: main-session
Worktree: /Users/bamboo/Githubs/Lite-OmniJev
Blocked by: s3:01, s0:05

## What to build
按.scratch/s4/spec.md与iteration-policy.md，完成实际训练/开发筛选/冻结候选/同条件教师与外榜/私有holdout/阶段门。数据gate工程复用s0:05，不重新按来源或探针拆票。

## Acceptance criteria
- [x] s0:05实际验收有效；训练新组合/变体严格来自train，来源/历史曝光和组件别名可追溯。
- [ ] 每轮冻结假设、预算、数据/配置/adapter、dev切片与退出条件；监控和中间检查点实际产生，小批量无收益停止，有收益才放大。
- [ ] 六cohort及保留来源切片的最强允许教师同条件比率达到.97；原cohort不被补充来源总分掩盖。
- [ ] 私有完整封存/匹配dev差≤.03，完整度、来源、模型/模板/token/输入哈希及失败拒绝证据齐全。
- [ ] Decision Index、JevBench真实可复现测量均不低于同尺寸基线，缺测不算通过。
- [ ] configs/gates.json S4全部门通过，资源/成本/创新结果有实证并独立集成复核；不以CPU/小样本工程报告代替阶段验收。

## Evidence
200对TRAIN-only图像+音频AND组合已真实构建并通过完整库存隔离，报告reports/s4-crossmodal-train-build-001.json（train SHA d3b0be2a…，inventory SHA244a5c4f…）；构建器87c766c，主树22针对测试通过。首个30步候选已完成（wall319.167秒、peak36.94GB、300曝光、真实逐步指标/online），adapter6bb86b09…；原crossmodal固定50从33→35、补充保持40，五旧cohort保留筛选亦通过（40/50、50/50、49/50、37/50、99/150；audio较S3少1题，其他持平），仅获得进一步验证资格；reports/s4-pilot-screening-001.json。报告reports/s4-joint-pilot-train-001.json及s4-pilot-crossmodal-{original,supplemental}-001.json。第二轮视频113/150、音频39/50，但原crossmodal33/50未守住35门，候选不推广；保留首轮。真实canonical曝光核对38报告通过。pilot-003两比例已真实试完但各有保留门失败，均未推广；B3视频140同权重单变量对照99→96/150且耗时增加，路线停止；A4原生路由100题raw/conditional概率门失败停止；CE+reference KL真实30步video113/150、原cross33/50仍失败，不推广。额外60条TRAIN跨模态覆盖候选已完成360draw/30步（449.90秒，adapter0deef8cf…），video108/150通过但cross34/50失败，未推广；配对纠正1/退化2，准备不更新参数的TRAIN梯度诊断；42份已知更新报告曝光核对通过，inventory s4-pilot-exposure-003。保持既定门槛。Decision Index私有原料materialize已执行，rebuild/许可闭包仍待。未通过S4质量门。已有S3及S4公开诊断只用于形成实验假设；见plan/STATUS.md。新实验在本票内记录，质量门保持。

## Out of scope
S5校准正式验收、S6/E2B正式验收、5b、付费/租卡开关、生产部署及大预算自动扩量。

2026-10-16短对照收口：受控claim数据CE更新后cross10073、supp50 43、video150109、old4持平，作为工作候选4b8e422f…；pair目标同样准确率但概率收益不一致、显存增加，停止。曝光004含46份已知更新报告且完整隔离。继续时序课程与同16帧预算的独立推理小试；正式S4未通过。

2026-10-16诊断与新覆盖：12TRAIN固定记忆可从10→12/12、CE降98.8%，但DEV50 30→28且Brier恶化，不推广；CE+Brier同数据对照无稳健收益，停止。新增不同60视频family、保持cross/old回放与配置的首段10步已完成，固定DEV50 30→33、Brier改善但coverage@5%下降，已接视频150进一步筛查，尚无晋级结论。精确证据与作业入口见STATUS和reports/s4-temporal-coverage-{preparation,training,dev50}-001。终局门槛保持。

2026-10-16完整来源诊断：第二段新family10步173.657秒后，video150仍116/150但yes偏移100→82纠正，DEV50为36/50（父33）；原视频500单遍356/500（71.2%）/363.467秒，420秒硬预算内完成，仍未达S4门。暂停第三段自动训练，先补保留检查；同30视频family与同replay的极性匹配对照CPU准备已真实隔离通过，但未授权运行/未有质量结论。保持短段、观察后才放大。

2026-10-16 10:34：固定1200draw池全局100步不变，真实10步暂停后完整AdamW/RNG恢复到20步；累计240draw，非1200。首DEV50 34/50/Brier.424180，第二DEV50运行session14990，禁止自动第3段。曝光014真实通过，015递推审计进行；原包装失败001与CPU recovered002永久保留。公开image500=410/500，不达S4。程序语义诊断支持原生事件条件/事件不存在分支是主要残余错误，不支持窄时间间隔主导；创新下一试须围绕此真实瓶颈。DecisionIndex已27/44eval-only、发布三档目录CPU模板完成，仍不代表模型/校准/端侧交付。

10:39补充：step20 DEV50真实33/50/Brier.476134/ECE.144538，较step10退化，停止第3段rawcoverage；015真实isolated/unknown0、actual240draw/fullcheckpoint持续绑定通过。gradient仅CPU准备官方原生事件条件课程（事件不存在/存在、before/after），先有界CPU核验后才启动唯一10步GPU，不碰私有cal/holdout。

本轮用户澄清：主线与自由探索继续并行相互促进，未暂停探索；模型出来前不再做HF。native课程10step完成，DEV50 33持平/Brier改善/ECE变差，fixedvideo150 session9982复核中，另CPU原生关系组和旧题型配对诊断，不预排下一段。发布S5registry仅CPU准备非激活；正式S4仍未过。

11:13：native video150113vs116，原生关系22持平、absence8退化，停止配方，保留f2db不扩500。主线与探索并行授权保持，HF模型前不推进。下一仅CPU准备video-only10step/120draw（60原题两遍）以检验视频监督强度和混合回放稀释；不冒称严格单因子因果、旧模态尚未通过，不替换生产单LoRA。016实际曝光pin是CPU审计前置，拒绝用stale015绕过。

11:19：016真实isolated/71known/unknown0完成；video-onlyCPU审计20.307秒通过并以f2db parent启动session4260/GPU PID61963。仅10step/120draw/60原题两遍，无新family或新gold；资源估值按实测6→12video/step，唯一hardcap240→360秒、质量门不动。结束先DEV50再决定，不预排新段；训练终态才跑017。

本轮已收终态：video-specialist DEV50=30/50vs父33，Brier/ECE变差，停止不扩；既有continuous step20宽video150112vs116，确认不续第3段。parent保持f2db。017因turn中断已恢复helper实际核对，无updates才audit。并行只读TRAIN检查精确“同题异视频相反官方gold”对照容量，尚未构建/训练；不重复同video flip路线。

本轮textmatched174.789秒completed10，DEV50 32vs父33且Brier/ECE差，无采用，停止扩大。018实际30.442秒/isolated/75known/unknown0闭合。准备6same-text/oppositegold TRAINpairs的视觉依赖/优化能力before-after诊断，避免把独立问题memorization当视觉可学性，尚未启动新训练。图像来源审计确认同源AOK，VQA增广已试，不重复错误假设。

本轮pairedTRAIN12 before7/12→after9/12、CE降16.78%、输入mapping12一致，仅部分拟合非泛化。固定300pairs课程CPU完整审计通过，global100仅上限，首次10step已DEV34vs33/Brier.424982/ECE.080932；019实际78known/unknown0/isolated/fullcheckpoint闭合。root在固定20min窗口12:20–12:40内续step10→20 session46306、保持完整AdamW/RNG，结束接固定DEV50；不每次重置/换data、不盲全100，所有原质量门保持。

本轮固定matched课程step10宽video150111vs父116、step20DEV30vs33，窗口结束不续第三段。曝光020真实35.934秒/80known/unknown0闭合。r64容量创新CPU保前缀/scale通过，但真实GPU初始化12输入概率差.030967超过.001，停止训练前置，准备fresh r16/r64控制诊断；不降低数值/质量门槛。LoopConfig限定16/32及64/128，P0不变，配置支持不等于实验获准启动或模型通过。

独立r64候选决策：受控GPUfresh16/重复16/历史全一致，r64概率差.030967，不能等价替换。按ADR0015改为新容量候选独立自身DEV50 baseline（session98268/180秒），不继承f2db指标、不将失败parity改pass；固定独立研究政策0cc0d96a…，正式S4/S5/S6及pack .001均保持。root已部署b97edc0配置源/archive636aca42…，仍未训练r64。

独立r64自身DEV50真实34/50/Brier.442514，达到研究筛选；rootreview8468c583…仅准first10/360秒。CPUpreflight29.159秒ready、actualderivedconfige0191602…已固定，训练session81664/PID259607/SwanLab nkksm663，终态须自身+2题（36/50）/Brier改善才宽video150，不自动续训。新跨视频条件排序objective仅CPU数学提案/准备，与旧同family pair contract隔离，尚未训练或声称创新收益。

r64 first10实际188.714秒结束，DEV31/50vsown34、Brier略降/ECE变差，政策shortlistfalse，停止续训/150/500，不采用07c2。下一仅实际exposed6pairs机制诊断和有界条件排序CPU集成；021曝光独立profile准备/实际audit，当前无GPUlive，不宣称质量进展。

实际exposedTRAIN12显示rank64CE可拟合（7→11/12、direction6/6）但DEV退化；批准一次条件排序比较归纳偏置，不诊断成无梯度/缺容量。newcore7b75、实际CPU准备300pairs/1200ledger通过；rootdecision6b7ecc32…限first10/240秒，session94264/PID289855/SwanLab e4jvjvwx，DEV≥36且Brier<.4401167682才宽复核。021完整isolated/unknown0已闭合rank64CE120draw；排序活跃时不跑全inventory，终态接022。131 root回归+24数据契约/准备+9wrapper、mypy/ruff通过，仍无S4质量通过。

条件排序first10实际191.140秒completed/paused，DEV32/50vsCE31/own34，Brier.453986变差，shortlistfalse，停止扩量/采用。此小试未取得可用创新收益。新只读证据确认300对严格in/out，缺motion/startend/collisiontarget，而旧rawcontinuous只是temporal关键词；下一官方程序分层课程先CPU准备/退出条件，不重复失败随机长跑。022完整曝光CPU审计中，当前GPU均terminal。

022实际52.531秒isolated/86known/unknown0闭合，四条链分别240/240/120/120，旧bindings/snapshots保持。下一官方程序四分层课程由gradient仅CPU构建，600视频4组各150且各75yes/no、exact600replay；保留f2db/rank16/LR/CE，20min含评估、10/20条件推进，未启动GPU。原窄in/out/关键词池失败证据不覆盖；容量/排除不足不得私自重复补齐。

2026-10-16用户新决策ADR0023：完整分层课程100step/1200draw已完成（600原TRAIN视频+600回放，非全部8299视频）。候选e465，video500372/500=74.4%vsparent70.8%，image50082.8%vs82.0%，audio50079.8%；原视频/图像质量gate未达，保留真实失败。继续text/screenshot/cross完整保留复核、必要独立汇总，再按实测质量局限决定S4出口并接S5/S6；不假称原门通过。曝光028与fresh UID inventory已真实isolated通过，calibration Linux6项边界实测通过，真实fit尚待。详见STATUS与reports/s4-program-stratified-*。

2026-10-16 19:07外榜收数：候选e465 JevBench231真实GPU两遍exit0/status passed/97.554秒，accuracy76.62%/macro74.93%/Brier.396538/ECE.165738/ordinalMAE.3745/paraphrase一致性94.44%(36对34双对)/invalid0/两遍重复性0差异(records_sidecar SHA750ef69a…)。同尺寸父模型基线同231题/同协议/同BF16运行中（s4-parent-jevbench-001），完成后才能判定“不低于同尺寸基线”门；不与教师195混比。校准fit003：JobSpec已按最新029曝光(c2743076…)+fresh receipt(a4f606c8…)+producer 19bb739c/883a9640重新准备，job SHA6bda6ad6…，UID64801/64802确认unused，待GPU空即启动。

2026-10-16 19:16外榜基线收数：父模型(无adapter冻结E4B)JevBench231同协议两遍exit0/74.7秒，accuracy76.19%/macro77.39%/Brier.427304/ECE.173385/paraphrase86.11%。"不低于同尺寸基线"门按官方headline判定：候选76.62%≥76.19%通过；macro−2.46pp如实记录(≈97%来自ambiguous家族n=7，2/7vs5/7，λ组合分析见reports/explore归档由fable/mainfable产出)。候选Brier/ECE/paraphrase全优，父ordinalMAE更优(.2705vs.3745)。S4原质量门(教师.97比率)仍未达，按ADR0023保留。

2026-10-16 21:30 创新包整合与GPU窗口排期:执行代理CPU包已合并(5dd2b66):①概率级融合/对比解码全λ全负→两子路线终止(oracle +7.8pp只在输入/目标维可取);②B1+I1 WSFT格clevrer:5218全量校验126题、292行软目标、三臂240s JobSpec就绪,门=video150≥121/150或关系/存在≥+3pp且DEV50不回退;③R1 GRPO经CPU闭式推导α/γ分解回CE→按不假称规则降级为B1耦合辅助损失,不排采样式GRPO;④I2孪生分歧U在video为诚实负结果(基座熵非单调)不排前向;⑤B2 caption/条件化管线就绪(采用需ADR4001+用户批);⑥I5教师缓存150题JobSpec就绪(~15min GPU,熵前置检查过)。GPU窗口排序(B1+WSFT 240s三臂>I5教师15min>B2 caption 20-30min>帧边缘化可砍)于holdout完成后、03:30硬停前执行;全部失败即停留证。调研代理10-idea报告已入主树(eb42cdc)。

2026-10-16 22:40 用户决策(ADR0024):关闭U作为产品输出特性与U中心创新线,接受模式回归Jev式(Choice/Score/yes-no);原方案"p(U)单独返回"降级为内部量,发布说明按ADR0024口径。GPU窗口重排:I5/D3P教师软目标蒸馏(150题~15min+240s学生切片)>B2 caption(需ADR4001+用户批)>R1-B-gold;WSFT/R1-B-wsft/孪生U前向取消,B1格与R1-B CPU证据保留为探索记录。外榜/校准/holdout协议不变。

2026-10-16 23:40 video围剿增量(用户令:全维度打败20pp差距):新增契约内未试输入预算探针——**32帧uniform video150**(8/16帧测过,32帧从未测;--video-frames 32,~6-8min GPU),门=≥121/150(vs 16帧116)即视频输入预算直接增益并触发video500确认;排holdout后第一GPU项(先于D3P,8分钟)。视频版SAS(程序切面辅助监督:count/color/relation/motion/start-end切面,gold=CLEVRER程序真值,非U,ADR0024合规)由执行代理CPU准备中——把CLEVRER当结构化监督源而非QA源,检索零先例。ADR轨道(目标中心token,4002草案)为最高天花板项,待B2/切面证据后提请用户快速决策。
