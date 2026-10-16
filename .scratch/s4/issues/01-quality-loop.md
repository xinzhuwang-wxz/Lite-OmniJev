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
