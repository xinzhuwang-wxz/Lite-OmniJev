# STATUS

更新：2026-10-15（Asia/Shanghai，18:34）。**S2图像原门仍未过：v2 78.8% vs80.6%；其它三门通过。停止旧池六条无扩量收益路线，新增1811条已许可/同train-family人工一致问题；新gold监督两臂均未达扩量条件，停止。S3训练技术路径已通，质量仍需改进。**

## 已验证（不重跑）
- S1通过；S2 v2 image78.8%/cf88.6%、screenshot96.6%/cf98.4%、text99.4%，各dev500双遍一致；reports/s2-stage-acceptance-v2-001.json仅image FAIL。
- cf统计修复bbdc041，ADR0014结构性结论失效。旧v1重算cf91.8%/96.6%。
- E01–E06 dev150分别121/124/122/121/122/122，对照v2=123，均未达125扩量规则，全部停止；见reports/explore/。训练集97.33%而dev78.8%，不是继续堆步数的依据。
- 现有OneJev4教师600训练图像标注77.78秒，513与gold一致、87保留硬gold；E06无准确率收益，停止。
- VQA2四官方ZIP共42,962,121 bytes已哈希，网络加速+分段54.95秒下载；ADR0016准入同原train-family CC-BY2图片配CC-BY4标注。新gold1811（颜色183/整数281/yes_no1347），十答案一致；其它family/模糊答案/重复均排除，程序干扰项透明。reports/s2-vqa2-{staging,gold}-001.json。
- 新训练入口：warm-start+谱系932ade4；CE+Brier可选c725485（zero与旧CE兼容）。108项集成测试、ruff/mypy通过。VQA构建17项回归通过。
- S3：CPU/音视频GPU前向通过；30步五模态训练242.96秒、峰值28.64GB。dev50对照v2→新模型：audio70→52%、video62→58%、image78→76%、screenshot100→98%、text98→98%；暂停扩大该候选。reports/s3-quality-probe-summary-001.json。
- v2 video dev500单遍53.4%，未达69.8%×.9；不能免除音视频训练。reports/s3-v2-video-dev500-diagnostic-001.json。

## 当前唯一GPU链 / 下一步
- SSH root@region-42.seetacloud.com:48926；mux /tmp/omnijev-ssh-root-final；远端 /root/autodl-tmp/s1-text-lora-001 代码2508a6f，运行中不切代码。
- E07已结束：CE123/150、CE+Brier124/150，未达125扩量条件；reports/s2-vqa-{ce,brier}-{run,dev150}-001.json，停止扫参。S2仍78.8% vs80.6%，未放行。
- S3 lazy/eager真实GPU一致性完成：30步loss完全相同、adapter SHA相同e4caf62d…；243→213秒；cache峰值134011035 bytes<128MiB，GPU峰值28.64GB。reports/s3-lazy-parity-comparison-001.json；不是S3质量通过。原作业PID243896已终止。
- 全池21184行lazy CPU预检完成：29.54秒、进程峰值1559654400 bytes；reports/s3-lazy-full-preflight-001.json及resource报告。仅全量metadata+4个代表样本，不宣称全部媒体读取。
- E08准入完成：400新family，颜色160/数字160/yesno80；排除原AOKtrain全部family、现有ledger与3832张图片SHA。首批下载400张62.63MB/252秒，其中coco:265843解码截断，已记录排除并确定性补选1张。最终plan artifacts/s2-new-family-plan-004，候选SHA8a9cf9a5…；媒体未删除。
- 构建输出artifacts/s2-new-family-gold-001，train SHA7d0f1583…；混合池artifacts/s2-new-family-probe-001为1000行（新image400/旧image400/text100/screenshot100），SHA88dab83e…，四ledger验证通过。构建器2508a6f远端16测试通过；无活跃子代理。
- E08训练完成100步/357.78秒，adapter d5ca009c26495a1fc5bd8c9eb84da554b436c629a8cef2cf7c6ff22545c2490c；dev150=125/150（83.33%）、ECE .14844、35.44秒，达到扩量规则。reports/s2-new-family-ce-{run,dev150}-001.json。
- E08完整image dev500单遍=399/500=79.8%，较v2提高1pp但低于80.6%；停止此候选完整阶段扩量，保留adapter。reports/s2-new-family-ce-dev500-screen-001.json，77.97秒。当前最好图像单遍不代表三模态整体最优。
- E09 dev150=126/150=84.0%，ECE .14332，68.44秒；同v2默认123/150，达到扩量条件。reports/s2-image-budget1120-dev150-001.json。
- 当前唯一GPU作业：/root/autodl-tmp/image-budget1120-screen.sh，timeout600，日志artifacts/s2-text-lora-001/image-budget1120-screen-001.log；研究wrapper+v2 adapter+1120预算，500题单遍，报告reports/s2-image-budget1120-dev500-screen-001.json。尚非阶段验收。
- 原生子代理image_budget_config独立worktree /Users/bamboo/.codex/worktrees/image-token-budget/Lite-OmniJev，负责train/eval/HF服务显式预算配置及测试，按ADR0019；无模型资产修改，默认仍官方值，训练报告/checkpoint追踪非默认。主会话负责集成、GPU验收、票据。
- 下一步：收取500题结果与子代理代码；若达.806，正式配置复验双遍+cf、screenshot及text，统一门报告；若不达则保留对照结论，不能重复扫预算。s2:01保持claimed，S3依赖不解除。

## 授权与研究
- 原生goal active；用户强调按开机总时长计费，CPU准备与GPU重叠、停止无收益路线、避免重复验证/空转。
- 最新授权ADR0015：训练/RL/架构/训推/产品均可自主提出并验证创新，RL时点由负责人决定，不硬性排在微调之后；社区作对照，创新/SOTA须证据。目标先产品达标，再吸收有益创新。
- 官方Jev公开称RLCD但未披露奖励公式/优化器；不混同社区OneJev CE+Brier或臆称复现。原R1需明确动作相关奖励及梯度路径，见ADR0017；当前实际作业仅监督训练。
- 不push/付费/自动开关卡/生产部署；许可和封存隔离保持。暂无需用户操作的阻塞；凭据不入仓。GPU链已启动，未请求关卡。

## 本轮并行准备
- E08原训练PID266665及原链已结束。CPU开发题型审计reports/s2-question-family-audit-001.json：颜色19/计数18/why54/其它409，属于粗词法分组，不当语义真值。
- E09官方1120视觉token预算研究对照已启动（E08完整筛查未过门），见reports/explore/S2-E09-official-image-budget.md；独立进程覆盖processor参数，不修改模型资产/默认评测/运行中代码。单train样本CPU280→1120处理预检通过：实际image token260→1107，prompt339→1186。远端/root/autodl-tmp/image-budget1120.sh已准备，E08无收益且原链终止才timeout300启动。
