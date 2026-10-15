# STATUS

更新：2026-10-15（Asia/Shanghai，18:59）。**S2候选v2+1120预算image500单遍82.0%（门80.6%）；正式image双遍82.0%/cf93.8%精确重复已通过，截图/文本仍运行，尚未关票。高预算HF/native CPU100例精确一致通过；实际HTTP与服务性能待验证。**

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
- SSH root@region-42.seetacloud.com:48926；mux /tmp/omnijev-ssh-root-final；远端 /root/autodl-tmp/s1-text-lora-001 代码ea3695e，运行中不切代码。
- E07已结束：CE123/150、CE+Brier124/150，未达125扩量条件；reports/s2-vqa-{ce,brier}-{run,dev150}-001.json，停止扫参。S2仍78.8% vs80.6%，未放行。
- S3 lazy/eager真实GPU一致性完成：30步loss完全相同、adapter SHA相同e4caf62d…；243→213秒；cache峰值134011035 bytes<128MiB，GPU峰值28.64GB。reports/s3-lazy-parity-comparison-001.json；不是S3质量通过。原作业PID243896已终止。
- 全池21184行lazy CPU预检完成：29.54秒、进程峰值1559654400 bytes；reports/s3-lazy-full-preflight-001.json及resource报告。仅全量metadata+4个代表样本，不宣称全部媒体读取。
- E08准入完成：400新family，颜色160/数字160/yesno80；排除原AOKtrain全部family、现有ledger与3832张图片SHA。首批下载400张62.63MB/252秒，其中coco:265843解码截断，已记录排除并确定性补选1张。最终plan artifacts/s2-new-family-plan-004，候选SHA8a9cf9a5…；媒体未删除。
- 构建输出artifacts/s2-new-family-gold-001，train SHA7d0f1583…；混合池artifacts/s2-new-family-probe-001为1000行（新image400/旧image400/text100/screenshot100），SHA88dab83e…，四ledger验证通过。构建器2508a6f远端16测试通过；无活跃子代理。
- E08训练完成100步/357.78秒，adapter d5ca009c26495a1fc5bd8c9eb84da554b436c629a8cef2cf7c6ff22545c2490c；dev150=125/150（83.33%）、ECE .14844、35.44秒，达到扩量规则。reports/s2-new-family-ce-{run,dev150}-001.json。
- E08完整image dev500单遍=399/500=79.8%，较v2提高1pp但低于80.6%；停止此候选完整阶段扩量，保留adapter。reports/s2-new-family-ce-dev500-screen-001.json，77.97秒。当前最好图像单遍不代表三模态整体最优。
- E09 dev150=126/150=84.0%，ECE .14332，68.44秒；同v2默认123/150，达到扩量条件。reports/s2-image-budget1120-dev150-001.json。
- E09 image dev500单遍=410/500=82.0%，ECE .14820，194.81秒；reports/s2-image-budget1120-dev500-screen-001.json。超过图像门但未完成阶段验收。
- 正式配置ea3695e已整合并同步：训练eager/mixed/lazy、eval --image-max-soft-tokens、gateway LITE_IMAGE_MAX_SOFT_TOKENS与SHA绑定的vLLM images_kwargs；默认checkpoint身份保持。主树83测试通过/15真实fixture跳过；子代理160通过/53跳过，native CPU100例已精确通过，实际HTTP高预算概率与性能待验证。
- 当前唯一GPU链：/root/autodl-tmp/budget1120-full-gates.sh，timeout1800；日志artifacts/s2-text-lora-001/budget1120-full-gates-001.log。原v2 adapter，image/screenshot预算1120，各500题双遍+cf；text500双遍。每cohort检查质量/repeat失败即停。报告reports/s2-dev-eval-{image,screenshot,text}-v2-budget1120-001.json；输出同名artifacts/s2-text-lora-001目录。运行中不切远端ea3695e。
- 独立CPU工作树/root/autodl-tmp/image-budget-cpu-check-001，代码60528f2：HF100例41.03秒/native vLLM0.26 100例70.13秒均passed，精确token/tensor匹配；reports/s2-budget1120-{hf,native}-cpu-001.json。本机CPU工作树无需保留训练产物，暂留证据；不加载权重、不占GPU。
- e48dd73加固S2报告同模型/adapter、冻结devSHA、图像预算一致性，S3回归也拒绝预算变化。52项门槛测试通过，历史v2指标逐值不变。GPU主树仍ea3695e，运行中不合并。
- 独立review已完成，无新预算/训练代码阻断；两项P2已处理：ADR0020用真实曝光记录替代过时独立训练清单；S3 gate绑定全部五cohort冻结SHA与严格权重身份，拒绝研究override。72项门槛回归通过，reports/s2-integration-review-001.md。无活跃子代理，review工作树已请求归档。
- 实际joint曝光审计reports/s2-training-exposure-audit-001.json：smoke300步image/screenshot各750曝光；v2 1600步image3108/screenshot930，覆盖1341/420全部family，step800对照齐全。没有声称完成原独立300步两条训练，也未改质量门。
- 里程碑工程检查：全仓ruff、410文件format、104源文件mypy、wheel/sdist构建通过。21个实验驱动格式整理经AST/compile核验；历史运行哈希保留，未重复GPU实验。
- image正式报告reports/s2-dev-eval-image-v2-budget1120-001.json已拉取：82.0%、cf93.8%、双遍逐记录一致。链继续screenshot与text，输出路径同上；结束后主会话用最新本地S2 gate组装机器门。远端运行目录仍ea3695e，勿合并。

## 授权与研究
- 原生goal active；用户强调按开机总时长计费，CPU准备与GPU重叠、停止无收益路线、避免重复验证/空转。
- 最新授权ADR0015：训练/RL/架构/训推/产品均可自主提出并验证创新，RL时点由负责人决定，不硬性排在微调之后；社区作对照，创新/SOTA须证据。目标先产品达标，再吸收有益创新。
- 官方Jev公开称RLCD但未披露奖励公式/优化器；不混同社区OneJev CE+Brier或臆称复现。原R1需明确动作相关奖励及梯度路径，见ADR0017；当前实际作业仅监督训练。
- 不push/付费/自动开关卡/生产部署；许可和封存隔离保持。暂无需用户操作的阻塞；凭据不入仓。GPU链已启动，未请求关卡。

## 本轮并行准备
- E08原训练PID266665及原链已结束。CPU开发题型审计reports/s2-question-family-audit-001.json：颜色19/计数18/why54/其它409，属于粗词法分组，不当语义真值。
- E09官方1120视觉token预算研究对照已启动（E08完整筛查未过门），见reports/explore/S2-E09-official-image-budget.md；独立进程覆盖processor参数，不修改模型资产/默认评测/运行中代码。单train样本CPU280→1120处理预检通过：实际image token260→1107，prompt339→1186。远端/root/autodl-tmp/image-budget1120.sh已准备，E08无收益且原链终止才timeout300启动。

用户最新强调：快、低成本仍是核心产品目标；1120预算是已出现质量收益的候选，不是最终默认。正式全质量验收后必须联评服务性能/成本，批评测时长不冒充API延迟；不未经证据宣称优于Jev。

## S3下一候选就绪（未启动）
- reports/s3-stable-probe-preparation-001.json：五cohort各200行共1000、五ledger验证；数据SHA594be68b…、config SHA5c52f0b6…，远端artifacts/s3-stable-probe-001。
- CPU预检reports/s3-stable-probe-preflight-001.json ready；lr2e-5、100步、lazy128MiB、图像1120/视频16帧。GPU入口/root/autodl-tmp/s3-stable-probe-train.sh硬性要求reports/s2-stage-acceptance-budget1120-001.json为passed/S2/exit0，当前无此通过报告，未运行训练。
- 有界试验与退出条件见reports/explore/S3-E01-small-update-replay.md。S2完成后先提取现成video500 sidecar的固定150题基线，再启动训练与筛查，不重复GPU基线。
