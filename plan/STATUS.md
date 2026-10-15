# STATUS

更新：2026-10-15（Asia/Shanghai，18:08）。**S2图像原门仍未过：v2 78.8% vs80.6%；其它三门通过。停止旧池六条无扩量收益路线，新增1811条已许可/同train-family人工一致问题；新gold监督两臂均未达扩量条件，停止。S3训练技术路径已通，质量仍需改进。**

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
- SSH root@region-42.seetacloud.com:48926；mux /tmp/omnijev-ssh-root-final；远端 /root/autodl-tmp/s1-text-lora-001 代码656ff65，运行中不切代码。
- E07已结束：CE123/150、CE+Brier124/150，未达125扩量条件；reports/s2-vqa-{ce,brier}-{run,dev150}-001.json，停止扫参。S2仍78.8% vs80.6%，未放行。
- 当前唯一GPU作业：S3 lazy/eager数值一致性canary，PID243896，/root/autodl-tmp/s3-lazy-parity.sh；日志artifacts/s3-lazy-parity-001.log；输出artifacts/s3-lazy-parity-001，报告reports/s3-lazy-parity-001.json。复用原500样本/30步/同seed同v2初始化，仅改lazy+128MiB cache；比较原reports/s3-training-probe-001.json的losses及adapter哈希。不是S3质量扩量。
- 内存证据reports/s3-prepared-memory-probe-001.json：全池tensor估计272857235778 bytes，大于主机161061273600限制。656ff65实现按需准备/缓存上限，86项CPU训练回归通过。缓存上限不等于进程总RSS，真实GPU/CPU吞吐仍待本作业。
- 子代理bounded_media_training已完成并整合，暂无运行子代理。s2:01仍claimed owner main-session；s3:01仍依赖S2，仅做独立工程准备。
- 下一步：收取lazy canary比较同序训练数值；S2转向增加视觉family覆盖的可行性与错误机制证据，先核对许可/隔离/成本，不继续旧池局部超参循环。任何新family必须另记准入范围，ADR0016仅覆盖现有train-family。

## 授权与研究
- 原生goal active；用户强调按开机总时长计费，CPU准备与GPU重叠、停止无收益路线、避免重复验证/空转。
- 最新授权ADR0015：训练/RL/架构/训推/产品均可自主提出并验证创新，RL时点由负责人决定，不硬性排在微调之后；社区作对照，创新/SOTA须证据。目标先产品达标，再吸收有益创新。
- 官方Jev公开称RLCD但未披露奖励公式/优化器；不混同社区OneJev CE+Brier或臆称复现。原R1需明确动作相关奖励及梯度路径，见ADR0017；当前实际作业仅监督训练。
- 不push/付费/自动开关卡/生产部署；许可和封存隔离保持。暂无需用户操作的阻塞；凭据不入仓。GPU运行中，未请求关卡。
