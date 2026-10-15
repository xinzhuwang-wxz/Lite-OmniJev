# STATUS

更新：2026-10-15（Asia/Shanghai，17:00恢复）。**S2 v2三评测完成，原门3/4通过：image78.8%仍差1.8pp；S2未关闭。反事实统计bug已修复，cf88.6%通过，ADR0014结构性结论失效。S3 CPU准备通过。**

## 已验证结果（不重跑）
- S1全门：reports/s1-stage-acceptance-001.json（dev50099.6%，ADR0013）。
- S2 v2：image78.8%/cf88.6%、screenshot96.6%/cf98.4%、text99.4%，各500题双遍逐记录一致；reports/s2-dev-eval-*-v2-corrected-001.json。
- S2门：reports/s2-stage-acceptance-v2-001.json绑定哈希，eval.gate退出1；只image_vs_onejev_delta失败(-4.8pp<−3pp)，其余通过。禁止以旧ADR0014跳过质量门。
- 反事实修复bbdc041：65项相关测试、ruff/mypy通过；历史sidecar哈希重算v1 cf91.8%/96.6%，reports/s2-counterfactual-rescore-001.json。
- E01中途检查点：step800 dev150=121/150，低于step1600=123/150；34.65秒，按停止条件弃用，不跑全量。reports/explore/S2-E01-checkpoint-selection.md。
- S3汇总入口scripts/s3_gate.py与18项行为/CLI测试完成；同版本、数据一致性及8/16帧校验；uv build通过。
- S3准备：reports/s3-joint-preparation-001.json，21184行联合池+500行均衡探针、五ledger通过、两次生成一致；reports/s3-probe-preflight-001.json官方处理器500例通过、最大1181 token、94.66秒。未开始训练。

## 当前作业与恢复
- 原生goal active；s2:01 claimed，owner main-session，工作树 /Users/bamboo/Githubs/Lite-OmniJev。
- SSH root@region-42.seetacloud.com:48926，mux /tmp/omnijev-ssh-root-final；凭据仅环境/会话，不入仓。
- 远端 /root/autodl-tmp/s1-text-lora-001 代码932ade4；v2及历史链已完成。冲突原报告保存在 /root/autodl-tmp/s2-evidence-recovery/pre-bbdc041/，未stash。
- E02已完成124/150，只多对1题、未达125/150扩量条件，停止；35.19秒H20。报告reports/s2-dev-image-average-v2-probe-001.json。
- v2 train-image150诊断完成97.33%，35.43秒；reports/s2-train-image-v2-diagnostic-001.json。E03定位为低LR模态再平衡泛化实验，不盲目加步数。
- S3音频/视频各10条训练样本GPU前向canary通过：reports/s3-v2-{audio,video}-forward-canary-001.json（20.31/22.77秒）；仅管线，不是质量门。
- warmstart_impl已交付并整合932ade4：训练初始化绑定adapter+SHA、检查配置/权重/冻结主干、checkpoint保存谱系；96项集成回归及ruff/mypy通过。
- E03完成100步，dev150=122/150，无改善停止；训练+评测382.13秒（0.10615 H20卡时），reports/s2-image-focus-{run,dev150}-001.json。
- E04金标平滑链PID72782运行：/root/autodl-tmp/image-smooth-chain.sh；日志artifacts/s2-text-lora-001/image-smooth-chain-001.log；100步+image150，timeout20分钟。报告s2-image-smooth-run-001.json和s2-image-smooth-dev150-001.json，输出image-smooth-001。600图像目标epsilon=.1、gold始终最大、text/screenshot硬目标保留，三ledger检查通过。
- 候选adapter artifacts/s2-text-lora-001/average-0800-1600-001/adapter；由step800/1600矩阵50/50平均，仅CPU导出，无重训；方案reports/explore/S2-E02-adapter-average.md。

## 下一步
1. 检查E04链与报告，不重复启动；dev150至少125/150且ECE不差于v2同切片才扩大旧模态探针，再决定完整验收。失败/无改善停止这次候选；续训须重复--init-adapter及固定SHA以保留谱系。
2. 图像质量门未过则继续票内有据的小实验；保留原失败。S2全门与集成证据通过后才关闭票、解除S3依赖。
3. S3候选配置与准备脚本在.scratch/s3/；先小批量GPU前向/训练对照，再主训练。同步代码须等待在跑链完成，保留远端未跟踪报告。
4. 旧handoff中的cf结构性结论/直接关票指令过时，以本状态和修正ADR0014为准。

## 授权与待我决策
- 新授权ADR0015：按阶段/实测瓶颈自主考虑成熟迁移、组合创新、探索与新idea；先写预算/对照/停止条件，小批量有收益再扩量。5b多步仍需范围决策。
- 不push/付费/自动开关卡/生产部署；不改基座/分类头/推理引擎；封存集仅接收gate汇总。质量目标调整须独立实测依据，不能贴合当前分数制造通过。
- 暂无需要用户操作的阻塞。GPU实验结束记录资源并通知；不自动关卡。
