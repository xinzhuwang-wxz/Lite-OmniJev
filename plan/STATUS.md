# STATUS

更新：2026-10-15T19:17:09.310866+08:00。**S2已通过原四质量门；S3小规模训练链已启动。原生goal仍active，产品尚未交付。**

## 当前阶段与证据
- s2:01 done；reports/s2-stage-acceptance-budget1120-001.json：image82.0%、screenshot96.4%、text99.4%，cf93.8%/98.6%，全部500题双遍一致。5000条原始记录复核；冻结E4B ee0ef602…、v2 adapter db45fb13…、图像预算1120。
- 文本复用原同权重/数据/前向/运行库双遍报告；本次总链1800秒在重复文本时中断，不记通过。reports/s2-text-evidence-reuse-001.json。图像与截图的新报告完整。
- 高预算HF/native CPU100例精确一致；实际HTTP LoRA概率和API时延/吞吐/显存仍pending，未将1120定为最终低成本默认。
- 独立review两P2已处理，ADR0020记录joint执行调整；全仓lint/410文件format/104源文件mypy/build通过。新S3 gate严格绑定五cohort冻结数据和模型身份。
- 里程碑交接入口：plan/handoff-2026-10-15-s2-complete.md。无活跃子代理。

## 当前唯一GPU链
- 远端/root/autodl-tmp/s1-text-lora-001，运行代码ea3695e；SSH mux /tmp/omnijev-ssh-root-final，端口48926。运行中不切代码，最新本地gate改动待后续同步。
- /root/autodl-tmp/s3-stable-chain.sh，外层timeout1800；日志artifacts/s3-stable-probe-001/chain.log。先s3-stable-probe-train.sh（内层timeout1200），后evaluate-stable-probe.py；训练入口已核验S2 passed，勿重复启动。
- 训练池artifacts/s3-stable-probe-001/train.jsonl，1000行（五cohort各200），SHA594be68b…，五ledger核验；lr2e-5/100步、lazy128MiB、image1120/video16、warmstart v2。检查点50/100；输出artifacts/s3-stable-probe-train-001，报告reports/s3-stable-probe-train-001.json。
- video150固定基线77/150（既有GPU预测CPU提取），扩量门95/150。后续筛查报告reports/s3-stable-probe-screening-001.json；video不达立即停，达标才audio/旧模态dev50。小屏幕不替代S3五模态500双遍/8帧对照。
- s3:01 claimed owner main-session；试验说明reports/explore/S3-E01-small-update-replay.md。下一步检查同一进程/报告，依据结果决定是否扩量。

## 保留决定与未完成项
- 原S3 lr1e-4/30步候选audio70→52、video62→58、旧图像/截图各降2pp，已停止。v2 video500单遍53.4%<62.82%，需要训练。已有输入/forward/lazy一致性证据复用。
- S2 E01–E08记录均保留；E08新family模型79.8%未达门，正式采用v2+1120，不能把E08与v2结果混拼。
- 用户允许各阶段创新、RL时点自主决定；快/低成本仍是核心目标。RL尚未运行，官方RLCD未公开配方，不臆称复现。探索有界、小批量有收益才扩大。
- S4预算、S5校准/封存、S6量化/E2B/API单卡端侧发布与真实浏览器/性能最终验收尚未完成。生产部署、付费、租卡开关卡、不明许可仍不自动操作；不push。
- 凭据仅项目外。暂无需用户操作的阻塞，GPU链运行，未请求关卡。独立CPU检查工作树/root/autodl-tmp/image-budget-cpu-check-001保留原始证据，未影响GPU checkout。
