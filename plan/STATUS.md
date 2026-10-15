# STATUS

更新：2026-10-15T19:17:09.310866+08:00。**S2已通过原四质量门；S3小规模训练链已启动。原生goal仍active，产品尚未交付。**

## 当前阶段与证据
- s2:01 done；reports/s2-stage-acceptance-budget1120-001.json：image82.0%、screenshot96.4%、text99.4%，cf93.8%/98.6%，全部500题双遍一致。5000条原始记录复核；冻结E4B ee0ef602…、v2 adapter db45fb13…、图像预算1120。
- 文本复用原同权重/数据/前向/运行库双遍报告；本次总链1800秒在重复文本时中断，不记通过。reports/s2-text-evidence-reuse-001.json。图像与截图的新报告完整。
- 高预算HF/native CPU100例精确一致；实际HTTP LoRA概率和API时延/吞吐/显存仍pending，未将1120定为最终低成本默认。
- 独立review两P2已处理，ADR0020记录joint执行调整；全仓lint/410文件format/104源文件mypy/build通过。新S3 gate严格绑定五cohort冻结数据和模型身份。
- 里程碑交接入口：plan/handoff-2026-10-15-s2-complete.md。无活跃子代理。

## 当前唯一GPU链
- 远端/root/autodl-tmp/s1-text-lora-001已在上一链结束后同步至b6b827b；SSH mux /tmp/omnijev-ssh-root-final，端口48926。运行中不切代码。
- 100步S3候选训练完成843.25秒、峰值显存30.14GB，adapter1bebae31309f799395d01499fb9deec5e44622599d0c082473ea8b5a014e5755，artifacts/s3-stable-probe-train-001/adapter。
- 小筛查全部通过：video99/150（基线77）、audio38/50、image40/50、screenshot50/50、text49/50。reports/s3-stable-probe-screening-001.json qualified=true，仅允许扩量，不代表S3通过。
- 当前/root/autodl-tmp/s3-full-gates.sh，外层timeout4200、各cohort独立时限；日志artifacts/s3-stable-probe-001/full-gates.log。顺序image→video→screenshot→text→audio（各dev500双遍）→video8（500单遍）。image/screenshot1120、video16/8，失败即停；阈值仍原S3，不扫配置。
- 输出artifacts/s3-dev-eval-{image,video,screenshot,text,audio,video8}-stable-001；报告reports/s3-dev-eval-同名-stable-001.json。新runner有progress.json与records.partial.jsonl，只供观察，完整通过报告才作门槛证据。勿重复启动。
- s3:01 claimed owner main-session。下一步收取真实全门报告，用scripts/s3_gate与configs/gates.json组装；旧模态基线是S2 1120的image/screenshot报告及原text v2-corrected报告。

## 保留决定与未完成项
- 原S3 lr1e-4/30步候选audio70→52、video62→58、旧图像/截图各降2pp，已停止。v2 video500单遍53.4%<62.82%，需要训练。已有输入/forward/lazy一致性证据复用。
- S2 E01–E08记录均保留；E08新family模型79.8%未达门，正式采用v2+1120，不能把E08与v2结果混拼。
- 用户允许各阶段创新、RL时点自主决定；快/低成本仍是核心目标。RL尚未运行，官方RLCD未公开配方，不臆称复现。探索有界、小批量有收益才扩大。
- S4预算、S5校准/封存、S6量化/E2B/API单卡端侧发布与真实浏览器/性能最终验收尚未完成。生产部署、付费、租卡开关卡、不明许可仍不自动操作；不push。
- 凭据仅项目外。暂无需用户操作的阻塞，GPU链运行，未请求关卡。独立CPU检查工作树/root/autodl-tmp/image-budget-cpu-check-001保留原始证据，未影响GPU checkout。

## 本轮新增证据
- S3原高学习率失败候选已停止；当前小更新候选有真实video收益且小样本未明显遗忘，扩大正式验证。
- b6b827b修复百分点边界：按十进制准确率相减，避免0.01被表示成0.010000000000000009误判；没有给比较器加宽容差，74项门测试确认刚好1pp通过/1.2pp拒绝，原阈值不改。
- 评测分段日志c243ed3已随同步进入当前GPU作业；源输入/前向顺序未变，主树60项回归通过，真实开销待本轮观测。
- reports/s3-video-error-audit-002.json为CPU诊断：训练标签平衡，AUC .578，temporal169/329；未据开发集拟合输出阈值，未读封存。
