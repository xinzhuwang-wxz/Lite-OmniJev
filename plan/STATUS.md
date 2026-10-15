# STATUS

更新：2026-10-15，S3图像与16帧视频正式评测已完成。**S2已通过原四质量门；S3正式评测链运行中。原生goal仍active，产品尚未交付。**

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
- 16帧video500双遍完成：product_accuracy63.6%（318/500）≥62.82%，repeat相同，723.67秒；reports/s3-dev-eval-video-stable-001.json及1000条sidecar审计已收取。当前截图子作业timeout PID583291，后接text/audio/video8。
- 输出artifacts/s3-dev-eval-{image,video,screenshot,text,audio,video8}-stable-001；报告reports/s3-dev-eval-同名-stable-001.json。新runner有progress.json与records.partial.jsonl，只供观察，完整通过报告才作门槛证据。勿重复启动。
- s3:01 claimed owner main-session。下一步收取真实全门报告，用scripts/s3_gate与configs/gates.json组装；旧模态基线是S2 1120的image/screenshot报告及原text v2-corrected报告。

## 保留决定与未完成项
- 原S3 lr1e-4/30步候选audio70→52、video62→58、旧图像/截图各降2pp，已停止。v2 video500单遍53.4%<62.82%，需要训练。已有输入/forward/lazy一致性证据复用。
- S2 E01–E08记录均保留；E08新family模型79.8%未达门，正式采用v2+1120，不能把E08与v2结果混拼。
- 用户允许各阶段创新、RL时点自主决定；快/低成本仍是核心目标。RL尚未运行，官方RLCD未公开配方，不臆称复现。探索有界、小批量有收益才扩大。
- S4预算、S5校准/封存、S6量化/E2B/API单卡端侧发布与真实浏览器/性能最终验收尚未完成。生产部署、付费、租卡开关卡、不明许可仍不自动操作；不push。
- 凭据仅项目外。暂无需用户操作的阻塞，GPU链运行，未请求关卡。独立CPU检查工作树/root/autodl-tmp/image-budget-cpu-check-001保留原始证据，未影响GPU checkout。

## 本轮新增证据
- 发现并修复Noul统计tie顺序：JSON按字母排序导致metrics使用no/yes，与产品规范yes/no不同；不改变前向、产品预测或S3门。2条回归先失败、修复后41项通过，ruff/format/mypy通过。运行中远端仍b6b827b不切代码。
- reports/s3-video-metrics-correction-001.json基于完整sidecar CPU复算：15条top概率tie，通用accuracy .626→.636、ECE .126459→.125338，product_accuracy始终.636；原GPU报告不覆盖。后续audio/video8同样生成修正汇总；正式S3 gate原本使用product_accuracy不受影响。全sidecar审计须在旧远端metrics环境完成后再同步代码，或显式核验原值/修正值，勿以不匹配默认为GPU坏结果。
- 私有CPU作业统计全部已下载Clotho train/val/test，reports/s3-audio-reserve-capacity-001.json只导出汇总：未分配1613题/670family，已有calibration196题/80family、holdout107题/40family；合计1916道上界，距3000+1000至少缺2084。还未验证完整历史/时长/media别名，不作准入；不改变池、不读取预留payload文件、不导出源题/答案。需后续补充可靠来源，ADR0004允许已准备TTS但不能冒称真实音频或重复计数。
- sidecar审计负向验证reports/s3-sidecar-audit-rejections-001.json：有效复制件接受；重新算SHA后篡改真值、概率、缺记录、错adapter均拒绝。仅一次性dev复制件，原GPU产物未改。
- CPU完成S3图像1000条原始记录审计：输入/权重/sidecar哈希、真实dev目标、逐条logits概率读出和汇总指标一致；reports/s3-image-sidecar-audit-002.json仅partial，不代S3全门。.scratch/s3/verify-final-sidecars.py已准备全六组审计，远端外置脚本/root/autodl-tmp/s3-verify-final-sidecars.py，未改运行中checkout。
- 训练曝光重建reports/s3-training-exposure-audit-001.json确认100步/1000次、每模态200次，image199family、其余200；首/末10步loss .722695→.207186。不是原300步/10000次预算完成，正式门通过后明确执行调整ADR再关票。
- 下一阶段独立准备reports/s4-gold-gate-implementation-assessment-001.md已整合；识别四池全历史family/media/component联通审计和私有作业日志隔离缺口。未实现/运行真实gold gate，未读封存，子代理已结束，工作树归档。
- S3 image dev500双遍完成：82.0%、两遍一致，与S2持平，reports/s3-dev-eval-image-stable-001.json；其余cohort和总门仍待完成。396.30秒是离线双遍耗时，不是API延迟。
- 用户再次强调快/低成本：发布配置按质量达标前提下的真实API延迟、吞吐、显存和单位请求成本选择，1120仍非最终低成本默认。
- CPU仅统计分池ledger，未读calibration/holdout载荷：reports/s3-reserve-inventory-001.json。text储备3000/1000；image623/211、screenshot120/60、audio80/40、video900/300（校准/封存题数），crossmodal未识别冻结ledger。S4/S5须补齐许可清晰且family隔离的可靠gold，不把现有储备称为就绪、不降数量门槛。
- S3原高学习率失败候选已停止；当前小更新候选有真实video收益且小样本未明显遗忘，扩大正式验证。
- b6b827b修复百分点边界：按十进制准确率相减，避免0.01被表示成0.010000000000000009误判；没有给比较器加宽容差，74项门测试确认刚好1pp通过/1.2pp拒绝，原阈值不改。
- 评测分段日志c243ed3已随同步进入当前GPU作业；源输入/前向顺序未变，主树60项回归通过，真实开销待本轮观测。
- reports/s3-video-error-audit-002.json为CPU诊断：训练标签平衡，AUC .578，temporal169/329；未据开发集拟合输出阈值，未读封存。
