# 05: 交付现成gold封存数据与自动评测gate

ID: s0:05
Milestone: S4
Category: enhancement
Status: ready-for-agent
Execution: claimed
Owner: main-session
Worktree: /Users/bamboo/Githubs/Lite-OmniJev
Blocked by: p0:04

## What to build
按ADR0012从已准备/可取得的许可明确来源选用已有gold或程序真值，交付独立holdout及自动汇总gate。保留ID供历史依赖；不再是需用户真人/账号的外部阻塞。当前已进入S4；沿用S1前预留pool/family，补齐真实数量、来源和自动gate后才允许S4放量。

## Acceptance criteria
- [x] 六cohort各1000独立holdout、各500dev来源/许可/revision/标签规则/hash可追溯；不得把现有dev/公开已调参题兼作封存，数量不足如实报告。
- [x] 真值为已有可靠数据集gold、公开benchmark gold或可验证程序真值；拒绝缺失/非法标签和把教师输出冒充评测gold。新增真人/双标/κ均不要求，κ为not_applicable。
- [x] train/dev/calibration/holdout按原始family及媒体别名去重，历史dev排除完整；S1前预留未训练family，禁止事后删训练记录伪造隔离。
- [x] 自动gate独立作业读取封存内容，开发agent仅收数量/hash/来源/质量与模型指标汇总；实际权限/日志/错误拒绝测试通过，不依赖用户提供外部人工账号，不把root同机chmod当外部安全隔离证明。
- [ ] 绑定真实数据、固定配置/种子/代码与运行环境，S4/5/6评测汇总可复现；模拟canary不代实际数据验收。S4放量及FINAL依赖本票。

## Evidence
现有四池family机制、历史dev union与来源manifest可复用。旧v2人工双标canary仅历史工程证据，当前不是首发必走合同；新的实际gold gate尚未实现，不宣称通过。详docs/adr/0012-existing-gold-without-new-human-labeling.md与docs/runbooks/sealed-gate.md。

2026-10-15：私有清单作业已实现并实跑，92项相关回归通过。40个输入文件覆盖已索引历史训练、五模态当前预留、六cohort dev与旧截图版本、crossmodal组件。首次扩展发现历史dev与截图calibration/holdout各一个连通组冲突；完整保留train/dev及原预留文件，在新派生清单隔离两条后复审零冲突（reports/s4-development-inventory-005.json、s4-reserve-quarantine-summary-001.json、s4-inventory-provenance-001.json）。这不是数量/来源gold/全历史完整性/权限验收；截图暂余119/59，仍需补足。后续只从远端artifacts/s4-gold-inventory-001/with-development-clean-reserves-manifest.json接续，并沿用私有quarantined-reserves/exclusions.json，禁止直接重用旧截图预留。

2026-10-15：新增官方CLEVRER validation真实gold储备3000 calibration/1000 holdout，一视频一原始题、固定seed42，所有输入SHA绑定。与40文件历史清单合并后113938观测、零跨池连通组。证据reports/s4-clevrer-gold-build-001.json、s4-clevrer-gold-isolation-001.json、s4-clevrer-gold-provenance-001.json。接续私有with-clevrer-gold-manifest-001.json；其余cohort与权限/完整准入未完成，整体验收不勾选。

2026-10-15：音频真实构建3000cal/1000hold（Clotho1400/500+speech1600/500），原120Clotho预留题完全保留，旧文件作为历史曝光保留。83项构建回归通过；reports/s4-audio-gold-isolation-002.json合并历史零跨池冲突，provenance001绑定1f02a8a代码包。最新私有总清单with-audio-gold-manifest-001.json SHA0d583307…；speech原始别名在allocation.jsonl。截图仅补179标注文件后测得4395新原题，容量足够，构建/完整权限及其余cohort仍未通过。

2026-10-15：GUI实际构建001拒绝；历史实际使用4141图片RGB审计零跨池冲突，已修复未使用帧引入虚假曝光边并启动002。VQA唯一图选择修复后计划002为4500唯一图；官方HTTPS503后003按固定COCO元数据原始HTTP传输执行。远端37项GUI/VQA回归通过，代码5096382；GUI002已成功构建3000cal/1000hold，178原预留完整保留，91.44秒；报告s4-gui-gold-build-002.json。VQA下载结果待收集，完整准入仍不勾选。

2026-10-15：CLINC原始Parquet全链重放匹配原dev/train/3000cal/1000hold哈希（s4-text-gold-source-replay-001.json）；GUI002全历史隔离通过。跨模态及媒体整理构建器集成126测试通过，实际组装待VQA下载完成；冻结策略.scratch/s4/crossmodal-gold-policy.md。VQA003仅因20分钟预算结束，1161图已校验复用，004续传剩余3339图。

2026-10-15：音频媒体整理实跑4000引用，2991文件全部SHA校验；原始artifact保留historical后合并全历史126438观测零跨池冲突，训练原resolver4000引用通过。新总清单with-audio-relocated-inventory-001.json SHA48d41081…；媒体与隔离报告均为001，非完整gold/权限准入。

2026-10-16：VQA4500图全量齐全并materialize3000cal/1000hold/500dev通过；crossmodal3000cal/1000hold组装通过，依赖组610/206单列，完整隔离142938观测零跨池。独立UID固定audit002和真实live/reused UID/保留目录访问拒绝探针通过，001权限claim已纠正；完整source准入及日志错误注入未完成。补充crossmodaldev002成功新增500题，原500dev保留；当前总清单crossmodal-dev-002/inventory-manifest.json SHA6eb32249…。

2026-10-16：最新总清单RGB002（12463图像/144938观测）零跨池，private-job003最新清单通过，Linux真实失败/日志探针002的13项通过，001脚本API错误保留。结构组合器73测试通过；实际选集/source replay映射准入尚待组装，不能凭元数据声明关票。

2026-10-16：数据侧验收1–4通过（范围为现成gold数据与独立数据审计job）。reports/s4-gold-data-admission-001.json绑定所有实际证据：六cohort各3000cal/1000hold及原dev500、源代码128绑定、source链154工件、Clotho120完整映射、13500行仅URI变化、已知34训练报告/18输入/7dev版本覆盖、RGB/权限/错误探针。第5项仍待固定私有模型推理入口及真实GPU工程验证；不把数据准入当S4模型质量通过。

## Out of scope
新增真人标注/招聘/付费外包、要求用户提供真人或外部gate账号、模型标签冒充gold、开发agent读取封存载荷、提前S4模型评测。
