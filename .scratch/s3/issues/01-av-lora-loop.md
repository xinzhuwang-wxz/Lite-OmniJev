# 01: 音频/视频数据→联合决策LoRA→dev500同条件评测闭环

ID: s3:01
Milestone: S3
Category: enhancement
Status: ready-for-agent
Execution: claimed
Owner: main-session
Worktree: /Users/bamboo/Githubs/Lite-OmniJev
Blocked by: s2:01

## What to build

按.scratch/s3/spec.md完成:联合池(五模态21184行)→音频/视频训练分派→joint主训练→audio/video/image/screenshot/text五重dev500评测(含8/16帧对照)→S3机器门报告。

## Acceptance criteria

- [x] 联合池冻结:s2-joint 12559+audio 326+video 8299=21184行,拼接报告绑定五输入SHA,双遍字节一致;训练入口五ledger并集校验。
- [x] 按ADR0021调整执行预算：100步五模态均衡回放真实GPU完成，loss下降、全部1000例前向；未冒称完成原300步smoke。
- [x] 主候选1000次曝光（每模态200）可追溯，step50 midway/step100 checkpoint、SwanLab真实指标；原10000次预算由ADR0021替代，质量门不变。
- [ ] dev500双遍评测:audio与video各双遍(Noul三向读出),image/screenshot/text各双遍(旧模态回归);视频8帧对照一套。
- [ ] 门槛报告:eval.gate原值判定原样记录(audio≥34.2%、video≥62.8%、旧模态≤1pp、frame_ab);未达标如实保留,结构性不可达按ADR0013格式另录。
- [ ] lint/type/build+新增回归;STATUS更新;commit带Lore trailers。

## Evidence

Evidence: reports/s3-joint-preparation-001.json：21184行五模态联合池及500行均衡探针，五ledger核验通过，两次生成数据哈希一致；reports/s3-probe-preflight-001.json官方处理器500例预检通过（最大1181 token）；仅CPU准备，S2依赖保留；规格.scratch/s3/spec.md。池证据artifacts/s3-pools-{audio,video}-001(远端);候选驱动与扩展扫描日志留远端。基线:教师Jev-Omni audio 38.0%/video 69.8%,E4B零样本audio 63.4%/video 55.6%(s0-baseline-summary.md)。

## Out of scope

量化/E2B(S6)、封存holdout构建(s0:05)、新数据源、门槛静默调整。

独立准备证据：656ff65按需预处理，86项CPU训练测试通过；reports/s3-lazy-parity-comparison-001.json真实GPU同30步loss与adapter哈希完全一致、cache峰值低于128MiB。仅工程可行性，不解除S2依赖、不代替S3训练质量门。
