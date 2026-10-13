# 06: 教师音频临时输出的无卡修复

ID: offline-prep:06
Milestone: S0
Category: bug
Status: ready-for-agent
Execution: claimed
Owner: teacher-audio-cpu
Worktree: /tmp/lite-omnijev-teacher-audio-cpu
Blocked by: p0:04

## What to build
固定教师源码以NamedTemporaryFile创建输出后调用未加-y的FFmpeg，可退出0却留下0B WAV。保持官方原件及教师结构，提供项目持有的音频预处理适配入口，并验证输出有效性；原音频30秒/mono/16kHz契约保持。

## Acceptance criteria
- [ ] 从自有合成音频经公共适配接口完整解码成功，含已有空输出的真实FFmpeg回归；不依赖模型/GPU。
- [ ] 仅覆盖适配器独占的临时文件，失败拒绝空/不可读输出，不触碰用户文件；错误/超时可清理自己临时资源。
- [ ] 官方固定源码保持原SHA；明确后续教师调用接入方式，未接入/模型等价部分保持pending。
- [ ] 定向行为测试、Ruff/format/typecheck通过，实跑命令/环境/退出码证据入报告；不宣称教师forward通过。

Evidence: reports/jev-omni-mvbench-protocol.json；docs/research/jev-omni-mvbench-protocol.md。Mac FFmpeg8.0.1原样exit0/0B，-y候选有效3278B；Linux补充排队并非通过。
