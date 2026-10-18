# 01: 恢复并分发release-002可核验候选

ID: s6-release-serving:01
Milestone: FINAL
Category: enhancement
Status: ready-for-agent
Execution: claimed
Owner: main-session
Worktree: /Users/bamboo/Githubs/Lite-OmniJev
Blocked by: None

## What to build
接续2026-10-18交接，将既有最终候选的发布工件、真实启动、基础工作台和材料完成收口。
本票完成不代表delivery:01、S5或S6原门全部通过；完整产品化缺口仍需单独实证。

## Acceptance criteria
- [x] 5平面分片、2130张量、逐字节来源一致及SHA256回执。
- [ ] 该分片真实vLLM启动；三类型/五模态基础浏览器操作与cURL重放。
- [x] 产品默认不输出U，内部诊断显式配置；未校准服务如实标记unfitted。
- [ ] 发布模型卡、数据卡、训练记录、部署命令、许可/哈希与实证一致。
- [ ] HF允许列表文件上传并核验revision/大小/哈希；不混入密钥、日志或私有行。
- [ ] 定向测试/静态/构建及一次集成review，更新FINAL缺口并通知关卡。

## Latest scope
2026-10-18用户确认卡仍开着，要求尽快完成全部HF/GitHub产品化；候选冻结，不展开新训练。
使用现有卡对来源绑定的新启动脚本做整包100案例双遍复验，与分发上传并行。

## Evidence
reports/s6-flat-remap-recovery-001.json；reports/s6-gguf-modality-matrix-001.jsonl。
63+1网关/校准/数值脚本公共行为测试通过；2启动dtype测试及4中断恢复测试通过。

## Recovery
真实远端main=c4432b9，仅README；旧push5已结束且空日志，不能当后台作业。
remap收尾safe_open.close异常且glob遗漏__tmp前缀；逐张量核验已有完整临时文件后恢复，无重新下载或删除来源。
HF上传改既有token+allowlist的resumable HTTP路径，避免git-lfs复制耗尽磁盘；token不入仓。
