# P0 GPU 接续：D004数值兼容边界

先读plan/STATUS.md和plan/proposals/p0-numerics-decision.md。当前P0未完成，不进入S0；native goal仍active。

- 已通过的训练、完整resume、浏览器API契约及其报告见STATUS；不要重做下载和训练。
- 关键新证据：reports/p0-fp32-base-full-comparison.json，100题max原始概率差7.9049e-6；BF16 base/adapter TRITON报告都失败。
- D004异步问题已发送；未收到明确批准不能应用三行兼容提案，不得静默改阈值或服务器合并LoRA。
- 当前22张票均无frontier，p0:04因D004 blocked，root已释放认领；S0草案已整合commit01b895c。
- 当前实例仍开卡；自建GPU测试进程和失配网关已清理，未关机/关卡。SSH使用既有ControlPath，凭据不在本文件。
- 本地artifacts/debug/gpu-20261013/remote-diagnostics.tar.gz保留探针/命令/日志；artifacts/browser/p0-fp16-001保留真实截图trace；训练完整备份路径见STATUS。
- 所有子代理已结束，成果保存后已清理本轮三个worktree；无GitHub写操作。

## Suggested skills
- diagnosing-bugs：使用已有小样本红色复现，先区分后端、精度与LoRA变量。
- verify：隔离候选通过后完整base+adapter100题，再浏览器验收，不能用小样本替代。
- handoff：每个闭环保存plan接续，用户目录要求优先于默认临时目录。
