# 无卡准备恢复交接

本交接记录已完成的外部原料和共享CPU输入闭环，不能作为开卡或S0通过通知。

先读 `plan/STATUS.md`、`plan/LOOP.md`、`docs/02-project.md` 和当前票。用户最新再次要求：把开卡前能做的都做完；实例已租，仍保持无卡。不重新租机，不自动开关卡/付费/push，不降低门槛、不更换基座。原生终局goal仍active。

## 可恢复证据

- `offline-prep:02` 外部原料闭环已关闭：见 `.scratch/offline-prep/issues/02-data.md` 和 `reports/project-data-offline-acceptance-001.json`。29个来源逐项区分已准备、批准排除、未来生成与外部gate；训练准入仍单独验证。
- GUI监督作业已结束0，`reports/gui-odyssey-full-decode-cpu-001.json` 与主树脚本SHA相符；勿重复启动78分钟全量解码。
- `s0:09` CPU共享输入闭环已关闭，见本票及 `reports/s0-prepare-integrated-acceptance-002.json`。canonical HTTP独立证据见 `reports/s0-canonical-http-acceptance-001.json`，真实媒体/官方processor/100×2，不含真实vLLM logits。
- 媒体间接引用和答案映射篡改已修复并独立复核，见 `reports/s0-media-protocol-review-001.json` 和42325d7。归档诊断脚本的既有SHA必须保留。
- 固定公开视频完整输入已冻结，见 `reports/public-video-protocol-freeze-001.json`；benchmark runner初版e39ede0正在按root审查修复，不能拿fakeforward作为效果证据。

## 继续工作

当前各worktree/owner以STATUS和原生agent实际状态为准，不靠本交接推定完成。主要剩余：026完整fresh离线重建、媒体网关、严格parity collector审查补强、teacher benchmark runner审查补强、缓存测量runner、实际native vLLM CPU processor canary及统一Linux快照复验。代码切片完成仍须主树整合/公共行为测试，最后才执行offline-prep:04。

冻结的历史HF/vLLM失败与未批准D004仍有效。GPU forward/质量/显存/内核/真实性能、人工gate/双人标注、端侧真机均未通过。S0全绿前禁止S1；不得将无卡准备阶段完成标为终局goal complete。

## Suggested skills

用 `verify` 核实每个具体完成主张；新的真实失败按最窄诊断路径处理。必要再次写交接可用本机 `handoff`，用户要求优先保存 `plan/`。不启动重型OMX/superpowers流程，也不重做需求访谈。
