# STATUS

更新:2026-10-18 09:0x(CST)。**负责人:交付链冲刺(主会话)。GPU 侧全部完成、已通知关卡;HF 推送后台跑;剩 ⑥文档终稿 + ⑦终局验收 + 与"打磨"(codex)协作。**

## ⚡ 接续点(第一动作)

1. **HF 推送(push5)监控**:盒 s6-hpush,log `/root/autodl-tmp/s6-hf-repo/push5.log`,完成=远端 main 到 bc2ca94(分片权重+MLX 4bit+GGUF 三档+adapter+回执,~61G)。multipart 大文件通道依赖用户 token(env 注入,未入仓)。
2. **打磨协作**:数据全齐后把 demo/图表/README 交"打磨"(codex session)做视觉与叙事终审;素材清单见下"现役资产"。
3. **⑥ 收尾**:README(GitHub ae520f1)已含赛跑 demo+图表+量化矩阵;s4-perf-report-002 待补 MLX/量化运行时行;⑦终局验收票待写。

## 数据终态(全实测,报告在盒 s6-release-serving-001/artifacts + 仓 reports/)

- ③ serving:parity 精确可重复 + 5.75e-6<1e-3;网关逐位一致;浏览器 live 过;SHA256SUMS 全验。
- ④ 量化:**文本一致性 f16/q8/q4 全 20/20 零退化**;Q4_K_M 5.34G(≈f16 的 1/3);RAM 实测 15/7.9/7.1G;视觉 image 队列三档 16-18/20(噪声级,screenshot 空输出=llama.cpp 运行时问题含 f16);llama.cpp 音视频验证 pending(--jinja 模板修复已验证,CPU 争抢超时);MLX 4bit **卡上转换完成** 6.2G(6.667bpw,运行测试 Mac 侧按用户决定暂缓)。
- ⑤ HF/GitHub:GitHub main=ae520f1(README+LICENSE CC-BY-NC-SA-4.0+demo+图表)已推;HF 推送进行中。
- ⑥ perf:评测态+serving 态+网关计时齐;量化运行时行待补。
- **Decision Index:外部阻塞**——官方发布套件仓不可得(401/404,带 token 同),rebuild 缺 3 个上游源(hellaswag 404 等);照实标 pending,不阻塞发布。

## 已闭(勿重跑)

③ 全链(票 issues/02 终态);serving 形态=merged-fp32(分片);GGUF 转换+一致性;MLX 转换;bf16 merged/compact 键/fp32+LoRA 形态证伪;vLLM 0.26 前缀缓存默认开;套件 rebuild 路线(源缺失,证据 dlsuite/rebuild 日志)。

## 现役资产

- 盒:s6-release-serving-001(repo+envs/serve+artifacts 全回执)、s6-quant(llama.cpp+GGUF×3+mmproj+MLX 4bit)、s6-index-work(env+workroot+日志)、s6-hf-repo(推送中);vLLM/网关/llama-server 已全部停止。
- 本地:main=ae520f1;docs/demo 四段录屏;docs/charts 8 张 SVG(light/dark);reports 全回执。
- 密钥:HF 盒上 SSH key + 用户 write token(仅 env 注入,未入仓)。

## 待我决策

- Decision Index:是否接受"上游不可得,保持 pending"进发布物;若你侧有该 gated 套件的访问授权,给我可用数据源即可补测。
- ⑦ 终局验收票在 ⑤ 推送落地后开。
