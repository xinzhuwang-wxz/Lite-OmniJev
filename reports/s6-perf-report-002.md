# 性能报告 002(serving 态实测;交付链环节 ⑥)

生成:2026-10-18 凌晨,release-002 serving 复验实测(承接 `s4-perf-report-001.md` 的评测态时延)。
配置:官方基座平面布局 + D3P-600(767be09b)融合的 **merged-fp32-flat** 工件,vLLM 0.26.0,
float32/eager/单序列/max-model-len 8192/kv 4GiB/**禁前缀缓存**(S0 已准入数值配方,
`reports/s0-full100-gpu-acceptance-001` 口径);网关 canonical-chat(S0 输入准备栈)。
**该配置是 parity 验收配置,非生产并发/延迟声明。**

## 资源(serving 稳态,nvidia-smi 实测)

| 项 | 值 |
|---|---|
| vLLM 引擎显存 | **39,056 MiB**(权重 ~30G fp32 + kv 4GiB + 激活) |
| GPU 总显存 | 97,871 MiB(H20) |
| 盒 RAM 使用 | ~56 GB(含页缓存) |

## 端到端时延(真实浏览器 → SSH 隧道 → 网关 → vLLM,Playwright 实测)

每请求 3 题串行(choice/score/noul);`total`=网关墙钟;`upstream`=vLLM 前向+调度;
`preprocess`=CPU 媒体/输入准备。报告:`reports/browser-s0-live-001/report.json`(盒上副本
`gateway/receipts/browser-s0-live-001-report.json`)。

| 模态 | preprocess | upstream | total |
|---|---|---|---|
| text | 12 ms | 392 ms | **973 ms** |
| image(8帧截图口径) | 167 ms | 1111 ms | 1815 ms |
| screenshot | 2383 ms | 716 ms | 3606 ms |
| audio | 471 ms | 1321 ms | 2416 ms |
| video 8 帧 | 1249 ms | 1256 ms | 3294 ms |
| video 16 帧 | 1572 ms | 1713 ms | **3543 ms** |

注意:total 含 SSH 隧道与家庭网络往返;screenshot/video 的 preprocess 为媒体解码/帧拼装,
生产部署同机时可消去隧道项。单题 vLLM 前向(评测态 p50 162ms,见 001 报告)与服务态
upstream 之差主要为调度/协议/串行 3 题。

## 吞吐参考(parity 链实测)

- HF 参考:100 案例×2 遍 = 230 s(评测栈)
- vLLM serving:100 案例×2 遍 HTTP = **161 s**(约 0.8 s/请求,fp32 eager 单序列)

## 工件大小(release-002 + 量化)

| 工件 | 大小 |
|---|---|
| model/merged(fp32,平面键) | 31.8 GB |
| model/merged-fp32 / merged-fp32-flat | 31.8 GB ×2 |
| GGUF f16 | 15.05 GB |
| GGUF Q8_0 | 8.03 GB |
| GGUF Q4_K_M | 5.34 GB |

原方案"Q4 内存 ≤3GB"按 E2B 口径;E4B Q4_K_M 实测 5.34 GB 如实记录,达标判定留给 ④ 票。

## 待补(下一窗口)

- bf16 serving 质量/时延(当前 fp32 为唯一 parity 过配方;bf16 已证不过 <1e-3 门)
- 生产并发/压测、同机无隧道端到端、Production 长度配置
