# STATUS

更新:2026-10-18 07:3x(CST)。**负责人:交付链冲刺(主会话,用户全权)。③已收口,④⑤⑥并行推进,⑦待 ④⑤ 完。**

## ⚡ 接续点(第一动作)

1. **⑤ HF 推送进行中**:盒 `/root/autodl-tmp/s6-hf-repo`(main=7e268da:README+LICENSE(CC-BY-NC-SA-4.0)+merged-fp32 31.8G+adapter+GGUF f16/q8/q4+gateway 回执)→ `git@hf.co:Chinzhu/Lite-OmniJev`,LFS 52G 上传中(screen s6-hpush;完成=远端 main 到 7e268da)。盒上 HF 专用密钥 `/root/.ssh/hf_chinzhu_ed25519`(用户已加 HF settings)。
2. **Decision Index**:rebuild-008 跑到 ToolRet 归一化(env=/root/autodl-tmp/s6-index-work/env;workroot 符号链接拼装 quarantine 树;全量检出 20/20 后重启)。出套件后用官方 http 引擎直打网关 `/v1/systemone`(网关仍在 8000 端口跑着,勿关 vLLM)。
3. **④ 一致性**:GGUF f16 文本 20/20 选答一致(`artifacts/s6-gguf-text-consistency-f16-001.json`);llama-server(f16,8021 端口)在跑;多模态 projector(mtmd/gemma4uv)与 Q4/Q8 全量一致性待做。GitHub 侧已换 LICENSE+README(本地 56bf521,未推远端——推 GitHub 需要用户凭据或明示)。

## 已闭(勿重跑)

- **③ 产品化全过**:serving parity(两遍精确 0.0,vs HF 5.75e-6)、网关 parity(conditional Δ=0.0)、真实浏览器 live 验收;入包(gateway/+receipts/merged-fp32-flat,SHA256SUMS 47 文件全验 exit 0)。票 issues/02 终态。
- serving 形态=merged-fp32-flat;GGUF 转换(llama.cpp 0.7.0-dev)三档已产出。

## 现役资产

- 盒:s6-release-serving-001(repo=main、envs/serve、artifacts=全部回执)、s6-quant(llama.cpp+GGUF 三档)、s6-index-work(env+workroot)、s6-hf-repo、s6-hf-staging;vLLM(8001)+网关(8000)+llama-server(8021)在跑。
- 本地:main=56bf521;reports/ 已收 ③ 全部回执;s6-perf-report-002。

## 证伪/勿重试(新增)

- vLLM 0.26 默认开前缀缓存(破坏两遍精确性);console-script shebang 指旧 env(hardlink 克隆必踩);bf16 merged/compact 键/fp32+LoRA 三种 serving 形态不可行(见票 02);`pkill -f` 自匹配。

## 待我决策

- GitHub 推送(56bf521)未执行——远端是 https,需要凭据或用户自己推;HF 推送用盒上密钥已完成认证。
- GPU 用完关卡时点:Decision Index 打分 + ④ 多模态验收完成后通知(预计还需数小时卡时)。
