# D004：P0 数值兼容例外提案（未获批准，未应用）

## 要决定的唯一边界
允许在隔离的vLLM0.31候选环境中试验一个最小FP32 LoRA兼容补丁，并把FP32仅作为高精度一致性基准；不改基座、不重新训练、不合并LoRA、不放宽任何门槛、不宣称FP32满足最终BF16/FP8/24GB交付性能。若未通过，回滚候选环境，不扩大修改。

根AGENTS.md规定“不改推理引擎；偏离须ADR与用户决策”。当前提案触及这条边界，用户的持续开卡授权不等于已批准此例外。

## 已有证据
- BF16十步训练/完整恢复通过；真实Chromium候选网关契约通过（reports/p0-browser-fp16-001.json）。
- 默认FlashAttention BF16完整100题失败；HF自身BF16 SDPA/eager亦不逐位一致。
- HF版本/分词输入一致；FP16基座enable-lora开/关输出逐位相同，排除LoRA包装。
- 只将FP16 attention改TRITON，原始概率最大差由三例0.36911降到0.005796；后端和精度都影响结果。
- BF16 TRITON完整100题：基座max条件/U0.061477、raw0.057493；adapter0.044019、raw0.003428，均未达严格<0.001。
- FP32无LoRA+TRITON+禁TF32三例max条件/U3.7643e-6、raw2.2014e-6。随后完整100题基座诊断通过：max条件5.7091e-6、raw7.9049e-6（reports/p0-fp32-base-full-comparison.json）；未证明独立adapter或P0通过。
- FP32 LoRA启动被shrink/expand的dtype断言拒绝；强改成BF16 LoRA仍不符合shrink的输入/权重同dtype要求。
- HF复用同FA3/FA4仍失败；标准Flex候选在512维相关编译处NoValidChoicesError，未作为验收通过。

## 可审查补丁
同目录vllm-0.31-fp32-lora.patch只增加两个文件中三处dtype断言的float32允许值；尚未应用/编译/运行，不能称修复成功。
执行同时设TRITON_F32_DEFAULT=ieee、NVIDIA_TF32_OVERRIDE=0、FP32 KV与model dtype、TRITON_ATTN；原BF16权重文件不可改写。Triton点积是否完全支持相关shape需实际验证，必要时失败即停，不能继续扩大引擎改动。

## 验证与退出条件
1. 隔离候选副本及原文件哈希，先对小矩阵LoRA前向和base无LoRA路径建立真实GPU对照。
2. 检查FP16/BF16未改变原行为，再运行FP32模型base+adapter三个失败样本，保持LoRA独立加载。
3. 通过后以可追溯precision参数生成新的HF reference，完整100题base+adapter严格<0.001，实际API/浏览器及耗时/显存证据齐全。
4. 任一失败留报告并回滚；不默改阈值，不把FP32基准当最终低精度/24GB产品完成。工程路线影响未来发布时另写正式ADR与回归门槛。

## 原始文件哈希（vLLM0.31.0，已读取安装源码）
- lora_shrink_op.py: f4551143d4c9244fb7a4ad6cdd5330f13e302332aca2c4eb97b8d84a84fe8918
- lora_expand_op.py: aced1e1480052cb401497313042a611d376f23746face87879e969f98856568d
