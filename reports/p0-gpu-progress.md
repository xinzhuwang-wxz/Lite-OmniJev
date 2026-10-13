# P0 GPU 验证进展（未通过集成门槛）

2026-10-13，新克隆实例H20/97871MiB、driver580.65.06、16CPU/150GB；数据盘800GiB（750付费）。

已完成：真实100样本10步，67.17秒，峰值17,296,830,976字节；仅34,881,536个LoRA参数训练。checkpoint第5步独立恢复后，全部loss与最终adapter哈希逐位一致。报告gpu-train-smoke.json、p0-resume-equivalence.json，产物本地备份p0-gpu-backup.json。
HF基座/adapter各100题reference完成；不是跨引擎一致性通过，也不是模型质量通过。

已证实推理阻塞：
1. vLLM0.26+TF5.19读取Gemma4全局head_dim抛AmbiguousGlobalPerLayerAttributeError；服务降到官方0.26发布组合TF5.13，最小配置probe通过。
2. 绝对路径启动vllm不激活venv/bin，FlashInfer JIT找不到已安装ninja；补PATH修复，native服务实际启动。
3. 原生后端完整100题：最大条件答案/U差0.14160200594007444，最大原始概率差0.3908445359157475，超过0.001；报告p0-base-vllm-001.json。不能宣称P0通过。
4. 相同HF诊断在TF5.13/torch2.11与TF5.19/torch2.14的100题输出完全一致；输入token哈希相同，排除该HF版本与分词差异。
5. vLLM0.26官方Transformers backend在全局/局部异构head处产生2048x4096与2048x2560矩阵形状错误，不能作为已验证替代。

当前候选环境：独立infra/serve-next固定vLLM0.31.0/TF5.17.0/Torch2.13.0。官方Transformers backend已成功启动，异构head形状问题解除；三个失败样本的BF16概率对照仍失败，不能放行。

数值诊断：
- HF自身仅从SDPA切到eager，三个样本最大条件/U差分别0.028525、0.009507、0.019509；其中原始概率最大差0.013737。BF16跨核误差已被实测，不是只靠引擎源码推测。
- 当前vLLM日志明确选择vllm_c RMSNorm；其CUDA内核FP32乘权重后转回BF16。先前“native中间BF16舍入导致服务错误”的猜测不能成立；孤立default VllmConfig的norm探针选native，不能代表服务实际CUDA路径。
- 正在测试双方FP32推理、关闭TF32、vLLM官方TRITON_ATTN；保留训练BF16和原始权重文件，不改引擎、不放宽严格<0.001阈值。此候选未通过前不作为正式服务基线。

用户已明确保持GPU开启，由负责人连续执行有价值的就绪作业；不再要求切无卡，也不自动开关卡。浏览器脚本和S0草案已准备，P0一致性门槛通过后再放行下游。

新增诊断结果：FP32基座3例最大条件/U差3.764255e-6、raw差2.201440e-6（p0-fp32-3case-probe.json），明显低于1e-3；但未达到100题完整验收。vLLM LoRA shrink只支持FP16/BF16输入，FP32启动LoRA失败，不能直接用FP32候选替代服务器不合并LoRA的原设计。HF官方attention注册调用同一vLLM FA3/FA4后BF16仍失败（p0-hf-vllm-fa-probe.json），不能只靠attention包替换解决。当前测试官方FP16配置，保持LoRA独立；尚未通过门槛。

最终单变量结果：FP16开启/关闭LoRA的基座输出逐位相同，排除包装；仅FA→TRITON使raw三例差0.36911降到0.005796，后端与精度都影响结果。BF16 TRITON完整100题基座和adapter仍失败，分别max条件0.061477/0.044019、raw0.057493/0.003428。

FP32基座完整100题诊断已通过：最大条件/U差5.7091125e-6，最大raw差7.9049396e-6；报告p0-fp32-base-full-comparison.json。它不包含独立LoRA，也不作为P0或最终低精度产品放行。下一步D004兼容例外提案待用户决定；patch尚未应用。标准HF Flex额外候选编译NoValidChoicesError，未声称通过。

本地136测试、Ruff、24文件mypy、wheel/sdist构建通过；22张票验证有效。真实Chromium契约通过，见p0-browser-fp16-001.json，artifact截图/trace在artifacts/browser/p0-fp16-001。三个已完成worktree清理；本轮自建模型/网关进程停止，付费实例保持用户要求的开卡状态。
