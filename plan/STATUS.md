# STATUS

更新时间：2026-10-13，GPU数值诊断轮。
当前里程碑：P0，p0:04真实集成尚未通过；终局仍为原方案S0–S6及FINAL。
Native goal已核实为active；终局尚未完成，不更新为complete。

## 当前票与认领
- bootstrap:01–08、p0:01–03：done。
- p0:04：blocked，待D004推理兼容例外决定；owner释放，已有训练/恢复/浏览器证据保留。
- s0:01–08：规格和票据已落盘，全部依赖p0:04；s0:06真实人工标注blocked，s0:08需后续基线展开后再triage。
- s0-data:01：CLINC准入未执行；依赖p0:04。
- delivery:01：尚未完成。当前22张票依赖校验通过；没有可执行frontier；三个子代理已结束，成果整合备份后worktree已清理。
- S0规划已整合commit01b895c；浏览器脚本已整合072ce5c；真实FP16候选API契约验收通过，报告reports/p0-browser-fp16-001.json，截图/trace在artifacts/browser/p0-fp16-001。不是数值parity/质量通过。

## 用户授权与实例
- 用户明确保持当前GPU开启；负责人按现有方案自主推进，减少总租卡时间。只跑有证据价值的就绪任务，不为利用率跳过门槛。
- 新实例SSH root@region-42.seetacloud.com:48926；H20 97871MiB、driver580.65.06、16CPU/150GB RAM。
- 数据盘/root/autodl-tmp总800GiB（750GiB付费）；不是750GB内存。旧无卡实例不是当前工作目标。
- 项目/root/autodl-tmp/lite-omnijev；复用SSH ControlPath=/tmp/omnijev-ssh-%C。凭据不入库、不输出。
- SwanLab key仅远程/root/.config/lite-omnijev/swanlab.key（0600）；project Lite-OmniJev / workspace maxen。
- 训练.venv：Python3.11.17/torch2.14.1/TF5.19/PEFT0.21.2；服务serve：vLLM0.26/TF5.13；隔离候选serve-next：vLLM0.31/TF5.17/torch2.13。
- 环境入口source infra/bootstrap/autodl-env.sh；缓存/root/autodl-tmp/cache，模型artifacts/models。

## 已验证证据
- 无卡15项通过，E4B revision ee0ef6023621cff504d758262d4e04895a5af4a2全部8文件16,024,822,159字节哈希通过；33答案token/官方模板通过。
- 本地与Linux各136测试通过；wheel工具另7测试、Ruff/mypy/build通过（最新环境配置改动待最终再检查）。
- 真实100样本10步训练完成：67.17秒，峰值17,296,830,976字节，仅34,881,536 LoRA参数训练；reports/gpu-train-smoke.json。
- 训练artifacts/training/p0-20261013T131751Z；step0005独立恢复到10步，loss序列和最终adapter逐位一致；reports/p0-resume-equivalence.json。
- adapter SHA256 97db02cd2bb3a4990fb68ceb3c3e863b9a2cb72669406926745a9e4c052b1b42。
- SwanLab真实训练：https://swanlab.cn/@maxen/Lite-OmniJev/runs/1mia2y76 。程序真值fixture不是模型质量证据。
- final adapter和step0010完整optimizer/RNG checkpoint已备份本地artifacts/training/p0-gpu-backup，约561MiB，hash已核验；reports/p0-gpu-backup.json。
- HF BF16基座/adapter各100题reference完成；CLINC raw train15250行哈希/许可通过，尚未训练准入，未读holdout。

## 当前阻塞与诊断
- vLLM0.26/TF5.19配置不兼容：服务固定TF5.13解决；ninja PATH修复后native服务成功启动。
- native0.26完整100题max条件/U差0.141602、raw差0.390845，未过严格<0.001；reports/p0-base-vllm-001.json。
- HF5.13/torch2.11和HF5.19/torch2.14的100题输出完全相同，输入token hash相同；排除这组版本及分词差异。
- 0.26 Transformers backend异构head形状失败；0.31官方Transformers backend成功启动，但BF16三个代表样本仍失败。
- 运行中0.31 eager明确使用vllm_c RMSNorm，CUDA内核FP32乘权重后cast；不能把IR native舍入差异误归因为实际服务失败。
- HF内部SDPA→eager的BF16对照已超过0.001，证明跨核数值差异需处理；双方FP32+禁TF32+TRITON_ATTN三例max条件差3.764e-6/raw2.201e-6，证明支持配置可以数值收敛，精度与attention后端仍需单变量区分；这只是诊断，不改训练/权重/引擎/门槛。
- FP32服务启用LoRA会触发内核dtype断言，不能替代原方案不合并LoRA的服务。HF注册同一vLLM FA3/FA4后BF16仍失败，排除只换attention构建即可修复的猜测。
- FP16+FA有/无LoRA基座逐位相同且都失败，排除包装；仅切TRITON后raw三例差0.369→0.0058，仍失败。
- BF16 TRITON完整100题base/adapter分别max条件0.061477/0.044019、raw0.057493/0.003428；两组失败。
- FP32 base完整100题诊断通过：max条件5.7091e-6、raw7.9049e-6，reports/p0-fp32-base-full-comparison.json。缺FP32独立LoRA，非P0放行。
- 标准HF Flex候选编译NoValidChoicesError；未修改引擎。D004的三处dtype补丁仅作为提案保存，尚未应用。
- 并行HF释放显存会令vLLM初始化memory profiler失败，已停止并行GPU探针后重试；以后vLLM初始化期间禁止其它GPU进程分配/释放显存。

## 下一步
1. 先查看D004用户回复；未批准不得应用plan/proposals/vllm-0.31-fp32-lora.patch，也不得换基座、合并服务LoRA或放宽门槛。
2. 若批准：正式ADR记录例外→隔离候选最小kernel行为验证→FP32 base+独立adapter新正式reference和100题严格门槛→真实API/浏览器。若拒绝：只根据新的上游修复证据继续，不能反复重跑已失败配置。
3. P0全通过才关闭p0:04，里程碑复核/handoff后进入S0。当前浏览器仅候选契约通过，不能替代概率门槛。

## 待我决策
- 当前开卡和接入已解决，不再重复询问。
- D004待用户批准：plan/proposals/p0-numerics-decision.md，隔离试验FP32 LoRA三处dtype断言兼容补丁。异步问题已发送；未收到回答，不能把时间流逝当批准。
- S0真实人工标注来源后续需满足双人独立标注要求，agent不可伪造κ；先完成不受其影响的已准入工作。
- S4放量预算、方向/阈值变更、生产发布须用户决定；当前不默认扩大。

## Git与资源卫生
- 仅本地commit和SSH文件传输；未push、未创建GitHub issue/PR，不高频gh轮询。
- 本轮GPU作业完成，已停止自建候选vLLM7581/engine7607与网关6099并取消18080转发；仅停止自建进程，未关卡、关机或更改付费实例。再次执行前核实进程，不凭历史PIDkill。
- 保留模型/数据/缓存/训练产物。清理合并worktree前确认无未保存成果。
- 历史无卡handoff：plan/handoff-2026-10-13-no-gpu.md；loop细则plan/LOOP.md。
