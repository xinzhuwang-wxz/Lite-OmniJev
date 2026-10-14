# S0缓存阻断：官方Triton FP32乘积候选

Owner: main-session-gpu-transition
Parent: s0:04
State: protocol prepared; not launched or admitted

当前最优官方async eager冷T10/T1为2.3164/2.3335，原≤2仍失败。下一候选只有TRITON_F32_DEFAULT从ieee变tf32x3，其余权重/模型/引擎/FP32dtype/NVIDIA_TF32_OVERRIDE=0及模板/答案token不变。已有实际安装源码证据reports/s0-tf32x3-source-assessment-001.json。此改变部分隐式Triton dot数值实现，不当作IEEE等价，不覆盖IEEE参考，不修改引擎源码。它针对当前必需性能失败，不启动第十一节可选研究或5b。

教师005仍活跃，先完成固定5600记录和落盘；活作业不得为本候选中断。GPU无活跃教师后再单独启动服务，独立Triton/VLLM/Torch编译缓存目录，保存启动cmdline、/proc环境、runtime/source/cache路径与hash、GPU占用和退出码。若仍有他人GPU进程或服务不匹配，停止调度，不自动关卡。

## 最小证据顺序

1. 新precision服务先cache-off，所有冻结public100输入/媒体/答案映射保持。复用已准入IEEE HF参考；跑两遍native100，四概率max严格<.001且重复完全一致。实际prepare/参考codehash必须匹配，不能以CPUprobe或不同源码摘要替代。失败即停止此候选，不启动cache-on/gateway/off额外测量。
2. 精度通过后cache-on按既定bounded-scheduled-v1先单题prime再9并发，冻结cache-probe-0..9。先同图1/10两repeat冷canary，实际inflight、KV媒体token命中及输入/四概率绑定完整。每次冷比≤2；任一失败停止，不扩展gateway/off，不用仅warm比或平均比抵消冷失败。
3. native冷canary通过才做完整native on/off和真实ASGI→HTTP GPU gateway on/off。所有正式重复/媒介覆盖/四概率和产品cold政策按既定协议，不跳项。外层wall包含验证/预处理/准备/请求/清理，工作时间另报，不伪造跨题CPU并行wall。

## 退出与范围

最多一个该精度候选，单次外层预算900秒（含服务准备与最小实验；到时正常收尾保存证据）。任何真实数值/重复/时延门槛失败，不重复相同配置、扩大benchmark或降低阈值；给出实测结果与下一必要技术判断。启动前确认冻结完整参考与可执行collector已存在，计时编译冷启动与steady-state分别记录，未知费用cost=null，不估充已付账单。

CPU准备和本规格不是GPU通过。新候选未实测前仍s0:04 claimed/failed，不可进入S1，正式S0交接仅在s0:08全绿后产出。
