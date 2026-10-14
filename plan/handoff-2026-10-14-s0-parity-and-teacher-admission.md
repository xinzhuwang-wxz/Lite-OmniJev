# 有卡接续：S0 数值闭环和教师技术准入已完成

2026-10-14主会话关闭s0:03和s0:01；不表示S0全部通过。用户要求总体按原方案自主推进并减少GPU时间，保持基座/引擎/原阈值；不代开关卡/付费/生产部署/push，不重建heartbeat。当前运行态先读plan/STATUS.md，不能直接重放本文件历史PID。

## 已完成且不重跑

- `reports/s0-full100-gpu-acceptance-001.json`：实际H20、primary vLLM0.26/Torch2.11/TF5.13，HF100与native100分别两遍精确；raw/joint/conditional/pU最大差0.0000322938/0.0000322830/0.0000167780/0.0000322830，全部严格<0.001。网关100两遍精确、conditional/pU差0。实际Chromium153六类媒体×三Primitive及坏媒体422通过，独立verifier通过，25定向回归通过。
- 原始证据本地`artifacts/s0-parity-full100-20261014/`、`artifacts/browser/s0-live-fp32-ieee-001/`；远端DF树`artifacts/hf-fp32-full100-001.json`、`vllm-fp32-full100-001.json`，gateway runner树`artifacts/gateway-fp32-full100-002.json`。摘要逐文件SHA绑定，浏览器trace和截图保留。
- `reports/teacher-technical-admission-gpu-001.json`/`teacher-head-reference-gpu-001.json`：固定13文件23,955,723,321B、原README声明、隔离原head转换、safehead实际GPU与原值hash链/审计公式完全相同。真实text/audio/video各两遍+counts2/3/4/20/256全exact，47.418秒exit0；31安全/语义回归通过，独立verifier通过。teacher/job真实运行时Torch2.13/TF5.17/PEFT0.21.2。
- 不在正常进程反序列化rawpickle或执行jev_omni.py；GPU参考为先前隔离转换的原tensor bytehash+独立原公式。发布者仅披露Gemma4-12B-it/30k fine-tuning及非Jev输出声明，没有完整训练源清单；这不放行原始训练数据或教师效果。

## 数值参考配置

E4B固定revision `ee0ef6023621cff504d758262d4e04895a5af4a2`，未加载adapter；官方模板关闭thinking，native raw_logprobs完整答案+U，maxlen8192、单序列。HF和native均FP32；native `TRITON_F32_DEFAULT=ieee`和`NVIDIA_TF32_OVERRIDE=0`，官方hf-config-path副本仅root/audio/text/vision四个dtype从BF16改FP32。原配置/权重未改，不修改vLLM引擎。历史BF16 full100漂移失败照留。

这是数值参考；BF16 LoRA训练不变，最终服务精度/性能和cache-on正确性仍须独立验收。原验收服务PID94918/130178已停止。实时proc/cwd/命令/精度env/critical code SHA与health曾采集到live-process-binding-001；只有operator/process binding，没有签名remote attestation。

## 可复用执行树和资源

SSH `root@region-42.seetacloud.com:48926`，当前mux`/tmp/omnijev-ssh-root-final`；凭据不保存。本机仅小报告/公开fixture，不搬权重。

- 资产树`/root/autodl-tmp/lite-omnijev/artifacts`保持不覆盖；H20/97871MiB，cgroup150GiB/16CPU，盘余约420GB。
- `/root/autodl-tmp/gpu-reference-fp32-complete-001`冻结HEADdfdfd935：正式HF/native、教师benchmark断点、cache runner。
- `/root/autodl-tmp/gpu-gateway-contract-001`HEAD0ca8ff1：正确chance-adjusted confidence网关validator；API公式未变。
- `/root/autodl-tmp/gpu-teacher-head-reference-001`HEAD1f3c140：safehead参考脚本及真实GPU报告。
- 服务解释器`/root/autodl-tmp/envs/serve-vllm026-offline-check/bin/python`；教师解释器`/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python`。使用已安装完整解释器，避免uv重整optional依赖或升级Torch。

## 下一闭环

s0:02完整5600条两遍教师评测：pilot首1397条后resume到1653条，再SIGINT暂停做cache独占短测；JSONL仍在DF树`artifacts/teacher-benchmark-pilot-001.jsonl`。首遍MMAU620/1000=62.0%在原63.1±2pp窗口内，MVBench部分不判质量。两个进程均退出，绝不能以中断报告status=running当活跃作业。缓存测试结束后以同源码、同输入与显式--resume新日志继续，不重复已有题、不改冻结树。模板命令在DF树`artifacts/resume-teacher-002.sh`；下一次使用新日志/exit，保留前次失败记录。

s0:04已认领；同Linux官方processor冻结协议`artifacts/cache-fp32-001/native.json`/gateway.json，2重复/2预热/10个尾问题、commonprefix309/mediaendpoint265/零输出容差。on/off单独启动、独占GPU、实际KV命中及native/gateway计时。FP32 eager on首次测量冷比约3.93、暖比约14.5，命中288；这是真实未达2倍，不能关票。后续off/网关完整结果和官方CUDAgraph参数优化以STATUS新证据为准。禁止把并发结果说成顺序通过或把不同冷暖条件相除。

s0:05/06仍受真实独立gate身份、两真人6000题/12000判断和κ≥.7资源约束。不得agent生成假人工真值/κ；封存只接收隔离汇总。S0全绿前不进S1，终局仍未complete。
