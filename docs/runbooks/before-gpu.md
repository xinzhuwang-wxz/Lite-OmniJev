# 开卡前恢复与验收

当前不能通知开卡：必需外部模型、数据、运行时下载闭环已完成；native vLLM CPU processor及Linux真实视频网关在2GiB无卡条件下仍未验收。总账见 `reports/project-model-offline-acceptance-001.json`、`reports/project-data-offline-acceptance-001.json`、`reports/project-runtime-offline-acceptance-001.json` 和 `reports/before-gpu-cpu-validation-005.json`。旧 `reports/no-gpu-readiness.json` 仅P0历史范围。

SSH恢复后用完整Git bundle建立了独立源码目录 `/root/autodl-tmp/before-gpu-release-4366197`。目录名固定，不代表当前版本；先核对 `git rev-parse HEAD`，后续同步receipt才是准确版本。不要覆盖原项目或借祖先仓库HEAD填实验版本。当前依赖缓存仍在 `/root/autodl-tmp/lite-omnijev/artifacts/`；大权重/媒体/缓存不进Git。

## 已实跑的离线重建

以下命令已经以fresh-001的全新env/report/log实际运行，耗时224.9秒、基础安装/overlay/mandatory import/probe/pip check退出0。再次运行必须改成新的目录，入口会拒绝任何已有env/report/log；不得把重新建环境放到计费GPU阶段。

```bash
cd /root/autodl-tmp/before-gpu-release-4366197
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 \
flock /tmp/omnijev-cpu-heavy.lock \
/root/autodl-tmp/envs/root-main-offline-check/bin/python \
  -m scripts.run_before_gpu_teacher_runtime_canary \
  --source "$PWD" \
  --env /root/autodl-tmp/envs/before-gpu-teacher-job-final-001 \
  --log artifacts/before-gpu-teacher-job-final-001/install.log \
  --report artifacts/before-gpu-teacher-job-final-001/report.json
```

可直接复用的完整teacher/job解释器：`/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python`。Torch2.13、Transformers5.17、Pillow12.3、torchvision0.28、PyArrow26、PEFT0.21.2、Accelerate1.15、pytest9.1.1，完整证据 `reports/before-gpu-teacher-job-fresh-001.json`。入口按已缓存manifest验证exact overlay wheels，使用`--no-deps`保护基础Torch版本；不是借另一个env的site-packages。不要用不带extra的`uv run`重新同步而丢失可选训练/媒体依赖。

服务primary仍是 `/root/autodl-tmp/envs/serve-vllm026-offline-check` 的vLLM0.26/Torch2.11/Transformers5.13；0.31环境只作准备/诊断，未经决策不能把它提升为服务默认。跨引擎比较须验证实际同权重/adapter/processor/prompt IDs/dtype与原始概率服务配置，不能因两个环境都能import就当作等价。

## 已验证的CPU闭环

- 本机带真实公开媒体/官方processor全量560pass；仅Linux root隔离skip，已在租机正确隔离reader单项补验通过。静态检查、类型检查、构建及32票依赖验证通过。
- 真实Chromium媒体/三种primitive/422隐私/503拒绝通过，backend明确fake，截图和trace见browser004。模型/性能证据另行验收。
- 租机轻量公共契约52pass；全套测试137保留。TF5.17视频缺可选framecount导致的422已修复；修复后Linux视频仍137，未声称Linux模型管线通过。
- native processor的正常及同官方类进程内诊断均零cases失败，Docker官方下载及SSH中转失败后停止重试；无engine/权重/GPU执行，未改用户容器/daemon。

当前2GiB cgroup不能用宿主机free替代；所有远端重CPU遵守重锁/线程1。已有Linux137时不要不断重复同一配置。需要更大无卡Linux资源或恢复官方下载通路后，先完成native六case、再full100；具体命令及比较字段见 `s0-vllm-processor.md`。4GiB以上仅建议起点，是否足够仍需实测，资源调整由用户决定。

## 获得开卡授权后的顺序

本节是待执行流程，不是GPU通过证据或开卡建议。当前不要运行大模型GPU作业。

1. 确认当前开卡前总账已真正pass；检查持久盘剩余、代码/资产/配置hash和许可，保留150GB checkpoint及18.16GB模型临时预算。
2. 用户开卡后先做驱动/CUDA/显存与已锁运行时小smoke，失败保留报告；不自动换版本、放宽阈值或长跑。
3. 继续既定S0教师真实两遍、100题HF/vLLM/网关概率、缓存及真实浏览器验收。历史maxTV失败与未批准D004不因CPU通过而消失。
4. 原P0真实训练/resume相关缺口按原票证据决定补验，不能把CPU小模型恢复等价当GPU恢复通过。训练入口见 `training.md`，教师/概率/缓存各用其runbook。
5. S0所有门槛包括独立gate身份/真人标注/五模型基线通过后才展开S1；不提前启动未知阶段实验。
6. 作业结束确认checkpoint/报告持久化、记录版本/时长/实际账单与续跑位置，通知用户关卡；不代用户开关卡。

2026-10-14最新收口：primary026 LinuxHF100题首次通过（94.092秒），第二遍90题后137；native零case137，释放重复媒体tensor后的网关两视频仍137。新汇总`reports/project-before-gpu-readiness-001.json`明确false。用户选择保持当前实例，当前交接为CPU检查点；真正可开卡后才写专用开卡交接并通知。新执行代码快照在`/root/autodl-tmp/before-gpu-release-32fb559`，准确HEAD/完整源码bundle见最终sync receipt。
