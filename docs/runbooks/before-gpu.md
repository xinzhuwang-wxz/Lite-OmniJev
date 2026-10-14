# 无卡准备与模式切换

当前无卡环境可自主完成的准备已完成，offline-prep:04已收口；可以由用户切换有卡模式。统一判定 `reports/project-before-gpu-readiness-002.json`，新session直接按 `plan/handoff-2026-10-14-gpu-transition.md` 接续。代理不执行开卡；模式切换就绪不等于GPU兼容、模型质量或S0验收通过。

用户2026-10-14澄清：先做完当前无卡可做工作，再切有卡补资源受限验证。旧readiness001及CPU检查点仍保留真实失败历史，操作结论由002替代。未完成代码/必需可获取资产不能用资源理由跳过；当前所有此类准备均已有回执。

## 已完成与可复用

模型7库70文件121,094,724,397B、29来源数据处置总账、训练/服务/教师/量化/前端/Chromium/端侧依赖各有真实回执，见三个 `reports/project-*-offline-acceptance-001.json`。原料下载不表示训练许可/来源准入；研究专用不混入商用，人工封存内容禁止开发agent读取。

实际代码CPU560pass/1Linux-onlyskip、Linux隔离reader补验、真实公开媒体官方processor、Chromium fake backend、静态/类型/构建/32票验证已有证据，见CPU-validation005/browser004。teacher/job fresh no-index从空env重建224.9秒、mandatory probes/pip check0：`reports/before-gpu-teacher-job-fresh-001.json`。不把这些结果说成真实模型/GPU验收。

执行源码 `/root/autodl-tmp/before-gpu-release-32fb559`；准确HEAD以最终transition-sync侧车为准，不以目录名推断。原树只作资产缓存，不覆盖。直接用完整teacher/job解释器 `/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python`；primary026 `/root/autodl-tmp/envs/serve-vllm026-offline-check/bin/python`；root `/root/autodl-tmp/envs/root-main-offline-check/bin/python`。031仅诊断，不静默升级。不要用无extra的uv run删除可选依赖，不混借site-packages。

既有fresh重建命令和不可覆盖规则见 `docs/runbooks/offline-runtime-preparation.md` / fresh001回执；只有环境损坏或版本变更才用新env/report/log离线重建，不在计费GPU阶段无故重装。

## 切换后顺序

1. 用户操作模式切换；先只读GPU/驱动/显存、cgroup和磁盘，保留150GB checkpoint和18.16GB临时空间。旧2GiB的宿主机free不能代表配额。
2. 先补primary026同Linux HF100两遍、native正常六case→full100、实际网关视频8/16→完整组。详细命令、环境/fixture和新报告命名在专用交接；保持离线、单线程和重锁，CPU检查CUDA_VISIBLE_DEVICES为空。
3. 历史HF重复137/native零case137/网关137均未通过，killer未知；不声称OOM已证、不覆盖历史。`--same-process-inspection`成功也不能替代正常初始化。失败先诊断并保留退出码，不盲目长跑；备用Docker/ARM失败路径不用继续耗时重试。
4. 通过首批前置后最小CUDA/模型smoke、既定P0真实10步/恢复缺口、S0教师真实两遍/100题严格概率/缓存/真实模型浏览器。各专用runbook定义不变；历史maxTV失败及D004未批准照留。
5. 人工gate/双人标注κ、数据训练准入仍为阶段条件；S0全绿前禁止S1。未来LoRA/校准/量化/阶段派生集是待生成产物，不是漏下载。
6. 每作业落盘版本、配置/数据hash、时长、成本、退出码与断点；结束通知用户关卡，不代开关卡或删资源。无卡准备结论不授权全项目放量预算。
