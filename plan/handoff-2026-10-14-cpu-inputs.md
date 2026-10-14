# 开卡前准备恢复交接

完整目标仍为原HTML的P0/S0–S6/FINAL，终局未完成，先get_goal核对原生实际状态；当前仅无卡准备，不自动开关卡/付费/生产/push。先读 `plan/STATUS.md`、`plan/LOOP.md`、`docs/02-project.md` 和当前票。

必需模型/原始数据/运行时外部准备分别由offline-prep:01/02/03关闭，真实证据与逐报告SHA见三个 `reports/project-*-offline-acceptance-001.json`。03已完成新版teacher/job从空env完整离线重建，不再依赖手动overlay。04由root统一集成，不能因下载完成就关票或通知开卡。

最新源代码CPU全量560pass/1Linux-onlyskip，Linux隔离reader单项已补验；`reports/before-gpu-cpu-validation-005.json`。真实Chromium及截图/trace见browser004，backend明确fake。独立review/probability/取消与资源预算回归已完成；不要重复同范围全量测试，除非代码改变/失败/新风险。

primary026仍primary，不被031替换。TF5.13/tokenizers0.22.2独立Mac target100×2通过；版本差异完整见 `reports/primary-transformers513-cpu-full100-comparison-001.json`，不把同字节不同shape称tensor等价。LinuxTF5.17旧422字段误判已修复，修复后137保留；nativeCPU正常及同官方类进程内诊断也未成功，不能称OOM原因已证或CPU不支持。

新的官方wheels.vllm.ai通路已取得完整ARMwheel并核对官方SHA；但官方GHCR固定ARM基础镜像pull180秒超时，零container/install/HF/native cases，已停止。报告 `reports/s0-official-arm-cpu-alternate-003.json`；partial/cache留待恢复，不删用户镜像或改daemon。此条不意味着全部无卡路线已就绪。

当前远端已恢复SSH mux `/tmp/omnijev-ssh-root-final`；凭据不写任何文件/报告。隔离Git源码 `/root/autodl-tmp/before-gpu-release-4366197`，目录名不是版本，每次核对实际HEAD；最新源码/报告同步receipt待root最终记录。运行与续跑顺序见 `docs/runbooks/before-gpu.md`，已有环境/报告默认拒绝覆盖。

若接续时STATUS中仍有真实监督句柄，先poll原句柄/实际进程与报告，终态或句柄缺失才处理恢复。primary026 LinuxHF100题首次通过，第二遍90题后137；native零cases137，释放重复tensor后网关两视频仍137。所有作业已终止，无需poll或重启。用户选择继续保持当前实例；04及S0:01/03释放claim为blocked。唯一开卡判定见reports/project-before-gpu-readiness-001.json，仍false。最终代码sync receipt是远端准确版本依据，目录名不能替代HEAD。

独立封存gate/两真人标注/未来S6真机仍为外部资源缺口；历史概率maxTV失败、未批准D004及所有阶段门槛保持不变。S0全绿前禁止S1，未来产物不假称下载缺口，终局goal不能complete。

## Suggested skills

使用`verify`核实具体完成主张；新可证失败用最窄诊断路径。必要写交接用本机handoff并按用户约定保存在plan。用户已禁止重型OMX/superpowers流程及重复访谈，不启动自动heartbeat。

用户最新要求：真正收尾到可开卡时才写专用开卡交接，确保无缝接续。本文件只是未就绪的CPU检查点；不得凭此启动GPU或当作开卡授权。
