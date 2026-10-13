# 全项目无卡准备交接

用户最新要求：整个项目需要的外部模型/数据/依赖全部下载校验后再开卡。实例已实际切无GPU，同SSH端口48926，项目盘完整；严禁自动启GPU。

先读STATUS.md、.scratch/offline-prep/spec.md和对应四票。模型清单configs/project-model-assets.json（固定七项约121GB）；数据账本configs/project-data-assets.json（29来源，物理下载与训练准入两轴）。运行时agent成果仍待root复核，尤其必须探测真实envs/{data,serve,serve-next}路径。

活跃作业：远程模型flock1827、Valen原料flock4435，先ps确认再读reports/project-model-download-001.json和valen-source-download-001.json，不能根据旧快照重启。学术加速source /etc/network_turbo不输出环境；HF_ENDPOINT官方。源码scripts.stage_models已进Git；Valen受控临时脚本remote artifacts/stage-valen-source.py保留、hash入报告。

教师CPU安全转换报告已合入，但父进程输出路径symlink问题由teacher_head_conversion在独立worktree修复中，尚未准入教师。CLINC修正后正式证据是reports/clinc-admission-corrected-001.json及clinc-semantic-preservation-001.json，旧报告不用于准入。S0严格parity仍失败、S1未放行。

建议技能：research用于尚未明确的源许可/媒体；to-spec/to-tickets/triage按既定本地tracker扩票；handoff用于闭环记录。独立任务fork_turns=none，显式workdir，主会话独占票据/STATUS。无远端push。
