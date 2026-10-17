# M0 → 无卡准备交接

本记录按用户的 handoff 约定保存在 plan/，覆盖技能默认临时目录。详细状态唯一入口：`plan/STATUS.md`。

## 阅读顺序

AGENTS.md → STATUS → docs/02-project.md → 当前spec/ticket → 相关ADR。
源方案不改，方案漂移/兼容边界见 docs/research/2026-10-13-feasibility.md。

## 已落实与证据

代码/文件不在此重复，查看M0票与reports。CPU数据/读出/gate/HTTP/资产/队列测试和真实Chromium范围明确；GPU、真实质量、媒体和端侧没有通过声明。
review曾发现覆盖率与资产ready误判，已修复回归；复核结论在 reports/m0-review.md。

## 下一会话

先确认用户是否已有无卡Linux实例。若有：按docs/runbooks/compute-handoff.md检查SSH与持久盘，下载本阶段必要权重/数据，锁Linux训练/服务环境，完成no-gpu-readiness报告后再通知开卡。
若没有：不重复下载tokenizer或跑同一套已通过检查；只处理新证据/可独立推进的票，其余保持明确外部阻塞。
封存集人工标注与私有榜单、教师来源/格式均有未解边界，S0阶段必须拆成实际票，不能自行宣称已解决。

## Suggested skills

setup-matt-pocock-skills 已配置，无需重跑。后续优先 to-spec、to-tickets、triage、handoff；实验优化有固定指标与预算才使用 autoresearch。不要激活superpowers或假设Sonnet/Haiku可用。

## 资源与 Git

默认本地票据/提交，不push；关联origin已核验，避免gh高频调用。
每个并行执行者独立worktree，leader集中关票。结束时仅清理本次创建资源，保留artifacts缓存；模型资产内容不进Git。

## M0最终出口
111项测试无跳过通过；具体命令/退出码见reports/bootstrap-validation.txt。已具备真实基座十步训练入口及27项CPU训练验证，实际GPU训练0步。当前唯一可推进前置动作是用户提供无卡实例；无需重复setup或重做本地调研。
