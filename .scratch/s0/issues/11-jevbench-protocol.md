# 11: 冻结并适配JevBench公开231题协议

ID: s0:11
Milestone: S0
Category: enhancement
Status: ready-for-agent
Execution: done
Owner: none
Worktree: none
Blocked by: p0:04, 01, 03

## What to build
使用已staged的fstandhartinger/jevbench固定八文件，完成来源与输入/标签/计分协议闭环；不联网重下、不读取模型预测或私有集。

## Acceptance criteria
- [x] 核验既有revision/8文件hash/许可，审计公共金标来源，拒绝以Jev API或模型输出作标签；所有231决策逐项可追踪。
- [x] 统计实际primitive/答案范围/媒体依赖/选项语义；适配到既有合法契约且语义保真，不能静默删题、裁选项、离散化改变原评分；不兼容项明确blocked。
- [x] 评分前冻结原发布计分规则和输入哈希；literal148 groups、easy按独立场景可得195的候选解释、教师协议未证实边界分别保存，不按模型分数挑口径。
- [x] CPU构建可重复且原标签/计分往返验证；保存分母/来源与输出hash/配置/代码/退出码。独立同条件公开基线不得写成教师195口径已复现。
- [x] 定向行为测试与ruff/mypy通过；输出后续模型实跑依赖和阻断，不在本票运行GPU。

## Evidence
已执行；规格见 `.scratch/s0/baseline-expansion-spec.md`。固定种子、源码/配置/数据hash、命令/退出码/耗时/成本必须留证；缺项不能pass。

## Out of scope
租卡/开关卡/付费/GPU推理、S1训练、暗改门槛、封存内容暴露、假分数/假人工一致性。

## Comments
2026-10-14主会话由s0:07展开：01/03技术前置已实测，02/04仍是S0总出口门槛但不阻塞该CPU独立闭环；用户已授权按原方案自主技术安排。与现有票概念核对后新增，不复制s0-data:01或05/06的人类/身份工作。

2026-10-14已核八文件SHA和231行结构：139Choice/18离散Score/74Noul，全部合法数量与非空gold，35object state应确定性JSON呈现；无媒体引用。发现旧元数据误把48easy的group=null写成easy；已更正当前config/doc，旧收据不覆盖。147非空group、literal含null148，官方仅36original双行组用于paraphrase consistency，不是195场景准确率；Score实现argmax与docstring冲突已记录。来源为发布方运行前编写/审核声明，hard含模型作者不充人工双标。审计reports/jevbench-public-protocol-audit-001.json；输入适配/计分程序尚未实现，本票仍claimed。

2026-10-14 CPU闭环完成：data/jevbench.py、eval/jevbench.py、scripts/s0_jevbench.py及7行为回归已整合。父会话使用已核八文件的公开源执行validate、build两遍及oracle score，均exit0；两遍231行SHA均11223d982606c0d26eadee412d4f1fbadfe29f1eeae8d9d75df8d446d803c75d。证据reports/jevbench-adapter-{validation,build,oracle}-001.json及build-002.json。oracle为公开expected生成的计分测试输入，100%只验证scorer，不是模型成绩；未用GPU。7 tests、ruff/format、82源mypy、uv build通过。
后续实际模型基线须依赖s0:10兼容探针、固定全部231题输入/标签映射及相同模型模板/参数；每模型保存完整原始非U概率和pU再还原source labels计分。195场景教师口径仍未证实，不作为本票阻断或虚报匹配；此票完成的是独立公开协议，模型forward和质量准入留s0:07/08后续依赖。接续plan/handoff-2026-10-14-jevbench-cpu.md。
