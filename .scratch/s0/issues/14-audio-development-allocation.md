# 14: 构建Clotho原始Noul音频500开发题与整family排除

ID: s0:14
Milestone: S0
Category: enhancement
Status: ready-for-agent
Execution: done
Owner: none
Worktree: none
Blocked by: 03, 13

## What to build
按audio-development-spec.md复用已有公开TRAIN标签/许可允许WAV，完成原真值→CPU媒体→dev500→全family训练排除的独立闭环。

## Acceptance criteria
- [x] 固定TRAIN/metadata SHA/Zenodo版本与QA/逐音频许可、attribution；不打开validation/test/holdout，不新下载。
- [x] 恰好3原行一致yes/no、原gold优先；6行组明确排除，confidence不作答案/概率；原sound ID family映射核对，500独立family/每family一题固定哈希选择。
- [x] 选中音频全部CPU有界decode、时长/采样/hash与payload绑定，原答案不进输入；开发ledger脱敏，dev全family/别名/其他QA从未来训练排除。
- [x] 两遍真实LinuxCPU输出逐字节相同、500精确、来源/代码/配置/输出hash及环境/命令/退出码/耗时齐全；不把构建或三匿名票当模型质量/真人κ。
- [x] 公共行为拒绝/隔离/许可/投票回归、ruff/mypy/build通过；明确Noul-only与完整音频任务分布缺口。

## Evidence
Evidence: reports/s0-audio-development-build-001.json; reports/s0-audio-development-build-002.json; reports/s0-audio-development-acceptance-001.json
待执行；s0:12公开TRAIN有界审计已证明原候选数量，不等于已构建。无GPU/付费/模型推理。Spec: .scratch/s0/audio-development-spec.md。

## Out of scope
GPU、教师打标、任何封存载荷、伪造双人一致性、补题变体或更换门槛、把单primitive切片当完整音频准入。

2026-10-14实际Linux源码5842217在/root/autodl-tmp/s0-audio-dev-001两遍完成，15.106/3.315秒exit0，四输出逐字节相同。500道原Noul/500独立sound families，选中500WAV完整有界PCM读取/hash通过、最长29.968秒，yes314/no186；973原TRAIN family中500全排除、473train_available（非新训练准入）。6行异常组排除且未取前三，confidence未作答案/概率，固定alias与许可/URL/署名严格匹配。9定向测试/ruff/mypy通过，不新增全检或review。仅原一致Noul，未代音频完整分布、其他cohort、模型质量或真人κ；后续统一归入07主闭环。
