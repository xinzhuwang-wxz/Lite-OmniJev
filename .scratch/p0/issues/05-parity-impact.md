# 05: 区分数值波动与决策影响

ID: p0:05
Milestone: P0
Category: enhancement
Status: ready-for-agent
Execution: done
Owner: none
Worktree: none
Blocked by: 03

## What to build
按用户要求重新评估未实测门槛：将已有HF/vLLM报告的影响分析做成可复现命令，保留全部样本与原始差异，拒绝错配或被篡改概率，不改引擎和gate。提出独立定标/验证数据要求。

## Acceptance criteria
- [x] 同输入/模型provenance检查、从logprob核对概率，坏报告拒绝。
- [x] 输出条件/raw/p(U)/合法联合TV、top-1变化、分析分层与来源SHA，明确不是业务质量放行。
- [x] 复现现有base和adapter结论，CLI不覆盖已有证据，行为测试和项目检查通过。
- [x] 记录代表性数据与独立验证要求，P0不自动关票。

## Out of scope
修改阈值、引擎补丁、模型训练、S0放行；无标注时不能报告准确率/ECE。

Evidence: scripts/parity_impact.py; reports/p0-base-impact-001.json; reports/p0-adapter-impact-001.json; tests/test_parity_impact.py（9项）；全套145通过，Ruff/mypy25文件/build通过；协议见plan/proposals/p0-parity-policy-review.md。
