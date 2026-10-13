# 06: 独立数值定标集与低精度实测

ID: p0:06
Milestone: P0
Category: enhancement
Status: ready-for-agent
Execution: open
Owner: none
Worktree: none
Blocked by: 05

## What to build
执行plan/proposals/p0-parity-policy-review.md的数据协议：准备与十步训练生成族独立的程序真值定标/确认数据，各100族；先仅对定标批运行HF/vLLM BF16基座及独立LoRA，收集业务影响证据并提出门槛方案。

## Acceptance criteria
- [ ] 两批按生成族隔离、与训练生成族不重叠，固定schema/真值/题型配额/来源hash；不读取产品holdout。
- [ ] 最小公共接口验证真值、缺信息/矛盾事实处理、选项数/等级数、样本族重叠拒绝；任何配额无法满足如实阻塞。
- [ ] 定标批HF/vLLM用相同模板、固定资产与未修改引擎BF16/TRITON；base与adapter报告完整且可追溯。
- [ ] 输出联合分布、条件分布、p(U)、答案/Score变化、真值指标和边界覆盖，不宣称程序题代表生产业务。
- [ ] 提出带业务理由的候选门槛和下一张独立确认票；门槛冻结前不读取确认批模型输出，不修改既有gate或放行P0。

## Out of scope
引擎补丁、再训练、调整既有gate、生产质量承诺、S0启动。确认批验证待候选门槛决策后进行。
