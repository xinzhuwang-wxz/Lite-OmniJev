# 07: 冻结影子规则后的独立确认

ID: p0:07
Milestone: P0
Category: enhancement
Status: ready-for-agent
Execution: open
Owner: none
Worktree: none
Blocked by: 06

## What to build
以configs/candidates/parity-shadow-v1.json为预注册候选，在已固定但尚未推理的confirmation/dev数据上运行官方FP16 HF/vLLM base+独立LoRA，并输出明确的拒绝/尾部复核/影子支持结论。只验证候选，不改正式gate。

## Acceptance criteria
- [ ] 确认推理前固定policy SHA、确认数据SHA；入口明确验证pool=dev、生成族不同、只能使用已冻结候选，不允许将产品holdout读入。
- [ ] 身份/有限值、决策集合变化、准确率/Brier/ECE差、mean/p95 TV与尾部复核均有公共接口行为测试。
- [ ] 100题base和adapter完整GPU报告；缺题/运行错误不是通过，不按输出选择子集。
- [ ] 所有尾部异常保留逐题证据并解释，不能用均值抵消；基座和adapter分别判定。
- [ ] 形成正式gate修订建议或保留失败结论，明确尚需批准事项；确认数据不再用于选择v1阈值。

## Out of scope
修改现有gate、引擎补丁、P0自动放行、生产质量声明；独立确认是影子实验，不是审批替代。
