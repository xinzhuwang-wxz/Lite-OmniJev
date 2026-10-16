# 01: 真实校准与稳健性到同版本服务闭环

ID: s5:01
Milestone: S5
Category: enhancement
Status: ready-for-agent
Execution: open
Owner: none
Worktree: none
Blocked by: s4:01

## What to build
按.scratch/s5/spec.md完成真实校准、独立评估、稳健性和强绑定服务闭环。前置S4出口未关闭；当前并行准备和部分实现不宣称阶段通过。

## Acceptance criteria
- [ ] 七真实组18000独立校准题完整推理与温度拟合，11组明确未校准，完整身份/资源/失败证据。
- [ ] holdout只评估；原ECE/coverage/掉分/正反/置换门真实汇总，缺测不代过，质量局限按ADR0023单项记录。
- [ ] C1–C4/R1按瓶颈有界实测及采用理由，不能用CPU等价数学原型冒充真实RL收益。
- [ ] 默认raw/unfitted与require_calibrated错误可恢复；实际部署runtime/model/template/token/rawreceipt绑定，未经数值验证不能跨engine/dtype激活。
- [ ] 真实GPU/网关/浏览器验证和独立集成复核，S5出口报告有效。

## Evidence
reports/explore/s5-calibration-job-preparation-001.*、reports/s5-calibration-inventory-001.*、reports/s5-calibration-linux-boundary-001.json、reports/s5-calibration-private-snapshot-boundary-001.json。当前fit未执行，原门未通过。
