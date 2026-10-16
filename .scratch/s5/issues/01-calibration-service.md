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

2026-10-16 19:17–20:10 fit003–007 排障链(全部真实证据,失败报告保留):fit003死于ordinary_identity_probe——生产traverse_acl复现发现8个s1/s2/s3老池sealed行文件0644可被worker经/root穿行打开(真实泄漏,probe正确拒绝),收紧0600后LegA/LegB双过;fit004死于runtime_acl_negative_probe——/root/.ssh为0755目录(worker可open目录),收紧0700/authorized_keys2 0600(对sshd只严不松);fit005死于runtime probe子进程导入——沙箱只拷scripts/train/eval/data而import链需serve.gateway.app,producer修订fa83acc(沙箱+code_hash补serve,沙箱模拟+远程导入闭包CPU验证,ruff/mypy过);fit006死于media_snapshot missing_file——gui-gold-002 119行校准媒体(保留原178中部分)内容不在bundle且不在盘,从quarantine源审计目录(25809文件receipt 00f3248b)逐个sha256验证后恢复到行引用路径(root:root 0600,原件不动),报告s5-gui-media-restore-001.json,119/119零失败。fit007(job SHA af4b7b48,UID65301/2,producer fa83acc,029+receipt002绑定不变)已真实进入inference:text 18000中,GPU~49%/15.8GB,监控中。UID 64801/2、64901/2、65001/2、65201/2、65301/2已永久消耗,诊断用65101/2一并记录;不得复用。诊断全程只读元数据/代码/路径存在性,未读私有行/worker日志。后续:fit007完→导出七温度+runtime_recipe→holdout评估(same machinery JobSpec另备)→S5聚合(原门如实观测,未达按ADR0023单项披露)。
