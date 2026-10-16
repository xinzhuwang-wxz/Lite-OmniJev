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

2026-10-16 20:20 fit007后接续序列(预登记,防断session):
1) fit007终态核验:reports/s5-calibration-fit-007.json status=fitted、processed=18000、runtime_recipe/七温度导出于reports/s5-calibration-fit-007.artifacts;elapsed/exit记录。
2) holdout准备:PYTHONPATH=/root/autodl-tmp/s5-calibration-code-fa83acc runtime python .scratch/s5/prepare-private-holdout-job.py --calibration-job /root/autodl-tmp/s1-text-lora-001/artifacts/s5-calibration-job-004/job.json --calibration-job-sha256 af4b7b48924a8b537b0277b2ab3f3e7ab9b1aac9879d7e93ba8807577a4a6b8c --producer-revision fa83acce6f4b0345a2694ec883a81647a6279f8d --output /root/autodl-tmp/s1-text-lora-001/artifacts/s5-holdout-job-001 --gates configs/gates.json --calibration-receipt <fit007 receipt路径>(需先scp .scratch/s5/prepare-private-holdout-job.py 到远端);脚本是元数据级。
3) holdout启动:private_gold_job --job holdout-eval --spec <job-001/job.json> --sha256 <hash> --work-root /var/tmp/omnijev-calibration-jobs-001 --report .../s5-holdout-eval-001.json --worker-uid 65401 --probe-uid 65402 --timeout 14400(先核UID free)。
4) S5聚合:原门(ECE≤.03/coverage@5%≥.65/掉分≤.5pp/置换≤.02/正反≤.05)如实观测,未达按ADR0023单项披露;服务registry按fitted/unfitted 18分区激活,同engine才激活。
5) GPU排期:holdout优先于创新试点;B1试点240s、B2 caption在S5验证后、01:30硬停前酌情插入,由main现场裁决。

2026-10-16 21:10 holdout媒体预检与修复(全部sha256验证,零失败,原件不动):gui-gold-002 holdout 59张原始名png从quarantine GUI源恢复;clotho-gold-001 holdout 40个原名wav从quarantine clotho-aqa/approved-audio-002恢复(root:root 0600)。holdout行URI布局与校准池不同,prepare-private-holdout-job.py已按文件级media_root补丁(clotho→clotho-gold-001,speech→crossmodal-source-bundle-001,vqa→s4-gold-inventory-001根);七文件全量预检0缺失。新墙钟:用户21:00改口"10-17 06:00前准备好模型即可,性能超榜,方式不限,全权负责"——创新GPU放宽至~03:30硬停,04:30冻结,04:30–06:00导出,06:00–09:00验收(见.scratch/s4/wallclock-budget.md)。

2026-10-16 23:05 里程碑:fit007真实完成(status=fitted,18000/18000,7203.8秒,exit0):七组温度全部NLL改善——text 1.433(.0339→.0276)、image 1.740(.2557→.1998)、screenshot 1.848(.1088→.0822)、audio-choice 0.05边界(近零NLL)、audio-noul 2.426(.659→.471)、video-noul 1.728(.571→.530)、crossmodal-noul 3.076(.833→.453,正对应ECE.25过自信);导出目录+raw receipt+全绑定链核验。holdout链路:001死于gates相对路径(unclassified),002死于main误用截断sha(已kill,UID未消耗),**003运行中**(UID65501/2,job sha 4f08542c…,绝对gates修复,6000行,已过全部probe进media_snapshot);音频DEV绑定缺冻结原版报告(79c09b0a≠61774ce2)四模态绑定+披露。D3P教师蒸馏为holdout后第一GPU试点(ADR0024对齐)。

2026-10-16 23:20 交付策略与放大阶梯(用户要求平衡增量与大量训练、限时超榜):关键约束=任何新权重使fit007温度失效,重校准18000题需~2h,今晚不可容纳→**今晚可交付模型=e465+fit007七温度**;D3P/B2/SAS试点产证据与下一代候选,不替换交付物,除非后续窗口允许完整重校准。D3P放大阶梯(预注册):①150题试点(教师~15min+学生240s)门=video150≥121/150或关系/存在≥+3pp且DEV50不回退;②过门→教师扩600题(~25min)+学生480s/80步,再过同门;③再过门且距03:30≥1.5h→教师扩1500题(~60min)+学生900s;全8299(4.8h)留下一窗口。无权重服务侧增益(按证据采用):w≈0.25混合(需双前向,ADR)、PKI插值(零训练)、C1共形包装——均记入发布说明可选增强。e465+温度为S5校准/汇总对象。

2026-10-17 00:20 S5封存首批汇总(reports/s5-sealed-aggregate-summary-001.json,holdout 6000题2707.4秒exit0):**准确率校准零损失六模态全过(≤.5pp门)**;ECE门3/6过(text .0013/screenshot .0117/image .0193),audio .0308差.0008/video .0557/crossmodal .0480未达→ADR0023披露;Brier/NLL全改善(crossmodal NLL −.422最大);同源族封存/开发差video 2.7pp≤3✓;cov@err video .163/crossmodal .002为下步攻击点;runtime_recipe(bf16/transformers/lora)落盘。D3P:教师150/150记录+学生40步完成(adapter 82525f39),video150门评估中;32帧探针因eval CLI仅{8,16}首跑即拒,已建s4-eval-32f-001补丁树(单行扩展)重发。holdout链累计消耗UID 65401-65502。
