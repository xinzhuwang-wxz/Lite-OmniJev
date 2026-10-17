# STATUS

更新:2026-10-17 20:25。**负责人交接:SOTA冲刺2(额度将尽)→ SOTA冲刺1 接任负责人。交付候选仍为 D3P-600(封存校准完成)。SOTA 口径已锁定 ADR0025;GPU 当前空闲,盒上无运行作业。**

## ⚡ 接续点(新负责人第一动作)

1. **口径**:只按 `docs/adr/0025-sota-benchmark-set.md` 判定 SOTA——Decision Index(含视觉)、JevBench 231、MMAU test-mini 1000、MVBench 九任务 1800(决策子集)、我们公开的多模态决策评测集;准确率+可靠性+成本;须超过 OneJev-4B、Valen-2B、Jev-Omni-12B、OneJev-27B(同协议同题我方实测)。CLEVRER/封存池/dev 切片不得称 SOTA。
2. **第一优先(GPU)**:在 MMAU test-mini 与 MVBench 九任务上实测 D3P-600 与基座,教师同题已知 62.0% / 54.22%。需要先写"教师同题 → 学生样本"转换器(建议 `scripts/public_bench_samples.py`:复用 `scripts/s0_teacher_eval.py` 的选行逻辑,断言输入 manifest 哈希一致;再用盒上 `s5-calibration-code-fa83acc` 的 `scripts.s1_dev_eval` 打分)。该 fable 任务已在 20:22 停止,无产出。
3. **第二优先**:Decision Index 27 个已准入子集打分(`scripts/decision_index_offline.py` 只准备输入;打分协议按官方 decision-index 代码),JevBench 对照模型补测。
4. 出分榜差距表后再定攻坚方向;创新方法库见下。

## 本窗口(SOTA冲刺2)结论,全部留证于 reports/explore/

- RL 形式化器级联终判 118/150 < D3P 123 → RL 线停;置信度仲裁留出半 +0 → 弃(`s4-sota-takeover-diagnosis-001.md`)。
- PTR 零训练逐帧列物体 121–124 vs 123,门未过(`s4-ptr-probe-001.md`);G1 原子门 presence AUC .949 < .97(`s4-ccd-oracle-001.md`)。
- CCD 自写 CLEVRER 执行器+物理(eval/ccd_*.py):真值世界 D100/E100/P97/C97;敏感度:位置误差 σ≤.05 才可用 → 精确几何须确定性 CV;CV 几何模块未完成,WIP 在分支 `worktree-agent-a35be9dbe956d68a1`(941320c,未验证)与盒 `/root/autodl-tmp/ccd-geometry-work`。
- D3P-600 在 CLEVRER 四类题(dev2 debug40)直答:D 54.3 / E 57.5 / P 58.3 / C 49.0(per-option),说明直答对非 exist 题型接近随机。
- 创新方法库(可迁移到判定榜):CCD 校准原子组合、贝叶斯决策引擎 BDE、主动感知(`s4-ccd-paradigm-001.md`);决策树 `s4-contingency-tree-001.md`;接手总览 `s4-sota-macro-plan-001.md`(其中 CLEVRER 为主战场的表述已被 ADR0025 取代)。
- 封存纪律:曾误打印 CLEVRER validation 少量题(已披露并排除 video 10000/10001);validation 只经私有聚合作业读。

## 盒上资产(SSH 别名 seeta,凭据不入仓)

- 四类题评测集 `artifacts/clevrer-fourtype-dev2-001/`;原子集 `artifacts/ccd-atoms-dev-001/`;RL 终判 `artifacts/s4-ns-cascade-002/final/`;PTR `artifacts/s4-ptr-probe-001/`;训练注释 `/root/autodl-tmp/clevrer-train-ann-001`;validation 解压 `/root/autodl-tmp/clevrer-val-001`(仅私有聚合)。
- 启动脚本禁止 `rm` 输出;长作业用 `screen -dmS`;`timeout` 按实测速率设。

## 高效开发精神(本窗口收敛的实践,传承)

1. **多维度监控,不等跑完**:每个训练/生成立即部署多层监控(逐视频奖励/checkpoint F1 gate/中途 video150 评估/OOM/停滞),3 分钟巡检,异常 3 分钟内捕获处置。监控脚本模式见 /root/autodl-tmp/multi-monitor.sh(reward-check.py 辅助)。
2. **并行分片 + 幂等保护**:独立任务分片并行(3 shard × 37);发射脚本内嵌进程计数守卫(pgrep 检查,重复发射拒绝),防重叠写坏。
3. **集中资源**:竞争进程全杀再单点发射;5 进程争抢的教训——先 pkill 再 launch,一个脚本原子完成。
4. **资源纪律**:达不到 SOTA 的训练不花 GPU(用户令);六条路线证伪即停(集成/COMBO混合/SAS全族/自适应取帧/温度转移/纯gold RL),全部留证。
5. **预注册判定门**:每条路线先定数字门(F1 +0.2/cascade ≥124/oracle ≥94.7%),不到即停,不移动球门。
6. **ssh 不稳定对策**:长命令必死 → scp 脚本 + 短命令执行;screen -dmS 会话免疫断连;setsid+nohup 子进程在容器 ssh 退出后被清(GPU 挂载前),**长作业必须在 ssh 存活期内完成 GPU 挂载**(GPU-wait 循环)。
7. **路径教训**:$O 变量多重展开混淆致 3 轮错误目录失败——脚本用显式绝对路径;heredoc 必须引号定界(<<'EOF')防远端展开。
8. **中途评估器**:训练不停,checkpoint 落盘即自动 video150 评估(midtrain-eval.sh 模式)——训练中的真实性能曲线实时可见。

## 全面解禁(ADR0024 修订二,用户决策)

**内部管线完全自由**:推理时结构化执行、U 作内部信号、多阶段管线、程序执行——全部可用。**唯一硬约束 = 产品输出形态 Jev 式**(Choice/Score/yes-no,无 U 概念)。选型唯一标准 = 最可能超过外部榜单。

## 三层阶梯(实测锚点)

```
96-99%  符号执行档(程序在推理时答题)
94.2%   教师线(OneJev-27B 直答)← SOTA 判定线
90.7%   级联 oracle(执行器 v4 + 完美场景,已证可达)
~82%    ★ 当前直答(D3P-600,dev 切片 123/150)
70-77%  纯直答档(已超越)
```

**超榜路径(已验证组件)**:执行器 v4 oracle 94.67%(超教师)+ RL 形式化器(F1 .36→.46 三连涨,60/60 完成,final 已存)+ 场景生成(进行中)→ 级联实测。

## 已证伪路线(勿重试,证据在 reports/explore/)

同骨干集成(错误同源)、COMBO/SAS-full 混合(稀释)、自适应取帧(平局)、温度跨候选转移(ECE 变差)、纯 gold RL(≡CE)、K3 自一致(事件缺口非采样噪声)、v3 三信号并集(回归)。

## 下一窗口旗舰队列

① RL 形式化器训练(fable 包全备:GRPO G=4+KL 锚+SFT 混 0.3,硬闸门内嵌,commit a541499)② 级联部署(形式化器见效后)③ COMBO 顺序课程(stage1 SAS→stage2 D3P,接力非混合)④ D3P-1500(切片/runner 已备)⑤ S6 量化/E2B ⑥ Playground

## HF 发布(整体 DEFERRED,等最终模型)

**原生生长,不改编预写件**。硬要求(用户令):传播学融入(精致 Demo/榜单排名凸显/注意力设计)+ 易用性(多格式部署:adapter/merged/GGUF 端侧/Mac 兼容+快速上手)+ HF 重点/GitHub 引流/暂不做网站。许可:CC-BY-NC-SA-4.0 推荐(研究自由+商用需同意,SA 堵下游绕过);基座 Apache-2.0(README 声明,gemma_4_license 链接待网页确认)。hf-repo/ 旧件降级为参考资料(CONVENTIONS/许可清单可复用)。上传需用户 HF token。

## 卡片与接续

- 卡片=关机非销毁(AutoDL 磁盘保留),SSH 地址开卡后恢复:`ssh -p 48926 root@region-42.seetacloud.com`(密钥 ~/.ssh/id_ed25519 已装,~/.ssh/config 有 seeta 别名)
- 远端关键产物全保留:模型 112GB、数据池、s5-calibration-code-fa83acc 工具链、formalizer 全套、RL-150 训练目录
- 私有作业前必清 /var/tmp overlay 媒体快照(30G 限制)
- 最近 handoff:plan/handoff-2026-10-15-s3-complete.md(旧);本 STATUS 即最新接续入口
