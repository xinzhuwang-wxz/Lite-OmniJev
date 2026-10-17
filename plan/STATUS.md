# STATUS

更新:2026-10-17 19:20。**交付候选 D3P-600 已封存验证(完整);SOTA 冲刺进行中:RL 形式化器训练完成(F1 三连涨),级联判定场景生成待一条命令续跑(连接抖动中断,重启命令已备)。原生 goal 仍 active,SOTA 判定数 = 级联实测 vs 教师 94.2。**

## ⚡ 19:40 更新(SOTA冲刺2 接管,SOTA冲刺1 副手监控)

- **勿再运行 launch-rest*.sh / launch-staggered.sh**(后者会 rm 正在写的分片)。三分片在跑,数据经 /proc fd 抢救到 `s4-ns-cascade-002/rescue/`;`autogate` screen 采样结束即自动出 `s4-ns-cascade-002/final/gate.json`。
- 诊断与同题锚点见 `reports/explore/s4-sota-takeover-diagnosis-001.md`:教师 141/OneJev-4B 133/D3P 123;差距 15/18 在 enter 题族 → PTR 逐帧存在时间线探针进行中(`artifacts/s4-ptr-probe-001/`)。
- 下方旧接续点的 launch-rest2 命令已作废。

## ⚡ 当前接续点(新 session 第一动作)

盒子开机后(SSH: `ssh -p 48926 root@region-42.seetacloud.com`,密钥已装 `~/.ssh/id_ed25519`,详见 ~/.ssh/config 的 seeta 别名):

```bash
bash /root/autodl-tmp/launch-rest2.sh   # 已在盒上,启动剩余 109 场景生成(~2h)
```

完成后:
```bash
cd /root/autodl-tmp/s1-text-lora-001/.scratch/s4 && PYTHONPATH=/root/autodl-tmp/s1-text-lora-001:/root/autodl-tmp/s1-text-lora-001/.scratch/s4 /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python ns_quality_gate.py /root/autodl-tmp/s1-text-lora-001/artifacts/s4-ns-cascade-002/dev150-questions.jsonl <merged-scenes> /root/autodl-tmp/s1-text-lora-001/artifacts/s4-ns-cascade-002/oracle-scenes-v4.jsonl /root/autodl-tmp/s1-text-lora-001/artifacts/s6-d3p-student-600-001/eval1/records.jsonl
```

**SOTA 判定**:级联实测(cascade_accuracy)vs 教师 94.2%。≥94.2 → SOTA 达成,转部署/HF;<94.2 → RL-80 加训或换方向(按数字共同定)。

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
