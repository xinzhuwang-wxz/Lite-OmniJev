# Handoff: SOTA 冲刺窗口 (2026-10-16 19:00 → 2026-10-17 20:30)

## 唯一接续入口
- plan/STATUS.md (287f312, 全面重写含八条实践/解禁/阶梯/队列/HF/许可/卡片)
- 本文件补充当日细节

## 卡片状态
- **关机非销毁**: AutoDL 磁盘保留, 开卡后 SSH 地址恢复 `ssh -p 48926 root@region-42.seetacloud.com`
- SSH 密钥已装(`~/.ssh/id_ed25519`), `~/.ssh/config` 有 `seeta` 别名
- 连接不稳定时用: `ssh -o ControlPath=none -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519 -p 48926 root@region-42.seetacloud.com`
- **开卡后所有远端产物原样**: 模型 112GB / 数据池 / s5-calibration-code-fa83acc / 全部脚本与证据

## 本窗口完成

### A. 交付候选(不变)
- **D3P-600** (adapter 767be09b) + FIT011 七温度
- 封存 6000 题: video 72.6%(+0.6), 零掉分, ECE 4/6 过门
- 发布目录 lite-omnijev-release-001/ 六件套完整

### B. 执行器 v4 oracle = 94.67%(超教师线)
- CLEVRER 四类题真值世界 CHECK: D100/E100/P97.3/C97.1(per-option)
- 版本迭代: v1 89.3 → v2 92.0 → v3 90.7(回滚) → **v4 94.67**
- v4 修复: 链位平局符号翻正 + motion 窗口语义(被引用时间窗内运动)
- 证据: reports/explore/s4-ccd-oracle-001.md, oracle-v4-verdict.json(盒上)

### C. RL 形式化器训练(60 视频)
- F1 三连涨: .362 → .438 → .459(all continue)
- GRPO G=4 + KL 锚 0.05 + SFT 混 0.3, peak 22.9GB
- final adapter: s6-formalizer-rl-main/final
- **级联实测(RL 场景 + 执行器 v4): 118/150(78.7%) < 直答 123(82%)** — 按预注册树停止
- RL 场景数据: 42(solo)+ 109(3 分片 rescue)= 150 全量, merged-rl-150.jsonl

### D. 三轮调研(前沿/维度/第一性原理)
- reports/explore/s4-frontier-research-00{1,2,3}.md
- 核心发现: 96 档=符号执行(推理时), 纯直答天花板 70-77, 我们 82 已超纯直答
- 8+10+8=26 个 idea, N1 形式化器/N3 仲裁/N2 聚焦抽取为 top3

### E. 证伪路线(七条,全部留证)
同骨干集成 / COMBO+SAS-full 混合稀释 / 自适应取帧 / 温度跨候选转移 / 纯 gold RL / K3 自一致(事件缺口非采样噪声) / v3 三信号并集(回归)

### F. SOTA 榜单范围(ADR0025 已入 AGENTS.md)
- **判定榜**: Decision Index + JevBench 231 + MMAU test-mini 1000 + MVBench 决策子集 + 公开评测集
- **判定维度**: 准确率 + 可靠性 + 成本
- **对照**: OneJev-4B / Valen-2B / Jev-Omni-12B / OneJev-27B(全部超过)
- **不是判定榜**: CLEVRER(试验场)/ 封存池分数 / dev 切片
- SOTA冲刺2 已提交 ADR0025(06da082), AGENTS.md 已指向

### G. HF 发布(整体 DEFERRED)
- 用户指示: 等最终模型出炉后从第一性原理原生生长
- 硬要求: 传播学(精致 Demo/榜单排名凸显)+ 易用性(多格式部署+快速上手)
- 许可: CC-BY-NC-SA-4.0 推荐(研究自由+商用需同意,SA 堵下游绕过)
- 基座: Apache-2.0(README 声明, gemma_4_license 链接待确认)
- hf-repo/ 旧件降级为参考资料

## 未完成(下一窗口)

### 1. MMAU/MVBench 评测(四榜空白补齐)
- **需要**: scripts/public_bench_samples.py 已上传盒上 scripts/ 目录
- **待做**: 转换 MMAU parquet + MVBench 数据 → Sample 契约 → s1_dev_eval 评测 D3P-600 + 基座
- **MMAU parquet**: /root/autodl-tmp/lite-omnijev/artifacts/data/research-eval/mmau/ccd9696c0111ea7060827598f310558df0b71b0a/test_mini.parquet
- **MVBench**: /root/autodl-tmp/lite-omnijev/artifacts/data/research-eval/mvbench/230a2d4fac8900333c61754641c7a13e069ac9c6/
- **帧清单**: /root/autodl-tmp/before-gpu-release-4366197/artifacts/public-video-protocol-freeze-001/canonical-frame-manifest.json
- 教师同协议: MMAU 62.0% / MVBench 54.22%

### 2. Decision Index 27 子集打分
- 已准入 27/44 子集; 评测待终版模型

### 3. JevBench 对照模型补测
- OneJev-4B / Valen-2B adapter 是否有公开权重可跑

### 4. RL 线(已停, 下一窗口评估)
- RL 形式化器 60 视频训练完成(F1 .36→.46)但级联 118 < 123
- 更多数据(150/全池)/更多步数可能改善但未证
- 执行器 v5(EXIT_REL 4 错)可提 oracle 上限但不解决感知瓶颈

### 5. S6 量化/E2B + Playground
- 未启动; S6 = 量化/E2B/端侧; Playground = 浏览器验收

## 证伪路线(七条, 勿重试)

1. 同骨干候选概率集成(错误同源, 无红利)
2. COMBOFULL 三源混合 4093 行(稀释, 118/150)
3. SAS-full 全族 3215 行(稀释, 116/150)
4. 自适应取帧(平局 122 vs 123)
5. 温度跨候选转移(ECE 变差 .084→.095)
6. 纯 gold RL(数学等价 CE, 无新信号)
7. K3 自一致(事件缺口为感知瓶颈非采样噪声)

## 关键教训

1. **ssh 不稳定**: 长命令必死; scp 脚本 + 短命令执行; screen -dmS 免疫断连
2. **heredoc 引号**: 远端写脚本必须 <<'EOF' (引号定界)防变量展开
3. **路径教训**: $O 变量多重展开混淆致 3 轮错误目录; 脚本用显式绝对路径
4. **发射幂等**: launch 脚本内嵌 pgrep 守卫, 防重复发射写坏输出
5. **SSH 会话清理**: 容器 ssh 退出会杀未 GPU-attach 的子进程; GPU-attach 后存活
6. **并发模型加载**: 3×16GB 同时加载会 I/O 争用 + cgroup 内存挤压; 错峰 75s+ 不够, 需 3min+

## 盒上关键路径

| 内容 | 路径 |
|---|---|
| 基座模型 | /root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2 |
| 教师模型 | /root/autodl-tmp/lite-omnijev/artifacts/models/(OneJev-27B) |
| D3P-600 final adapter | /root/autodl-tmp/s1-text-lora-001/artifacts/s6-formalizer-rl-main/final |
| RL 训练 rl-log | /root/autodl-tmp/rl-logs/main.log |
| RL 场景(solo 42) | /root/autodl-tmp/s1-text-lora-001/artifacts/s4-ns-cascade-001/scenes-k3-rl.jsonl |
| RL 场景(rescue 109) | /root/autodl-tmp/s1-text-lora-001/artifacts/s4-ns-cascade-002/rescue/ |
| 合并 150 场景 | /root/autodl-tmp/s1-text-lora-001/artifacts/s4-ns-cascade-002/final/merged-rl-150.jsonl |
| 质量门终判 | /root/autodl-tmp/s1-text-lora-001/artifacts/s4-ns-cascade-002/final/gate.json |
| 级联 oracle v4 | /root/autodl-tmp/oracle-v4-verdict.json |
| 执行器 v4 脚本 | /root/autodl-tmp/s1-text-lora-001/.scratch/s4/ns_cascade.py |
| RL 脚本(修复版) | /root/autodl-tmp/s1-text-lora-001/.scratch/s4/run_formalizer_rl.py |
| FIT011 温度导出 | /root/autodl-tmp/s1-text-lora-001/reports/s5-calibration-fit-011.artifacts/ |
| 校准工具链 | /root/autodl-tmp/s5-calibration-code-fa83acc/ |
| MMAU parquet | /root/autodl-tmp/lite-omnijev/artifacts/data/research-eval/mmau/ccd9696c0111ea7060827598f310558df0b71b0a/test_mini.parquet |
| CLEVRER 注解 | /root/autodl-tmp/lite-omnijev/source-audit/clevrer-official/files/annotations/ |
| CLEVRER 视频 | /root/autodl-tmp/s3-media-candidates-001/video/media/ |

## 报告/证据索引(本地仓)

| 文件 | 内容 |
|---|---|
| reports/s5-sealed-aggregate-summary-00{1,2,3}.json | 三版封存汇总(e465/D3P-150/D3P-600) |
| reports/s5-holdout-eval-00{3,5,8}.json | 三版封存 holdout 原始 |
| reports/s4-parent-jevbench-001.json | JevBench 父基线 |
| reports/s4-program-jevbench-001.json | JevBench D3P-150 |
| reports/explore/s4-frontier-research-00{1,2,3}.md | 三轮调研(26+ idea) |
| reports/explore/s4-leaderboard-sota-map-001.md | 榜单映射 |
| reports/explore/s4-sota-macro-plan-001.md | 宏观冲刺方案 |
| reports/explore/s4-ns-quality-gate-001.json | 级联质量门 |
| reports/explore/s4-ccd-oracle-001.md | 执行器+物理 oracle |
| docs/adr/0024-u-scope-decision.md | U 范围决策(含两次修订) |
| docs/adr/0025-sota-benchmark-set.md | SOTA 判定榜锁定 |
