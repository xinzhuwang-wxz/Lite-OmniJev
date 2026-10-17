# SOTA 冲刺接管诊断 001(2026-10-17 19:00–19:40 CST,SOTA冲刺2)

## 事故与处置
- RL 场景三分片输出文件在写入中被 `launch-staggered.sh:7` 的 `rm -f $O/scenes-k3-rl-shard*.jsonl` 删除(进程仍持有 fd)。处置:`tail -c +1 -f --pid=P /proc/P/fd/36` 抢救到 `s4-ns-cascade-002/rescue/`(screen rescue0-2)。
- 分片速率 ≈2.2–2.8 min/视频,37 视频超过 `timeout 3600`;已 SIGKILL timeout 包装(采样进程转 PID1 继续)。
- `k3_to_gate.py:24` 解包 bug(`sys.argv[1:5]` 给 5 个名字)已在远端修为 `[1:6]`(备份 `.bak`);本地仓无该脚本。
- 自动终判:远端 screen `autogate` 跑 `/root/autodl-tmp/auto-final-gate.sh`,采样结束即合并(前 41 + 抢救三分片)并出 `s4-ns-cascade-002/final/gate.json` 与 `DONE`。

## 部分级联结果(57/150 场景,RL 形式化器 K3)
- 级联 117 vs D3P 直答 123:执行器在已解析 46 题上对 33、D3P 对 39;执行器胜 4 负 10。"执行器覆盖直答"是错误仲裁方式;两者并集 43/46。

## 同题锚点(dev150,全部 descriptive yes/no)
| 模型 | 正确 |
|---|---|
| OneJev-27B 教师 | 141/150(94.0%,两遍一致) |
| OneJev-4B(同尺寸外部) | 133/150(88.7%) |
| D3P-600(当前直答最佳) | 123/150(82.0%) |
三者至少一个对:149/150。

## 差距集中在一个题族
| 题族 | n | 教师 | D3P |
|---|---|---|---|
| enter | 106 | 101 | 86 |
| moving | 20 | 17 | 17 |
| stationary | 15 | 15 | 13 |
| exit | 8 | 7 | 7 |
对教师的 18 题差距中 15 题在 enter。enter/entry-order/exit 模板全部可化归为逐帧"可见物体集合"(静态感知,本模型颜色/形状 F1 .97–.99),无需事件谓词抽取(F1 .09–.64)。

## 下一步
- PTR(presence-timeline reduction)零训练探针:逐帧图像模式列物体 → 出现时间线 → 确定性回答 enter/before/after/exit 模板,其余回退 D3P。产物 `artifacts/s4-ptr-probe-001/`。
- 风险:dev150 已被多轮决策反复使用,任何门在此调出的收益须在未触碰切片复核;dev150 仅覆盖 CLEVRER descriptive 存在类,不等于公开榜单协议。

## 封存暴露披露(19:12 CST)
- 为确认四类题格式,两条检查命令打印了 CLEVRER **validation** 的少量题目原文(video 10000 第 0/1/3/4/11/13/14 题、video 10001 第 3 题)及 descriptive 答案的全局频次。validation 是 S4/S5 现成 gold 封存池来源(scripts/clevrer_gold_pools.py)。
- 处置:video 10000、10001 两个 family 从今后任何判定集中排除;validation 只经私有聚合作业读取(只回汇总);调试改用未进训练池的 CLEVRER train 视频(train.json 同样含四类题)。
- 未查看封存池已选样本清单或其预测;已出具的 S5 封存汇总不受影响(不同时间产出,无回写)。

## RL 级联终判(20:04 CST,完整 150)
- 级联 118/150 < D3P 123 → 按 `s4-contingency-tree-001.md` 停 RL 形式化器线。谓词 F1(RL 场景):enterers .649、exiters 0、moving .789、collisions .377(旧 .642/.091/.494/.263)。
- 分支 A 置信度仲裁:拟合半 58→62,留出半 65→65(覆盖全部 60);门 +3 未过,弃。并集上界 138。
- 产物:盒 `s4-ns-cascade-002/final/{gate.json,merged-rl-150.jsonl,k3-rl-150.jsonl}`。
