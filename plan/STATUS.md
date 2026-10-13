# STATUS

更新时间：2026-10-14。当前里程碑：S0；P0工程冒烟完成，终局S0–S6及FINAL未完成。

## 当前票与结论
- bootstrap:01–08、p0:01–07：done。P0验收见reports/p0-engineering-acceptance.md。
- 原始P0/S0阶段归属已审计纠正；严格概率门槛始终失败，S1不放行。
- s0:01–08按依赖推进；s0-data:01已解锁。实时前沿以scripts.loop list为准。
- s0:03保留数值失败；s0:06需真实双人标注，agent不能伪造κ；s0:08待基线展开再triage。
- delivery:01未完成，教师、多模态、端侧和终局不能由P0替代。

## 最新证据
- 真实100样本10步67.165秒；完整恢复loss/adapter逐位一致；远端/本地完整备份约561MiB。
- reports/p0-browser-final-triton-001.json：退出0，真实HTTP正负流程通过，无页面错误；model_quality_verified=false。
- FP16独立确认各100题：联合决策变化0，base/LoRA meanTV 0.000705/0.000764，最大TV 0.01129/0.02292。
- 严格gate failed，影子review_required。LoRA一例p(U)跨0.8，不能由argmax相同推断业务等价。
- reports/p0-confirmation-review.md解释尾部；确认批不再用于选阈值。configs/gates.json未改，D004未应用。
- a9ec4ae前置175tests/Ruff/mypy/build通过；身份及源码hash见报告。

## 下一步与加速
1. 认领s0:01教师来源/环境/head准入，同时独立推进s0-data:01 CLINC准入。
2. 教师准备通过才做GPU小样本/基线；批量准备后集中GPU运行，复用权重，避免重复加载。
3. s0:03只验证新的可检验原因，不重复已失败配置；正式改门槛需证据+ADR+用户决定。
4. 全S0通过才进入S1，不为利用率跑无验收价值训练。

## 实例与资源
- 用户明确保持GPU开启；SSH root@region-42.seetacloud.com:48926。
- ControlPath=/tmp/omnijev-ssh-%C；项目/root/autodl-tmp/lite-omnijev。
- H20 97871MiB、16CPU/150GB RAM；盘800GiB（750GiB付费），约741GiB空余。
- 快照vLLM9031/网关10404、GPU占29493MiB、无训练；操作前核实PID。
- serve-next隔离vLLM0.31/TF5.17/torch2.13，FP16/TRITON独立LoRA；训练.venv独立。
- E4B revision ee0ef6023621cff504d758262d4e04895a5af4a2；adapter SHA 97db02cd2bb3a4990fb68ceb3c3e863b9a2cb72669406926745a9e4c052b1b42。
- SwanLab https://swanlab.cn/@maxen/Lite-OmniJev/runs/1mia2y76；密钥仅远程私有配置。
- vLLM初始化时禁止其它GPU进程分配/释放显存；初始化后先核实余量再顺序HF作业。

## 待我决策
- 当前无需租卡/开卡操作，不重复询问；D004搁置，不将质疑门槛视为补丁批准。
- 正式数值门槛修订尚缺充分方案，先保留失败并推进独立工作。
- 真实人工标注后续需合规人工资源；继续其它票。
- 放量预算、付费、生产、删数据、方向/门槛改变仍须用户决定。

## 卫生与交接
- 仅本地commit/SSH，无push、远端issue/PR或gh轮询。
- 保留模型/数据/缓存/训练产物，不关卡，不碰用户进程。
- 交接plan/handoff-2026-10-14-p0.md；循环plan/LOOP.md。
