# STATUS

更新时间：2026-10-14。当前里程碑：S0；P0工程冒烟完成，终局S0–S6及FINAL未完成。

## 当前票与结论
- bootstrap:01–08、p0:01–07：done。P0验收见reports/p0-engineering-acceptance.md。
- 原始P0/S0阶段归属已审计纠正；严格概率门槛始终失败，S1不放行。
- s0:01 claimed：teacher_head_conversion，/tmp/lite-omnijev-s0-teacher；真实CPU chroot/dropuid/seccomp转换已完成并合入，host canary/网络拒绝通过；root复核父进程复制输出的symlink风险，补丁仍待合入，不准入教师推理。
- s0-data:01 done：修正后远程15250条/33答案token全部通过，逐行语义保真和agent抽检完成；正式证据clinc-admission-corrected-001.json，旧报告不作准入证据。
- s0:03 open：原媒体子代理未交付指定切片，已结束；等待离线媒体依赖准备，严格gate仍失败。
- s0:05 blocked：canary导入/双标κ/脱敏工具已合入，12个sealed-gate测试通过；真实隔离身份仍缺。
- 其余S0按依赖推进；实时前沿以scripts.loop list为准。
- s0:03保留数值失败；s0:06需真实双人标注，agent不能伪造κ；s0:08待基线展开再triage。
- delivery:01未完成，教师、多模态、端侧和终局不能由P0替代。

## 最新证据
- 真实100样本10步67.165秒；完整恢复loss/adapter逐位一致；远端/本地完整备份约561MiB。
- reports/p0-browser-final-triton-001.json：退出0，真实HTTP正负流程通过，无页面错误；model_quality_verified=false。
- FP16独立确认各100题：联合决策变化0，base/LoRA meanTV 0.000705/0.000764，最大TV 0.01129/0.02292。
- 严格gate failed，影子review_required。LoRA一例p(U)跨0.8，不能由argmax相同推断业务等价。
- reports/p0-confirmation-review.md解释尾部；确认批不再用于选阈值。configs/gates.json未改，D004未应用。
- reports/clinc-admission.json：CLINC train 15,250条转换为合法Choice样本，原始人工intent保留；oos不是U，输出SHA256 5aace71f0c9af2b30630e2b2b379881e1ebf946221c59355e733fe496e5971d8。
- reports/sealed-gate-canary.json：封存gate canary路径通过，输出只含汇总；真实隔离身份和真实人工κ仍未通过。
- e73f6d0当前验证：`uv run pytest tests/test_sealed_gate.py tests/test_admit_clinc.py tests/test_core.py tests/test_assets.py` 79 passed；相关ruff通过；`uv run mypy eval scripts tests/test_sealed_gate.py tests/test_admit_clinc.py` 通过。

## 下一步与加速
1. 继续s0:01教师来源/环境/head准入；已到小文件先做head安全转换/隔离探针，等待权重完成后再GPU小样本。
2. 固定教师14文件约24GB隔离下载中，远端PID11060，当前直接流式下载会断开，正在验证Range分块恢复路线；不等同admitted。教师与学生媒体GPU探针由root串行协调。
3. s0:03只验证新的可检验原因，不重复已失败配置；正式改门槛需证据+ADR+用户决定。
4. 全S0通过才进入S1，不为利用率跑无验收价值训练。

## 实例与资源
- 用户此前明确保持GPU开启；SSH root@region-42.seetacloud.com:48926。当前GPU无进程占用，新增GPU作业暂停，待权重完整和媒体/教师探针准备好再使用。
- ControlPath=/tmp/omnijev-ssh-%C；项目/root/autodl-tmp/lite-omnijev。
- H20 97871MiB、16CPU/150GB RAM；盘800GiB（750GiB付费），约741GiB空余。
- 实例重启后No devices found；旧11060/1408已终止。全模型队列flock PID1827已确认运行，报告reports/project-model-download-001.json；60文件约125MB已校验、0失败，大权重仍在下载。
- serve-next隔离vLLM0.31/TF5.17/torch2.13，FP16/TRITON独立LoRA；训练.venv独立。
- E4B revision ee0ef6023621cff504d758262d4e04895a5af4a2；adapter SHA 97db02cd2bb3a4990fb68ceb3c3e863b9a2cb72669406926745a9e4c052b1b42。
- SwanLab https://swanlab.cn/@maxen/Lite-OmniJev/runs/1mia2y76；密钥仅远程私有配置。
- vLLM初始化时禁止其它GPU进程分配/释放显存；初始化后先核实余量再顺序HF作业。

## 待我决策
- GPU作业全部暂停，待下载/哈希/教师环境/媒体探针准备齐再通知是否值得用卡跑；用户若自行切无卡，需提供新的SSH或确认旧连接仍有效。D004搁置。
- 正式数值门槛修订尚缺充分方案，先保留失败并推进独立工作。
- 真实人工标注后续需合规人工资源；root开发身份不能凭chmod实现不可读封存，需独立受控身份。先完成canary协议/工具，不冒称隔离通过。
- 放量预算、付费、生产、删数据、方向/门槛改变仍须用户决定。

## 卫生与交接
- 仅本地commit/SSH，无push、远端issue/PR或gh轮询。
- 保留模型/数据/缓存/训练产物，不关卡，不碰用户进程。
- 交接plan/handoff-2026-10-14-p0.md；循环plan/LOOP.md。

## 本轮执行纠正
- 网络直连超时报告已保留；SSH反向仅远端loopback18081复用本机代理，官方HTTP200，固定教师权重下载可续传。切换重启后需重建转发。
- 部分子代理曾越界写main；已中止CLINC代理，现有产物保留待root复核，不将未审查training_admitted当完成证据。后续所有命令显式workdir，主会话独占票据/STATUS/合入。

## 最新用户范围：全项目下载后再开卡
- 2026-10-14用户明确要求全项目所需下载齐备再开卡，新增offline-prep:01–04规格/票据；不再只准备S0。
- 已核实实例重启为无GPU，项目盘完整；root恢复SSH同endpoint/port，无需新凭据。原下载PID11060已不存在，按缓存续传；反向代理需重建。
- 模型/数据/依赖/集成准备分票；未来训练生成的LoRA/校准/量化产物列待生成，账号/许可/封存身份缺口保留blocked，不假报离线全部就绪。

## 全项目离线准备执行前沿
- offline-prep:01 claimed project-model-assets；七项模型/依赖共约121GB，configs/project-model-assets.json已合入，教师固定5addda86不变；scripts.stage_models两线程下载/续传/哈希，pt只进quarantine，不加载权重。
- offline-prep:02 claimed project-data-assets；configs/project-data-assets.json覆盖29来源，已纠正OneJev身份和下载/训练准入边界。Valen约6.36GB已启动隔离原料下载，flock PID4435，报告valen-source-download-001.json；只用于来源审计，不作训练准入。Clotho有1659条CC0/CCBY音频可准备，精确白名单提取待实现；其余媒体/账号/许可缺口继续核实。
- offline-prep:03 claimed offline-runtime；训练/教师/服务/媒体/量化/端侧工具与离线依赖审计执行中。
- offline-prep:04依赖前三票完成，全部必需资产/CPU准备通过后才通知开卡，缺口不准pending冒充ready。
- 最新本地205tests/Ruff/format/mypy/build全通过，实际输出reports/offline-prep-local-validation.txt。
- 当前优先下载链路：source /etc/network_turbo（不输出环境凭据）+HF_ENDPOINT官方；16MiB Range实测206、3.30秒、约5MB/s。备用loopback18081转发32MiB需278秒，优先不用。

- 本轮新增教师CPU转换实际10tests通过，固定头4tensor/shape/dtype/bit-roundtrip有报告；GPU教师加载/复现仍未执行。
- 实时下载证据已留本地snapshot（project-model-download-snapshot-001、valen-source-download-snapshot-001），均为running，不作下载完成证据。
