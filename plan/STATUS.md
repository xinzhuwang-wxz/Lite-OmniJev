# STATUS

更新:2026-10-17 21:05。**负责人:SOTA冲刺3(本轮起)。主线:拼最新预发布候选 → ADR0025 榜单实测差距 → 按宏观规划定提升。GPU 空闲可随时发射;盒上 build 样本作业进行中。**

## ⚡ 接续点(第一动作)

1. **等 MVBench 两份报告落地**(~23:15,`reports/s4-public-bench-mvbench-{d3p600,base}-001.json`)。**MMAU 已出(同协议,单遍确定性)**:基座 65.9% > **教师 Jev-Omni-12B 64.6%**(修复加载后同协议实测,report `s4-public-bench-mmau-jevomni12b-001.json`)> D3P-600 60.6%;教师自家旧协议口径 62.0 仅作参考。**单遍制(用户 22:12 决定)**:两次 MMAU 均 repeat_identical=true,第二遍信息量为零;链式已改 `--passes 1`。
2. **分数齐后填 ADR0025 差距表**,关键议题已浮出:D3P-600(视频 LoRA)在 MMAU 音频上 −5.3 vs 基座,发布形态需考虑按模态路由 adapter(内部管线自由,ADR0024 修订二)或重校准;ECE 未套发布校准,勿直接比较。
3. 同协议对照矩阵:harness 已实现并**真机冒烟通过**(OneJev-4B MVBench 5/5,commit cc2cb25)。关键事实:**OneJev-4B/27B 是 Qwen3-VL,无音频塔,MMAU 对其结构性不可评**(判定表该格记"不可评");其原生榜=MVBench(视频路径 + boundary_tail 尾部边界 + enable_thinking=False)。全量 OneJev-4B MVBench 跑中(screen `cmp-4b`,单遍)。s1_dev_eval 已打**架构感知选类补丁**(Gemma4Unified checkpoint 否则会静默随机初始化双塔,已写进代码注释)。Valen-2B(1.16GB checkpoint.pt + Qwen3.5-2B 底座)正在流式中转。打分数学与 s1_dev_eval 同源(read_probabilities 复用);Jev-Omni 无 comparator 族、无双源交叉,OneJev 系分数一律带 harness 披露;Valen 为 RLCD 自定义 checkpoint,需移植其加载码,矩阵先记 conversion pending。
4. **探索队列(宏观规划 × 今晚证据的对账结果)**:①23:15 MVBench = 最大分叉(D3P≥基座 → 发布形态=模态路由+攻可靠性;D3P<基座 → 下一个训练目标改为榜单形态通用视频决策数据,不加码 CLEVRER);②Decision Index 27 子集打分(唯一未测榜,输入已 staged);③同协议对照矩阵(harness 就绪,烧卡待用户);④剩余预算进 CLEVRER 线(原子训练 ~3 GPU h 门 AUC≥.97,或 D3P-1500 ~25 GPU h,二选一,预注册门不到即停)。

## 现役资产(已核验)

- **基座**:`/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef60…`(112GB 全量在盒)。
- **产品候选 = D3P-600**:`s6-d3p-student-600-001/run/adapter`,adapter sha `767be09b…`(rank16/α32,600 CLEVRER 样本,80 步,video 模态);S5 封存证据 `reports/s5-holdout-eval-008.json` + 校准 `reports/s5-calibration-fit-011.artifacts`(均绑 767be09b)。
- `lite-omnijev-release-001` 内是**旧 adapter-d3p150**(82525f39)——已过时,仅作结构模板;`s5-registry-export-001` 同样绑 d3p150,勿直接复用。
- RL 形式化器是内部管线组件(级联线已闭,见下),**不进发布包**。

## 已闭线(勿重跑)

- **RL 形式化器级联:场景 150/150 全覆盖,终判 118/150 < D3P 123,线停**。`s4-ns-cascade-002/final/gate.json` 为准;`rescue/` 三分片 37/37/35 覆盖 rest-109。`/root/autodl-tmp/launch-rest2.sh` 是 18:54 的过时计划(rescue 已完成同工作),**不要再跑**。merged 文件有 1 行重复 id(clevrer-train-03324-5),不影响判定。
- 已证伪路线清单见 reports/explore/(同骨干集成、COMBO/SAS 混合、自适应取帧、温度转移、纯 gold RL、K3 自一致、v3 并集)。
- G1 原子门未过(presence AUC .949<.97);PTR 门未过。

## MMAU/MVBench 样本管线(本轮修复记录)

- 设计定案脚本:盒 `.scratch/s4/public_bench_samples.py`(632 行,自校验:FFmpeg 无损拼装 16 帧 mp4 + RGB24 回读断言 + 冻结 S5 loader 终检)。MVBench 原始视频不在盒,用教师协议冻结的 canonical 16 帧(`public-video-protocol-freeze-001`)拼装。
- 本轮补丁:①MMAU 46 行内嵌 MP3 → ffmpeg 解码为 pcm_s16le WAV(Media sha 覆盖转码文件,provenance 留 meta);②>30s 音频样本级裁剪(既有约定)。已过 trim 崩溃原因:旧版 stream-copy 超冲,新版 wave 精确切。
- 已知协议差(如实记录):`load_media` 会把帧 scale 到 640 内而教师路径未做;MMAU 46 行 mp3→wav 转码;报告见 `build-summary.json` 的 `trimmed_rows_over_30s`/`transcoded_rows_nonriff`。

## 判定口径(ADR0025 + ADR0026 修订)

- **四公开榜**(ADR0026 移除自建榜):Decision Index(含视觉)/ JevBench 231 / MMAU test-mini 1000 / MVBench 九任务 1800;准确率+可靠性+成本;四对照(OneJev-4B、Valen-2B、Jev-Omni-12B、OneJev-27B)同协议同题全超才称 SOTA。
- **JevBench 为外部提供**(repo `fstandhartinger/jevbench@7625ecb`,公开 original/easy/hard 231 行,未用于训练)——不自建。CLEVRER 按用户指示保留在评测表(方法试验场),不作为 SOTA 声明口径。

## 盒上纪律(传承)

- 长作业一律 `screen -dmS` + 显式绝对路径 + 幂等跳过(report 存在即 SKIP)+ 禁止 `rm` 输出;发射脚本先 `pgrep` 防双跑;`timeout` 按实测速率。
- 多维监控:发射即挂监控(逐阶段 log 关键行/异常/退出码),不等跑完。
- 私有作业前清 /var/tmp overlay 媒体快照(30G 限制)。
- SSH 别名 `seeta`;凭据不入仓;Python env `/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python`(pyarrow/PIL/pydantic/torch 全)。

## 下一窗口:SOTA 收口计划(负责人已定)

**头号标的 = JevBench 缺口 −9.5**(教师发布 86.15/micro 87.45 vs 我方 D3P-600 76.62)。根因是训练规模:对方全参微调 3 万题(12B)/9.9 万题(OneJev-4B),我方候选 = 600 样本 rank16 LoRA(基座 76.19,LoRA 仅 +0.4)。

1. **先创新后规模化(用户令)**:JevBench 错题解剖(e465 记录)显示差距**集中在 long_policy 52.6%/ambiguous 28.6%/multi_hop/judge_hard,简单族 100% 全对**——与 CLEVRER enter 族同构。首个干预 = **CCD-for-decisions**(宏观规划创新库平移):长输入原子分解 → 逐原子校准判断 → 组合决策(ADR0024 结构化执行解禁,协议内多前向),预期 +4~7 分收掉大半缺口,零训练;预注册门:四弱族合计 ≥80%,全榜 ≥84,不达即弃。之后才轮到规模化蒸馏(D3P 管线 + teacher-soft,600→3 万,teacher-cache 已建;门 JevBench ≥86,约束 MMAU ≥63.10)。
2. **本期发布照发**:release-002,模型卡报全部实测数 + 对照公开数 + 差距(ADR0025"失败项照实披露")。
3. **DecisionBench Medium 排除**:两轮检索无公开数据集(Jev-Omni 卡的 80 场景/293 题公开网不存在)= 私有评测,按 ADR0026 排除;引用 87.57 时标"私有评测,不可验证"。
4. **MVBench 口径差**:教师卡 14 任务/2786 vs 我方 9 任务/1800,引用双列标注;同协议教师 MVBench 数由单遍链补。
5. Valen:权重在盒(Qwen3.5-2B 底座 ✓ + RLCD 自定义 checkpoint),需移植其加载码;对照格记 conversion pending。

## 待我决策

- 下一窗口规模化蒸馏的卡时预算(预估 12-30 GPU h:数据构建 + 训练)。
- 开卡/关卡、HF 上传 token 照旧须用户。
