# s4-b1-grid-prep-001 — B1+I1(WSFT) 程序格:CPU 构建与 240s GPU 试点规格

状态:**CPU 构建完成并全量校验(含 WSFT 目标层);GPU 试点待 main 排期(本任务不启动 GPU)。**
升级记录:2026-10-16 晚按 main 合并指令并入前沿报告 I1(WSFT);WSFT 作为目标函数层叠加,原 126 题格数据不变。

## 问题

候选 e465 残余错误集中在 同视频关系(enter/collision 顺序)与事件/对象存在;8 条已停路线都在
逐题目标内打转,没碰过 (a) 同视频极性孪生的结构一致性与 (b) 程序真值可判定的不可答(p(U))监督。

## CPU 构建(零 GPU,已实跑)

- 脚本 `.scratch/s4/build-b1-consistency-grid.py`(stdlib;远端 /tmp 实跑 21:0x,输出拉回 worktree)。
- 输入三件全部 SHA 钉住:官方 questions/train.json `11181da6…`、annotation_train.zip `e7f9185a…`、
  TRAIN 池 train.jsonl `912c871f…`(8299 video family,只用该池,禁新数据源)。
- **选中 family:clevrer:5218(video_05218.mp4,TRAIN 池内,media sha 76bd…见 manifest)**,126 题:
  - order_direct before/after 孪生 12(入场帧序,逐题 gold=帧比较)
  - order_official(官方句式 relational enter)12
  - exist_begin/end 12、exist_absent(可答"no")16
  - collision_order 孪生 20、enter_collision 混合序 30
  - **u_premise_order 24(程序构造不可答:不存在对象的前后序,gold=U,InsufficientTarget)**
- 单 family 上限实测 ~120(入场对象稀少),126=1 family 恰好进带;若扩家族,select_groups 按
  孪生原子+优先级确定性扩,不需改代码。
- 产物:`artifacts/s4-ensemble-oracle-001/b1-grid.jsonl`(sha `41ebacc4…`,126 行)+ 同名 manifest
  (sha `3bac13b6…`,含每题程序推导、twin 组、transitivity 组)。
- **校验实跑通过**(`.scratch/s4/validate-b1-grid.py`):126 行全部过 Sample 契约(pydantic);
  37 对孪生 gold 严格互补;order_direct gold==帧数值比较;6 个传递三元组一致;24 U 题
  kind=insufficient;media sha/uri 与 TRAIN 行一致;license CC0-1.0、pool=train 不变。

## 与已停路线的区别

collision 顺序纯 CE gold(TRAIN12)已停(泛化无收益);B1 新增维度=极性孪生结构 + 真实 U 监督 +
混合事件序;试点门直接测 video150 泛化,不重跑已停配置。

## WSFT 目标层(I1,已实跑构建)

- `.scratch/s4/build_b1_wsft_targets.py`:对 126 题逐项算谓词可观测性——用该视频真实
  16 帧均匀采样索引(128 帧总长,采样点 0,8,…,127)判定 order/collision 谓词在输入上是否可分辨:
  - 两个事件帧之间无采样帧隔开 → `unresolvable`(世界真值确定但感知不可分辨)→ 行复制 7:3 hard:U;
  - 恰有一个隔开采样帧 → `bracketed_tight` → 17:3(≈0.15 U 质量);
  - 可分辨 → SoftTarget [1−ε, ε](ε=0.05,远低于报告软质量上限 0.3,退火由 LR 承担);
  - 程序不可答(fake premise)保持 InsufficientTarget。
- 产出:`b1-wsft-grid.jsonl` **292 行**(hard 138 / soft 88 / insufficient 66,全部 pydantic 通过,
  id 唯一化 `-wsft{k}` 后缀)+ `b1-wsft-ledger.jsonl`(292 行全声明 train,sha 48ccd7a7…)。
- 审计:`b1-wsft-grid.jsonl.audit.json` 逐题记录 observability 与 U 质量;**14 个序关系题在
  16 帧输入上本就不可感知**(10 unresolvable+4 tight)——这批题在纯 hard B1 里是"必错标签",
  WSFT 把它们转为部分 U,正是 I1 命中的谓词组合/可答性结构。
- 契约零改动:SoftTarget 与 InsufficientTarget 都是既有目标类型;部分 U 用整数比复制表达
  (混合 CE 的量化近似,7:3/17:3 精确可复算),不新增目标 kind。

## 240s GPU 试点 JobSpec(main 排期;三臂同种子同数据族,只差目标/损失)

- 数据:`b1-grid.jsonl` + **补充 allocation ledger**(grid id→train,一行一 `{"id":…,"pool":"train"}`,
  随原始 ledger 一起传 `--allocation-ledger`, trainer 可重复参数;不写 cal/holdout)。
- 臂1(B1-CE,现有 trainer 零改动,数据=one-hard 版 126 行):
  `timeout --signal=INT --kill-after=20s 240 <python> -m scripts.train_s1 run --config <cfg> --model-dir <E4B ee0ef60…> --asset-manifest artifacts/manifests/e4b.json --data b1-grid.jsonl --allocation-ledger <原ledger> --allocation-ledger <b1-grid-ledger> --output-dir …/s4-b1-ce-pilot-001 --report … --init-adapter …/adapter-invocation-step-0100 --init-adapter-sha256 e46550143b28…`
- 臂1w(**B1+WSFT,主推**):同命令,`--data b1-wsft-grid.jsonl`(292 行)+
  `--allocation-ledger b1-wsft-ledger.jsonl`;sample_count=292、steps=60。
- 臂2(R1 降级耦合损失,可选第三臂,见 s5-r1-grpo-lattice-prep-001):α·CE(WSFT 目标)+
  β·孪生耦合 `<p_L,p_R>`;需 trainer 小扩展,数学已 CPU 单测,接线未上 GPU,窗口紧可砍。
  config=课程 config 克隆,仅改:sample_count/steps(臂1 126/40,臂1w 292/60)、accumulation=1、
  learning_rate=2e-6、modality=video、media_roots=[/root/autodl-tmp/s3-media-candidates-001/video/media]、
  expected_data_versions=["clevrer-train-video-b1-grid-v1"](wsft 行 data_version 同源)。
- 评测(两臂各自):video150(s3-stable-probe-001/video-dev150.jsonl,sha e9b91fed…)+
  DEV50 固定切片,两遍一致才算数;门判脚本用现有 receipt 对比。
- **预注册门(失败即停,证据保留)**:video150 ≥121/150 **或** 关系/存在子项 ≥+3pp,
  且 DEV50 不回退(≥33/50 基线 0.66…按 e465 实测DEV50 34/50,回退>1 题即停);
  WSFT 附加判据(报告 I1):关系切片增益 ≥+3pt 且其余切片回退 ≤1pt 且 ECE 不变差;
  放弃线:臂1w 与臂1(one-hot)差异 ≤0.5pt(2 seeds)→ WSFT 层不采用。两臂都不过 → B1 路线停。
- 预算:训练 240s×2 + 评测(video150 ~110s + DEV50 ~60s)×2 ≈ **≤15 min GPU**,窗口内可容纳。

## 风险与边界

- 单 family 泛化风险:门直接测 video150 全官方句式,不以 TRAIN 拟合度报喜。
- LR 2e-6 是课程 LR 的 1/10、40 步有界;DEV50 回退即停,不做无监控续跑。
- 不触碰 calibration/holdout;补充 ledger 只声明 TRAIN 派生,不改原冻结账本。
