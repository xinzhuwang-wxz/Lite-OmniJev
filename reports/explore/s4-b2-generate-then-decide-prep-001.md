# s4-b2-generate-then-decide-prep-001 — 感知-决策解耦 generate-then-decide:CPU 准备

状态:**CPU 管线(提示/解析/条件化构建)已实现并自测;GPU 步骤(caption 抽样→条件化切片)
待 main 排期。采用需 ADR(4001)+用户批(输入契约变化);试点只测 DEV。**

## 思路与依据

- oracle-001 证明:模型对融合取不出的 +7.8pp 余量在"输入维"——决策层看的是 16 帧+短问句,
  关系/存在错误源于感知信息没进决策前向。B2 把"看"与"判"拆开:冻结基座自由生成结构化
  感知摘要,决策 LoRA 在 [问题+自生成摘要] 上出单 token 分布(训推一体:训练时用的摘要
  =推理时同一基座产生的摘要,无 exposure mismatch)。
- 服务化潜力:summary-only 变体决策 pass 不带视频 token → 端侧/低成本形态;这是 ADR 4001
  的核心卖点,也是契约变化点(决策输入契约从"媒体+问题"变为"摘要+问题(可选媒体)")。

## CPU 交付(全部实跑)

1. `.scratch/s4/caption_probe.py`
   - 感知提示:要求仅输出 JSON,字段=objects(first/last_seen 粗时相)、entry_order、
     collisions(when 粗时相)、present_at_start/end;不确定的排序必须省略而非猜。
   - `validate_summary`:严格结构校验(名字一致、时相枚举、entry_order 引用真实对象、
     不允许自环);`--self-test` 实跑通过(含 4 个反例)。
   - 生成循环镜像已验证的 s0_development_baselines 加载路径(offline、官方 chat template、
     enable_thinking=False、贪心 do_sample=False、max_new_tokens 512);**前向部分未上 GPU,
     接线按现有 API 写好,main review 后运行**。
2. `.scratch/s4/build_b2_conditioned.py` + `test_b2_conditioned.py`(fixture 实跑通过):
   - caption JSONL + 源池行 → 条件化 Sample 行(prefix.state 追加
     "Perception summary (model-generated, probe evidence): {json}");
   - 两变体:`+b2-summary-video-v1`(保留视频)与 `+b2-summary-only-v1`(决策 pass 无视频,
     契约允许:media⊆modality,空 media 合法,pydantic 校验通过);
   - 无效/缺失摘要丢弃并计数,不伪造。

## GPU 步骤(main 排期,预注册)

- 步骤1(caption 抽样,~20–30 min):
  `caption_probe.py --model-dir <E4B> --data <TRAIN pool video rows> --media-root /root/autodl-tmp/s3-media-candidates-001/video/media --limit 100 --output captions-100.jsonl`
  验收:summary_valid 比例(目标 ≥80%;<50% 停);人工抽检 10 条对照程序图
  (annotation_train.zip 真值可对照,scene 级 objects/entry_order)。
- 步骤2(240s 条件化训练切片):用 build_b2_conditioned 产出 TRAIN 侧 summary+video 行,
  复用 B1 JobSpec 形态(lr 2e-6、40 步、init e465);门同 B1:video150 ≥121/150 或
  关系/存在子项 ≥+3pp 且 DEV50 不回退。
- 步骤3(可选,summary-only 形态零训练评测):直接用现有 s1_dev_eval 在条件化 DEV 行上
  评 e465(不训练),测"摘要是否已携带决策所需信息"——若 summary-only 掉幅小,
  ADR 4001 的服务化论据增强。

## 风险与红线

- 摘要是模型生成物,不是真值;错误摘要会系统性误导决策(步骤1 的人工抽检+valid 率是闸门)。
- 决策输入契约变化=采用需 ADR+用户批;试点只在 DEV,不碰 cal/holdout。
- 不改基座、不加分类头、决策读出仍单 token;唯一新自由度=prefix 文本(官方 template 内)。
