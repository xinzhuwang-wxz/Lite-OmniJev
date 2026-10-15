# 时序课程120draw CPU准备 001

2026-10-16。真实CPU构建22.2838秒，exit0，`ready_for_preflight`；全库存审计isolated。未训练/GPU，不是GPU就绪或质量通过。产物 `/root/autodl-tmp/s1-text-lora-001/artifacts/s4-temporal-curriculum-pilot-001`，机器证据见同名preparation JSON。旧输入/库存未改。

最终采样固定，不扫比例：原建议40temporal/10motion/10existence改为50/5/5，以83.3%时序形成明确课程变量。temporal25yes/25no，motion2yes/3no，existence3yes/2no；共60个唯一视频family、30yes/30no。每step前5时序按3yes2no与2yes3no交替，第6项motion/existence交替；接同family cross正负两项（10唯一group，两种negative rule各5），最后audio/image/screenshot/text各1。10step×12accum，seed42 inverse-shuffle；逐step和全部120sample SHA独立核验通过。视频从完整TRAIN8299按固定hash选择；没有全历史unseen排除，不声称未见。其他样本严格复用claim-pilot001的已隔离120输入，按(id,data_version)/canonical全字段SHA/来源ledger和显式cohort映射绑定。gold、媒体、组件未改。

初始化由root选择CE0：`artifacts/s4-claim-pair-ce0-train-001/adapter`，实际weights SHA `4b8e422f3ca29e863999d2cd713bb9fd3f0a46e5360faa259a3bb8a4a9814bed`，实际adapter_config SHA `c7570d0569abbb333fb7339fe6aa024a19ea81581c440959396d89027b348574`；与真实completed10/exit0训练receipt SHA `c30ba04f01aae54dd24e816f7713e4c3bcbcc8e0e78706b3b985a65a30aa7675` 绑定，r16/alpha32与该训练config一致。原coverage0deef只是早期默认，未用作本产物init。

`config-ce.json`与原claim CE配置字节SHA完全相同：`3bca7236bc4e2fe8db9ef4742858f5a3ad257288f10d714693e16380e1ea2145`。steps10、sample120、accum12、midway/checkpoint5，lr2e-5、bf16、image1120/video16及所有媒体/优化器超参保持；pair/refKL/brier均默认0。每arm最多240秒，由root launcher控制。本入口不启动作业。

## 既有来源范围准入链

`source_admission=verified_existing_scope`，没有改原report的`training_admitted=False`。以下精确文件SHA、代码行和事实全部进入provenance/admission_trace：

- `scripts/s2_media_pools.py:392` 的build_report对所有状态无条件写False：builder不自行授予准入，非CLEVRER特定拒绝。
- `docs/adr/0005-permitted-first-release-media.md:3,7` 是已accepted的CC0 CLEVRER official train首发商用训练决策；官方README真实SHA `a4838cc68b79ba049b54f95def1714cdbbd1380193d7d2a5d1a85440ac862e7b` 再核验，声明dataset CC0。
- 官方questions/train.json实际SHA `11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4` 再核验，仅哈希字节，不解析题目行。原视频ZIP沿用已下载source receipt，不另重哈希12GB；所选60个TRAIN媒体逐文件实际SHA验证。
- `scripts/s0_media_dev.py:311,331,458,475,490` 的转换只选官方descriptive/exist、yes/no，instruction原文、answer.casefold；source_hash绑定question/program/media/revision。该exist是原官方question_subtype，不能将本次基于词匹配的temporal/motion heuristic称为官方新类别。
- `.scratch/s3/spec.md:9,11` 冻结全8299视频TRAIN；`reports/s3-joint-preparation-001.json` 实际SHA `ad2ea7b72f307166214e7944dd54d4ea676c38a0248c1e14e14cd94f3402e6da` 再核验：全8299 source `912c871f...`、ledger `3049be...`、pool report `e4f095...`，联合21184的完整ledger校验；实际joint-train字节SHA也再核验。并非用旧150实际曝光证明全池准入。
- `docs/adr/0021-s3-bounded-replay-execution.md` 及 `reports/s3-stage-acceptance-stable-001.json` 接续接受该S3冻结池执行范围；不把S3验收扩大成S4质量/封存/发布验收。本轮依据已有范围，没有新许可接受。

当前exposure004 SHA绑定后，新候选对重叠current TRAIN来源只在派生manifest标historical，新增候选current，保留component ledgers；完整audit仅回aggregate。prepared data SHA `9b163f5bd0f6e57eab86922f6210e2d241cdee7149e0630f9918171ec39dadff`；ledger SHA `50e536c75d33aae41540d21309adbfffaa5735a2ba47c1432bc71feca73bc583`；inventory SHA `3b55ada87f02a7c7736d8cf342e8c7239da8cc54f46e59cee76946be1dfa6dbd`。

## 验证与复现边界

5项新/复用CPU测试通过，覆盖最终配额、唯一family、输入顺序不影响选择、每step布局/总体30:30、容量不足、非TRAIN和family复用拒绝；ruff/format、py_compile通过。实际独立核验全部outputSHA、120真实shuffle/样本SHA、每step5temporal+1minor+2cross+4old、10group/5:5，以及本builder所属文件600权限。root开始在同目录添加后续preflight文件，权限核验明确只覆盖本builder文件，不修改他人产物。

实际代码通过SSH stdin运行，导入固定已验证 `PYTHONPATH=/root/autodl-tmp/s4-reference-code-fix-001`。bundle携带新入口、既有prepare-claim helper及准入证据文件SHA元数据，wrapper对收到源码bytes分别SHA绑定；两源码SHA在report中。SOURCE_EVIDENCE为provenance中的admission_trace去掉运行实测附加字段即可恢复，单独本地运行没有该证据会fail closed。原始metadata/报告与本地现有决策文件分别核对，不将本次复用指令充当准入依据。

下一步由root运行相同SHA数据/ledger/config的CPU训练preflight，并在≤240秒预算内决定上卡；本切片不启动训练，不做效果结论或新颖性主张。成本null。
