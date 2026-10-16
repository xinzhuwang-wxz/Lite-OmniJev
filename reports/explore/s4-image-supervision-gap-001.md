# S4 image supervision gap — bounded CPU audit

2026-10-16; no GPU, new data, training construction or private gate payloads. Actual remote canonical TRAIN data and parent run/config/hash chain were inspected; only aggregate metadata returned. JSON binds runtime files/reports and source evidence.

**原image500是A-OKVQA，不是VQA2。** `scripts/s0_media_dev.py:24–42,244–254,416–438` 固定官方A-OKVQA问题与四个原生选项、correct_choice_idx真值；image dev SHA38efa78…。f2db在1120预算410/500，与S3同分，S4门84.778%需至少424/500，仍差14题。当前训练image与该dev同源；不能把持平归因为“缺少同源VQA2监督”。同源不等于同family或同样本，原分池保持。

|保留谱系（旧→新）|真实image draw|实际unique image family|图像预算|
|---|---:|---:|---|
|S2 joint-v2|3108|1341|config未显式记录；E09前默认280，不能将推断写成实测字段|
|S3 stable|200|199|1120|
|S4 joint（1200存储、实际只训练300draw）|50|50|1120|
|S4 crossmodal retention|30|30|1120|
|S4 claim-pair CE0|10|10|1120|
|S4 coverage001/f2db|10|10|1120|

全部保留谱系image版本均aokvqa-train-image-choice-s2-pools-v1：累计3408draw，2778unique ID、1341unique family。S4四段合计100draw、72unique ID/family，并非1200池全部曝光；这些统计仅独立image，不把crossmodal中的image组件监督冒充同任务image draw。S4跨模态确实可能更新图像表示，本审计未归因其因果贡献。现有原image500聚合持平只支持“未观察到总准确数提升”，不能证明逐题预测完全相同。

ADR0016批准审计VQA2官方CC-BY4标注，但照片许可仍按原COCO逐图；派生Choice干扰项是程序构造，不是官方多选题。已实际准入TRAIN1811题/970family（color183/integer281/yes_no1347；文件SHA48d0ef…），来自原1341train图片范围。它是容量，不是当前模型的已训曝光。E07所选400题实际338family，和AOK400/text100/screenshot100合成1000；两个实验真实1000draw/100steps。其余池规模不能计入训练。VQA2开放问答高共识→颜色/整数域选项与yes/no映射，和原AOK知识/常识四选一协议有差异。

S2已测试：E01早停121/150；E02平均124；E03同池图像聚焦CE122（原训练150已146/150，非明显欠拟合）；E04平滑121；E05弱化delta122；E06gold优先教师122。E07旧family VQA2新问题CE123、CE+Brier124，对照123，均未达到125扩量。E08新400视觉family VQA2 CE125/150但完整399/500，未达原S2门。E09只增加官方视觉预算1120，原v2达到410/500并正式采用。**不能把VQA2 augmentation当未试过的新方向，也不能把E08独立分支结果或容量误报为当前谱系贡献；它是独立未采用分支。** S4新vqa-gold-001只有dev500/cal3000/hold1000；不是TRAIN池，内容未读。

至多一个低优先级候选：若root仍需要image对照，只考虑当前f2db、1120实际训练预算下的单次图像聚焦CE，10steps/120draw/seed42/≤240秒：固定hash60unique AOK TRAIN family + video20/cross10/audio10/screenshot10/text10，保持标准gold/模板/LoRA。来源可用canonical `artifacts/s2-text-lora-001/joint-v2-001/data.jsonl` 中2778 AOK题/1341family；不按dev错题挑训练。它只检验当前parent和1120输入下的图像强调，和E03旧parent/旧预算不同；不是新数据创新，不承诺提升，不复跑100步或扫CE/Brier。当前已有S3/S4在1120学习，故该区别证据较弱，若更强假设可用则不优先耗卡。

本报告未构建候选，尚需最新曝光inventory+原source allocation ledger/媒体SHA核验、明确120选择/实际draw顺序、完整family隔离及父同输入image筛选基线，才可谈preflight。现有数据存在和历史准入不代表新派生池自动ready。保留视频与其余模态回放不能保证不退；若screen无原DEV收益即停止，不改门。

可复核只读命令：将本地 `.scratch/s4/inspect-image-supervision-gap.py` 通过SSH stdin给现有远端训练python，`--root /root/autodl-tmp/s1-text-lora-001`。脚本只读保留谱系run.json及已知TRAIN data.jsonl、两个已准入TRAIN容量文件；不遍历sealed目录，不输出正文/ID/family列表。exit0；Ruff通过；拒绝非TRAIN输入、runtime SHA不符、父adapter不匹配。缺少逐样本prepared tensor身份和全图像错误分布归因，不能推出视觉表示不足的唯一机制。
