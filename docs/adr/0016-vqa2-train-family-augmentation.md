# ADR0016：审计VQA v2人工标注以补充现有许可训练图片

状态：source-audit-approved，2026-10-15；训练候选须通过下述准入检查。依据用户授权按实测瓶颈自主采用组合/探索方案，且按开机时长压缩交付时间。不是变更基座、质量门或数据池。

S2已在小A-OKVQA池上完成真实训练/多条有界实验，最佳图像78.8%仍低于80.6%。新增标注信息比继续扫同池训练参数更值得先做低成本检查。仅从官方源下载VQA v2 train/val问题与答案，共42,962,121压缩字节；不下载新图片、不取test、不付费。

官方依据：VQA Consortium标注条款 https://visualqa.org/terms.html 明确CC BY 4.0，https://visualqa.org/download.html 提供文件及schema；许可正文 https://creativecommons.org/licenses/by/4.0/ 。照片仍依原COCO逐图许可，标注许可不覆盖图片。COCO官方源 https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/download.htm 说明2017复用2014图片但重划train/val；因此按numeric image_id匹配，不能沿用外部train/val名字推断项目分池。

准入：固定四文件URL、字节数和SHA；仅对已经在s2图像train ledger内、且图片SHA/许可已验证的family提取新问题。项目dev/calibration/holdout及其他family全部排除，不向agent输出这些family的问题/答案。保留VQA question_id、十名回答与多数答案的统计、源revision或归档SHA、图片许可与原family；禁止重复计数。

先报告eligible重合数、去重后题量、回答共识与题型分布；训练只选高共识gold。yes/no可映射Noul；其余是开放问答，不能假称官方提供了Choice干扰项，需单独固定构造方法、排除所有已知可接受答案、保留来源。归属问题集的attribution和变更说明纳入数据卡；不把许可审核、下载成功当模型质量证据。

下载落quarantine；通过上述逐项检查才产出train派生池。单次小样本验证后才扩量，仍遵守S2原图像门与旧模态回归。此次只放行公开标注与现有CC-BY2许可图片的组合，不放行NC/SA/ND隔离媒体。
