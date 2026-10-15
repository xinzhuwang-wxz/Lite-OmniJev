# ADR0018：以新增视觉family检验S2泛化瓶颈

状态：accepted-for-bounded-pilot，2026-10-15。用户授权按阶段自主探索；保留S2质量门。

E01–E07未找到可扩量收益。E07新增1811问题仍来自同1341图片family，不能检验视觉覆盖不足。利用已经下载且有哈希的官方COCO train2017元信息与VQA train2014标注，CPU计数找到8744个具有严格一致颜色/数字/yes-no问题的新family；见reports/s2-new-family-feasibility-001.json。

准入范围在ADR0016基础上明确扩展：仅COCO train2017、license=4且许可URL精确为CC-BY2；VQA官方train标注CC-BY4，十人一致且multiple_choice_answer一致；不使用外部val/test。排除整个原A-OKVQA train源16540个family，以及传入现有allocation/exclusion ledger的全部family与媒体SHA。先过滤family再处理问题答案，训练前下载哈希、解码、重复媒体与谱系核验；保留逐图URL、Flickr attribution及QA来源。不得以只排除开发集代替预留隔离。

先固定seed42、唯一family、颜色160/数字160/yes-no80共400个图像问题，下载上限600张、每张20MiB；不下载整套COCO。与E07相同旧池400image+100text+100screenshot、v2初始化、lr2e-5/100步/CE对照，固定dev150至少125/150才扩大，旧S2全门保持。程序Choice干扰项须明确标注；不伪称官方多选。一次训练+评测15分钟H20上限；失败停止本小候选，不扫参数。此为原单步产品数据覆盖实验，不改基座、引擎、封存策略或质量阈值。

数据下载与构建不代表获准训练：报告绑定所有输入、输出、图片SHA、ledger，训练入口独立验证后才上卡。CPU准入与卡上后续准备并行；不为占卡而启动无收益GPU工作。
