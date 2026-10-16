# 匹配极性对照：先保存普通续训控制

control10steps/120draw在174.272秒训练完成，losses全有限，第5步adapter8803039c…与恢复checkpoint实际落盘；final adaptera70b7fc6de3e6283bdca3037d32df346e3cb22da6db303359870a9c744764bee，init f2db。固定原时序DEV50单遍completed，36/50（父33/50），Brier .42867056（父.44748602）、ECE .17170445（父.10419439）、coverage@5% .1（父.12），50逐题inputhash核验通过。真实SwanLab run r3ntciqc，未知费用null。

此控制是同30physical TRAIN family原题重复，不是极性创新收益。下一polarity arm必须相同f2db初始化、相同物理视频/replay/10steps与预算，以control而非仅父模型比较。唯一主要变量是同family第二题改为询问原题正确答案是否为no，并按gold确定yes/no；没有新增真值、RL或训练头。

退出规则：身份/非有限/OOM/timeout立即停止；质量在50题成对正确数、双向错误与概率loss综合判断。一题不机械定输赢；若准确率及概率loss均未比control改善，不扩video150/500、不扫参数。若有可解释的额外改善，最多先原video150/180秒并检查existence-no退化，再决定保留筛查。小切片不替代S4最终门。CPU曝光010通过才启动单个polarity，不预排第三arm或新coverage段。
