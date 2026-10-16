# 匹配极性对照：先保存普通续训控制

control10steps/120draw在174.272秒训练完成，losses全有限，第5步adapter8803039c…与恢复checkpoint实际落盘；final adaptera70b7fc6de3e6283bdca3037d32df346e3cb22da6db303359870a9c744764bee，init f2db。固定原时序DEV50单遍completed，36/50（父33/50），Brier .42867056（父.44748602）、ECE .17170445（父.10419439）、coverage@5% .1（父.12），50逐题inputhash核验通过。真实SwanLab run r3ntciqc，未知费用null。

此控制是同30physical TRAIN family原题重复，不是极性创新收益。下一polarity arm必须相同f2db初始化、相同物理视频/replay/10steps与预算，以control而非仅父模型比较。唯一主要变量是同family第二题改为询问原题正确答案是否为no，并按gold确定yes/no；没有新增真值、RL或训练头。

退出规则：身份/非有限/OOM/timeout立即停止；质量在50题成对正确数、双向错误与概率loss综合判断。一题不机械定输赢；若准确率及概率loss均未比control改善，不扩video150/500、不扫参数。若有可解释的额外改善，最多先原video150/180秒并检查existence-no退化，再决定保留筛查。小切片不替代S4最终门。CPU曝光010通过才启动单个polarity，不预排第三arm或新coverage段。

polarity最终completed/exit0，adaptere3e155f884a85928315f75373ce2ff87a658d4057bd9b368edc43716ba633f93。相比control同50输入：32vs36正确、纠正1/退化5，NLL .626589→.639859、Brier .428671→.449876、AUC .7488→.6832、coverage .1→.06更差；ECE .171704→.134885单项好转不抵消。yes预测31→37，FP10→15、FN4→3。停止这条特定问句/预算路线，不扩150/500、不扫参；不把失败泛化成所有极性监督无效。011隔离实际通过60已知更新报告/unknown0，见相应机器receipt。

下一主线只运行已准备第三60新family（仍50temporal+5motion+5existence、同cross/old replay），父模型选第二coverage250ef而非此失败branch，10步/240秒后原DEV50/180秒，不自动长串续训。第二段的小temporalDEV改善与完整原视频诊断提供一次额外新family短段依据；video150持平/cross概率退化的警戒保留，第三段若无可解释改善不扩大原rawcoverage路线。程序化时间顺序对照仅CPU可行性调查，不把普通训练增益称创新。
