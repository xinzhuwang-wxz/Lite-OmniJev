# 第三覆盖段：小切片微升但更广筛查退化，停止追加

第三段以250ef为父，原data-only准备保持不变；120draw/10步173.053531秒完成，adapter5cff2c3c2bc80018615891760fcf4fb883aad6f8666cae65d70d7f43071d5821。固定DEV50同输入hash通过，37/50vs父36、Brier .439262vs .442754微改善，coverage .06vs .08退化。随后原video150仅108/150vs父116，故不推广、停止同配方第四段/扩大原rawcoverage。报告s4-temporal-coverage-training/dev50/video150-003.json。

这再次说明反复使用50题筛查会有选择偏差，不能按单点升分自动放量。既有父段/首段/CE0/S3保留，程序gold事件顺序路线只CPU构建。为选后续父模型，对首段f2db做其首次fullvideo500，single pass/420秒cap/每25题，session56050；首段曾video150116/cross10074/old4收齐，扩大资格来自完整小筛查。第二段已有full356/500；不重测第二段。full1的实际部署wrapper保存在Git171c63f，后续仅docstring长度修复不影响逻辑，receipt绑定实际runnerSHA。费用null，不把全500开发评估当S4通过。
