# 固定TRAIN时序可学习性诊断

冻结2026-10-16，实际before10/12、mean joint gold CE .43352499286023277、mean gold-vs-opposite margin1.375。非饱和，允许10steps/120draw。不是DEV或封存质量证据，也不用于选产品候选。

12题从已审计temporal120 TRAIN中按固定hash选择6yes/6no、12family，不看预测对错；datae4153a3aacdd71ed4ab951ad9e67c77705728bec20f8a6b7afc2faa89aee6f8a。sample_count12、accum12、10steps使每轮循环同12题；init CE0 4b8e422f…，LR2e-5/CE/seed42/rank16/alpha32/bf16/16帧/step5checkpoint。CPU精确canonical子集和既有ledger验证，不新增样本或媒体；真正更新后需加入曝光发现。

before/after各90秒，训练240秒；before12/12且CE≤.05则不训练。本次未饱和。训练完成且权重SHA改变、前后输入哈希/gold/mapping相同才比较：CE相对降低≥20%且margin增加≥.5为可学习信号，另报是否≥11/12且CE≤.1；若CE改善<5%且margin增加<.1，只提示检查优化/输入，不能宣称表征/模型上限。其他为inconclusive，不自动扫参。训练改善不能自动证明DEV泛化；后续视证据做固定DEV短查。

S3实际仅训练200视频family/200draw，报告s3-training-exposure-audit-001；不把完整8299池已备妥等同全部训练。该诊断用于决定下一段训练，而非继续无依据换loss/取帧参数。

部署source9062f65，archive5e8ee814150578614557b5ec2774c7c38ed70f1270ecf626391fe96a79b40666；before/after复用同源码及官方template、单token映射，记录完整prepared张量身份。root launcher .scratch/s4/launch-train-learnability.py 严格检查before非饱和与data/sidecar身份，训练成功后才执行after。真实before和CPU准备报告已存reports/s4-train-learnability-{before,preparation}-001.json；训练/after待终态。
