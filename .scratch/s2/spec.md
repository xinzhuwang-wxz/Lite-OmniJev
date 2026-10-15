# S2 图像/截图决策LoRA业务闭环规格

Owner: main-session-2026-10-15
Parent: S1通过（reports/s1-stage-acceptance-001.json + ADR0013）
State: 一个闭环；池/训练扩展/评测/修复都在票内清单

## 目标

图像与截图各自：池冻结→多模态训练扩展→smoke→主训练→dev500同条件双遍评测+文本回归+反事实翻转，通过S2机器门。延续ADR0013校准原则（用户2026-10-15授权）。

## 数据与池冻结

- 图像：A-OKVQA允许集（revision 42e8c44f…，train_tar 3992b488…，manifest e20747c1…，2738候选QA/2641 family，QA Apache-2.0，媒体CC-BY-2.0）。dev500（SHA 38efa78e…，family已排除）之后全部候选为可分池输入；媒体在quarantine stage2-sources/aokvqa提取树（uri相对media image/目录，逐文件sha校验）。
- 截图：GUIv2协议（source hflqf88888/GUIOdyssey 61632d0f…，CC-BY-4.0），排除dev500 family与v2排除ledger后可分池；媒体复用quarantine已有截图存储，不复制。
- 分池协议与s1相同：key=SHA256 `s2-<cohort>-pools-v1:42:<family_id>`，升序取前H个为holdout-reserve、次C个为calibration-reserve，其余train。规模：图像 holdout200/calib600、截图 holdout110/calib330（与文本9.3%/27.9%比例近似；s0:05的正式1000/cohort封存gold将优先采用各来源官方val/test现成gold，此处为fallback预留）。
- 输出jsonl+allocation ledger+报告绑定输入/输出/媒体manifest SHA；双遍字节一致；训练入口核ledger并拒绝dev/预留family与历史排除媒体SHA。

## 训练协议扩展

在s1:01已验证机制上最小扩展：LoopConfig.modality增加image/screenshot；样本经官方processor产生pixel_values（媒体按sha256校验后读取，总量/尺寸预检沿用P0媒体约束）；vision/audio tower保持冻结+eval，LoRA仍只挂decoder q/k/v/o+gate/up/down；损失仍为最后位置全部合法答案logits（含U）FP32 log-softmax交叉熵。thinking关、固定seed42、lr1e-4。
- 执行预算按ADR0020实测调整：一次300步三模态joint smoke；图像/截图均真实参与，loss下降只作工程信号。原“各300步”不再单独重跑。
- 主训练采用可追踪的joint主训练与文本回放，保留midway对照；现有候选v2为1600步/16000样本曝光，image3108、screenshot930，分别覆盖1341/420个全部训练family。具体输入/adapter SHA、各run曝光与调整理由见ADR0020和reports/s2-training-exposure-audit-001.json。
- 是否推进仍取决于以下原质量门，不能用曝光或loss代替质量验收。

## 评测协议

- dev500同条件双遍（image与screenshot各一），协议同S0基线（同文件/模板/读出/U联合归一/pU单列），精确重复。
- 文本回归：S2多模态adapter评text dev500，acc≥98.6%（=99.6%−1pp门）。
- 反事实翻转：每dev题按固定种子置换选项顺序（答案字母随内容重映射），重新读出，预测选项内容与原预测一致的比例≥0.80；置换评测同样双遍确定性。

## 门槛（configs/gates.json S2原值）

| 门 | 值 | 基线 |
|---|---|---|
| image_vs_onejev_delta | ≥−3pp（≥80.6%） | OneJev4 image 83.6% |
| screenshot_vs_onejev_delta | ≥−3pp（≥63.2%） | OneJev4 screenshot 66.2% |
| text_regression | ≤1pp（text dev500 ≥98.6%） | s1:01 99.6% |
| counterfactual_flip_rate | ≥0.80 | Jev公开弱点7%置换 |

未达标如实记录；结构性不可达时按ADR0013格式另录。

## 证据

同s1:01标准：每run报告含命令/版本/数据与媒体与adapter hash/seed/环境/耗时/退出码/cost null；门槛eval.gate机器可读；SwanLab真实指标；验收后STATUS+commit。

## Out of scope

音频/视频（S3）、E2B/量化（S6）、封存holdout构建（s0:05）、打包差<1e-3部署门（S6前不启用）、未准入的新数据源。VQA2现有family与新family有界扩展按ADR0016/0018执行，不移动原数据池。
