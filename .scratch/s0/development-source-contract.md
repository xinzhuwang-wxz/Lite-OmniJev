# 六类开发集来源合同与实际缺口

Owner: main-session-gpu-transition
Parent: s0:07（原12已合并）
State: all six cohorts500 actually built and deterministic; model matrix in progress

ADR0009固定六评测cohort，每类500dev/1000holdout；物理输入仍四值。以下只规划公开TRAIN候选与自有程序真值，不读取validation/test/项目holdout。来源的下载、媒体数、原标签数均不等于开发题准入或人类κ。当前五cohort实际构建见STATUS与reports；来源子项统一07，不再建单来源票。

| cohort | 固定来源与实际候选 | family与池排除 | 首个合法题型/尚缺 |
|---|---|---|---|
| text | CLINC plus/train revision155b9c710419136e17307b80d0a13e68cd46b4ec，CC-BY-3.0，15250已准入公开训练题 | normalized原句family；dev500从旧全量train排除，余14750；s0:13 | 原始intent Choice500；不等于Score/Noul或完整文本任务分布 |
| image | A-OKVQA Git42e8c44ff40d13cf272776a3197365f582137772，标注Apache-2.0；允许TRAIN清单2741题/2644 COCO image_id，全部license_id4/CC-BY | COCO image_id；同图全部题及变体同池，训练不能保留该图的其他QA | 原始choices+correct_choice_idx可转Choice；逐图attribution/许可join/500媒体decode与任务保真尚待执行；3832总图含其他split不能作TRAIN候选数 |
| screenshot | GUI-Odyssey HF61632d0f3f4d51d7e9561ce4f84347dd06b2019d，CC-BY-4.0；选中TRAIN1679episodes/25809PNG已全decode | 整episode，保留官方app/device/task边界；禁止逐截图随机分池 | all_annot只有episode元信息，不能当step action gold。先定位固定TRAIN annotations并核step/action/坐标/截图对齐；Choice候选语义未冻结 |
| audio | Clotho-AQA Zenodo6473207，QA CSV MIT，允许WAV CC0/CC-BY-3.0；TRAIN17514答题行/973audio；严格恰好3行一致yes/no为2267QA/947audio | Freesound原sound id/audio；同音频所有问题/派生WAV同池，复用固定两处filename alias | 原三行一致gold可Noul，每family选1；6行异常组不得取前三行，confidence是自报确信文本不是概率。匿名三行不是固定两真人/κ证据。500构建与逐WAV decode未执行 |
| video | CLEVRER CC0，官方train.json SHA11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4；10000videos/152572QA，descriptive exist20000 | scene_index/原视频，全部QA、帧、反事实同池 | exist可直接Noul；color/shape/material可冻结词表Choice；choices[].answer多项独立correct/wrong，禁止直接转唯一单选。500构建/所选视频逐decode未执行 |
| crossmodal | 未来组合程序真值，继承全部组件许可；现无500已生成题 | 组件family构成联通分量；先全局分配组件再组合，任何组件不得同时进train/dev/calibration/holdout | 必须真实多输入共同决定标签；禁止忽略某输入或拼接现有parity变体充数。组合程序与语义探针未实现，不能提前claim500 |

## 已核的精确路径与来源边界

远端根/root/autodl-tmp/lite-omnijev。Image标注是artifacts/data/quarantine/stage2-sources/aokvqa/aokvqa_v1p0.tar.gz内aokvqa_v1p0_train.json，17056原TRAIN行；允许清单同目录aokvqa_train_coco_ccby_publicdomain_image_manifest.jsonl。依question_id/image_id精确join媒体路径/SHA与attribution，不从HF镜像洗原媒体许可。

GUI计划source-audit/stage3-source-audit/gui-odyssey/app-split-train-media-plan-under-19_5gb.json；pinned-core/all_annot.json只有episode_id/category/app/device_name/h/w/instruction/step_length。截图位于artifacts/data/quarantine/gui-odyssey-source-audit/media/app_split_train_under19_5gb/screenshots。已启动最小公开TRAIN step annotation审计，无新媒体下载；在证据产生前不虚构动作gold。

Clotho路径artifacts/data/quarantine/clotho-aqa/6473207/{clotho_aqa_train.csv,clotho_aqa_metadata.csv,approved-audio-002}。QA5837 distinct(file,question)：5836组3行、1组6行。过滤只依许可/原标签格式与一致性，不能依模型表现选题。原vote行hash保留为provenance；不将3票一致称人工双标准入。

CLEVRER路径source-audit/clevrer-official/files/questions/train.json与videos/train/video_train.zip；后者SHAc7401bc86f7ff230bda3b193619f41a79828bed1f0890948b16dd57c89dfdd3f。descriptive109952：count40000/exist20000/color18454/shape16884/material14614。question program仅真值provenance，不进模型输入。

## 分配与最终退出

seed42、版本化family ledger先于模型评分。全局组件/重复媒体SHA归并后分池，同源不同数据集镜像必须归一family。dev可以多题同family但不能跨池；题数500与独立family数量分报。初始单来源单primitive切片必须显式记录覆盖缺口；最终配比与模型支持矩阵冻结后才形成S0基线，后续改dev必须升版本并同条件重跑全部必需模型，不能择优保留成绩。

calibration不在本次构建中偷偷分配；将其family分配标pending，未来校准候选必须未参与实际训练/dev，逐family/媒体对完整训练账本查重拒绝；不能事后删训练记录冒称未训练。holdout载荷与标签由外部custodian处理，只给开发返回源/数量/hash/隔离/双标κ汇总。未知组件交叉关系、媒体许可或原gold冲突均阻断相应构建，不以程序补标签伪装成原标签。

证据来源：s0-data:01及reports/clinc-admission-corrected-001.json；reports/stage2-remote-completion-001.json、stage2-media-cpu-validation-001.json；reports/gui-odyssey-download-acceptance-001.json、gui-odyssey-full-decode-cpu-001.json；reports/clotho-aqa-audio-preparation-002.json、clotho-range-recovery-004.json；reports/clevrer-source-download-snapshot.json、clevrer-media-cpu.json。补充bounded source-map读取了公开TRAIN元数据/标注以核数量/schema，未读其他split/封存或教师预测；该审计未做模型推理、媒体decode或训练准入。

2026-10-14小型官方TRAIN动作审计已取得：reports/gui-odyssey-step-annotation-audit-local-002.json。固定首episode16steps全部连续且16非空截图basename逐一对齐；CLICK12/TEXT2/SCROLL1/INCOMPLETE1，11CLICK有合法目标bbox/点击点几何一致，另一CLICK info为字符串。只有单target bbox，无完整候选元素集合；开放参数/INCOMPLETE含义及不泄漏输入规则未冻结。可继续审计粗动作类型Choice，或明确以bbox几何程序真值构造新任务，但不得把自行负例或粗动作类别当原完整GUI动作gold。仍只有1family，不是500或16独立family；001远端无网失败保留，002为Mac成功替代路径，未请求另外两份。

2026-10-14当前实产物取代上表旧“未执行”描述：image原Choice500、video原exist Noul500已在6e9c467 Linux双遍实际decode/hash与整family排除通过；crossmodal image+audio合取Noul500已在c4ce657双遍通过，yes250/no250、1000媒体hash核验，引用原image/audio dev组件同池。三个acceptance报告分别reports/s0-{image,video,crossmodal}-development-acceptance-001.json。单primitive覆盖限制仍明确，不能据此宣称Score覆盖或质量通过。截图目标定位Choice500实际构建在进行。

截图最终实产物已完成：reports/s0-screenshot-development-acceptance-001.json；目标bbox来自原CLICK，3个同尺寸不相交负例为程序构造，Choice4仅目标定位切片，非完整GUI动作能力。630份固定TRAIN标注得到500合法独立episode，各500截图实际RGB decode/双构建确定性通过。
统一开发排除入口scripts/s0_development_bundle.py绑定六份构建报告，以媒体SHA连接原始与跨模态family并输出dev-exclusion.jsonl；后续训练须同时检查source_dataset/family_id与媒体SHA。该清单不冒称calibration/holdout分配已完成。
