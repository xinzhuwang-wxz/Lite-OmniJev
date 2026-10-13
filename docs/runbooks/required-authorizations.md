# 全项目必须本人办理的授权与资源

核对日期：2026-10-14；范围为首发阶段0–6。依据 `configs/project-data-assets.json` 全部29项、`configs/stage3-source-audit.json`、`configs/video-eval-assets.json`、`configs/decision-index-assets.json`、`configs/runtime-cpu-builds.json` 和 `docs/02-project.md`，逐项区分访问手续、原媒体权利、工程准备、后续验收资源。本文没有接受协议、提交申请、发邮件或付费，也没有读取封存题。

这里列的是负责人一次性办理清单，不是开卡通知。账号获批、字节下载、逐文件哈希、无卡验收、训练准入是不同状态；全部必需外部准备与无卡验收未通过前不开GPU。

## 现在必须本人处理

下表前3项直接覆盖阶段2/3/6；后5项属于现有MVBench全项目外部清单，先统一暴露。教师14任务/2786题的精确协议仍未公开，不能先假定这些受限任务都可删，也不能把全20任务都说成已证实必需。负责人不具备机构/学术资格时应如实申请替代授权，不能虚填身份。

| 项目与官方入口 | 本人动作、限制与交回证据 | 影响范围；是否阻碍下载 |
|---|---|---|
| DocVQA：[官方说明](https://www.docvqa.org/datasets/docvqa) → [RRC挑战17](https://rrc.cvc.uab.es/?ch=17) → [Download](https://rrc.cvc.uab.es/?ch=17&com=downloads) 的 **Single Document Visual Question Answering** | 注册/登录RRC，阅读实际下载页原数据条款，保存版本/文本与获准下载方式。官方说明确认图片来自UCSF Industry Documents Library；HF镜像Apache标签不能代替原文档权利。此次RRC读取超时/证书校验失败，未取得登录后条款全文，不能声称原数据已商用许可。 | 阶段2文档题；**是，原数据访问/条款仍缺**。已下载Valen中的docvqa派生文件不解除此缺口。 |
| AudioCaps：[固定revision原音频说明](https://github.com/cdjkim/audiocaps/blob/d004db3ea1b01cf4fd0347dd8d27db90cadc8809/dataset/README.md) → [现行申请表](https://forms.gle/2wF54Y1Ft2LtPdhW8) | 由本人填写真实身份、用途并申请原音频/视频，保存审批回复与许可范围。[主README](https://github.com/cdjkim/audiocaps)明确学术用途；商业用途另取明确授权及原媒体依据。表单短链本次返回401，正文未读取；需要实际账号访问，不推断问题字段。旧清单 `8sFHc3XJ32ERWxqF8` 实测404，不能再作为办理入口。 | 阶段3音频；**是，原音频需申请**。公开caption CSV无需另申请，已准备。 |
| Android SDK/NDK：[NDK下载及完整SDK协议](https://developer.android.com/ndk/downloads)、[SDK协议独立页](https://developer.android.com/studio/terms) | 负责人以有权代表项目的身份审阅并明确接受Android SDK License Agreement，记录主体/日期/版本；随后agent可缓存和哈希指定工具链。协议2.1–2.4约束接受与代表权限；3.5另行保留组件OSS许可。无需购买开发者账号或Google Play上架账号来做本地构建。 | 阶段6 Android工具链；**是，官方二进制下载/使用协议未接受**。不是OpenJDK的独立签约阻塞。精确版本见下节。 |
| Moments in Time：[官方项目页](https://moments.csail.mit.edu/) | 本人从官方Download入口办理原媒体申请/研究条款；本次官方站连续超时，无法核实当前表单URL或条款文本，故不能给出猜测短链。现有 `video-eval-assets.json` 将其列为未接受官方表单。收到正文后先核对项目用途是否符合，再提交。 | MVBench `Moments_in_Time_Raw.zip`；**当前是**，不能以HF镜像绕过；最终是否必需取决于冻结评测协议。 |
| Something-Something v2：[官方页及Downloads](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)、[Research Use协议PDF](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/jester_something_something_exercise_research_license_final_qti_28jul2022.pdf) | 本人阅读研究协议并通过官方下载流程办理账号/接受条款；不把研究许可解释成商业训练许可。官方联系 `research.datasets@qti.qualcomm.com`，需要扩大用途时由负责人申请。 | MVBench `ssv2_video.zip`；**是，研究协议未接受**；最终必要性待协议冻结。 |
| TVQA原视频帧：[官方Download第4节及其form](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa.html) | 本人按页面表单提供本人及advisor信息、签协议，获批后保存下载链接/条款。官方明确不与他人分享帧；项目团队/隔离评测身份的访问范围也应确认，不能将镜像公开视为允许共享。 | MVBench `tvqa.zip`；**是，签署申请未完成**；标注/特征公开不等于帧可用。 |
| Matterport3D/VLN-CE：[项目页](https://niessner.github.io/Matterport/)、[Terms of Use PDF](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) | 本人确认学术资格与用途，填签协议，按官方指示发送至 `matterport3d@googlegroups.com` 申请；agent本轮不代发。商业项目不能靠接受学术协议自动获得商业权。 | MVBench `vlnqa.zip` 原场景；**是，申请/适用资格未确认**；最终必要性待协议冻结。 |
| NTU RGB+D：[ROSE官方说明及条款](https://rose1.ntu.edu.sg/dataset/actionRecognition/)、[账号后Request](https://rose1.ntu.edu.sg/dataset/actionRecognition/request) | 本人注册、提交request并接受Release Agreement，等待审批；官方学术/非商用限制明确。官方第8.3节提供TVBench相关320视频申请途径，与当前配置320引用应再逐ID核对，不能把整个136GB RGB库设成必要下载。 | 当前MVBench/相关视频评测原料审计中的NTU媒体；**是，账号审批缺失**；320视频身份及任务必要性由agent核实。 |

### LiteRT-LM v0.9.0具体需要什么

官方[固定版本构建说明](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.9.0/docs/getting-started/build-and-run.md)要求Android原生构建 **NDK r28b或更新**；该版本[CI实际固定r28b](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.9.0/.github/workflows/ci-build.yml)。官方[SDK repository索引](https://dl.google.com/android/repository/repository2-3.xml)将r28b映射为 **`ndk;28.1.13356709`**、`android-sdk-license`；Linux归档 `android-ndk-r28b-linux.zip` 为722,264,879B、官方SHA1 `f574d3165405bd59ffc5edaadac02689075a729f`（下载后仍需本地SHA256）。因此无卡准备采用 **r28b** 作为可复现选择；不要为“必须最新”再要求r30。官方说明还要求 `adb` 连接真机。

该原生CLI路径没有在说明/CI固定Android SDK platform API或build-tools版本；[WORKSPACE](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.9.0/WORKSPACE)声明SDK repository而没有指定 `api_level`/`build_tools_version`。不能编造“必须SDK35/36”。agent下一步应按实际目标冻结完整SDK包版本/哈希并CPU构建；如新增APK/Gradle宿主，依据那个宿主配置再锁compileSdk/build-tools。当前协议处理覆盖官方SDK/NDK发行包，包清单未冻结是工程缺口，不是让本人另办一个账号。

已有OpenJDK17 Temurin（GPL-2.0 with Classpath exception）、Bazel7.6.1（Apache-2.0）、CMake、GCC缓存是普通OSS履约与工程准备，不要求本人去接受Oracle JDK商用协议；本项目并未选择Oracle JDK。已有版本/哈希证据见 `configs/runtime-cpu-builds.json`。Linux host CPU构建亦不因Android协议未接受而必须停下；不能声称全部剩余工作只有开GPU可做。

## 需产品权利/资源决策

这些不是可以一次点击解决的账号阻塞。agent继续做逐样本原来源映射、权利证据整理与已允许下载；本人只需决定商业用途、必要的权利人申请或明确范围变更。保持原指标，替换/删除设计来源必须记录ADR和本人决策。

| 项目 | 官方证据与需要的决定 | 下载与阶段影响 |
|---|---|---|
| NExT-QA / VidOR | [NExT-QA固定README](https://github.com/doc-doc/NExT-QA/blob/2432e9724f88ed9f40010e2989f104570a91de4e/README.md)公开[NExTVideo原档](https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view)，明确视频来自VidOR且给映射文件；[VidOR](https://xdshang.github.io/docs/vidor.html)来源YFCC100M。agent须映射原Flickr视频、逐项CC许可/署名；负责人对NC/不明权利样本决定取得权利人许可或另提范围方案。没有证据表明NExT-QA本身必需新注册账号，不能假称登录即解决。 | 阶段3；标注已准备，原媒体下载/准入暂阻塞于逐媒体权利，非单一表单。 |
| LLaVA-Video-178K所选5k | [固定官方card](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/blob/6d8c562dc26d70042a0d9704d1cae58c94b89098/README.md)同时写Apache与学术/教育用途限制，并列YouTube、NExTQA、ActivityNet、Perception、LLaVA-Hound等来源。负责人应取得覆盖目标商业训练用途的明确授权/澄清，agent逐样本核原媒体和生成标签条款。接受NC用途不能代替商业权，作者也不能替第三方授予其没有的媒体权。 | 阶段3；受限媒体暂不下载；全量约1.28TB不在授权范围。 |
| VGGSound | [固定官方README License](https://github.com/hche11/VGGSound/blob/9bbc7491c054a203e53584c0065df336e1322bb7/README.md)明确数据集可商用/研究CC-BY-4.0，同时原视频版权归原持有人。因此不能误称整个VGGSound禁止商用，也不能推导任意YouTube原片都获许可。agent应先查所选原视频许可和取得方式；未清项由负责人决定取许可或替代方案。 | 阶段3；CSV已准备，所选原媒体权利/平台取得依据仍缺；没有已证实的统一账号申请。 |
| AndroidControl、Multimodal-Mind2Web | [AndroidControl官方datasheet](https://papers.nips.cc/paper_files/paper/2024/file/a79f3ef3b445fd4659f44648f7ea8ffd-Supplemental-Datasets_and_Benchmarks_Track.pdf)含非商用条件；[Mind2Web官方card](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web/blob/1b4c6a8cf9f77b7a5e0d641959935c80c4a05889/README.md)声明研究用途。负责人需明确商业路线取许可或提出来源范围方案；Apache/OpenRAIL标签不能单独解除矛盾。 | 阶段2；既定研究隔离下载可继续，商业训练/发布不准入；不要求无证据的额外账号。 |
| ScienceQA、ChartQA、A-OKVQA/COCO、Clotho | [ScienceQA原库](https://github.com/lupantech/ScienceQA)、[ChartQA原库](https://github.com/vis-nlp/ChartQA)、[COCO条款](https://cocodataset.org/#termsofuse)、[Clotho-AQA记录](https://zenodo.org/records/6473207)须区分代码、题目、原图片/音频。agent整理SA/GPL/署名及逐媒体许可，负责人决定发布物义务方案；此处不作“所有衍生模型必然GPL/SA”的法律结论。Clotho仅选已知CC0/CC-BY，COCO逐图隔离NC/未知。 | 阶段2/3；无需新增账号，已允许原料准备继续；有条件的训练/发布准入单独结案。 |
| STAR/Charades、MMAU、FunQA、MVBench | [Charades原协议](https://prior.allenai.org/projects/data/charades/license.txt)、[MMAU官方mini](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini)、[FunQA官方库](https://github.com/Jingkang50/FunQA)、[MVBench官方card](https://huggingface.co/datasets/OpenGVLab/MVBench)各保留NC/研究/传播限制。已有内部研究评测下载不等于商业训练或对外打包许可；负责人要扩大用途时须取相应许可。 | 阶段0/3/4–6；已允许评测媒体继续下载/验收；商业训练/媒体再分发受限。 |
| MVBench `data0613.zip`、`scene_qa.zip` | 当前 `configs/video-eval-assets.json` 保留来源/版本不明（scene_qa为MovieNet/MoVQA冲突）；agent须先核原任务和媒体ID，再给负责人精确权利人及申请入口。不能把不明来源包装为“请同意条款”。 | 影响匹配评测准备；当前暂缓媒体；这是agent审计缺口，不是可立即办理的人类账号。 |
| OneJev-Data、JevEmbed、Open-Jev、Valen与Decision Index各底层来源 | [OneJev](https://huggingface.co/datasets/OmniJev/OneJev-Data)、[JevEmbed](https://huggingface.co/datasets/HIT-TMG/JevEmbed-Data)、[Open-Jev](https://github.com/Zefan-Cai/Open-Jev)、[Valen](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k)、[Decision Index](https://github.com/apolinario/decision-index)不能用总许可证洗白原料；agent继续逐行来源/标签审计，JevEmbed中的PKU-SafeRLHF等NC源排出主池。发现确需额外许可才给负责人精确申请，不要求泛授权；禁止的Jev来源不因本人同意而准入。 | 阶段1–6；公开原料隔离下载和审计可继续；混合聚合库不整包准入。 |
| 额外磁盘、标注付费、租卡、生产与公开发布 | [项目验收契约](../02-project.md)及根AGENTS保留本人决策；GUI媒体约92.64GB需要agent先做选择性恢复/磁盘计划，未证实需要新增付费容量。 | 不把存储规划缺口说成账户缺口；任何实际付费/开卡/生产发布另有明确授权才执行。 |

## 后续验收资源

依据[项目验收契约](../02-project.md)与 `configs/project-data-assets.json` 内部池要求，这些需要负责人安排人员/设备，但不阻碍公开模型、工具链和已准许媒体的当前下载。

| 资源 | 本人需安排 | 何时阻塞 |
|---|---|---|
| 真实独立封存gate身份 | 独立受控身份、存储和读取权限；开发agent不得读取封存内容，只收汇总门槛报告。 | 当前S0隔离验收与阶段4/5/6封存门槛。 |
| 真实双人标注/合规现成标注 | 独立标注者、任务/分歧裁决、真实κ证据；开发每模态500、封存每模态1000、校准每模态3000–5000按契约形成样本族隔离。合成真值/agent自标不能冒充人工一致性。 | S0标注一致性及后续池验收。 |
| 阶段6端侧真机 | 符合选定Android后端的真实设备、可授权adb调试、操作者及测量窗口；agent准备已锁包及脚本。购机/租机另属付费决定。 | E2B内存/时延/模态覆盖真实验收；不能拿CPU构建通过代替。 |
| 最终运营观测/外部用户 | 生产部署授权、两周SLO观测窗口、20位真实外部用户等契约资源；不通过当前下载任务提前扩张生产。 | FINAL/P4运营证据，不阻塞当前无卡公开准备。 |

## 已完成授权与无需本人再办的项目

- CommonVoice MDC与HLE：本人已完成授权，实际官方资产下载和哈希通过，分别见 `reports/mdc-commonvoice-download-002.json`、`reports/hle-source-download-001.json`。不再索要MDC key、HF token或重复接受条款；旧配置中的account/terms blocked为历史记录，不能覆盖新证据。
- 7项模型：按主会话最新全文件哈希报告核验；现有会话已获得下载权限，不新增Gemma账号授权要求。模型forward、LoRA、量化/校准产物仍按阶段生成与验证，不能虚称可提前下载。
- GUI-Odyssey：官方[固定HF card](https://huggingface.co/datasets/hflqf88888/GUIOdyssey/blob/61632d0f3f4d51d7e9561ce4f84347dd06b2019d/README.md)公开CC-BY-4.0；CLEVRER：[官方README](http://data.csail.mit.edu/clevrer/README.txt)明确CC0。无需假设额外账号/人工签约；agent继续媒体、哈希和署名清单准备。
- Perception：[官方许可](https://github.com/google-deepmind/perception_test#license-and-disclaimer)为CC-BY-4.0；JevBench公开题、CLINC等已允许原料无需再申请统一账号。其评测协议、精确抽样/裁剪与隔离由agent继续处理。
- Playwright项目自有合成页、变体/反事实、软标签、蒸馏和发布包是未来产物，非外部账号。阶段5b环境、ESC-50/RAVDESS研究扩展不在首发必需外部下载范围。
- Python/uv、Transformers/PEFT、vLLM、llama.cpp、Ollama、llm-compressor及公开OSS工具：继续按锁版本下载、许可证/哈希归档与CPU验收；未发现需要新增本人账号的证据。SwanLab已有项目凭据流程不重新索要；网络失败、未编译、缺依赖闭包都由agent推进，不转成泛化“授权缺口”。

## 验证范围与限制

本轮复核官方网页/固定源码，未读取评测隐藏答案、pickle或封存题。DocVQA登录后下载条款、Moments当前表单正文未取得；AudioCaps现行短链需要登录，旧短链404。文中保留这些边界，不承诺申请必获批或存在统一商业许可。后续收到条款/审批后由agent保存原文哈希、主体、适用资产及用途，再继续允许下载；每项哈希/CPU结果应写实际报告，本文不是这些结果的替代。
