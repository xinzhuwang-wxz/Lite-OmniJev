# Jev-Omni 固定版本的公开评测协议审计

截至本次审计，未从固定教师 `akhilaaa3/Jev-Omni@5addda86ddee081a68fb067477ea100c221b8917` 的公开接口和说明找到可重建的精确14任务／2,786行 manifest 或基准评测脚本。同条件复现继续缺少上游协议；这不阻止已获许可的公开媒体准备。此次只审计公开源代码、配置、公开任务引用和自有合成音频，不加载模型，不读取隐藏答案、私有 split 或预测，不按发布总数/分数选择排除项。机器证据、源文件 SHA256、官方20任务表和 CPU复现保存在 `reports/jev-omni-mvbench-protocol.json`。

## 已证实范围与精确缺口

[固定模型卡](https://huggingface.co/akhilaaa3/Jev-Omni/blob/5addda86ddee081a68fb067477ea100c221b8917/README.md) 报告14任务、2,786问、accuracy53.10%、micro53.09%，MMAU1,000问；这些是发布值，本项目没有复测结果。完整递归 [固定树](https://huggingface.co/api/models/akhilaaa3/Jev-Omni/tree/5addda86ddee081a68fb067477ea100c221b8917?recursive=true&expand=false) 有22个条目，含目录、loader、example、配置、权重与展示资产，但没有 eval脚本、任务名单或逐行选择清单。权重、head、验证结果文件和展示媒体未为此次协议审计下载/读取。模型卡没有指向可追溯的另一官方基准入口；公开讨论索引是标签PR和页面认领主题，没有提供评测清单入口。这里的“不存在”仅指所检查的公开固定源，不能证明作者没有未公开文件。

缺口需要由上游补充：

- 14个任务JSON的精确名称、dataset revision及源哈希；按原JSON标识的2,786行有序清单，以及相对公开源缺行的原因。
- 评测入口、state与question拼接、选项原顺序、提示、解码失败/缺媒体策略及聚合分母。
- 有界任务的时间窗、FPS/取整/采帧数/短视频处理、图像列表或原生视频输入路线。
- MMAU实际1,000公开ID/version、音频截断/混音/重采样/输出编码、运行时与依赖版本。

本次固定官方源20任务每项200行，共4,000行。如果教师14任务使用这个版本的完整原行，应为2,800行；差14行本身不证明它们是哪14行、为何缺失，也不证明教师使用了这个dataset revision。不能把STAR缺失12条、时间越界3条或音频>30秒66条当成教师排除项，不能枚举任务组合匹配53.10%。

## 可冻结的公开参考映射

[官方 MVBench 数据](https://huggingface.co/datasets/OpenGVLab/MVBench/tree/230a2d4fac8900333c61754641c7a13e069ac9c6/json) 固定 `230a2d4fac8900333c61754641c7a13e069ac9c6`；[官方映射和解码代码](https://github.com/OpenGVLab/Ask-Anything/blob/4dd8210b030374bb0c510a57dfbe916f70d3ef71/video_chat2/mvbench.ipynb) 固定 `4dd8210b030374bb0c510a57dfbe916f70d3ef71`。只解析 notebook 的 code cell，不读取执行输出。20个映射 tuple与现有资产表逐项完全一致，20个JSON全部核对固定SHA和200行。报告冻结JSON SHA、原source prefix、media kind、是否有界，以及公开引用清单的规范JSON SHA。

行身份定义为 **task JSON SHA256 + 原始零基row index**，视频名不是唯一行ID。引用摘要保留原video/start/end，未导出问题、答案或预测。此公开4,000行参考映射可重建，不能命名为教师2,786行 manifest。

| 来源 | 官方任务 | 时间窗 |
|---|---|---|
| STAR | Action Sequence、Action Prediction、Object Interaction | 有 |
| Something-Something V2 | Action Antonym | 无 |
| Moments in Time | Fine-grained Action | 无 |
| FunQA | Unexpected Action | 无 |
| CLEVRER | Object Existence、Moving Direction、Moving Count、Moving Attribute、Counterfactual Inference | 无 |
| Perception | Object Shuffle、Action Count、State Change、Character Order | 无 |
| Charades-STA | Action Localization | 有 |
| Scene QA | Scene Transition | 无 |
| NTU RGB+D | Fine-grained Pose | 无 |
| VLN QA | Egocentric Navigation | 无 |
| TVQA帧 | Episodic Reasoning | 有，帧目录 |

保留官方 `Fine-grained Action` prefix中 `pyour_data_path/` 字面拼写，不能把修正路径误当成选择/排除规则。当前已验证STAR+补充、Perception/FunQA/STA九任务共1,800引用存在；其余来源的许可/媒体缺口仍按独立资产报告处理，不以教师任务未知掩盖这些具体缺口。

## 公开loader与官方processor的行为

[固定 `jev_omni.py`](https://huggingface.co/akhilaaa3/Jev-Omni/blob/5addda86ddee081a68fb067477ea100c221b8917/jev_omni.py) 可证实以下行为，但没有证据把它等同于上述发布分数的实际评测入口：

- 音频通过 FFmpeg先输出最多30秒、单声道、16kHz的临时WAV；未显式指定混音矩阵、输出codec或FFmpeg版本。对6通道音频不能假定FFmpeg自动混音等于逐通道算术平均。
- 视频 `_video_frames` 默认请求16帧，按 `round((total-1)*(k+0.5)/count)` 取中点，排序并去重；OpenCV逐帧读取、转RGB后作为多张 `image` content。去重或部分解码失败可使实际帧数小于请求数；没有start/end参数，也没有MVBench有界任务的裁剪实现。
- 提示将选项按输入顺序编号1…N；官方chat template设置 `enable_thinking=False`。发布基准的state内容、选项构造和完整入口仍不明。
- `load_jev_omni` 内部 `snapshot_download` 没有revision参数，直接运行可能跟随main；未来项目适配应显式用已核对的固定本地快照，不能覆盖官方原件。

[固定requirements](https://huggingface.co/akhilaaa3/Jev-Omni/blob/5addda86ddee081a68fb067477ea100c221b8917/requirements.txt) 要求Transformers5.17.0。官方tag解析到commit `856157a2f3e9594954310df18fdccc31ffddebe9`，相关文件用tag及commit分别下载并逐字节比对。其 [Unified processor](https://github.com/huggingface/transformers/blob/856157a2f3e9594954310df18fdccc31ffddebe9/src/transformers/models/gemma4_unified/processing_gemma4_unified.py)、[音频提取器](https://github.com/huggingface/transformers/blob/856157a2f3e9594954310df18fdccc31ffddebe9/src/transformers/models/gemma4_unified/feature_extraction_gemma4_unified.py)、[音频加载](https://github.com/huggingface/transformers/blob/856157a2f3e9594954310df18fdccc31ffddebe9/src/transformers/audio_utils.py) 支持源码层核查。教师processor配置16kHz、640 samples/token、audio_seq_length750（对应30秒），但提取器max_length默认None，仅在显式max_length时做截断；不能只凭配置替代loader的明确FFmpeg转换。原生audio加载会依安装后端选择torchcodec/librosa，可能有不同混音/重采样路径。

[固定processor配置](https://huggingface.co/akhilaaa3/Jev-Omni/blob/5addda86ddee081a68fb067477ea100c221b8917/processor_config.json) 的原生video processor num_frames是32，而公开loader实际走16张image路径；原生视频会构造时间戳占位内容，两条路线不能默默互换。本项目既有Gemma4Processor5.19.0 CPU冒烟也不等于该Gemma4Unified5.17.0教师路径验收。

## 待冻结的预处理候选

候选A是公开loader音频路线：保留全部1,000源行及完整原音频，推理输入取首≤30秒，FFmpeg单声道/16kHz；明确冻结FFmpeg/依赖、输出codec、混音策略及输出有效性。69份6通道、66份超过30秒、最长34.4785秒均已完整解码，属于合法来源，不作损坏或排除。本候选能复现公开loader意图，尚不能证明复现发布MMAU成绩。

候选V是官方MVBench有界采样：代码实际以num_segments16实例化，`start_idx=max(first_idx,round(start*fps))`、`end_idx=min(round(end*fps),max_frame)`，再按区间中点采样。STAR row92 EDXBD end14.0>12.864秒、row107 K47J5 end28.8>28.258005秒、row140 ALXUC end17.0>15.58秒，保留原注释、标签和分母，按此官方边界夹取可构造候选；是否为教师实际规则仍待确认。不能把公开loader的整视频16图路线直接声称为有界任务同条件复现。

这些候选冻结的是可审计的参考行为与决策边界，尚未批准替换项目评测协议、门槛或读出。下次执行前必须在独立验收票固定选用路线，并对原始/有效时间窗与实际帧数留证据。

## 音频临时输出的无卡复现

固定loader先 `NamedTemporaryFile(delete=False)` 创建空WAV，再用没有 `-y` 的FFmpeg写同一路径。在项目自有临时目录生成100毫秒、单声道16kHz PCM16合成WAV，原样调用上述选项，以DEVNULL stdin模拟非交互作业；不加载模型、不用真实题目、不覆盖用户文件。每次只创建并删除自己的目标。

实测平台、原命令、stdout/stderr、真实exit、文件大小/哈希和适配后的WAV帧数均在机器报告中。不能仅凭FFmpeg exit0声称成功；本机8.0.1实测拒绝覆盖时exit0但输出仍0字节。最小适配候选是在项目独占临时目录中创建自己的输出，限定仅对此路径加 `-y`，并在交给processor前检查常规文件、非空、WAV可读、mono/16kHz、预期时长/帧数；finally清理。保持原cap/混音/重采样和分类读出，不修改上游快照。

本次已证明的是macOS8.0.1的最小CPU行为：原样命令exit0/0B，`-y`候选exit0/3,278B，WAV可读且为mono/16kHz/1,600帧。Linux补充复核使用同一嵌入报告的合成驱动，`flock` PID25486排队等待共享重型锁，尚未启动Python，不能宣称Linux结果通过。远端自有目录 `artifacts/jev-protocol-probe/` 的 `audio-overwrite-probe.json` 将在实际执行后生成；主会话可据该终态补充独立证据，不能把排队快照算为成功。此次研究与本机最小复现已完成，无需为此重启任何作业。

这属于teacher真实执行的先决缺口，应并入 `s0:01` 或由主会话创建独立依赖票，补回归后再宣称音频入口可用；本研究切片没有实现适配或关闭该票。

## 上游联系草稿（未发送）

拟发到 [Jev-Omni公开讨论](https://huggingface.co/akhilaaa3/Jev-Omni/discussions)，由主会话在获得发送授权后处理：

> We are preparing a reproducible evaluation of Jev-Omni revision `5addda86ddee081a68fb067477ea100c221b8917`. Could you share the exact fourteen MVBench task JSON names, dataset revision/file hashes, and the ordered 2,786 public row identifiers (task file plus original row index), including reasons for omissions relative to that public version? We also need the evaluation entrypoint, prompt/state and option order, clip-bound/frame/decode-failure rules, image-list versus native-video route, audio cap/downmix/resampling settings, dependency versions and aggregation denominators. For MMAU, could you pin the public mini version and 1,000 IDs? Please share only public benchmark identifiers and protocol/code, with no hidden labels, private splits or model predictions. The public audio loader creates its output before invoking FFmpeg without `-y`; we observed an empty output after overwrite refusal. Was the reported evaluation run using a separate adapter or entrypoint?

无GPU、付费操作、外发消息或门槛更改；不能据此通知开卡。
