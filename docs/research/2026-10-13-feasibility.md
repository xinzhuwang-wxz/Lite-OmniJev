# 2026-10-13 方案可行性与上游契约核查

检索日期：2026-10-13（Asia/Shanghai）。范围：原设计的阶段 0–6、教师、概率读出、基线与三档部署。只读取公开元数据、文档和源码；没有执行上游代码，没有调用 Jev API，没有下载大权重，没有声称模型实测通过。

结论：可以继续本地工程准备；训练启动条件仍未满足。最先消除的风险是固定 revision 的模板/token 与 HF/vLLM 读出一致性、教师复现、数据许可/污染、真实人标封存集。远端榜单私有分和端侧内存目标也不能用本地合成测试代替。

## 1. 已核实的资产与当前文档漂移

以下 bytes 来自 Hugging Face `?blobs=true` 元数据；同目录四份 `*-inventory.json` 保存 revision、LFS SHA-256、文件清单和上游声明。它们是下载预期值，**不是已下载或本地校验通过的证据**。

| 仓库 | 固定 revision | 权重 bytes | 上游声明 / 判断 |
|---|---|---:|---|
| google/gemma-4-E4B-it | `ee0ef6023621cff504d758262d4e04895a5af4a2` | 15,992,595,884 | BF16，7,996,156,490 总参数；约 4.5B 有效参数不代表 4.5B 权重内存 |
| google/gemma-4-E2B-it | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | 10,246,621,918 | BF16，5,123,178,051 总参数；约 2.3B 有效参数不代表 2.3B 驻留权重 |
| akhilaaa3/Jev-Omni | `5addda86ddee081a68fb067477ea100c221b8917` | 23,919,549,408 + 3,966,079 的 head.pt | 当前已合并 BF16 主干；另有 FP32 decision head |
| OmniJev/OneJev-27B | `aec969d68e7e8e8b0df051e065c04304c0cd7163` | 54,713,608,144（两片） | 当前基座 **Qwen3.8-27B**；4B/9B 才仍列 Qwen3.5 |

四仓当前 API 均返回 `gated=false`。这只说明此次公共访问无需登录，不承诺未来权限不变。四模型权重合计约 104.9 GB（十进制），还没有计入 OneJev-4B 基线、缓存副本、数据、解压、训练产物和镜像。按阶段下载，不按有效参数粗估磁盘或显存。

证据：[E4B 卡](https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/README.md)、[E2B 卡](https://huggingface.co/google/gemma-4-E2B-it/blob/3e22461f65e89153144f8adb70e3b8c2cc9845a7/README.md)、[OneJev-27B 卡](https://huggingface.co/OmniJev/OneJev-27B/blob/aec969d68e7e8e8b0df051e065c04304c0cd7163/README.md)。

### Jev-Omni：源码优先于陈旧 README

固定 revision 的 [`load_jev_omni`](https://huggingface.co/akhilaaa3/Jev-Omni/blob/5addda86ddee081a68fb067477ea100c221b8917/jev_omni.py) 从一个 snapshot 装载完整 checkpoint，显式 `dtype=torch.bfloat16`，要求 `device="cuda"` 且 CUDA 可用；不再二次下载 Google 基座。README 的“自动下载原始多模态部件”和 decision_config 的 `dtype=float32` 是漂移元数据，不能据此沿用“约 50GB FP32 + 额外基座”的旧估算。

`head.pt` 通过 `torch.load(..., weights_only=True)` 装载；本项目“只加载 safetensors”不能被默默绕过。入库票须审计 tensor-only head，固定哈希，在隔离转换步骤导出 safetensors 并验证键/shape/数值等价，再由项目适配器加载；若需政策例外，用 ADR 和相应授权处理。此处没有执行转换或上游 loader。

上游 [requirements](https://huggingface.co/akhilaaa3/Jev-Omni/blob/5addda86ddee081a68fb067477ea100c221b8917/requirements.txt) 写 `torch>=2.10`、`transformers==5.17.0`；这属于教师独立环境候选，不能直接覆盖学生环境锁。模型卡声明没有使用 Jev 输出；这是一项可记录的发布者声明，不等于审计了其完整训练样本来源。

## 2. 原始概率读出可行，但必须测完整路径

核查的 vLLM 源码 revision：`c7a305ec10f3ec06c2784f1258162483c5fc8d73`。这是研究固定点，**不是本项目已验证的安装版本**；落地时选择含该能力的 release/wheel 后锁版本、平台、镜像 digest，并跑下述探针。

- [`SamplingParams.logprob_token_ids`](https://github.com/vllm-project/vllm/blob/c7a305ec10f3ec06c2784f1258162483c5fc8d73/vllm/sampling_params.py) 接受明确 token ID 列表，当前上限 128；返回这些 token 加 sampled token。21 个最大 Choice 标签（A–T+U）在此上限内。
- [`ChatCompletionRequest`](https://github.com/vllm-project/vllm/blob/c7a305ec10f3ec06c2784f1258162483c5fc8d73/vllm/entrypoints/openai/chat_completion/protocol.py) 暴露同名字段，要求 `logprobs=true`；指定 ID 优先于 `top_logprobs`。
- [`Sampler.forward`](https://github.com/vllm-project/vllm/blob/c7a305ec10f3ec06c2784f1258162483c5fc8d73/vllm/v1/sample/sampler.py) 在 allowed-token mask、penalties 和温度之前计算 `raw_logprobs`；`processed_logprobs` 是另一模式。启动参数应显式锁 `--logprobs-mode raw_logprobs`。
- 只有 `allowed_token_ids` 加 `top_logprobs` 不足：在 raw 模式，top-k 仍可能来自全词表，低概率 U/答案可能遗漏；processed 模式又改变分布。缺失 token 必须 fail，不能补 0 或给缺失项分摊质量。
- 对每题取 K+1 个目标 raw logprob 做稳定 softmax，即等价于对应原 logits 的 restricted softmax；将 U 单独返回，其余再条件归一化。不能对已截断的 top-k 候选假称“全答案分布”。

建议最小服务探针：Chat Completions 请求 `max_tokens=1, logprobs=true, logprob_token_ids=[本题答案及U的ID], chat_template_kwargs={"enable_thinking":false}`，不设 logit_bias/惩罚；如果用 `temperature=0`，也必须由 raw 模式证据证明读出不受其影响。解析时按 token ID 匹配，检查无重复、无缺项、数值有限；不要依赖返回顺序或仅凭解码字符串。已核实同 revision 的 [CompletionRequest](https://github.com/vllm-project/vllm/blob/c7a305ec10f3ec06c2784f1258162483c5fc8d73/vllm/entrypoints/openai/completion/protocol.py) 支持 `return_tokens_as_token_ids=true`。本项目文本网关可使用 `/v1/completions`，传共享模板渲染的 prompt、`max_tokens=1, temperature=1, logprobs=1, logprob_token_ids=[...]`；返回 `choices[0].logprobs.top_logprobs[0]` 的键是 `token_id:N`。其 [serving](https://github.com/vllm-project/vllm/blob/c7a305ec10f3ec06c2784f1258162483c5fc8d73/vllm/entrypoints/openai/completion/serving.py) 在指定ID时返回所有指定项，不受 HTTP `logprobs=1` 截断；转换到 SamplingParams 时不再传这一 top-k 参数。HTTP会把极低/-inf值钳为 `-9999`，应拒绝该哨兵以免伪造原始概率。多模态仍需独立验证其受支持的 Chat/processor 输入路径。以上是源码证据，所选部署 release 仍须实测契约。

HF 对照应来自**同一权重、adapter、processor、dtype、模板和 prompt token IDs**，取最后有效 prompt 位置预测下一 token 的 lm_head logits。保留 100 题跨模态结果，按原方案验证概率差 `<1e-3`。比较时关闭自动截断和 thinking，记录 padding 与多模态处理差异；重复运行一致性和前缀缓存门槛独立验证。

### 单 token 与 thinking

[官方 Gemma thinking 文档](https://ai.google.dev/gemma/docs/capabilities/thinking) 区分 E2B/E4B 与大模型模板；E 系列关闭 thinking 时不会自动添加大模型的空 thought 块。固定 [E4B chat_template](https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/chat_template.jinja) 的 generation prompt 以 model turn 开始。不能从“yes 看起来很短”推断单 token。

P0 可做 tokenizer-only 检查：A–T、0–9、yes/no、U，在最终 prompt 边界逐一验证恰好一个 token，检查 ID 唯一、解码往返与训练答案拼接一致。还须用 processor 验证多模态占位符，不以手写 token 字符串替代官方模板。若某标签失败，采用经探针验证的单 token 写法并保存映射版本；不变更业务选项语义。

## 3. `/v1/systemone` 兼容范围

权威是 [TypeSafe HTTP API](https://docs.typesafe.ai/api.md) 与 [SDK 响应类型](https://docs.typesafe.ai/sdk/python/api/types/responses.md)。独立实现对照为 [OneJev schema](https://github.com/OmniJev/OneJev/blob/6122519058af23504a33d944f0e63b2d868c8a4b/qev/schema.py)，其 [官方 SDK 示例](https://github.com/OmniJev/OneJev/blob/6122519058af23504a33d944f0e63b2d868c8a4b/examples/official_sdk.py) 设置本地 base_url。源码示例存在不等于本项目已通过 SDK 测试。

| 层 | 必需契约 |
|---|---|
| 请求 | `state` 为 string/object/array，`model` 为 string，`questions` 为调用方 ID → 问题的 map；ID 不得混入模型推理文本 |
| Choice 问题 | `type="choice"`，`instructions` 支持 string/object/array，`criteria` 为 option ID → 描述（允许 null） |
| Score 问题 | `type="score"`，`criteria` 是有序描述数组，2–10 级 |
| Noul 问题 | `type="noul"`，可选 `criteria={"true":描述,"false":描述}` |
| 响应外壳 | `model`，`answers`（仍按请求 ID），`usage={input_tokens,output_tokens}` |
| Choice 答案 | `type, choice, probabilities, confidence`；probabilities 按业务 option ID |
| Score 答案 | `type, score, legend, probabilities, confidence`；级别 key 是可转整数的字符串；score 是零起点等级期望值 |
| Noul 答案 | `type="noul", noul`；noul 是 yes 概率，不是 bool |

本项目限制单组 2–20 个 Choice，官方可到 255；分组淘汰必须单独验收，它不等价于一次 255 类联合分布。媒体、`p(U)`、不足证据状态、高可靠模式属于扩展，不应破坏 SDK 必需字段。SDK 兼容至少分请求解码、响应解码、字段语义、错误路径四层。

**已发现设计冲突：** [官方 confidence](https://docs.typesafe.ai/confidence.md) 对 Choice 用 peak 公式；Score 用“概率加权的等级到众数距离 / 均匀等级平均绝对偏差”。例如三级 `(0.5,0,0.5)` 的官方 Score confidence 为 0，设计统一 peak 公式为 0.25。不得静默改设计或称完全语义兼容：先保留冻结设计，用 ADR 声明兼容子集，将是否改 Score 公式作为用户决策。Noul 官方不返回独立 confidence。本项目的条件 yes 概率还须连同 p(U) 说明，避免误解为无条件正确率。

## 4. 许可、来源与污染门槛

Gemma4、OneJev 当前模型卡均声明 Apache-2.0；Google [Gemma4 许可页](https://ai.google.dev/gemma/docs/gemma_4_license) 是基座许可来源。应保存对应 LICENSE/NOTICE、revision 与文件哈希，但本报告不替代逐源许可核验。

[OneJev-Data](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/c991b20e70359acb4448147c2ce99eecccd08ac4/README.md) 明确为 **per-source license**：公开 94,707 条 / 原训练 99,193 条，未公开部分因来源禁止再分发而移除；100 条 starter sample 属训练集，不能当 holdout。其 [licenses.csv](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/c991b20e70359acb4448147c2ce99eecccd08ac4/licenses.csv) 包含 non-commercial、图像另受 source terms、none stated 等条目，也有 `onejev/distill`、jevlite 等需追溯标签来源的记录。

这是**来源待审计**，不是断言权重违法或所有蒸馏都不可用。处理方式：教师权重许可、上游训练来源声明、目标数据来源许可、标签生成方式分别记录；未澄清的来源不能被笼统的模型 Apache 标签洗白。研究-only 数据与可发布训练池隔离；优先程序化真值和许可清楚的原始硬标签，沿 sample family 去重/去污染。模型卡没有覆盖的训练来源不能自行编造“从未使用 Jev 输出”的保证。

## 5. 复现目标要冻结分母、分割和评分方式

| 目标 | 已核实边界 | 执行动作 |
|---|---|---|
| Jev-Omni MMAU 63.10 | 模型卡为 1,000 题；[MMAU upstream](https://github.com/Sakshi113/MMAU) 发布 1,000 test-mini + 9,000 test，后者无公开答案 | 冻结 test-mini revision、样本 ID、音频预处理；不能拿 9,000 blind test 或任意子集直接比较 ±2 分 |
| Jev-Omni MVBench 53.10 | [模型卡](https://huggingface.co/akhilaaa3/Jev-Omni/blob/5addda86ddee081a68fb067477ea100c221b8917/README.md) 限 14 tasks / 2,786 题，展示 group avg 53.10、question avg 53.09 | 取得精确 task/ID 清单、macro/micro 规则和媒体可用性；未拿到清单时标复现阻塞，不能自选容易的 14 类 |
| JevBench 86.15 | 同卡是 matched 195 groups / 231 decisions 的 group average；question average 87.45 | 匹配清单、答案语义和 group 聚合都锁定；不把题均值误作组均值 |
| Decision Index | [固定 kit](https://github.com/apolinario/decision-index/blob/f8552a13b643d39f61185f5951bb5fed16bedf51/README.md) 默认 edition 0.3；公开分 37 benchmark，经 chance/coverage 修正；Full 只有20%公开，其余维护者私有；Vision也有私有部分 | 配置锁 edition/manifest/哈希和评分器；本地只称 public index。Full排名依赖维护者运行，不能代理伪造；需要外部提交时另走授权边界 |
| 内部 holdout | 设计要求每模态 1k 人工标注，κ≥0.7，阶段4/5/6由隔离gate读取 | 确保独立标注者、真实分歧/裁决记录、访问控制和审计。程序化/教师/agent标签不能冒充人工双标 |

Decision Index 重建有许可/访问门槛（例如 HLE 需账号接受条款），不能让下载程序自动代表用户接受条款；其原始公共数据也不一定允许再分发。只保存可发布清单/构建配方。没有完整 coverage 时，不把测试成功数作为完整榜单分母。

当前设计“≤5B 模型 SOTA”与 Gemma E4B **约8B总参数**存在口径歧义。报告必须同时列 total/effective/active 参数、精度、硬件与输入长度，不可用 effective 口径与他人 total 口径混榜；这一对外宣称定义需要 ADR 冻结后才可发布。

## 6. 无卡与 GPU 阶段边界

| 阶段 | 可完成的证据 | 不能替代的证据 |
|---|---|---|
| 本地 macOS ARM / Python 3.11 | 数据 schema、纯数学指标、模板/token 探针、来源清单、下载校验、小样本管线、网关错误处理、真实浏览器交互 | CUDA/vLLM numerical parity、LoRA训练收敛、教师效果、真实GPU显存/延迟 |
| Linux 无卡实例 | 已锁依赖与镜像准备、权重/数据分批下载、哈希验证、磁盘/网络检查、作业模板与CPU预检 | CUDA kernels/FlashAttention/FlexAttention正确性与性能 |
| 经用户开卡后 | 教师复现、HF/vLLM原始读出与缓存、最小训练冒烟、阶段门槛与计费证据 | 跨真实手机的内存/功耗/端侧logit接口保证 |

[vLLM官方安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) 的 NVIDIA 路径以 Linux/CUDA 为准。Apple Silicon 已有独立 vLLM-Metal 路径；[插件安装页](https://docs.vllm.ai/projects/vllm-metal/en/stable/installation/) 要求 macOS 15、原生 ARM Python 3.12，不能说“vLLM绝对不能在Mac运行”，也不能将该路径当 Python3.11 CUDA部署的等价验证。本文未确认其 Gemma4 全模态 selected-token/LoRA支持；没有必要为P0扩大到新运行时。

## 7. 优先风险与分阶段补救

| 优先级 | 证据 / 尚待证明 | 处理与退出条件 |
|---|---|---|
| P0 | 工程P0要求模板/读出验证，训练S0又依赖P0，存在依赖环风险 | P0建立探针、tokenizer检查、环境与纯数学测试；S0做真实HF/vLLM和教师比较。P0不得伪标GPU检查pass；S0阻塞不阻止P0代码完成 |
| P0 | 当前上游教师包已改变，README/metadata有冲突 | 固定revision+清单，以实载代码与文件头为准；教师报告追溯到同一revision；未达原±2分门槛则不入教师白名单 |
| P0 | 人标κ和封存身份尚需真实主体 | 自动完成标注格式、分配、导出、汇总脚本；获取真实标注服务/人员需用户授权时入STATUS待决策，同时继续其他票 |
| P0 | top-k可能漏U/低概率选项 | selected-token全量返回、raw模式证明、100题 `<1e-3`；任何缺项立即fail |
| P0 | OneJev混合来源不等于可商用训练池 | 建逐源数据卡、可发布许可白名单与污染检查；不明项隔离，不牺牲阈值以填满数据量 |
| P1 | API Score语义、≤5B口径与设计不一致 | ADR记录，不擅改；实现可先保留明确声明的兼容子集，阻断不实对外宣传 |
| P1 | 原方案3GB端侧、24GB BF16服务、时延/预算都是假设 | 实际文件+峰值RSS/VRAM+KV/activation测量；Q4裸权重理论值不含元数据/缓存。未过就保留blocked，不能把量化下载成功当端侧验收 |
| P1 | 前缀缓存只复用共同前缀，媒体处理/并发调度仍有成本 | 同图1题与10题使用同样并发/冷暖条件，验证原 `≤2x`；不把命中日志等同于目标时延达标 |
| P1 | 公开子集不能证明私有榜单或内部holdout | 每报告明确 scope/edition/coverage；最终票逐项保留外部验收依赖 |

推荐接续：完成 P0 可执行资产/数据/模板/读出探针与独立环境锁 → 无卡主机分批预下载并校验 → 用户开卡后 S0 最小复现 → 只在 S0 全部门槛通过后展开 S1 实验票。原方案阶段门槛、预算和人工决策边界均不因本报告改变。
