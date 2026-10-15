# Decision Index evaluation许可事实审计 001

日期：2026-10-16。只读事实整理，不是法律建议或准入决定。44个固定构建catalog全部覆盖；远端 admissions 仍为 `[]`，未修改任何准入。未读取题目/答案数据行、封存、pickle，未运行上游或GPU。

机器证据：`decision-index-evaluation-license-evidence-001.json`；SHA256 `384966c8aaa2271c29a9a6222684c92b41c158e9d7801972242509e3919af206`。每项包括实际输入文件数、前三个source/sha/destination示例、完整该项binding集合摘要、固定recipe来源和许可范围。完整文件集合以远端manifest的SHA绑定，未把1530项重复入Git。

远端manifest SHA `44e6782bae205b655ddbe8f8d4ca3afb5095932e78b59f4af3e189a2993ed77b`；report SHA `47fa59533daa18bb0d1c3a2b38df74030bb431fc0dd23a5087d12b1266216c2c`。固定source `f8552a13b643d39f61185f5951bb5fed16bedf51`；原tar SHA `12719836807976e25e9ff5b5dee742679577b48508cbd601f7c57be9672f4082`。上游 `docs/suite.md` 是二级整理，明确要求自行核对数据权利，root MIT仅代码。

## 可供主会话按现有政策核对的分组

- 发布方数据声明证据候选（16）：10, 11, 12, 24, 26, 27, 29, 30, 31, 36, 43, 48, 50, 57, 62, 64。这不是admitted：11还需dataset page terms，64仅annotations并保留底层caption边界；具体证据在JSON。
- 限制用途/多来源需逐项边界（7）：22, 25, 45, 56, 58, 59, 61。HLE复用已有授权receipt，不能新接受条款；RAGTruth/Yelp不能重分发。
- 许可或数据适用范围未闭合（21）：1, 2, 3, 4, 5, 6, 9, 20, 21, 23, 28, 32, 33, 34, 37, 38, 39, 40, 41, 42, 44。不得用仓库代码许可证填数据准入。

## 固定catalog与实际来源

| ID | Benchmark | 文件数 | 固定recipe许可说明（非批准） | 审计边界 |
| --- | --- | ---: | --- | --- |
| 1 | BFCL | 41 | Apache-2.0 (repo LICENSE) | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 2 | ToolRet | 39 | see repository LICENSE; queries/tools aggregate many upstream tool datasets with their own terms | 固定Queries/Tools卡未发现许可字段；repo Apache不能清除聚合上游工具数据条款。 |
| 3 | API-Bank | 319 | see api-bank/LICENSE in the repo | 仅 api-bank/LICENSE Apache-2.0；DAMO根MIT及其他项目许可证均不适用自动推断。还需发布方确认其样本数据范围。 |
| 4 | BANKING77 | 2 | CC BY 4.0 | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 5 | CLINC150+OOS | 1 | CC BY 3.0 (per dataset card) | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 6 | RouterBench | 4 | see repository LICENSE | 固定HF卡未找到许可字段；转换JSONL不改变原数据权利。 |
| 9 | Home appliance simulator | 0 | MIT (this repository) | 本固定source generator及root MIT，先核对其生成内容来源范围；不是外部下载。 |
| 10 | SGD/SGD-X | 35 | CC BY-SA 4.0 | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 11 | ContractNLI | 1 | see repository LICENSE and dataset page terms | 本地LICENSE为CC-BY4；suite仍提示dataset page terms，准入前保留该附加核对。 |
| 12 | ANLI | 3 | CC BY-NC 4.0 | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 20 | BPoMP | 3 | per the Zenodo record | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 21 | Humicroedit | 1 | SemEval-2020 Task 7 terms | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 22 | POP909-CL | 909 | see repository LICENSE; POP909 underlying data is research-use | repo MIT不能覆盖POP909音乐底层研究用途限制。 |
| 23 | cfcolor | 1 | per the project page | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 24 | MMLU | 1 | MIT | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 25 | GPQA Diamond | 1 | CC BY 4.0; authors ask not to publish the questions in plain text | suite称数据CC-BY4，但本次当前官方仓库LICENSE为MIT；代码/数据范围与快照不同，不能据此批准固定zip。保留不得公开明文题目请求。 |
| 26 | ARC-Easy | 2 | CC BY-SA 4.0 | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 27 | ARC-Challenge | 2 | CC BY-SA 4.0 | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 28 | WinoGrande | 1 | see dataset card | 固定卡Licensing Information为More Information Needed；不能从suite see card变成准入。 |
| 29 | HellaSwag | 1 | MIT | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 30 | GSM8K | 1 | MIT | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 31 | ChessBench | 1 | see repository LICENSE (Apache-2.0 code) | README明确code Apache、weights CC-BY、data部分CC0/其余CC-BY4；不可把bag标Apache。 |
| 32 | MuSR | 3 | MIT | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 33 | SATA-Bench | 1 | see repository LICENSE | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 34 | SimpleBench | 1 | MIT | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 36 | BRIGHT | 24 | CC BY 4.0 | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 37 | Amazon ESCI | 2 | Apache-2.0 | README只写project Apache；本轮保守保留数据范围确认。 |
| 38 | ACOS | 6 | not stated in the repository | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 39 | FinEntity | 1 | not stated in the repository | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 40 | iSarcasmEval | 5 | see repository LICENSE | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 41 | VAST | 1 | not stated in the repository | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 42 | NLI4CT | 3 | SemEval-2024 Task 2 terms | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 43 | CRUXEval | 3 | MIT | README明确CRUXEval MIT；还包含codellama生成的distractor样本，记录这一来源边界。 |
| 44 | CLadder | 1 | see repository LICENSE | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 45 | HLE | 1 | MIT (per dataset card); gated access | 已有hle-source-download-001获授权研究评测receipt；本次匿名固定卡401，不能新推断门条款或商用训练准入。 |
| 48 | ForecastBench | 74 | see repository LICENSE | README明确datasets CC-BY-SA4；0.3运行已retired，本次离线build闭包仍包含，需区分构建与实际评分。 |
| 50 | Habermas Machine | 1 | see repository LICENSE | README把software Apache与all other materials CC-BY4分开；比较parquet为后者。 |
| 56 | PhishNChips | 2 | per the dataset's SOURCE_LICENSES.md (project content MIT; URL sources cleared for academic use with attribution) | 固定SOURCE_LICENSES明确academic research/attribution，多来源；不是MIT无限制，不替用户接受条款。 |
| 57 | MMLU-Pro | 1 | MIT | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 58 | BBH | 27 | MIT; carries the BIG-bench canary, never train on it | BBH repo MIT，但保留BIG-bench canary及不得训练边界。 |
| 59 | RAGTruth | 2 | MIT annotations; contexts from CNN/Daily Mail, MS MARCO (non-commercial) and the Yelp Open Dataset (no redistribution): rebuild locally, never republish the rows | MIT仅annotations；CNN/DailyMail、MSMARCO非商业、Yelp不可重分发需分来源处理，不能整体MIT准入。 |
| 61 | HoVer | 2 | CC BY-SA 4.0 (data), Wikipedia text CC BY-SA | repo LICENSE MIT与suite CC-BY-SA data/wiki存在范围差异；需原始数据页和Wikipedia署名链，不能MIT覆盖。 |
| 62 | When2Call | 1 | CC BY 4.0 | 上游suite说明不是发布方新增授权；仅repo LICENSE的项需确认适用数据范围。 |
| 64 | New Yorker | 1 | CC BY 4.0 annotations; no cartoon images used | 固定卡CC-BY4明确仅所贡献annotations，底层NewYorker captions权利边界仍需确认；未用cartoon images。 |

## 重要范围修正

- API-Bank仅 `repos/apibank/api-bank/LICENSE`，SHA `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`，为Apache2；忽略同仓库EPO、Loong、MaskedThought等LICENSE。当前证据仍不将样本自动批准。
- ChessBench的data不是Apache；Habermas比较parquet为CC-BY4而software是Apache。ForecastBench README直接声明dataset CC-BY-SA4；它在edition0.3评分已retired但重建闭包仍读其输入。
- 固定HF卡：BRIGHT CC-BY4、MMLU-Pro MIT、When2Call CC-BY4、ANLI CC-BY-NC4；ToolRet两卡和RouterBench未找到许可声明；WinoGrande固定card仅More Information Needed。
- PhishNChips固定SOURCE_LICENSES说明多来源academic用途/attribution；该发布方说明不等于本项目已获得所有第三方新授权。RAGTruth与HoVer必须保留底层文本条款。
- HLE仅复用 `reports/hle-source-download-001.json` 的已授权研究评测下载证据，匿名读取固定card401；未绕过门禁。

## 复核与剩余条件

本轮 Python urllib只获取明确README/LICENSE/SOURCE_LICENSES与Zenodo元数据（失败也记录），不取数据。Git/HTTP/HF revision来源由固定acquire AST literal与既有下载receipt提供；ARC固定recipe revision=None，实际receipt resolved210d026faf9955653af8916fad021475a3f00453，不能将latest称为recipe固定pin。GPQA存在suite数据CC-BY4与当前repo MIT的范围/快照差异，列为待核验；不要任选其一。新增publisher证据带URL、SHA和许可段行号；远端证据带绝对路径、SHA。current master/main页面明确标与原HTTP文件SHA不同的revision边界，不能假称同一快照。

主会话下一步：仅对适用范围明确并符合既定研究评测政策的项做显式admission记录；其余逐项取得发布方数据许可/条款事实。不要将downloadable_verified或本报告分类复制成evaluation_admitted。完整rebuild、质量门、商用/训练许可均未在本轮验证。
