# Decision Index evaluation-only准入提案 001

2026-10-16。仅复核既有16项publisher候选；未扩大到其他28项调查、未读题/答案行或封存、未加载pickle、未运行上游/GPU、未下载数据、未接受新条款/账号/付费。以下是root可审查的提案，不是实际准入或法律建议。远端原manifest admissions仍为[]。

**13项数据级发布方证据可支持现有范围内的evaluation-only提案；3项明确保留缺口。** 精确Admission数组为 `.scratch/s4/decision-index-admission-proposal.json`，SHA `11e9bc766804206a8d9d5ed2310e14adffd9da5082f181db99b86a5f0efadded`。其中evaluation_admitted=true是待root批准草案字段，evidence字符串明确root_review_required；未应用到任何manifest。每项绑定原source revision/archive/input manifest SHA、对应源文件集合SHA、发布方README/card/LICENSE URL/revision/SHA及用途边界。完整结构化事实见同名JSON（含145个实际文件path/SHA/destination）。

固定原source：f8552a13b643d39f61185f5951bb5fed16bedf51；原tar SHA `12719836807976e25e9ff5b5dee742679577b48508cbd601f7c57be9672f4082`；输入manifest SHA `44e6782bae205b655ddbe8f8d4ca3afb5095932e78b59f4af3e189a2993ed77b`、report SHA `47fa59533daa18bb0d1c3a2b38df74030bb431fc0dd23a5087d12b1266216c2c`，本轮远端实际再次核对；1530文件metadata未改变。本提案不把已安装的derived source patch称为原版未修改。

| ID | 范围 | 数据许可证据 | 本提案边界 |
| --- | --- | --- | --- |
| 10 | SGD/SGD-X test | 固定README242–247明确两个datasets CC-BY-SA4；LICENSE.txt SHA71a43a… | 保留署名/SA声明；仅本地eval |
| 24 | MMLU all/test | 固定cais/mmlu card license与Licensing Information均MIT | 只评测，不转训练 |
| 26/27 | ARC-Easy / ARC-Challenge各自test | resolved acquisition210d026…固定card CC-BY-SA4 | recipe revision=None不伪称pin；本项目实际取得revision和文件SHA固定；准确分开两文件 |
| 29 | HellaSwag validation | 固定Rowan card Licensing Information MIT | 本地eval，无转发原文本 |
| 30 | GSM8K main/test | 固定openai card明确GSM8K dataset MIT | 保留机械选项重建来源 |
| 31 | ChessBench test bag | 固定README286–291：部分CC0/其余CC-BY4 | 不误用软件Apache或权重CC-BY条款；混合数据许可完整保留 |
| 36 | BRIGHT examples/documents | 固定xlangai card CC-BY4 | 保留query/doc来源及机械检索变换记录 |
| 43 | CRUXEval data及固定released distractor ZIPs | 固定README89–90明确CRUXEval MIT；root LICENSE SHA055ca9… | 包含已发布模型生成distractors这一事实；不下载/执行CodeLlama模型，不推断另一次模型使用许可 |
| 48 | ForecastBench datasets | 固定README18–20明确datasets CC-BY-SA4 | 0.3评分retired；仍覆盖固定rebuild输入，不能冒称活跃评分项 |
| 50 | Habermas comparison parquet | 固定README76–78把all other materials列为CC-BY4 | 软件Apache与数据分开，不下载权重 |
| 57 | MMLU-Pro test | 固定TIGER-Lab card MIT | 仅本地eval |
| 62 | When2Call test MCQ | 固定nvidia card CC-BY4 | 仅本地eval |

sgd/searchless_chess/cruxeval/forecastbench/habermas五个原真实Git root、HEAD已再核对与固定acquire pin完全一致；上述README/LICENSE实际SHA再核对。HF卡沿用上一份已哈希、固定revision的官方元数据证据，不以动态main替代；未重复下载大数据。所有源文件绑定复用已验证输入manifest及其逐catalog集合摘要，本轮未重哈希5.90GB。

## 16项内剩余3项

- **11 ContractNLI**：实际只读固定zip的LICENSE/TERMS成员，不读test.json。LICENSE CC-BY4，SHA9e5f1b…；另有`contract-nli/TERMS` SHA `b86e5b9bd7d2c49fd0d60e1d0842b7f64746abc8b206451acbe9080027cc49c8`，明确访问/下载/使用被描述为同意Terms。未凭已下载推定root有额外条款授权，故此轮不提案。
- **12 ANLI**：固定官方card CC-BY-NC4；evaluation-only不自动证明本项目用途non-commercial。已有政策只写research/evaluation staging，尚无明确非商业用途判定，故不自动填true。
- **64 New Yorker**：固定card明确其贡献的annotations CC-BY4；recipe还使用候选caption文本。无cartoon images不等于底层caption权利已清楚，保留该范围缺口。

没有调查其他28项；它们仍沿用原pending，不得由这份部分提案批量准入。即root接受13项，固定44项仍缺31项（逐ID在JSON中），ready=false；当前实际全部44项仍未应用准入。工具schema没有用途字段，故每项evidence明确同一限制：offline local evaluation/rebuild only、no training/commercial release/dataset redistribution、保留作者/source/license/机械变更记录。署名与未来分享的CC-SA义务不因不redistribute而从来源证据删除；未来扩大用途另审。

## 实际schema与验证

`scripts/decision_index_offline.py:131` Admission仅允许catalog_id/evaluation_admitted/license/evidence四字段；本提案完全匹配。原PreparationManifest带完整1530file metadata加入该草案后也通过schema；`CATALOG_IDS`仍44，缺31条件明确保留，没有跑prepare/rebuild或改工具。

生成入口是离线读取已有证据与本轮只读metadata观察的独立小脚本：

```sh
PYTHONPATH=. .venv/bin/python .scratch/s4/decision-index-admission-proposal.py --evidence reports/explore/decision-index-evaluation-license-evidence-001.json --observed-metadata /tmp/index-proposal-bindings.json --proposal /tmp/decision-index-admissions-review.json --report /tmp/decision-index-admissions-review-report.json
```

`observed-metadata`是本轮SSH只读得到的{manifest,manifest_sha256,report_sha256,git}：读取远端s4-index-inputs-001两个JSON；五原Git root执行rev-parse --show-toplevel/HEAD并核对相等，README/LICENSE仅哈希；没有题目载荷。本报告记录该观察文件SHA。复核时可重新生成同结构观察，不依赖/tmp持久化或信任下载即准入。

实跑13 Admission/Manifest schema、相对路径安全检查、44项缺口不变；receipt SHA、archive SHA、publisher pin、catalog file SHA四种篡改均拒绝；ruff/format、py_compile通过。脚本拒绝覆写输出。root下一步只需逐项审查草案与3项具体缺口；本切片没有应用或授权任何准入，也不代表全量离线重建已ready。
