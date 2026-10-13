# 2026-10-13 公开训练数据准入审计

结论：P0 继续使用现有 100 条程序真值。OneJev-Data 整库和 starter 均不准入训练；可先下载并校验 **CLINC 的单个公开 train 文件**，保留原始人工标签，待转换和样本族隔离验证后再入库。合成教师数据保持 pending。本轮不下载训练载荷，不读取任何 validation/test/holdout 内容，不改阶段门槛。

## 审计边界与证据

按 research 技能做有界一手资料核查。OneJev 固定 `c991b20e70359acb4448147c2ce99eecccd08ac4`；读取 README、licenses.csv、公开 train starter JSON 的字段/来源/许可/teacher/meta，未展示或分析题目答案。树 API 仅查文件元数据。读取直接来源的数据卡与 CLINC 原始 README/LICENSE。网页工具未能打开 HF 固定页面，改用 HTTPS 读取同一固定 raw URL；没有执行上游代码。

[OneJev README](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/c991b20e70359acb4448147c2ce99eecccd08ac4/README.md) 指明逐源许可，18 个混合来源训练分片约 17.7GB；100 条 starter 是 seed=0 训练样本而非封存评测。[逐源表](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/c991b20e70359acb4448147c2ce99eecccd08ac4/licenses.csv) 的许可证声明只作筛选证据，不能替代原始作者许可与转换链审计。

`configs/data-admission.json` 记录 127 个来源的初筛状态、原始声明、理由、三个已读取文件的 SHA-256，以及精确候选文件。它是证据登记表，**尚未接入下载器/训练器，不是已经实施的自动门禁**；`admitted_external_training_files=[]`。所有未逐源验证的条目保持拒绝默认值。许可不明与单独图像条款待审，research/non-commercial 来源排除出商用训练池；不得因模型权重 Apache-2.0 而放行数据。

## 可安全继续准备的原始人工真值

| 项目 | 固定值 |
|---|---|
| HF repo / revision | `clinc/clinc_oos` / `155b9c710419136e17307b80d0a13e68cd46b4ec` |
| 唯一候选文件 | `plus/train-00000-of-00001.parquet` |
| 预期大小 | 312,096 bytes |
| 上游 LFS SHA-256 | `30188119cf9f86fc9db27e1c22442d091cb5cb0913c9496f945fe11e7a02a28f` |
| 许可 | CC BY 3.0；保存作者、论文、许可、来源与修改说明 |
| 状态 | 可下载到隔离 raw 区并校验；尚未训练准入 |

[HF 固定数据卡](https://huggingface.co/datasets/clinc/clinc_oos/blob/155b9c710419136e17307b80d0a13e68cd46b4ec/README.md) 声明 train 15,250 条、151 类和 CC-BY-3.0；原始 [CLINC README](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/README.md) 说明语句由众包生成，[原始 LICENSE](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE) 确认为 Attribution 3.0。原始类别标签是本候选真值来源，没有采用 Jev API 或 JEV-27B 标签的证据；这不等于逐行数据已验证。

仅取上述 train 文件，不使用 `load_dataset` 默认全 split 下载；原始 GitHub `data_full.json` 混合多个 split，不应下载读取。下载后校验 SHA-256，验证 parquet schema/数量，保留原标签和 sample family，沿原句/改写族去重与隔离。151 类不能直接塞进 2–20 选项 Choice；需冻结符合语义的候选集构建规则，保留原类别真值。`oos` 是领域外意图，不自动等于证据不足 U。此数据不替代内部人工双标、κ、封存集或 GPU 验收。

## 合成候选与纠正过宽推断

- [typed-decisions-synth](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth/blob/7b86c5eda358b0d3bf4a45d503675ea1c16cfa00/README.md)：MIT；作者明确生成和标注来自 DeepSeek V4.1 Flash。精确 train 为 `data/train.jsonl`，17,670,034 bytes，LFS SHA-256 `1f3e0f751b6fdd1115cf6af4ea3ebc13545a1148614eda4f9f56ae8492c80675`。仅待审候选，不能把 gold 字段视为程序/人工真值；行级 teacher 一致性、许可保留和 case 隔离尚待验。当前优先人工真值，不启动该下载。
- [jevlite_dataset](https://huggingface.co/datasets/vagmi/jevlite_dataset/blob/b1c29cfe506a35579b68784b526a95dcf475dea2/README.md)：作者声明状态和标签来自本地 Qwen3.6-35B-A3B，不应因名称含 jev 就断言使用被禁止模型。`synth.train.jsonl` 为 7,885,118 bytes，Git blob `014536c6361d3d01ea0533599aefa1b211d5772e`（不是 SHA-256）。CC BY-SA 4.0 的署名、派生数据再分发与产物许可边界需完成包装审查，保持 pending；不读取 synth.eval 或混含 eval 的 all-labelled/raw 文件。
- OneJev starter 的 4 条 `onejev/distill` 和 2 条 `retention/onejev/distill` 元数据明确 teacher 为 `Qwen/Qwen3.5-27B`；6 条 typed-decisions 元数据指向 DeepSeek。可纠正“所有 distill 都被禁止”的推断，但这 12 条元数据不证明对应全量来源均合规，不据此放行混合分片。上游转换所用原始 revision 仍缺失。

## Starter schema 与隔离要求

[公开 train preview](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/c991b20e70359acb4448147c2ce99eecccd08ac4/sample/sample-100-preview.json) 实测 100 条均 `split=train`，36 个来源。键并集为 `id, source, task, split, state, media, question, target, meta, license, teacher`，teacher/meta 可缺失。JSON 的 media 为引用，不能当已下载图片；README 所述 Parquet 的图片字节需另行处理，本轮未读 Parquet。

来源名、许可证字符串或存在 target 都不足以入库。仍须来源 revision/hash、原始标签依据、变换记录、样本族键、train/dev/calibration 隔离，以及本项目 Choice/Score/Noul/U 契约验证。外部 benchmark 全量和封存内容不属于本轮范围。

## 验证与限制

机器表由固定 licenses.csv 生成并核对源数/行数总计、唯一 source、revision 与 SHA-256 格式；JSON 解析和 git diff 检查通过。不为静态研究文档添加镜像测试。未运行训练，未审核全部 127 个原始来源，未证明合成标签正确，未将任何新外部文件标为训练已准入。
