# S1 文本决策LoRA业务闭环规格

Owner: main-session-2026-10-15
Parent: S0通过（reports/s0-stage-acceptance-001.json, ADR0010/0012）
State: 一个闭环；来源/探针/运行只作票内清单，不另拆票

## 目标

文本数据→决策LoRA→dev500同条件评测，通过S1机器门。用户2026-10-15明确授权负责人按实测自主决策/调整非实测规划指标（ADR0010精神）；质量结论不暗降，调整须记录原值/证据/新标准。

## 数据与池冻结（训练前完成）

唯一训练来源：`/root/autodl-tmp/s0-text-dev-001/artifacts/text-dev-001-a/train.jsonl`，14750行，SHA256 `12a42005478ff1cd2f618fbe4951ced1af878b9e115a9a055c57d9d79578cb11`（s0:13冻结的剩余训练集，全部为单样本family，CC-BY-3.0，clinc_human_label现成gold）。评测集：同目录`dev.jsonl` 500行，SHA `8c7809e8b07673904b71bc4676a49a5e31abb22ee7126f2dc09e8fde45b4e070`，只评测不训练。

入训前冻结池分配（family级，单样本family=行级）：
- key = SHA256 UTF8 `s1-text-pools-v1:42:<family_id>`，升序、family_id平局；
- 前1000 → `holdout-reserve`（未训练封存候选，最终holdout来源由s0:05决定；CLINC官方val/test未下载，若s0:05改选官方test，预留family归还训练只能凭记录决策）；
- 次3000 → `calibration-reserve`（S5每模态3–5k校准预留）；
- 其余10750 → `s1-train`。
输出data/allocation ledger/报告绑定输入输出SHA；双遍字节一致；与dev500 family及历史dev排除union的family零交集（文本无媒体，媒体别名不适用）。训练入口必须校验ledger并拒绝任何dev/预留family样本。

## 训练协议（复用P0已验证机制）

E4B `models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2` 主干冻结；LoRA r16/α32 仅decoder q/k/v/o与gate/up/down投影；官方chat template、明确关闭thinking；答案token=A..T选项字母+U，训练前单token探针在真实generation boundary通过（含20选项最大情形）；BF16；固定seed。损失：最后位置全部合法答案logits（含U）先FP32 log-softmax，再对hard target交叉熵；无温度/采样/top-k；超max_length拒绝不截断。训练入口逐样本经data/schema校验并核池ledger。

## 运行计划

- R0 smoke：2k样本/300优化步，训练后测这2k训练样本acc（优化健全性信号，非产品质量）；原门≥98%。
- R1主训练：30k samples-seen（10750池约2.8epoch），固定调度单次运行；在10k样本点与30k终点分别保存adapter并评测。原门"30k较10k+2pp"操作化为同运行10k点vs30k点的dev500对比；独有数据上限10750为已准入来源的真实属性，如因上限未达+2pp按实测记录，不虚增来源。

## 评测协议（同条件）

dev500与S0基线完全同条件：同一dev文件（SHA 8c78…）、官方模板、关闭thinking、全部合法答案+U原始logits、FP32 log-softmax、U联合归一后答案条件归一、p(U)单列、选项原顺序。HF native加载base+adapter推理（P0实测HF/native四概率差<1e-5，属同条件可比范围，记录为依据）。双遍确定性运行，精确重复对=1500。ECE操作定义：10等宽bin，confidence=预测答案（U除外）归一概率，预测U计错误；此定义为S1新增测量，S0未测dev500 ECE。

## 门槛

| 门 | 值 | 依据 |
|---|---|---|
| smoke_accuracy | ≥.98 | configs/gates.json S1（2k训练样本自测） |
| text_vs_onejev | ≥−3pp（即≥93.6%） | dev500 OneJev4 96.6%（s0-baseline-summary） |
| text_vs_base | ≥+5pp（即≥98.6%，绑定门） | dev500 E4B零样本93.6% |
| text_ece | ≤.06 | configs/gates.json S1 |
| 30k_vs_10k | ≥+2pp | configs/gates.json S1 |

绑定门98.6%高于当前最强基线96.6%：首跑按原门实测；若未达且属结构原因，按ADR0010式记录原值/证据/新标准后调整并通知用户，不暗降、不按模型成绩改集/分母。

## 证据

每run报告：命令、代码版本、数据/adapter/模型hash、seed、环境、逐步loss、耗时、GPU时长、`cost: null`、退出码；失败留证据不覆盖。SwanLab（ADR0003，Lite-OmniJev/maxen）只传真实训练指标。门槛判定用`eval.gate`机器可读报告+阈值配置。验收后更新STATUS并本地commit。

## Out of scope

其他模态、E2B蒸馏、量化/端侧、生产服务改动、封存holdout实际构建与gate（s0:05）、教师打标、CLINC官方val/test下载。
