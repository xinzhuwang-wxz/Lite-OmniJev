# 固定BFCL/BBH离线评测范围应用

root复核BFCL固定data README的明确Apache-2.0，以及BBH MIT/数据README的禁止训练和canary后，将catalog1/58追加到独立003，18项准入、26项剩余，ready=false；不训练、不再分发、不执行外部API。原002/001不变。

实际验证68固定manifest输入文件hash/长度、四份固定publisher许可/数据范围文件hash，Pydantic清单通过；输出manifest SHA8ac03215c8497115c73626c3aa202113f472e436f6b50c0b868c3d7fc2292ce3。见reports/decision-index-evaluation-admission-003.json。

首版admit-index-third-evaluation.py在远端urllib读取官方metadata时timeout退出1，发生在003 mkdir之前，未改变任何manifest；替代入口admit-index-third-evaluation-002.py使用已有固定repo LICENSE/README，不再联网，exit0；源码与Git helper SHA5aec5944…一致。CPU-only，GPU训练control同时运行。许可范围不是模型质量或完整Decision Index评分证据。API-Bank/MuSR/ESCI三项新提案仍未应用。
