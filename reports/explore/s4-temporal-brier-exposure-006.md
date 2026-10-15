# CE+Brier终态曝光核对 006

2026-10-16。仅CPU，未在训练活跃时审计。root明确通知session40849已completed/exit0后执行。新增 `--temporal-brier`，默认旧行为不变；以真实005为base，核验其全部既有输入SHA，继续包含七个pilot的终态/数据绑定、完整historical index与未知更新发现。

Brier final/runtime completed10、seed42/120sample/10step/12accum、同temporal prep canonical全字段及CE0 init身份。对实际config-brier.json与training report config分别经LoopConfig验证，除brier_weight=.5外必须与SHA绑定config-ce.json规范化后完全相等。launcher必须终态，并以其prepared_report/config SHA绑定到实际文件；未把新Brier报告当unknown漏掉。

首轮 `/root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-006` 在0.19秒fail closed，原因是训练report config附带 `init_adapter` 执行元数据，LoopConfig extra-forbid拒绝；未进入完整audit。该目录与本地failure-001报告保留。修复只允许此字段在与prep/top-level/runtime身份严格相等时剥离，再做唯一Brier差异比较；未知字段/不相等身份仍拒绝。测试包含真实report结构的这一回归，不放宽正式LoopConfig。

真实重试目录 **`artifacts/s4-pilot-exposure-006-retry-001`**，19.8864秒/exit0：

- 完整inventory audit isolated；50份更新报告、unknown0；旧18个indexed input仍核验。
- Brier的120个canonical Sample与prep全字段相等；data9b163f5b…、ledger50e536c7…、init CE0不变。
- 新增artifact0：该相同canonical输入已在005中；两个新更新报告（final/run）仍分别保存实际10步，不能解释成没有新曝光/模型更新。
- inventory SHA保持 `996a51ce4849ff1d8510ac3dd4ce93f407ec7f40cd9988abeea559625b5bafff`；audit SHA `be2d8ddf14f8cd57df7b5aca753580e083d8ef14689103de3ec76c33e375e3a8`。
- discovery SHA更新为 `8ef6a86535b976c2c6da8a283c5a375f226bb73a8e3e418b7777714747e9ce49`。
- 部署/提交helper SHA `1eb5f61567afa0964535a2fb4ca9b858fda9cc602532e2d61d60efa57defb015` 一致；全部输出文件600、目录700。

真实命令（已执行，重试目录不可覆写）：

```sh
PYTHONPATH=/root/autodl-tmp/s4-claim-pair-code-001 /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python /root/autodl-tmp/s4-exposure-brier-code-001/reconcile-fixed.py --root /root/autodl-tmp/s1-text-lora-001 --legacy-root /root/autodl-tmp/lite-omnijev --output-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-006-retry-001 --temporal-brier
```

30项CPU测试通过：9项新增涵盖唯一Brier差异、same-input不重复追加artifact但保留更新、failed3steps记录、活报告/权重/其他超参/launcher hash/005既有输入变更拒绝，以及严格init元数据例外；ruff check/format、mypy通过。真实报告 `reports/s4-pilot-exposure-reconciliation-006.json` 指向重试成功目录；失败证据 `reports/s4-pilot-exposure-reconciliation-006-failure-001.json` 独立保存。不改原预测、数据或阈值，不读取封存载荷；未知外部未保存历史和质量门不在该证据范围。cost=null。
