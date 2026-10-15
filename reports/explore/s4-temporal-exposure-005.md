# 时序课程终态曝光核对 005

2026-10-16。主线b757213、738fe36已在独立worktree吸收（无冲突merge c93e29c）；本切片仅修改reconcile helper/针对测试/本报告，不编辑主树、STATUS或票据。实际仅CPU，无GPU。

新增显式 `--temporal-curriculum`，默认旧行为不变。此分支以真实exposure004为base（原SHA d28d98b2…），仍核验joint/video/crossmodal-retention/CE0/pair01五旧pilot、完整historical index与所有发现的update-bearing reports，另纳入时序终态。base004报告及输出manifest SHA双绑定；时序prep必须ready_for_preflight/isolated、source_admission verified_existing_scope，ledger/config/recipe/原manifest/audit输出SHA持续绑定。

时序final/runtime均completed10/exit0；固定seed42/120sample/10step/12accum、pair0配置一致，init身份与prep一致。runtime data SHA `9b163f5bd0f6e57eab86922f6210e2d241cdee7149e0630f9918171ec39dadff`，120个Sample与prep全字段相等；ledger `50e536c75d33aae41540d21309adbfffaa5735a2ba47c1432bc71feca73bc583`。未终态、短completed、错init/data/ledger/base或未闭合准入均拒绝；failed但steps>0仍依旧机制记录更新来源，不因训练失败漏曝光。历史TRAIN artifact表示该真实更新输入来源，不将文件总行数冒充失败run的实际forward次数。

实际执行19.8653秒，exit0；最新目录 `/root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-005`。发现48份更新报告，unknown0；18个旧indexed input仍完整核验；追加1个时序canonical历史TRAIN artifact。完整inventory audit isolated，最终再次验证全部输入/报告未变化。报告 `reports/s4-pilot-exposure-reconciliation-005.json` 留存完整小型绑定证据，仅aggregate/文件metadata，无封存载荷。

- 最新inventory SHA：`996a51ce4849ff1d8510ac3dd4ce93f407ec7f40cd9988abeea559625b5bafff`
- audit SHA：`be2d8ddf14f8cd57df7b5aca753580e083d8ef14689103de3ec76c33e375e3a8`
- discovery SHA：`3e40c2cf5dceae6cdca179778a6d980b29302acbfb1370f8daa91b99a1fa042c`
- 部署helper SHA：`7280c280417677de2efd0c45bc735d1d8155ceb7d4076bf6e9a46f0e1726c9ca`，与提交源码相等。

独立部署 `/root/autodl-tmp/s4-exposure-temporal-code-001/reconcile.py`，实际命令：

```sh
PYTHONPATH=/root/autodl-tmp/s4-claim-pair-code-001 /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python /root/autodl-tmp/s4-exposure-temporal-code-001/reconcile.py --root /root/autodl-tmp/s1-text-lora-001 --legacy-root /root/autodl-tmp/lite-omnijev --output-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-005 --temporal-curriculum
```

已执行，005不允许覆写；再次执行须全新目录。全部输出文件600/目录700独立复核。21项CPU测试通过（7项新增，覆盖004 base、保留全部旧pilot、真正新增canonical工件、活报告/输入/ledger/init/base/admission拒绝）；ruff check/format、mypy与CLI help通过。未知外部未保存历史、gold重新准入及S4质量不在本证据范围，成本null。
