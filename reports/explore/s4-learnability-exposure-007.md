# Learnability终态曝光核对 007

2026-10-16。独立CPU切片，无GPU、STATUS/票据变更或原件修改。新增显式`--learnability`，默认旧行为不变；以成功006-retry-001为base，先验证其全部既有输入SHA，包含此前七个pilot、historical index及完整未知更新发现，再纳入真实learnability终态。

真实final/runtime completed10/exit0。source canonical12 SHA `e4153a3aacdd71ed4ab951ad9e67c77705728bec20f8a6b7afc2faa89aee6f8a`，与prep12全字段相等；12个唯一family、6yes/6no，且是已审计temporal120的逐ID/逐完整Sample精确子集。prep source report/data SHA与temporal来源一致，allocation ledger完整复用源120字节SHA50e536c7…；config经LoopConfig验证并与真实report相同，report附带init_adapter仅在严格相等时移除该执行元数据。init仍是CE0，CE-only。

**12是唯一样本数，120是10step×12accum的总计划draw；不是120个独立新样本。** 实际completed10；重复训练仍保留真实曝光来源。新adapter权重实际SHA `02a579d5094d61d53018046a0c542edcef19b88b929fb3bdd8714448366d54c7` 与final/runtime元数据及指定权重一致，纳入最终不可变绑定。

实际19.6524秒/exit0生成 `/root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-007`：

- 完整inventory isolated；52份更新报告、unknown0；18个旧indexed input继续全部核验。
- 追加1份canonical12历史TRAIN artifact；原完整源/组件账本/所有历史保留，不重新分池。
- 最新inventory SHA `80b8b7d9a00d8ae442bc4c520b57c4d72f163ccea00248a8b980c0f9797d7efb`。
- audit SHA `77b2bf70a453a0d90e26791733b4110fc6a2bf2e38cf0bf9617ed689761b0bc8`。
- discovery SHA `3a32cbf288be4efdd6b70c2792962e694cd74fa7a4959cb375b3ff77ebfc5f30`。
- helper实际/提交源码SHA `bda29c7e5f69f9f6fcc66ea6b79c6d8b2ca23143363c0b75b2b3f4627e817373` 一致；输出文件600/目录700。

实际命令（已执行，007禁止覆写）：

```sh
PYTHONPATH=/root/autodl-tmp/s4-claim-pair-code-001 /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python /root/autodl-tmp/s4-exposure-learnability-code-001/reconcile.py --root /root/autodl-tmp/s1-text-lora-001 --legacy-root /root/autodl-tmp/lite-omnijev --output-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-007 --learnability
```

37项CPU测试通过（新增7项：精确子集/6:6、changed supervision、live/错误sample_count/accum/objective/短completed拒绝、全部旧路径保留），ruff check/format和mypy通过。完整真实绑定报告 `reports/s4-pilot-exposure-reconciliation-007.json`；仅aggregate及文件metadata，不导出题目、媒体或封存载荷。成本null；不代表learnability实验泛化收益、S4质量或未知未保存外部历史已完备。
