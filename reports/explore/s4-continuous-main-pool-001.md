# S4 固定连续训练池 CPU 准备 001

结果：`data_ready_only`，parent=null，training_authorized=false，实际训练曝光0。
远端 artifacts/s4-continuous-main-pool-001 真实CPU构建27.018秒，exit0；完整inventory一次审计isolated。
1200是固定池容量/100 global optimizer steps上限，不是已完成曝光，也不是100步训练授权。
下一次若root选择parent并授权，仍只先10步、240秒cap，DEV复核后再决定后继；完整checkpoint/exact-resume另由trainer实现。

- 固定seed42、accum12、CE、官方16frames、image1120、rank16/alpha32；其余原config保留，无新loss。
- 实际seed shuffle后的每step：6video+2cross+audio/image/screenshot/text各1；video每step3yes/3no。
- video600不同family：原temporal词规则500（250yes/250no）、motion50（25/25）、existence50（25/25）。固定namespace hash选择，不参考DEV错误。
- 只排除已核f2db lineage408 family（CE0 350 union首coverage60）。只称对该选定lineage新，其他失败branch可能见过；不是全历史未见。
- old四模态各100个不同family，来源为已审计S3联合池原TRAIN清单。
- cross复用首coverage里明确已审计20原题/10family，显式10轮aliasdraw共200。未声称200独立题；未扩组件组合、未造新gold。
- 总计1020源题，1200独立namespace draw ID。每条保留原family/source/data_version/问题/target/媒体；canonical TRAIN ledger绑定原题与draw SHA，cross alias组件ledger保留源image/audio组件。
- 915个唯一媒体SHA真实核验；复用训练器既有多根SHA匹配解析器，无媒体下载。
- 013报告SHA构建前后均5872a667ef0935436430e22f8c041c85f4e4f2c0c4ffc1e0076e4a9e6525b15e；前后无训练更新进程。仅内部inventory审计访问私有储备，执行者未读取cal/holdout内容。

## 证据

远端报告SHA bdf25dbe6955b983d263d13d3436f930481dc908241ef12b56526545156b1cc9；同内容保存为本目录JSON。

| 工件 | SHA256 |
|---|---|
| train.jsonl | aeee70458ca202e68f80f9d9b7dc617929a4bc1d4c2eb1e81113a25a207441a0 |
| allocation-ledger.jsonl | bc8fc32d4dba76534b9bd144f5d228b397db2e3511514f3722b945209be63d7d |
| config-ce.json | d8e9e5b0fc8427b32f39165c58486bfd93123fea1a584b5c0f8508efcb389414 |
| inventory-audit.json | 10cfda59e5329403e36d2882e5bd968dca7e5774503de2665448b61886a2330b |
| draw-recipe.json | f09c13c9d68ed548a0517f51a1550bff76ab7e6b613f2890be04e6d14914b779 |

CPU执行使用静态core /root/autodl-tmp/s4-claim-pair-code-001，builder/helper位于
/root/autodl-tmp/s4-continuous-main-pool-code-001。
Python /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python。
实际命令：在静态core目录设置PYTHONPATH为该目录，执行builder prepare-continuous-main-pool.py（默认root/output）。
成功日志 /root/autodl-tmp/s4-continuous-main-pool-002.log，exit0。
首尝试日志 /root/autodl-tmp/s4-continuous-main-pool-001.log，exit1：严格单路径media_path遇到历史cross目录同名文件，尚未创建输出；改用已有verified_media_file按SHA逐根匹配后通过，原日志保留。

本地 `uv run pytest tests/test_continuous_main_pool.py tests/test_temporal_curriculum_preparation.py -q`：8 passed。
`uv run ruff check` 与 `ruff format --check` 两个新Python文件：exit0。
真实独立CPU重放：1200 canonical alias/ledger匹配，100实际optimizer block配额与3yes3no检查全部通过，exit0。
来源、版本、许可、SHA及准入证据完整链保留在JSON，不复制原始题或媒体进Git。

未训练、未选父、未运行DEV/gate；无模型质量改善声明。训练时仍须绑定本池/全局config身份并使用独立invocation_steps实现有界exact-resume。
