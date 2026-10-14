# 首批文本开发集：CLINC500与训练排除

Owner: main-session-gpu-transition
Parent: s0:12; execution s0:13
State: frozen CPU slice; not complete text product coverage

复用s0-data:01已准入的CLINC plus/train，唯一输入固定为admitted/clinc-oos-plus-train-v1.jsonl，SHA256 5aace71f0c9af2b30630e2b2b379881e1ebf946221c59355e733fe496e5971d8，共15250行；其原始parquet、revision、许可与语义证据在reports/clinc-admission-corrected-001.json。只读该公开train派生集，禁止打开validation/test/项目holdout。不改原始已准入文件。

首批text cohort取500道原始Choice任务，保留原句/候选顺序/target与family_id。此来源只覆盖intent Choice，不能宣称Score/Noul或全部文本分布已验收；完整S0基线必须旁报primitive缺口，后续补充文本任务须独立版本和同条件重跑，不择优更换dev。

按family分配。所有family中先验证同family无冲突标签/输入，一family的所有样本只进一个池。确定性key为SHA256 UTF8 `s0-text-dev-v1:42:<family_id>`，按key升序、family_id作平局，选择前500个单样本family；若存在多样本family，明确拒绝该固定输入版本而不是裁family或补变体。不得以随机行切分忽略family。保留原始intent覆盖统计，500不能假称500类。

输出dev500、remaining train14750和allocation ledger。每个dev样本只改meta.pool/data_version，其他内容必须完全相同。训练输出必须排除全部dev family；输出报告列旧全量训练文件的SHA与superseded_for_training=true，旧准入文件保留为原始输入，不再允许下游以旧15250行版本训练。新剩余train不凭此工具自动training_admitted：未来训练入口须校验allocation ledger和精确输出hash。

ledger按DevelopmentRecord v2：item_id/modality=text/family_hash/source_hash/source_dataset/source_revision/license/label_origin=hard_gold；无label/题目/媒体。family_hash是family_id的规范SHA（输入已经64hex则直接用），source_hash是该原始Sample规范JSON的SHA，不得用全parquet同一hash冒充逐item绑定。另保存私有于公开dev的allocation ledger，记录原始/派生样本hash、family、pool及source revision，训练可验证。

入口必须拒绝源SHA/行数/来源/revision/许可/pool不符、未知/冲突target、重复id、同family冲突和任何已有输出；报告包含命令、种子42、源/代码/配置/输出hash、环境、耗时、退出码、成本null、gpu_used=false、holdout_accessed=false、quality_gate=not_run。CLI参数可支持测试源hash/count，但生产默认固定该协议，不能把测试覆盖当正式500通过。

验证：原始全部字段除pool/data_version逐item相等；两次输出字节SHA相同；500/14750精确且family交集0；训练排除ledger完整；source tamper、重复/冲突、不足500及输出覆盖拒绝。用公开真实文件在Linux CPU执行，不加载模型，不算训练/准确率证据。
