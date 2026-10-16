# S4 video-text-matched 单段曝光核对 018

真实CPU exit0，30.442秒，完整inventory isolated；75份已知update-bearing报告，unknown0。
artifacts/s4-pilot-exposure-018首次核对即成功，未重复训练或GPU作业。

- 30组exact native text/program、相反官方gold、不同media与family的pairs，共60视频family；原60 replay canonical相等，共120 TRAIN样本/120实际draw。
- 每step6video+2cross+4old、视频3yes/3no；实际seed42 shuffle在offset0/2/4形成三组相邻pairs。
- 原官方source/question/program/sourceSHA、ledger、pair metadata、每成员canonical SHA强绑定；非媒体prefix与question全部字段相等，target保留各自原官方gold。
- 非媒体字段验证不是prepared tensor完全一致证明；pair只作metadata，无新pairloss或teacher真值。
- 独立10step纯CE、原f2db初始化、oldcore3fe0ab2；runtime/final/config/adapter/preparation绑定。
- 唯一current TRAIN为text-matched runtime data.jsonl。017retry所有all_bound_input_sha256逐项完全保留，continuous两份immutable snapshots原样保留。
- 历史曝光仍分别为continuous240、native120、specialist120，当前text-matched另记120；不累加不同branch到同一训练链，不把1200池容量算已曝光。
- 旧failed001/recovered002及017失败链保留；私有内容未向执行者展示，完整隔离只取现有aggregate audit汇总。

018 report SHA：b70aa762a2d66492b53d5a44a03c1877d9d3fcd2861e7c5ad43c77b84715e659。
017retry report SHA：5b3d3a26bfad2f35dc85e3f166a5d77f3c0996469fa6cbd0cbe962aa98b7e8b6。
prepared report SHA：3e764378147c06816a3f14fe71967f53d83fb50871237e094ceb19587419db93。
训练adapter SHA：9dea6c259e8b45df8b9466e87ebeb6e8f0d3a48822c7104640286a75ab5bc6b1。
完整SHA链见同名JSON副本。

实际执行helper /root/autodl-tmp/s4-exposure018-code-3733f2d/reconcile-pilot-exposure.py；
Python /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python，
cwd/PYTHONPATH /root/autodl-tmp/s4-claim-pair-code-001。
CLI --video-text-matched，root=/root/autodl-tmp/s1-text-lora-001，
legacy-root=/root/autodl-tmp/lite-omnijev，output=artifacts/s4-pilot-exposure-018。
日志 /root/autodl-tmp/s4-pilot-exposure-018.log；实际exit0。

复用3733f2d已通过的44项相关CPU测试与lint/format证据，helper/输入未改；本次额外验证真实完整inventory、017retry全部旧绑定全等、continuous snapshots全等、唯一current与分别记录draw计数，均通过。
本报告只证明已知保存更新输入的曝光与隔离，不是质量晋级或S4通过。
