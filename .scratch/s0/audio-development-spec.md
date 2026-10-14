# Clotho首批音频开发500：原一致Noul金标与整音频排除

Owner: main-session-gpu-transition
Parent: s0:12; execution s0:14
State: frozen first source/primitive slice, not full audio distribution admission

只读已staged Zenodo6473207公开TRAIN：clotho_aqa_train.csv SHA237c692e21e015a3e9b4391edd716b1387d7eb2d068d7ecdb5a6d6ead7e7a992；clotho_aqa_metadata.csv SHA7505ba93c291d344b31e5c33e872b95d84255852cff93278bbdf549402acc1f5。根/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/clotho-aqa/6473207，媒体approved-audio-002；复用reports/clotho-aqa-license-plan.json和audio-preparation-002的两处固定名称alias，不模糊匹配或新下载。QA为MIT，音频仅CC0-1.0/CC-BY-3.0允许源，保留原URL/作者attribution；不洗其他来源许可。

已有有界TRAIN审计：许可过滤后17514原答题行/973音频family/5837(file,question)，5836组3行、1组6行。严格恰好3条原answer一致且为yes/no才有2267候选QA/947音频family；confidence是自报确信文字，不作概率或答案。6行组明确排除并报告，不能取前三行。一致性筛选只依原格式/标签和许可，不看模型预测。不声称匿名三行是固定两真人或κ验收。

family取原Freesound sound ID（从固定metadata original URL解析），同sound ID的所有文件别名/QA/派生音频归组；若多个不同文件共享ID须保持整family，不按文件随机切。构建前核实审计947按文件到sound ID是否一一对应，差异报告并拒绝把文件数作独立family数。按SHA256 `s0-audio-dev-v1:42:<family_id>` 排名前500family；每family按规范question文本SHA及文本排序取第一条合法原Noul题，目标原answer不变。不因数量不足补变体或借其他split。

每个选择WAV逐文件SHA/头部/真实完整有界decode验证：本项目支持<=30秒，必须按实际时长筛选前冻结可用family候选再哈希取500，不能先选后遇长音频静默换题。只读WAV，不写媒体副本；使用已有安全resolve_media/load_media或soundfile分块CPU decode，禁止启动模型/CUDA。样本媒体相对可信根，无symlink/绝对路径/逃逸，保存媒体sha/时长/sample rate/channels及具体许可。原问题作为Noul instruction，state只给必要音频任务说明，不含answer/投票/来源注释/程序。原vote内容只在公开provenance文件，以hash绑定，金标不入模型payload。

输出dev500 Sample JSONL、redacted DevelopmentRecord ledger、全部原TRAIN sound family allocation/exclusion ledger（dev所选sound的所有QA/别名都从未来训练排除）、逐选媒体/attribution manifest、报告。Meta pool=dev、modality=[audio]、source_dataset=Clotho-AQA、revision=zenodo-6473207、source_sha256逐QA原三行和媒体hash规范摘要、license明确QA MIT与媒体实际许可组合；target.source声明clotho_original_unanimous_three_answers，不能称新增真人。

固定种子42、输出拒绝覆盖/路径别名/symlink；SHA/许可/ID映射/缺媒体/解码失败均明确失败并留小型报告。两遍真实LinuxCPU输出字节相同；整family排除与样本真值/媒体绑定完整。报告command/code/config/source/outputhash、环境/耗时/退出码、cost=null、gpu_used=false、holdout_accessed=false、training_admitted=false、quality_gate=not_run、real_human_kappa_verified=false。只覆盖原一致Noul，不代音频Choice/Score/跨模态或完整S0音频质量。未来训练必须先应用exclusion ledger。
