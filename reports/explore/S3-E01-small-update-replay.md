# S3-E01：小更新与均衡旧模态回放

状态：qualified-for-full-evaluation；S2正式通过后已启动。

证据：原30步、lr1e-4、500均衡样本probe使audio dev50从70%降到52%、video从62%降到58%、image/screenshot各下降2pp，因此没有扩成主训练。默认全池有text10750、video8299、audio326，直接前1000随机样本会使audio曝光很少；先用均衡小批量判断训练方向，不重复原高学习率配置。

新候选：固定seed42，每cohort取200训练行，共1000；从通过S2的v2 adapter初始化，100步×10累积、lr2e-5、50步保存检查点，lazy缓存128MiB，图像1120/视频16帧（保留S2输入条件），CE默认。五份ledger验证，不读封存/校准样本。训练上限20分钟H20；这是学习率/曝光共同调整，不把收益归因于单一因素。

先评video固定dev150，要求达到教师门对应的至少95/150且比同样本v2增加至少3题，才继续audio与旧模态小样本筛查；video基线从已完成500题sidecar提取，避免重新跑。旧模态50题筛查最多新增1错只用来选候选，不能代替全500题≤1pp门。audio筛查至少达到原教师比率门；通过才扩大完整五cohort双遍+8/16帧对照。任何探针都不能直接关闭S3。

准备入口.scratch/s3/prepare-stable-probe.py；远端artifacts/s3-stable-probe-001/{train.jsonl,config.json,report.json}；GPU入口/root/autodl-tmp/s3-stable-probe-train.sh硬性读取S2通过报告，训练已在S2通过后启动。

训练实测：100步/843.25秒，峰值显存30,144,239,616 bytes，lazy全部1000样本访问，cache峰值133,553,889 bytes低于128MiB；adapter SHA1bebae31309f799395d01499fb9deec5e44622599d0c082473ea8b5a014e5755。视频150筛查仍运行，不能据训练完成宣称质量通过。

小筛查完成：video99/150、audio38/50、image40/50、screenshot50/50、text49/50，全部达到事先规则。已启动正式dev500双遍及video8对照，优先验证旧image回退；全部门齐全后才判断S3。
