# Claim-pair pilot CPU准备 001

2026-10-16，实际CPU执行40.9357秒，exit0；不训练、不GPU。远端产物 `/root/autodl-tmp/s1-text-lora-001/artifacts/s4-claim-pair-pilot-001`，小报告见同名preparation JSON。状态明确 `prepared_pending_pair_hashes`，未声称stage gate通过。

600行变体及component ledger先追加到exposure-003派生 `inventory-expanded.json`，未改旧库存/来源。新版本映射crossmodal；原600 allocation逐Sample canonical hash匹配；新ID不覆盖历史TRAIN source。`audit_manifest` 首次真实isolated才继续准备pilot。随后原current中与候选重叠的TRAIN artifact在另一派生manifest标historical，追加120current样本；第二次真实audit也isolated。自动audit仅输出allowlisted aggregate，未显示封存内容。

120draw实际分布：40cross、40video、audio/image/screenshot/text各10。20唯一group按固定hash排序选前20，交替两种negative rule，各10。每step两个pair在offset0/1及2/3；offset4..7四video；8..11依次audio/image/screenshot/text。其他样本只取coverage prep002既有TRAIN；以(id,data_version)和完整canonical SHA严格映射库存cohort，并要求source allocation derived_sample_sha256全字段一致。inverse shuffle让seed42真实训练顺序保持上述布局；独立只读复核120条sampleSHA、实际shuffle、每step offset、20唯一group、pair draw绑定和所有outputSHA均通过。

固定初始化coverage adapter `0deef8cfd3d94f4dd3b42cfdbe995c825c388230721df7e6716e5963cb9dc818`，实际本地权重验证通过。data SHA `72c9bbf0b3cd8b931feeb4ce3b63128634d714dc1caadeb2117a27aa583e81fe`；ledger SHA `c2376c1b83da4bfe45308e827690f95d71422232d925e4057da083f378c3bee8`。全部产物文件600，目录700。

`config-ce0.json` 为可解析公共LoopConfig；相对coverage仅seed42/steps10/sample120/accum12、midway/checkpoint5、候选expected_versions及追加pair媒体根发生显式记录的变化。LR2e-5、bf16、rank16/alpha32、image1120/video16及其他超参数不变，reference KL关闭。原lazy preparation保留。`pair-override-recipe.json` 只是weight.1、待填四字段列表与canonical data/ledger/seed/steps/accum绑定，不是正式pair LoopConfig或prepared sidecar。`draw-recipe.json` 保留draw/index/sampleSHA/group/expected components。未造preparedSHA；两arm各240秒预算由root冻结规格/launcher执行，本入口不启动作业。

下一步由root集成trainer helper后，基于这份train/ledger/CEconfig的真实prepared examples调用 `claim_pair_row`，完成严格header与20行sidecar并绑定SHA，再生成pair.1正式配置及同数据CE0对照配置。不可将recipe直接当sidecar。

独立部署运行入口 `/root/autodl-tmp/prepare-claim-pair-pilot-001.py`，运行SHA `e69d76cbb94a6f695b26f42d1d6f2c3d2a154dcf4c3dc1064402121f8b75be98`；导入core固定 `s4-reference-code-fix-001`。提交版本只后补类型注解及对应非空assert（已由require保证），运行源码未覆写。报告中运行源码SHA保持真实值。源码已有目标目录/报告拒绝覆盖；失败仅输出allowlisted原因/类型，已有partial保留。

```sh
PYTHONPATH=/root/autodl-tmp/s4-reference-code-fix-001 /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python /root/autodl-tmp/prepare-claim-pair-pilot-001.py --root /root/autodl-tmp/s1-text-lora-001 --output-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s4-claim-pair-pilot-001 --pair-report-sha256 8eb7ce290ab05f98332df601fa83151f822432eab24f7f7e5b7504dda57391be --exposure-report-sha256 f4d851c683bd9ec377c0893a093b60ff84dcf43d69e1049684e07dd80e4f3dd8 --coverage-report-sha256 4da8a49c512a2725368eb159e444107130487ec69d7d74970df6a479436ef882
```

命令已执行，不重复覆写001。验证：2项公共排序/严格ledger测试通过，涵盖错长度、pool/version/hash及冲突拒绝；ruff check/format与mypy通过。无训练/效果证据。成本null。
