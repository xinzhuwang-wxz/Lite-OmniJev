# S4 连续固定池首段曝光核对 014

真实CPU核对完成：exit0，26.878秒，完整inventory审计isolated。
发现67份已知update-bearing报告，未知输入0；完整013历史与当前固定池均保留。

- 首段原始runtime paused/exit0、累计10步；invocation0→10、实际完成draw区间[0,120)，累计120。
- 固定1200 draw仅是全局池容量/100步上限，没有把未曝光1080 draw计为已曝光。
- current TRAIN仅一个固定池条目，旧TRAIN保留history；本次追加1项。
- 原初始化仍f2db，非别branch权重。config100/sample1200固定，首段resume_from=null。
- step-0010 checkpoint元数据、model/optimizer/RNG三tensor SHA、全局config/model/assets/data和seed42完整shuffle顺序均实际核验。
- named final与不可变raw runtime snapshot通过；只允许CLI TRAIN ledger汇总差异，原始snapshot保留原字节。
- mutable train/run.json当次核验但未写永久all_bound_input_sha256；后继resume不会因此破坏历史证据。
- 原失败launch001和修复launch-recovered-002均原样保留。无GPU训练、无重跑、无私有样本内容查看；GPU同期只有root的只读DEV50作业。

recovered launcher SHA：83240d5ddc515f7f2bb6fb33c46a3d4f7128399b116198908668533f200fb124。
014 report SHA：2e3d89bbd4be8585057a5e197e3caadeaa5e4159f6cca51700bb2500fc12ac2c。
完整报告副本保存为本目录同名JSON；原始工件位于
/root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-014。

checkpoint metadata SHA：96c3942a4c74e85185425c5edddf0e289f692863ac0729298eb0ebbb6456d01a。
checkpoint config SHA：210dedebc367134f727ef4924aa6bbf5c764353611060550b8e12643ddce79be。
checkpoint order SHA：ea7c7cdcd8a3802a18beabc52e073b7cda778c8cb3ce5ee2159910c00d3353e4。
三tensor SHA详见JSON中的continuous_invocations[0].checkpoint.tensor_sha256。

执行：静态core /root/autodl-tmp/s4-continuous-code-235ad68 为cwd/PYTHONPATH，
Python /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python；
helper /root/autodl-tmp/s4-exposure014-code-f0c5ea1/reconcile-pilot-exposure.py，
参数 --continuous-main-pool --continuous-launch-report 指向recovered-002，
root为s1-text-lora-001，legacy-root为/root/autodl-tmp/lite-omnijev，output为exposure-014。
日志 /root/autodl-tmp/s4-pilot-exposure-014.log，实际退出码0；未修改helper。

本证据只证明已知保存更新输入的曝光与隔离，不是全外部历史证明、质量晋级或后续90步授权。
