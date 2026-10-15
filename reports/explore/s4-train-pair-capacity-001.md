# TRAIN对比对容量审计 001

2026-10-16，只读现有TRAIN；不新建样本、不访问dev/calibration/holdout、不读媒体、不GPU、不实现训练。实际CPU退出0；结果见同名JSON（SHA256 `1c74f228deae0eb73f3f271a7d21dea032c19f4b775922592903e0cd8e68c28a`）。

**当前严格容量为0，不能直接从已有200条组成所要求对比对。** both_true100、image_false_audio_true50、image_true_audio_false50；图像/音频各200个唯一item、family、source_hash、media_sha，每个组件最大复用1次。共享audio且相同proposal/option-text hash的正负对0，共享image同条件0；相同完整instruction对均0。该结论不按相似词句替代组件身份，也不把不同图像/音频的任意yes/no样本当受控对。

`scripts/s4_crossmodal_train.py:136–166` 先对每模态按family和media去重选200，再一对一zip，每对仅compose一种rule；因此三个rule集合的组件互不重用。`scripts/crossmodal_gold_pools.py:101–179` 由两组件proposal构造AND问题，both_true为yes，其余为no。三个输入SHA已逐文件核验；全部200项ID一对一覆盖、TRAIN pool、Sample schema、allocation canonical全字段hash、family、媒体hash、target/source与rule/ledger一致。原始组件gold来自既有准入构建证据，本轮没有重新标注或声称人工真值审核。

以后若另行批准构造对：共享未变组件和其claim只能保证半边语义不变；被改变claim必然改变完整instruction。必须明确是控制一个组件的结构化对比监督，不是同一完整语义下标签冲突。应基于同一组件二元组的合法proposal反事实并保持TRAIN家族归属；这会创建新样本，本轮未做。复用组件会让pair loss观测相关；计数应按独立component/family分组而非把对数当独立样本，不得跨池。当前200没有重复组件，因此现有数据内没有多次复用风险，但也没有该约束需要的配对容量。不据此推广训练、不称RL或创新。

审计脚本 SHA256 `21e0c32247aa401c811f06771c34dd1e63a81b3dce51a64b7d0d72fefce1fdfd`。复用 `Sample` 与 `canonical_sample_sha`，远端PYTHONPATH固定已验证core snapshot `s4-reference-code-fix-001`；本地ruff check/format check通过。失败验证：非TRAIN ledger、gold错误、allocation hash错、缺ledger、重复ledger均拒绝；另实跑错误train SHA，退出1且stdout为空。所有失败仅在内存改审计metadata或错误CLI参数，未写输入。

可复执行只读命令（本地worktree执行）：

```sh
ssh -o BatchMode=yes -o ControlPath=/tmp/omnijev-ssh-root-final -p48926 root@region-42.seetacloud.com 'PYTHONPATH=/root/autodl-tmp/s4-reference-code-fix-001 /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python - --root /root/autodl-tmp/s1-text-lora-001/artifacts/s4-crossmodal-train-001 --train-sha256 d3b0be2adeadbd12cbfc01daa5fa6ef74fb687b8fab6148bc750d695bbab6977 --components-sha256 e87043f345ae2dd515d04d92af6444a0d470384309a45a3bbf8a410e4fdaab46 --allocation-sha256 2fc9d8aa3de539502b9a138fb03eedbc39af70423b970d2ebad53ff3ed76be4c --check-failures' < .scratch/s4/audit-train-pair-capacity.py
```
