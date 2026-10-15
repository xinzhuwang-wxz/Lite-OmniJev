# TRAIN受控claim变体准备 001

2026-10-16。实际远端CPU构建退出0；未运行训练/GPU、未读dev/calibration/holdout样本，未修改原数据或准入。状态 `new_prepared_not_admitted`，完整inventory隔离待主会话接入 `s4-pilot-exposure-003` 后验。

实际产物 `/root/autodl-tmp/s1-text-lora-001/artifacts/s4-crossmodal-pairs-001`：600行、200个family，每family三变体；both_true200 / image_false_audio_true200 / image_true_audio_false200。图像/音频组件各200个唯一，每个复用3次。所有ledger pool均TRAIN，产物目录700/文件600。输出与输入SHA、源manifest inventory/source bindings及aggregate审计见 `s4-crossmodal-pairs-build-001.json`。原始provenance绑定SHA `01ee26024fa365a51ad23b95a11d0349b5f906bfd30042509367cc3bfaac68c4`。

完整组件来自原provenance指向、SHA绑定且登记为TRAIN的两个真实source文件，而非从ledger猜正文。入口复用Sample、canonical_sample_sha、compose、allocation_rows和原builder的write_rows。生成前对原train/components/allocation做SHA核验和一对一覆盖，再用完整组件重新compose原rule，要求原Sample全字段（URI按原builder归一）与component ledger完全一致。原gold不改；生成后新id/data_version，group记录原item、三variant、组件gold及完整原组件canonical SHA。新allocation保留当前TRAIN别名全字段hash。

共享audio的both_true/image_false对，以及共享image的both_true/audio_false对现在各200组。每组完整instruction有变化；只有未变组件及其claim保持相同，不能称为完整语义相同。三个变体相关，不能把600当独立600个family；任何未来对比loss应按group处理，保持全部同TRAIN，不跨池。原始人工gold未重新标注，本轮只验证既有gold/生成/ledger绑定。没有训练效果或创新性结论。

源码独立部署 `/root/autodl-tmp/s4-crossmodal-pairs-code-001/build.py`，SHA `984033bedd1faccea795641ca5e91e3c05089802e84809a5dee9096b613076e8`；核心导入固定 `PYTHONPATH=/root/autodl-tmp/s4-reference-code-fix-001`。只部署本helper，不改当前GPU作业源码。运行命令：

```sh
PYTHONPATH=/root/autodl-tmp/s4-reference-code-fix-001 /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python /root/autodl-tmp/s4-crossmodal-pairs-code-001/build.py --original-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s4-crossmodal-train-001 --provenance-sha256 01ee26024fa365a51ad23b95a11d0349b5f906bfd30042509367cc3bfaac68c4 --output-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s4-crossmodal-pairs-001 --report /root/autodl-tmp/s1-text-lora-001/reports/s4-crossmodal-pairs-build-001.json
```

以上已执行，不可原路径重复运行；入口拒绝覆盖。前置验证失败不创建output；写出阶段失败保留partial并在小失败receipt标注，不清理来源。成本null；未单独instrument构建耗时（非训练计时证据）。

验证：`pytest tests/test_s4_crossmodal_pairs.py tests/test_s4_crossmodal_train.py -q` 29 passed/1.03s；ruff check与format check通过；mypy新入口通过。针对性测试覆盖三个instruction/gold/共享组件差异、原件不变、私有权限、覆盖拒绝、错provenance SHA、源内容变动、ledger真值/非TRAIN和allocation mismatch拒绝、失败receipt不覆写。未验证完整全库存隔离或训练目标实现，主会话决定下一步。
