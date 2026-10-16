# 导出执行手册(2026-10-17 03:40-06:00 窗口;全部为已验证命令重放)

前置:FIT009 完成(status=fitted,18000/18000)。R=/root/autodl-tmp/s1-text-lora-001;E=$R/artifacts;PY=/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python

1. 核验 fit009:PY /root/autodl-tmp/s5-fit007-verify.py 不适用(fit007 专用)→ 改核 report 字段:status=fitted/exit0/artifacts=7/export_directory 存在
2. 清媒体快照(腾盘给 holdout):find /var/tmp/omnijev-calibration-jobs-001/<fit009-run-dir> -type d -name media -exec rm -rf {} +;df 确认 ≥15G 空闲
3. D3P 封存 holdout:PYTHONPATH=/root/autodl-tmp/s5-calibration-code-fa83acc PY /root/autodl-tmp/s5-holdout-prepare.py --calibration-job $E/s5-calibration-job-005/job.json --calibration-job-sha256 eee1c52d… --producer-revision fa83acce6f4b0345a2694ec883a81647a6279f8d --output $E/s5-holdout-job-003 --gates /root/autodl-tmp/s5-calibration-code-fa83acc/configs/gates.json --calibration-receipt $R/reports/s5-calibration-fit-009.json --development <四模态 D3P 留任报告,已上传版>;UID 65801/2;report s5-holdout-eval-004.json
4. registry manifest:PY /root/autodl-tmp/s1-text-lora-001/…build-s5-registry-manifest.py(已上传版)--fit-report s5-calibration-fit-009.json --calibration-version d3p-150-001 --registry-root <export-dir>
5. 网关 smoke:uvicorn serve.gateway.app(本地 bf16 bootstrap,registry 指向 export-dir)→ /docs + 一次决策请求(温度绑定生效=calibrated 标志)
6. 汇总:同 s5-sealed-aggregate 口径出 D3P 版 reports/s5-sealed-aggregate-summary-002.json;填模型卡 docs/release/model-card-draft.md 占位
7. 包:三档目录模板 + adapter e465→D3P 替换 + 本卡 + 数据卡 + ADR0023/0024 披露;缺项标 pending 不虚标
