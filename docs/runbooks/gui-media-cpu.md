# GUIOdyssey 下载后的无卡验收

固定来源为 `hflqf88888/GUIOdyssey@61632d0f3f4d51d7e9561ce4f84347dd06b2019d`。只验证既有 app_split/train 预算方案选出的 1,679 个完整 episode；25,809 张 PNG 共 19,499,790,993B。下载摘要和远端逐文件 manifest 的 SHA256 见 `reports/gui-odyssey-download-acceptance-001.json`。文件完整不代表训练/商业发布准入，也不代表模型效果。

下载入口 `scripts.gui_odyssey_media fetch` 将已有文件与 ZIP central directory 的尺寸/CRC 对照，损坏的普通文件重新下载，软链接拒绝。正常下载最终报告包含全部选中文件的 SHA256；只看到目录计数或进度日志不能关闭票据。

`scripts.validate_gui_media` 在无网络模式逐张重验尺寸/CRC/SHA并完整PNG解码，拒绝缺失、坏图、非PNG、软链接、重复episode和非train方案；不会删除或修复输入文件。复用已有独立media-cpu环境的Pillow，不修改训练/服务环境。它仅扫描方案中选中的文件，不遍历或解码test截图。

远端项目根的复验命令（报告路径须未存在）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/autodl-tmp/envs/media-cpu/bin/python -m scripts.validate_gui_media \
  --manifest source-audit/stage3-source-audit/gui-odyssey/central-directory-manifest.jsonl \
  --plan source-audit/stage3-source-audit/gui-odyssey/app-split-train-media-plan-under-19_5gb.json \
  --media-root artifacts/data/quarantine/gui-odyssey-source-audit/media/app_split_train_under19_5gb \
  --report reports/gui-odyssey-full-decode-cpu-002.json \
  --lock-path /tmp/omnijev-cpu-heavy.lock
```

重CPU锁按每张图片获取/释放，不包住整个下载或整个验收，其他串行CPU验收可以交错运行；图片逐张处理，避免把19.5GB媒体放进2GiB内存。报告包含代码/方案/源manifest哈希、Pillow/Python/平台、解码数、字节数、顺序逐文件收据摘要、耗时和退出码；异常保存失败报告，退出1。输出已有时拒绝覆盖，旧失败证据保留。

Linux独立快照 `artifacts/gui-full-decode-cpu-001/` 已实跑公共CLI测试，6 passed/0 skipped；覆盖完整解码、输入不变、拒绝覆盖既有报告及上述失败路径。本机无Pillow，解码相关两项明确skip，不能将本机测试视作媒体解码证据；Linux6项是实际Pillow验收。全25,809张实际结果须另看 `reports/gui-odyssey-full-decode-cpu-001.json` 与监督作业退出码，通过前保持pending。
