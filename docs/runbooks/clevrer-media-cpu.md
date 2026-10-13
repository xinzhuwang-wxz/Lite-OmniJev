# CLEVRER公开媒体的代表性CPU解码

`reports/clevrer-media-cpu.json` 保存官方train/validation/test各首、中、尾三个视频的真实Linux CPU结果，共9条。官方来源、三个ZIP的既有本地SHA256/整包CRC与准确路径来自 `configs/clevrer-assets.json`，本次固定快照SHA为 `4a0e3b9e003146f3e6edcd6e8c6863b2f7da0ccb05d13b1030e39ff53709887c`。上游未发布SHA256；不能把先前计算的本地哈希描述成上游校验和。本次不重复重哈希约25GB归档，另记录选定成员的真实SHA/CRC及源ZIP前后size/mtime/inode不变。

## 范围与有界操作

仅以只读模式打开以下三个视频ZIP，不打开annotation、questions、隐藏答案或封存数据：

```text
/root/autodl-tmp/lite-omnijev/source-audit/clevrer-official/files/videos/train/video_train.zip
/root/autodl-tmp/lite-omnijev/source-audit/clevrer-official/files/videos/validation/video_validation.zip
/root/autodl-tmp/lite-omnijev/source-audit/clevrer-official/files/videos/test/video_test.zip
```

先检查全部ZIP成员元数据，拒绝绝对/上级/非规范路径、重复项、symlink/特殊文件和加密项；单条目≤128MiB、每包≤20,000成员、声明总展开≤32GiB。不按归档路径写磁盘，不用extractall。对每split的全部MP4路径按字典序排序，固定取0、count//2、count-1；只将当前选定成员流式复制到独占临时目录中的 `sample.mp4`，读到EOF使标准库验证该成员CRC，提取字节不得超过声明大小或128MiB。

FFprobe检查元数据，FFmpeg以 `-xerror`、codec/filter线程1、不设时间/帧数截断、映射首视频流完整解码到null。要求退出0、progress=end、实际解码帧数为正且与容器声明一致（如果有声明）；宽高为正，fps/duration有限且为正。每条完成即清理临时视频，异常也由context清理，最多同时提取1条。本次视频提取空间峰值1,807,123B，低于128MiB上限。

## 已实跑命令与结果

驱动源码完整嵌入报告的 `execution_receipt.driver_source`，远端自有目录保留config/driver/log快照和SHA。实跑：

```bash
cd /root/autodl-tmp/lite-omnijev/artifacts/clevrer-media-cpu-001
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
flock /tmp/omnijev-cpu-heavy.lock \
  /root/autodl-tmp/lite-omnijev/.venv/bin/python validate.py
```

| split | 列出的公开视频数 | 固定选定视频ID |
|---|---:|---|
| train | 10,000 | 00000、05000、09999 |
| validation | 5,000 | 10000、12500、14999 |
| test | 5,000 | 15000、17500、19999 |

9条全部完整视频流解码通过：均H264/yuv420p、480×320、25fps、5.12秒、128帧；合计1,152帧/46.08秒。选定成员CRC和SHA完成，全部FFmpeg/FFprobe退出0，无FFmpeg错误输出，临时提取目录已清理，源ZIP size/mtime/inode前后相同。Linux驱动5.025秒、退出0；外层wrapper PID28166也以 `process_exit_code=0` 结束，重型锁已释放。Python/FFmpeg版本、0.5核/2GiB限制、每样本属性、原始结果/日志/脚本SHA见报告。

这是20,000公开视频中的9个固定代表样本解码证据，不是全20,000视频解码、逐像素有限值检查、全部任务媒体引用覆盖或模型/分数验收。完整归档CRC沿用先前证据，不能与本次采样解码混淆。没有模型/GPU、训练准入、标签、门槛、官方原件或环境更改。
