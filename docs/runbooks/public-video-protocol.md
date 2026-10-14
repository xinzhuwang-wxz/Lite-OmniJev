# 公开九任务视频输入冻结（CPU）

用途仅为已批准的独立公开 baseline 输入准备，不是发布 53.10 分复现、教师 forward 或质量门槛通过证据。不读取项目封存内容、不输出公开题目/选项/答案、不加载模型；研究媒体不因此准入训练或商业发布。

固定 MVBench `230a2d4fac8900333c61754641c7a13e069ac9c6`，官方任务顺序筛出 Action Sequence、Action Prediction、Unexpected Action、Object Interaction、Object Shuffle、Action Localization、Action Count、State Change、Character Order，各保留原 200 行。零基 row_index + task JSON SHA 是身份；video 名称不是唯一题号。规范化引用 JSON 为 UTF-8、sort_keys、无空白，固定 SHA256 `4530b25b34e464958d26d3f4b01abd73a4ebe53aa287702c5206048524e43efb`。仅提取 video 及有界任务的原 start/end，不保存题目或答案。

官方公式依据 Ask-Anything `4dd8210b030374bb0c510a57dfbe916f70d3ef71` MVBench notebook 的 `MVBench_dataset.get_index`：`lo=max(0,round(start*fps))`、`hi=min(round(end*fps),N-1)`；无界题用官方 ±100000 秒边界。`segment=(hi-lo)/16`，第 i 帧为 `int(lo+segment/2+round(segment*i))`，i=0..15，round 使用 ties-to-even。重复索引保留为 16 个输入。STAR/STA 时间窗原值保留；STAR EDXBD/K47J5/ALXUC 原 end 越界夹到最后帧，不删除行或更改标签。全量窗口核对另发现27条实质end超过stream duration（Action Sequence26、Object Interaction1），另20条仅六位小数duration舍入差；全部按同一公式夹取，不删行。帧数和平均帧率取固定 Linux FFprobe 容器元数据；这份新项目协议采用固定 FFmpeg RGB24 解码，不能声称像素与作者未知版本的 Decord 相同。

`eval.public_video_protocol` 复用 `scripts.public_eval_media.checked_zip_members`，全部 ZIP 先拒绝穿越、重复路径、链接/特殊文件和展开超限。每次仅提取一个 ≤128MiB 媒体，ZIP CRC 随提取验证；12 个官方 Charades 补充媒体必须匹配既有 SHA。媒体原件和 ZIP 不修改。每媒体 ffprobe 后计算所有引用窗口的索引并合并索引，一次 FFmpeg 顺序解码到最后一个所需帧，单线程输出原尺寸 RGB24；逐帧读出完整 `width*height*3` 字节并 SHA256。不是全视频 EOF 或所有未采样帧验证；全部 1800 行的全部 16 个模型输入帧均须通过，缺行或失败不缩分母。

实际采用 **Mac canonical lossless PNG**：FFmpeg 8.0.1 单线程输出原尺寸 RGB24 和 PNG（mixed predictor、level6），每张 PNG 再实际解码回 RGB，全部 SHA 与首次原视频 RGB 一致后上传。远端再次验证每张 PNG 文件 SHA，再原子提升整个媒体目录为只读缓存。32 媒体/543 帧比较显示 Linux FFmpeg4.4.2 从原 MP4 解码的 RGB SHA 全部与 Mac 不同，因此不得用 Linux MP4 重解码替代 canonical PNG。最终只读 frame-manifest 记录原行、时间窗、索引、尺寸及16张PNG路径/RGB SHA/PNG SHA；每媒体 record 保存源 SHA/ZIP member/CRC/decoder/协议hash。

远端在自有 artifacts 工作目录运行，使用已有 media-cpu Python，不装依赖、不导入 torch：

```sh
cd /root/autodl-tmp/lite-omnijev/artifacts/public-video-protocol-freeze-001/code
flock /tmp/omnijev-cpu-heavy.lock /root/autodl-tmp/envs/media-cpu/bin/python -m eval.public_video_protocol \
  --project /root/autodl-tmp/lite-omnijev --config ../config.json \
  --output ../output --plan-only
# 全量采样解码去掉 --plan-only；同一代码/来源的同一 output 可以恢复。
```

预检实际1800行/1606媒体通过，27997唯一选中帧，raw RGB 84,647,947,392B；567.86秒/exit0。32媒体实测Mac含传输/PNG93.41秒、Linux原视频采样136.93秒，按来源尺寸加权的全量PNG估算约18.19GB，只是估算。实际进度/命令/退出码/耗时/最终缓存体积见 `reports/public-video-protocol-freeze-001.json`；running/plan-only不算全量通过。Mac全量由报告内保存的自有supervisor/receiver脚本执行，两并发CPU worker、单远端流式ZIP reader（1MiB块）；每worker source+PNG+TAR保守≤500MB，总临时≤1GB，处理成功/异常均清自己TemporaryDirectory。远端canonical ledger硬限29GB，另留1GB给incoming/staging，总派生≤30GB；150GB checkpoint reserve不动。远端完整视频解码用共用heavy lock；Mac CPU独立，已批准的增量成员传输锁外。

GPU 前只能读取冻结 canonical PNG，并核对文件 SHA 与解码后的 RGB SHA；随后把同一帧对象/同一只读缓存交给教师和学生。任何 SHA 不符立即拒绝 forward，不接受新像素或静默重新冻结。原视频仍保留；需要修复缓存时须按冻结Mac decoder/协议重建并匹配已冻结hash，不能用Linux MP4重解码假称等价。本片未实现模型 forward 集成；`decode_frames`/`verify_png_frames` 实际验收公共采样和无损往返，后续消费者须接入完整验证。各模型官方processor可以不同，原题/选项顺序及16个RGB内容/顺序必须相同。S0 repeat、S3/S4等gate由主会话独立验收。

恢复任务前核实自有supervisor PID/terminal终态，不能重启活作业。报告嵌入运行时的自有driver源码与SHA；从报告恢复到忽略artifacts目录后使用已有专用媒体Python与该worktree的PYTHONPATH运行supervisor。supervisor启动读取remote ledger，对已完成媒体重新核全部remote PNG SHA和record SHA后复用，不重复解码或上传。失败保留逐媒体receipt，仍要求1800/1606/27997全部覆盖；不按失败/分数删行。远端receipt与小型汇总在Git，PNG/逐行manifest/进度大文件只存忽略artifacts。最终manifest SHA按UTF-8 sort_keys/无空白/无尾换行计算，实际文件含尾换行的文件SHA另记。
