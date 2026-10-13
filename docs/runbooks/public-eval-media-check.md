# 公开评测媒体 CPU 检查

此入口只检查固定公开 MMAU mini 与 MVBench STAR 原料；不读取私有/隐藏/封存标签，不准入训练，不产生模型质量分数。原始许可与版本见 `configs/video-eval-assets.json` 和 `docs/research/video-eval-offline.md`。

本次全量检查实际在 macOS arm64 上完成。专用 Python 3.11.14 环境位于忽略目录 `artifacts/public-eval-media-env`，PyArrow 26.0.0、SoundFile 0.14.0、NumPy 2.4.6，总计约 151 MiB；依赖通过官方 PyPI 安装，没有改项目主环境。只复制获批的两个公开文件到本机（共 1,941,294,873 B），大小/SHA256 均与远端 canonical 副本匹配。脚本不触发 HF datasets 远端代码。

本次实跑的全量命令：

```bash
cd /tmp/lite-omnijev-video-eval
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  artifacts/public-eval-media-env/bin/python scripts/public_eval_media.py \
  --data-root artifacts/public-eval-media \
  --sources /tmp/video-sources \
  --config configs/video-eval-assets.json \
  --audio-python artifacts/public-eval-media-env/bin/python \
  --report reports/public-eval-media-validation-001.json
```

Linux 仅做同字节 32 份代表音频的 FFprobe/FFmpeg 完整流检查，不使用 Arrow 读大文件：

```bash
cd /root/autodl-tmp/lite-omnijev
flock /tmp/omnijev-cpu-heavy.lock env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /root/autodl-tmp/envs/data/bin/python artifacts/public-eval-linux-check.py
```

该 driver 和样本 manifest 的 SHA256 绑定在最终报告；原始 32 文件位于忽略的 `artifacts/public-eval-linux-samples/`。选样优先保留最长音频并覆盖 codec/采样率/通道/域，再按 stride 31 补足；每份≤16 MiB、总计≤128 MiB，实际 75,691,030 B。样本均由固定 parquet 原字节复制，没有转码。

首次执行先确认脚本/config 与提交版本一致；本地源码为 `scripts/public_eval_media.py`，远端复制到 artifacts 后运行。报告记录两者 SHA256、环境、实际耗时、成本/GPU 使用和退出码。退出码 0 只说明本入口定义的检查完成；不能替代阶段 0 gate、完整 20 任务覆盖或教师精确 14 任务复现。

MMAU 检查：

1. 核对固定 parquet 大小与 SHA256，以及固定公开 JSON SHA256。
2. PyArrow 按批读取 `context` 内嵌音频和公开 metadata；官方 HF 文件使用 `context`、`instruction`、`other_attributes`，不是假设的 `audio`/`id` 顶层列。`context.path` 为 null 时仍可使用 bytes。
3. 1,000 公共 ID 一一对应，无遗漏、重复或其他 split；音频源文件名 stem 必须等于 ID。问题文本完全匹配；HF choices/answer 必须精确等于公开原选项按原顺序加 `(A)`、`(B)` 等前缀，不按含糊字符串清洗猜答案。
4. 所有 1,000 份音频交给已安装 media-cpu 环境中的单个持久 SoundFile worker；每次读取 65,536 帧 float32，检查全部样本有限、完整读到 EOF、读取帧数等于 libsndfile 报告帧数，并获取采样率/通道/format/subtype。IPC 每条有 30 秒超时，不为每条启动解释器。只有 libsndfile 明确的未知格式/不支持编码错误允许 FFprobe/严格 `FFmpeg -xerror` 转成临时 float WAV 后再由 SoundFile 全量检查；最多 32 次 fallback、每条预计展开≤128 MiB、转换超时 180 秒。其他解码异常/NaN/帧数不一致直接失败。按独立 row group 读取、批大小 1、`use_threads=False`、逐条解码；每个 row group 后释放 Arrow 未使用内存，临时文件每批清理，不保存或打印原始问答/音频内容。
5. 报告只输出覆盖数、采样率/通道/codec/时长统计与 ID→音频内容哈希清单的整体摘要。原始真值不被教师预测覆盖。

STAR 检查：

1. 固定 ZIP 大小与 SHA256 通过后，先遍历全部 member，拒绝绝对/上级/Windows/非规范路径、NUL、重复路径、symlink/其他特殊文件及加密条目。
2. 限制最多 10,000 members、单条目 256 MiB、总展开 2 GiB；检查完成前不提取任何条目。全 archive 流式 CRC 检查通过后才允许选样。
3. 用已锁 SHA256 的全部 20 个公开 task JSON 计算每个 row reference 的精确路径是否存在，以及独立媒体引用数。STAR 包的缺失不代表其余 17 个来源不可下载，也不能伪称覆盖了教师 14 个任务。
4. 按名称排序、固定 stride 选最多 32 个视频，提取到新建临时目录中的数值文件名；不使用 `extractall`，不信任归档路径作磁盘目标。按声明大小检查后，FFprobe 检查流，FFmpeg 完整解码选中视频流到 null，逐条删除临时视频。
5. 全 ZIP 路径/CRC/任务覆盖检查与最多 32 个视频的解码是不同证据。报告明确 `all_video_members_decoded=false` 时，未选视频仅有 CRC/存在性证据。也未测真实视频裁剪、8/16 帧处理器或模型 forward。

所有后续重型 CPU 验证通过 `/tmp/omnijev-cpu-heavy.lock` 串行执行；已有暂停作业只按主会话明确协调恢复，不能锁外并行或按观察超时重启。实际容器 CPU quota 为 50,000/100,000（0.5 核），cgroup memory.max 为 2 GiB，不能按宿主机 free 输出估计可用内存；parquet 有 10 个 row groups，最大约 382 MB 未压缩。早期并行运行没有完整退出报告，未据此宣称解码通过；改为分组/单条读取及释放 Arrow 缓存，使用 detached shell 的最终 `process_exit_code` 留存进程退出证据。

首次 schema 探针因 HF 选项/答案带 `(A)` 前缀而严格比较失败；公开 ID 和问题已全量匹配。保留失败报告，并按官方固定格式修正比较后重跑，不改变公开答案或 benchmark 版本。

若执行失败，读取报告错误类型与退出码后定位；不重启正在下载的作业。MMAU 和 STAR 的原料哈希已验收，STA/FunQA/Perception 的正在下载文件不能交给此入口冒充完整 archive。教师的精确 14 任务／2,786 row IDs 仍是独立缺口。

SoundFile 检查的是 libsndfile 支持且接受的完整音频流；帧数比对使用其报告的 frames，不能证明原始录音语义或没有在上游被裁剪。fallback 检查完整转换后流，原压缩格式的帧数语义可能不同，报告单列次数；不把它描述为所有原格式都经过 SoundFile。实际全量报告给出 decoder 计数与有限值检查数。STAR 视频仍使用 FFprobe/FFmpeg 全流解码，未作每像素有限值检查。

## 实测结果与边界

最终 `reports/public-eval-media-validation-001.json`：本机全量检查 13.90 秒、退出 0；1,000 public IDs、问题与精确格式化选项/答案全部对应。全部 1,000 原音频读到 EOF，帧数相符且所有样本有限，共 446,617,655 帧；866 PCM16、73 FLOAT、15 PCM32、46 MP3，fallback 0。69 份为 6 通道；66 份大于 30 秒，最长 34.4785 秒。它们是合法公开源样本，教师文档的 30 秒 cap 与学生/教师如何截断、转单声道及重采样属于待冻结的评测协议，不当作坏样本或暗自排除。

STAR 原 ZIP 425 members / 423 videos，全路径安全和 CRC 通过，32 个视频完整 FFmpeg 解码。原包覆盖 Action Sequence **188/200**、Action Prediction 200/200、Object Interaction 200/200；其余来源任务不在此包。12 个缺失视频的官方选定来源补齐另留报告；不把它们解释为教师 2,786 题清单的排除。

Linux 32 份覆盖四类实际音频格式、五种采样率、1/2/6 通道（含三份 6 通道）及最长 34.4785 秒样本；源哈希全匹配，FFmpeg 4.4.2 完整解码通过、9.78 秒、退出 0。Linux 未检查全 1,000 份有限值/帧数，不能把本机全量结果冒称 Linux 全量结果。

早期远端 full job 记录退出 137；当时 cgroup oom/oom_kill 计数为 0，原因未证实，未宣称 OOM。按主会话授权仅停止自己的暂停进程并转本机验证，保留原始失败/替换记录；未重启数据下载或改变公开标签。
