# S0 公开输入 CPU 准备

`train.multimodal.prepare_case` 是共享入口。它复用 `render_payload` 和
`check_answer_tokens`，用官方本地 `AutoProcessor` 关闭 thinking、不截断输入，构建
CPU 张量。`PreparedDecision` 的 `messages` 只含官方模板占位符；`prompt`、`inputs`、
`answer_token_ids` 仅留在内存。仅 `summary` 写报告，不落原文、token ID 或媒体内容。
占位符 messages **不是可直接提交的 vLLM Chat 请求**。vLLM processor/transport、
视频 JPEG 重解码像素及时间戳一致性仍待下游验收；本入口没有 forward、模型质量或概率结论。

公开 fixture：`tests/fixtures/s0-prepare-public-001/manifest.json`，每类20题，涵盖
text/image/screenshot/audio/video、choice/score/noul，所有合法答案集都含 U。
这些是程序自写问题，无真值、无教师标签；100条输入来自4个来源族（CLEVRER 视频与
其首帧同族、GUI train 截图、Clotho CC0 音频、程序文本），每类20个问题变体；20个视频问题分为10×8帧和10×16帧。
不代表100独立样本、准确率或泛化覆盖。来源与派生方法在 `provenance.json`。
原始媒体不进 Git，受控远端根为
`/root/autodl-tmp/lite-omnijev/artifacts/s0-prepare-public-001`。

源 split 与项目 pool 独立：此 fixture 全部项目 dev。未来训练构建器必须消费
`excluded-training-families.json`，排除对应源 family 及媒体 hash；不得因原来源是 train
再次入训练池。开发入口拒绝 holdout 和跨池同族。

```sh
uv run python -m scripts.s0_prepare \
  --manifest tests/fixtures/s0-prepare-public-001/manifest.json \
  --asset-root /path/to/verified/public/media \
  --model-dir /path/to/pinned/gemma4/local/files \
  --output reports/s0-prepare.json
```

需已有 training CPU 依赖、Pillow/numpy、ffmpeg/ffprobe；不新增依赖，不自动下载。
必须使用固定模型 revision 目录；加载设置 `local_files_only=True` 和
`trust_remote_code=False`，不加载模型权重。完整运行要求100题、五类各20题、三primitive。
`--allow-partial` 仅作显式调试，成功也写 partial，不写 complete_100_cases。

无卡远端运行整个 Python 进程时使用 `flock /tmp/omnijev-cpu-heavy.lock`、
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false`。
不要与高 RSS 下载/安装并行挤占2GiB内存。报告先写 pending，实例化前与每题前写
checkpoint；外部 SIGKILL 后 checkpoint 保留，但需监督者另记实际退出码，不能当完成报告。
失败记录：`reports/s0-prepare-resource-failure-001.json`。原137失败证据不可覆盖。

输入只接受显式根下的相对常规文件，拒绝远程 URI、逃逸、软链接、空文件、坏 hash；
单文件默认64MiB，图片40MP，音频30秒、视频60秒。视频仅8/16帧，由manifest固定，
本CPU probe按源帧索引均匀取样（含首尾）与最长边640像素解码；
显式传入源fps、total_num_frames、frames_indices、duration，关闭processor二次采样，
不依赖默认fps24；不声称复现任何benchmark视频协议。
每题摘要保存实际配置、来源、媒体/payload/prompt/答案映射 hash，以及每个张量
shape/dtype/hash。generation boundary 检查合法答案单token、非特殊/未知、可逆与
prompt续接；同时校验媒体展开后的最后8个token仍匹配模板生成边界。

两遍运行只比较 `cases` 的摘要和 manifest/model/source hash，忽略运行耗时。
CPU实际成功与两遍逐题一致前保持 pending；即使通过，也不解锁 GPU 概率门槛。

本机共享只读target（唯一安装者为此票所有者）：
`/Users/bamboo/.codex/worktrees/s0-multimodal-cpu/Lite-OmniJev/artifacts/mac-media-libs`。
固定Pillow11.3.0/TorchVision0.29.0 Mac arm64 wheels以`uv pip install --target ... --no-deps`
安装；wheels、PyPI SHA与上游许可metadata见`reports/s0-prepare-mac-media-wheels.json`。
以`PYTHONPATH=<target> /Users/bamboo/Githubs/Lite-OmniJev/.venv/bin/python ...`运行，
target优先，root既有site-packages随后；root Torch2.14.1/numpy2.4.6未改。
这是复用现有环境的macOS CPU验证，不是fresh离线安装、Linux/GPU兼容证明。
真实结果与两遍比较见`reports/s0-prepare-acceptance-001.json`；Linux137独立保留。

`tests/fixtures/s0-prepare-public-001/build_public_fixture.py`仅用于远端重建小媒体与固定
问题manifest，输入是已审计源资产；运行时也必须持有重CPU锁。它不读取题标签或
内部封存集，输出媒体留在artifacts；文本source_sha256绑定此生成器源码。
