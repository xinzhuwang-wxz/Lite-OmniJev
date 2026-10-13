# 教师音频的项目预处理入口

`train.teacher_media.prepared_teacher_audio` 修复固定教师音频分支“先创建空WAV、FFmpeg未允许覆盖”造成的空输出问题。官方 `akhilaaa3/Jev-Omni@5addda86ddee081a68fb067477ea100c221b8917` 原件不变，`jev_omni.py` SHA256前后均为 `11d761b0b6cefc8aac29757f43af4b2b02af8f9b6c6dad19834c0b14538180f0`。此入口只转换媒体，不加载模型、改结构或改变模板/概率读出。

## 调用与生命周期

要求本地可执行FFmpeg，Python3.11；无新Python依赖。已实跑的公共用法：

```python
from pathlib import Path
import wave

from train.teacher_media import prepared_teacher_audio

with prepared_teacher_audio(Path("input.wav"), timeout_seconds=30) as audio_path:
    with wave.open(str(audio_path), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 16000
    # 在此作用域内消费audio_path，退出后文件自动清理。
```

`temp_parent` 可指定已有临时父目录，适配器仍创建自己的私有随机子目录；调用者没有可指定覆盖目标的参数。只对该子目录内自有预建空 `audio.wav` 使用 `-y`；源文件与父目录内其他文件不变。`ffmpeg_binary` 可指定已审计的可执行文件路径，使用参数列表，无shell。

保留参考转换意图：首≤30秒，FFmpeg默认通道混音/重采样，mono/16kHz。明确PCM16 WAV（实测参考命令的默认WAV输出格式），输入/输出codec线程各1，非交互、遇解码错误拒绝。默认超时30秒，允许有限的(0,180]秒；转换失败、超时、输出无效或调用者异常均清理自有目录，未创建或清理调用者文件。

进程exit0不作为充分证据。返回路径前先拒绝symlink/特殊文件、空文件和超过2MiB的输出，再完整读取PCM WAV，检查mono、16kHz、16bit、未压缩、1…480,000帧及实际字节数与声明帧数相符。转换/输出错误抛 `TeacherAudioError`，不静默跳过样本或改变分母。

## 教师接入边界

后续项目教师音频入口应在此context内部，将返回路径作为 `{"type": "audio", "audio": str(audio_path)}` 交给固定教师processor的既有chat-template路径，完成张量构建后才离开context；保持原文本prompt、选项顺序、thinking关闭与head/概率读出不变。

**当前未完成教师forward接入。** 上游 `JevOmni.predict(media=..., modality="audio")` 仍会执行其原有转换分支；将这里的路径直接传回该方法会再次遇到原问题。需要项目持有的教师调用适配层消费此音频content，再独立验证模型forward及同条件输出；不能修改官方快照或假称本CPU入口已修复上游方法。精确MVBench14任务/2,786 IDs、MMAU版本和评测预处理决策仍按协议报告保留pending。

## 本次实测与复验

```bash
uv run pytest tests/test_teacher_media.py -q
uv run ruff check train/teacher_media.py tests/test_teacher_media.py
uv run ruff format --check train/teacher_media.py tests/test_teacher_media.py
uv run mypy train/teacher_media.py tests/test_teacher_media.py
```

`reports/teacher-audio-cpu.json` 记录命令、源码SHA、平台/FFmpeg、耗时和退出码。真实自有合成输入：0.1秒mono16kHz→1,600帧；32.25秒六通道48kHz→30秒mono16kHz、480,000帧。所有输出完整读取，源SHA不变，成功与失败路径的临时目录均清理。8项定向测试覆盖实际转换、exit0空/不可读输出、截断PCM/错误采样率、非零退出、超时和调用者异常；不使用真实题目或模型。

原命令的覆盖回归也实测保留：macOS FFmpeg8.0.1拒绝覆盖时exit0且0B，Linux FFmpeg4.4.2拒绝覆盖时exit1且0B；两个平台仅对自有目标加 `-y` 的合成候选均exit0，输出可读1,600帧WAV。Linux结果是先前排队的原命令/候选小探针终态，**不是新项目适配器的Linux实跑或教师forward验收**。新适配器本次实际运行在macOS；未使用GPU、模型、隐藏真值或付费操作。
