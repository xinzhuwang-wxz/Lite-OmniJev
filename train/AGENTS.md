# 共享输入与教师边界
- 预提视频帧出现默认fps24警告 → 只有帧像素不足以确定视频时间戳 → 绑定真实fps、total_num_frames、frames_indices、duration并关闭二次采样，保存逐帧与tensor摘要。
- CPU probe的均匀首尾采样不等于教师官方抽帧协议 → 同帧数仍可能看到不同输入 → 数值parity双方用同一冻结输入，benchmark及师生质量比较另绑定同条件协议，不能复用probe结果冒充复现。
- 官方教师round+set在某些帧数会去重少采样 → 不可静默补帧或自改公式 → 保持固定协议并明确拒绝原因；协议变更另记录依据，不把少采样误称媒体损坏。
- 已hash媒体也能包含外部文件引用 → ffprobe/ffmpeg会继续读HLS/concat等载荷 → 输入先限制自包含容器和file/pipe协议，拒绝播放列表；仅protocol=file不能封住本地逃逸。
- 单文件40MP不能约束多图总内存 → 同请求会同时持有解码图、传输bytes和tensor → 全部路径/hash和图像header先预检，默认总图像40MP/媒体8项，再启动任何完整解码。

- 管线脚本stdin被吞/视频提前退出 → FFmpeg默认轮询交互stdin → 媒体probe/decode子进程使用DEVNULL，真实stdin回归包含q输入与实时视频。
