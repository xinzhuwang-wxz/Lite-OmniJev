# 资产与调度入口
- metadata下载通过不等于权重齐备 → 清单可能缺权重或分片 → full ready前检查必需文件、分片index、上游hash；metadata-only永远weights_ready=false。
- OneJev-27B不等于JEV-27B → 子串禁用误伤候选教师 → 用明确模型身份/来源规则，教师许可仍需独立准入。
- queue claim不等于派工成功 → 中断可能留下claimed → 恢复前检查owner进程与worktree，不按超时抢占。
- HF LFS 原文件校验失败 → sibling.blob_id 是指针 Git 哈希而非原文件哈希 → LFS 用上游 SHA256+size，指针哈希另存；非 LFS 才比 Git blob。
- curl重试反复丢下载进度 → 内部retry会撤回本次写入再试 → 使用有界外部续传，每次从实际文件大小开始，完整校验前不提升partial。
- MDC urllib返回403但授权已完成 → 不能据此断言账号/条款缺失 → 复用官方curl POST流程；真实200/文件哈希才算下载通过，密钥走stdin配置且签名URL不落日志。
- 临时教师冒烟在昂贵加载后报字段错误 → Score/Noul没有Choice options，结果字段为prediction_index和概率dict → 加载前按Primitive验证选项，逐题保存数值；复用.scratch/s0/teacher-public-canary.py，失败不伪报模型通过。
- 截图负例网格造成标签捷径（无图坐标497/500）→ 同尺寸网格只负例对齐 → 使用确定性随机合法坐标、升数据版本并保留旧dev排除，旧截图分数superseded，仅重跑受影响cohort。
