# 资产与调度入口
- metadata下载通过不等于权重齐备 → 清单可能缺权重或分片 → full ready前检查必需文件、分片index、上游hash；metadata-only永远weights_ready=false。
- OneJev-27B不等于JEV-27B → 子串禁用误伤候选教师 → 用明确模型身份/来源规则，教师许可仍需独立准入。
- queue claim不等于派工成功 → 中断可能留下claimed → 恢复前检查owner进程与worktree，不按超时抢占。
