# release-002 集成复核

2026-10-18，原生独立代理只读复核；工作区隔离于`Lite-OmniJev-release-audit`。
范围：产品网关U边界、Playground输入/输出与cURL、部署、分片恢复和分发允许列表。

发现并修复一项阻断：启动脚本只检查派生config存在，可静默复用旧模型且删除无config的运维目录。
改由`serve/deploy/prepare_view.py`绑定模型/基座/权重文件身份和metadata摘要；不匹配拒绝，
仅创建/清理自己的临时目录。4个启动行为测试覆盖两种dtype、保护运维目录、旧版本拒绝与完整来源复用。
实物CPU验证5分片hardlink、标准化配置与GPU实测版本一致；用户后续确认卡仍开，已安排最终整包复验。

其余被审代码未发现可证实的必须修复项。最终回归记录以`release-final-validation-001.json`为准。
不将复核当作E2B、生产鉴权或完整原方案验收。
