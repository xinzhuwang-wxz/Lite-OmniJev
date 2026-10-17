# 01: clevrer_fourtype_eval.py mypy strict 债务

- 状态:open
- 来源:SOTA冲刺3 整理(2026-10-17)。该文件来自 3653009(上一冲刺),非本轮引入。
- 现象:`uv run mypy data train eval serve scripts` 报 8 错,全部在 `scripts/clevrer_fourtype_eval.py:48-58`(score_levels/options 的 `str | None` 收窄缺失)。
- 影响:全仓类型门不过;不影响该脚本已产出的评测结果(运行时行为已验证)。
- 验收:该文件 mypy strict 零错;不改变脚本输出。
- 优先级:低(下次触碰该文件或里程碑集成时顺手收口)。
