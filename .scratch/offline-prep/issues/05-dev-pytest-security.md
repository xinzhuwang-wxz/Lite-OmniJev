# 05: 开发依赖安全修复

ID: offline-prep:05
Milestone: S0
Category: bug
Status: ready-for-agent
Execution: open
Owner: none
Worktree: none
Blocked by: None

## What to build
独立安全审查报告本地pytest8.4.2受PYSEC-2026-1845/CVE-2025-71176影响，修复版9.0.3；先核对官方公告，仅升级这项现有开发依赖及必要锁闭包，不升级GPU环境/全依赖。该票非当前下载阻断项。

## Acceptance criteria
- [ ] 官方pytest修复PR14343及PyPA公告版本范围已核对。
- [ ] 本地开发锁固定不受影响pytest版本；保留Python3.11支持，不改运行时/GPU环境。
- [ ] 全测试、lint、类型、build实际通过；有失败先查兼容性，不静默skip。

## Evidence
审查来源：https://github.com/pytest-dev/pytest/pull/14343 与 https://github.com/pypa/advisory-database/blob/main/vulns/pytest/PYSEC-2026-1845.yaml 。当前修复尚未实施。
