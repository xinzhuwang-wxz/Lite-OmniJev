# P0 无卡准备审查

范围：SwanLab凭据/异常边界、checkpoint与恢复；独立checkout审查，不执行GPU作业。

- 6173c84发现：无效resume会覆盖已有run.json；旧checkpoint在训练后才撞上已有输出；checkpoint张量载荷缺哈希。
- 88f4a19修复：先验证resume再允许任何已有输出写入；提前拒绝旧checkpoint；模型/优化器/RNG载荷SHA256先验证再读取。
- 独立复核44项训练测试通过；直接重现无效恢复保留原报告字节、旧checkpoint提前拒绝、三类合法格式但被改动的safetensors载荷拒绝。
- SwanLab caught-error路径未发现额外凭据泄漏；远程reports与monitoring文件扫描无用户API key。
- 主会话另检查HF/vLLM对照门槛：原始概率及归一化答案/U均须严格<1e-3，不能用归一化掩盖原始概率漂移；8项CPU回归。

GPU载入、数值一致性和真实E4B断点恢复仍须开卡验证；本报告不放行完整P0或阶段0质量门槛。
