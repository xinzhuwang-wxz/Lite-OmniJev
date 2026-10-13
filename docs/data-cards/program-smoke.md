# program-balls v1（CPU工程夹具）

来源：本仓库 `data/synth` 确定性程序；内部生成事实，不含第三方数据/个人信息，随仓库Apache-2.0代码生成。
固定seed42生成100状态、400问题（Choice/Noul/Score/U各100）；真实数量/哈希见 `reports/cpu-smoke.json`。
标签来自红蓝球计数的程序真值；缺少材质证据的问题标U。family_id按事实哈希，跨池检验；这是smoke训练池，不是开发集或人工封存集。

限制：领域极窄；包含人为构造的标签到logit夹具，只验证工程概率与gate，不能计为训练效果、泛化分数或SOTA。
复现：`uv run python -m scripts.smoke --output reports/cpu-smoke.json`；产物存忽略的artifacts/data，不把大数据入Git。
负责人：项目agent；版本/哈希和seed记录于报告。
