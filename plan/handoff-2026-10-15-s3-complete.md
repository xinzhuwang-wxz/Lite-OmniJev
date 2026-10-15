# S3完成接续（2026-10-15）

## 当前结果
S3已通过，s3:01 done。唯一验收入口reports/s3-stage-acceptance-stable-001.json；原四门原值通过，独立复核reports/s3-integration-review-001.md无实质阻断。终局goal仍active，远未交付。

- 五模态500题双遍：image82.0%、screenshot96.4%、text99.4%、audio80.0%、video16帧63.6%。旧三模态与S2持平；教师audio38%/video69.8%同条件基线不变。
- video8帧500题单遍60.8%，保留16帧；6组共5500条原始记录全部通过来源目标/哈希/logits读出/重复/指标审计。
- 基座E4B ee0ef6023621cff504d758262d4e04895a5af4a2，adapter1bebae31309f799395d01499fb9deec5e44622599d0c082473ea8b5a014e5755；远端artifacts/s3-stable-probe-train-001/adapter。
- ADR0021替代原300步/10000曝光估值：实际S2 v2暖启动、lr2e-5、100步×10、各模态200次，step50检查点。不是旧预算完成；质量门未降低。
- 训练843.25秒，峰值30.14GB；本候选训练+正式评测3266.99秒（0.9075 H20活跃小时），不含失败探针/小筛查/S0–S2/空闲租时，不等价H100、不冒报费用。reports/s3-resource-recap-001.json。

## 必须保留的修正
Noul规范顺序yes/no，旧metrics按JSON字母排序no/yes导致并列概率统计差异；a2c84cb修复。产品预测不变，S3门一直读取product_accuracy。原GPU报告保留，audio/video/video8分别有CPU correction报告。旧远端b6b827b已完成全sidecar审计，再同步新代码不会丢失证据。不要用新指标函数直接断言旧错误accuracy/ECE相等。

## 下一阶段
s0:05已认领（S4真实gold/自动gate）。reports/s4-gold-gate-implementation-assessment-001.md是实现边界；eval/gold_gate.py目前只有严格元数据四池连通审计，79测试通过，不代表完整gold准入/权限/封存已完成。

- 库存reports/s3-reserve-inventory-001.json：文本3000校准/1000封存足量，其余不足，crossmodal尚未分配。
- audio全部已下载公开划分，在排除当前train/dev后最多1916题，至少还缺2084，仍须全历史/media别名审核；ADR0004允许已准备TTS，但不能当真实音频或重复计数。不能挪已训练family。
- video原train保留calib1800题/holdout600题；已下载validation有5000family/10000原始exist题候选，先做全历史/媒体准入，不再下载大视频集。
- 图像/截图可复用既有源与builder，需补实际可靠gold及完整历史曝光账本。开发agent不读封存载荷，只接独立作业汇总。
- S4最强教师97%、封存差、外榜，S5校准稳健性，S6量化/E2B/性能与最终三档包/真实浏览器均未完成；大预算/付费/租卡开关/生产部署不自动做。小规模创新/RL按实测自主选择，不臆称公开配方复现。

## 运行环境与GPU接续
远端/root/autodl-tmp/s1-text-lora-001；Python /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python；SSH mux /tmp/omnijev-ssh-root-final，端口48926。模型与凭据路径沿之前runbook，禁止将凭据写入repo。

原s3-full-gates.sh已全部完成；当前GPU使用以plan/STATUS.md和真实进程为准。b135176补s0_parity的--image-max-soft-tokens参数，准备对S3 adapter+1120运行真实100例HF/native概率对照；31相关测试通过，真正GPU证据待跑，不把旧CPU100一致冒充HTTP通过。此验证不是S4模型封存评测，也不证明低成本API性能。

主会话负责同步：运行作业时不改远端checkout；结束后用git bundle ff同步，把确切路径冲突的旧untracked证据移动到唯一recovery目录，不stash-u/reset/clean。旧权重/媒体不删，不push。

## 工程证据
reports/s3-engineering-validation-001.json：ruff全仓、430文件format、105源文件mypy、wheel/sdist通过。相关行为测试覆盖Noul/日志41、元数据隔离/旧canary79、parity31、教师顺序2；独立review另60项。普通后续修复只跑相关检查，不按轮次全跑。
