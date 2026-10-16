# 有限动作 exact expected reward：CPU原型与数学边界

依据ADR0017，只新增独立`train.decision_reward.expected_reward_loss`，不接trainer/CLI/checkpoint/Swanlab，不选λ、不扫参、不GPU、不读任何样本。返回有限FP32标量：`-Σ q(a)r(a) + β KL(stopgrad(ref)||q)`；q为全部legal答案加U的FP32联合softmax。参考KL使用PyTorch现成`functional.kl_div(..., reduction="sum")`，方向与现有trainer reference项一致，不另造KL库。

接口：

```python
expected_reward_loss(full_vocab_logits, *, question, token_ids, rewards,
                     reference_probs=None, reference_kl_weight=0.0)
```

Question提供现有choice/score/Noul合法答案契约；token/reward映射必须恰好覆盖全部答案及U，使用canonical答案顺序，与JSON插入顺序无关。奖励必须是完整的有限固定实数，明确由调用者的gold/业务代价提供；不接受可导reward tensor。reference可为完整映射或canonical序向量，始终detach；给reference必须同时显式正β。拒绝遗漏U、重复/越界token、不合法维度、NaN/Inf、无效reference。半精度输入转FP32保留梯度。奖励中心化后加回常数，保持原目标值，同时令全相同reward的梯度精确为零。

## 是否有实质差异

对动作logit `z_j`，令`R=Σq r`，奖励损失梯度为 `q_j(R-r_j)`；非legal词表位置为0。奖励整体平移不改变梯度。若`r_j=c`对所有动作相同，损失为`-c`，梯度0；把stopgrad的Brier分数当每个动作相同reward不会产生有效动作优势。

若仅gold reward=1、其他（含U）=0，则损失`-q_gold`，梯度为`q_gold × ∇CE_legal`。因此`CE_legal + λ(-q_gold)`只是把该样本CE梯度乘`1+λq_gold`：**不是新监督信息或新单样本方向，而且更强调已易样本**。跨样本权重仍会改变总梯度/优化轨迹，但不能据此宣称修复时序排序。该项不是Brier；三动作时Brier一般方向不同。显式不同业务代价（如错误动作和U代价不同）能改变动作间梯度方向，新增的是优化偏好而非事实标签。

`-q_gold`不是proper scoring：真实标签分布p下期望风险为`-p·q`，一般由把概率集中在p的mode最小化，而非q=p。加KL或CE也不能未经证明就宣称组合校准proper。当前trainer的`decision_loss`实际在selected legal+U上算CE；以上比例等价只对该CE成立，不能误写成full-vocab softmax CE的等价式。现有实现及未来full-vocab CE baseline均保持原样。KL也不新增监督；reference只是固定初始化策略，且原trainer仍有自己的correct-gold gate/固定pilot权重，本原型不改变它们。

helper自身β权重位于其返回值内；若将来外层使用`CE+λ*helper`，KL实际系数是λβ。实验必须把实际系数写清，不能无意缩放或把不同KL方向混称同一方法。

## 后续最小公平对照（未授权运行）

只有当前事件关系数据/覆盖瓶颈证据支持时，再由root冻结一个TRAIN数据、同初始adapter/sourceSHA、同prepared input/draw hash、seed42、官方模板/答案映射、同16帧/1120预算；两arm都固定120draw/10step/accum12，每arm最多240秒，沿用5步checkpoint。CE baseline不改；候选仅在原CE上加一次预先冻结的完整动作reward项，λ及业务reward policy只从TRAIN目标/明确产品代价确定，不能拟合DEV或扫多个值。不新增头、不采样额外动作、不多forward，完整动作reward已知时直接精确求和。

可选reference必须绑定同init、模板、完整动作分布及输入hash并stopgrad；如需预先计算reference，其实测耗时/卡时计入候选预算，不能隐去额外成本。首次只比较一个有明确理由的候选，是否使用KL另由root定，不自动跑第三arm。记录gold CE、expected reward、KL、有限/非零梯度、实际steps/time/memory；同一预定DEV切片看accuracy/NLL/Brier/ECE、U/coverage、原题与置换/顺序一致性、旧模态保留。超时或明显退化终止，不因创新名义放量。纯one-hot reward若无业务机制理由，数学上主要是重加权，应优先解释该限制，而非把它包装成新的RL方案。

## 验证

真实本地CPU PyTorch2.14.1：新35tests＋既有pair23tests，共58 passed。覆盖解析梯度、有限差分、reward平移、constant reward精确零梯度、U动作、one-hot与legalCE比例/非Brier、显式业务代价方向、reference标准KL/stopgrad、FP16/BF16→FP32及错误输入拒绝。ruff check/format、mypy新模块通过。无新依赖；无训练/模型质量证据，cost=null。不称官方RLCD、proper校准、SOTA或创新收益已验证。
