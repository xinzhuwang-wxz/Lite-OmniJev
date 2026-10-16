# CLEVRER TRAIN 严格事件顺序对照：可行，但尚未构建数据

**在固定24个候选TRAIN family中，22个满足安全collision顺序规则，提供44个合格事件对。** 可构造新的、以官方事件标注求值的关系gold；不能将现有no问句反转后猜另一关系为yes。此结果只证明标注层可行，不证明16帧模型能看清事件或训练有效。未创建训练数据、未GPU；cost=null。

## 来源与有限检查

复用ADR0005已批准的CC0官方CLEVRER TRAIN、现有8299 TRAIN pool和原builder `s0_media_dev.source_hash_for_video`。官方README明确dataset包含videos/annotations/questions、CC0、128帧，以及collision的object_ids/frame_id和原问题postfix program。

固定8299 pool SHA `912c871f7467b36ea1837276536250539f16f2ae6c5eaf32f867912b9e49e948`；原TRAIN questions SHA `11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4`；TRAIN annotation zip SHA `e7f9185ad3dde8139b839b00f0f75483374dd0325a3cbec2dac882078497784c`；官方README SHA `a4838cc68b79ba049b54f95def1714cdbbd1380193d7d2a5d1a85440ac862e7b`。全部实际重核。

候选从8299中已有temporal词规则选取，namespace `42:s4-order-feasibility:`，固定hash排序12yes/12no、24unique families，不按模型对错挑。只JSON解码这24个已准入TRAIN family的原场景；其他官方source对象仅结构扫描跳过。只读对应24个TRAIN annotation zip成员，未开任何cal/holdout文件、未取新媒体、未执行原程序或第三方代码。逐项原question/gold/program/media/source绑定24/24成立，原文不进入Git。

## 实际可用性

| 项 | 聚合结果 |
| --- | ---: |
| 候选family | 24 |
| collision标注事件 | 63 |
| 各自唯一发生的collision object-pair | 63 |
| 事件双方在对应帧inside_camera_view | 63 |
| 存在三属性重名对象的family | 0 |
| 存在同object-pair重复collision的family | 0 |
| 两事件同帧的被拒候选对 | 0 |
| 不同帧但间距不足16，被拒候选对 | 12 |
| 满足全部条件事件对 / family | 44 / 22 |

一个family不足两种collision，另一个只有过近事件对；均不合格。合格间距min16/max88/mean42.5帧。名义128→16均匀采样下，没有合格事件对落在同一个最近采样帧bucket；这只是可见机会，**不是两次碰撞已被采样帧捕获的证明**。事件双方在真实collision帧可见也不保证附近采样帧无遮挡。

原所选24题全为exist subtype，其中10程序带filter_before/after，15带filter_in、5带filter_out，0带filter_collision（计数可重叠）。因此现有temporal关键词课程混有入场/出场/时间状态过滤，不能把它们都当成直接collision先后关系题。完整原postfix executor本轮未实现、未验证；不能改其filter_before/filter_after或自然语言并沿用旧gold。

## 最小安全builder边界

只需新增独立TRAIN派生builder，不改生产loader/模板/采样：从官方annotation以unordered object-pair定义collision事件；要求被引用pair在整个视频恰好一次、对象存在且color/material/shape三属性组合在整个场景唯一、各参与者在该事件帧inside_camera_view为true。事件帧为合法0–127整数、不同且绝对差≥16。缺事件、重复事件、描述歧义、同帧/过近、不可见全部拒绝，不用first/last规避重复。

最小句式为：`Does the collision between the <color material shape> and the <color material shape> occur before the collision between the <color material shape> and the <color material shape>?` 配对句完整交换两个事件短语。object_id只用于内部标注查找，**不出现在模型输入**。

新程序应明确为 `resolve_unique_collision(A); resolve_unique_collision(B); less(frame(A), frame(B))`，两条关系分别计算 `<` 后映射yes/no，再验证互斥与一正一负。只有“两事件确实存在、各唯一、无tie”条件通过，才能得到互斥对；不是从旧问题no推断反关系。原问题program只作为来源绑定与独立审计资料，不冒称重执行旧程序。若未来扩到in/out、while、复合事件，必须另实现并对照官方gold验证相应语义，当前不扩。

builder还需将同video两变体绑定同family/组件/媒体SHA与TRAIN ledger，记录annotation成员SHA、事件选择规则、规范程序/求值结果、构建代码SHA；每family固定最多一个事件对，选择与命名顺序按hash，勿引入“第一句总yes”。不改已有source文件、不把44个事件对当独立family、不跨池分配。当前22family若各取一对可得44条新关系题，此数字不是已创建/已审计训练产物。

验证：真实CPU聚合5.729秒/exit0；7个合成严格规则检查通过（正常顺序、缺事件、重复pair、tie、不可见、属性歧义、间距<16），ruff通过。后续构建须先由root定范围，本轮不预排训练、不宣称新颖或收益。
