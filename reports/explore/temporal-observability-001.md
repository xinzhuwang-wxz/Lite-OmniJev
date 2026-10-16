# Temporal DEV50：残差集中事件条件，未见窄间隔alias主导证据

**下一重点应是原生事件存在/时间条件组合，而非再扫motion采样。** 固定公开DEV50上第三coverage的13错中9错来自22道原生事件before/after题；这9错由4个“目标事件不存在却答yes”和5个间隔≥16帧的关系题组成。唯一间隔<16的关系题答对。此为CPU只读诊断，不证明任何训练/模型收益，也不证明视觉表征充分。

## 真实输入与时间覆盖

复用已存在motion preparation、50个媒体motion manifests与两份完成的uniform16 raw readouts。两模型为CE0 `4b8e422f…` 和第三coverage `5cff2c3c…`，30/50→37/50；后者仍是候选，不是本报告新选工作模型。每题sample/source/gold/program/media、实际frame_indices与prepared input hash配对核验，manifest/receipt/records SHA严格绑定，未新下载、解码、推理或访问private。

视频均128帧/25fps/5.12秒。实际uniform16索引为 `[0,8,17,25,34,42,51,59,68,76,85,93,102,110,119,127]`，相邻间隔8–9帧（.32–.36秒）。程序中的50个时间anchor可在标注中定位，距最近采样最大4帧（.16秒），平均1.66帧。16个anchor恰好采中、34个在采样间隙；“恰好未采中”不等于事件不可识别。

本轮仅做受限、明确的程序结构与camera transition解释：attribute filters选对象；唯一false→true/true→false作为in/out时间proxy，start/end固定0/127；collision用官方collision.frame_id。未假称运行完整官方postfix executor。22道关系题的受限解释22/22与原官方gold一致，但这不自动推广其他程序、遮挡或初始已在场物体的in语义。初始inside=true不擅自造入场事件，多次/缺失anchor返回unknown。

## 任务结构解释了collision-order pilot的覆盖缺口

| 原生程序组 | N | CE0正确 | 第三coverage正确 |
| --- | ---: | ---: | ---: |
| in/out anchor的before/after事件存在关系 | 22 | 12 | 13 |
| 入/出事件时点的moving/stationary存在判断 | 15 | 8 | 12 |
| start/end时点的moving/stationary存在判断 | 13 | 10 | 12 |

22题中仅5题的target是collision，anchor仍是in/out；**没有collision–collision before题**。先前新order pilot要求两个collision都存在且唯一，因而未直接覆盖这50题的主体原生组合，更完全排除了下面8题的“目标事件不发生”分支。这是任务分布差异证据，不能单独证明该pilot无收益的原因。

22关系题中，8题没有目标事件（官方gold全no），CE0正确6、候选正确4；另14题有目标事件，关键关系间隔10–116帧。对existential yes使用任一充分且间隔最宽的witness，对no使用最近边界，明确该选择用于判断是否存在可分辨证据，不把所有不相关事件都当critical。0题两事件映射同一最近uniform采样bucket、0题gap<9帧；1题gap10且候选正确。13题gap≥16，候选正确8。因此残差没有集中在极近事件导致的采样时间排序alias上；短促碰撞仍可能落在间隙，实际捕获没有得到证明。

关系题gold yes11、候选yes12；错误包含5 false-positive/4 false-negative，不能只用一个全局yes偏移解释。相比之下28道时点motion题候选正确24，关系组合是更集中的错误源。

## camera/遮挡边界

全样本“anchor在间隙”错误11/34，“anchor恰好采中”错误2/16；但后组包含13道较易start/end题。关系题内只有1道anchor恰采中且答错，另21道错8；时点motion事件组只有2道恰采中且均对。小组和任务混杂不足以声称补采anchor必然改善。

11题在最近采样帧没有target对象inside-camera，其中7题是属性筛选后根本没有对象（候选全对），不能误称7个遮挡例。剩余4题有解析对象但最近帧全部outside-camera，两个模型都只对1题；这里确有信息缺失proxy，但仅4例。**inside-camera不表示无其他物体遮挡、足够像素、processor保留细节或模型看到了对象**。缺少逐对象mask/相机投影与处理后像素验证，本轮实际occlusion不可判定，也未把outside当作原gold错误。

## 唯一可逆TRAIN下一候选（提案，未构建/未授权运行）

在现有连续AdamW固定池工作流内，尝试一次**原生in/out-before/after gold课程覆盖**：只从已准入8299 TRAIN按固定hash选原题/program类别，保留原题和官方yes/no，明确覆盖目标事件不存在的原生no；不反转no、不重新标注、不改uniform16/模型/损失。沿用每步6video+2cross+4old、10步120draw、seed42、同预先固定init及对照预算，每arm≤240秒、5步checkpoint；由root在连续optimizer边界已验证后择时，不能借此自动上卡或追加多arm。

只比较这一种task-coverage变化，记录固定DEV50关系组双向错误/概率loss、总开发概率与旧模态保留；若关系机制与总体保留没有实证改善就停该配方，若有利仅扩大开发验收。不得调DEV阈值、扫采样或把小切片当最终质量门。本建议保留与当前主线的接点，不另造reward/分类头或图像引擎。

机器证据同名JSON：来源/模型readout SHA与全部预定义分层聚合，cost=null。真实CPU5.361秒/exit0，5个公共程序/时间覆盖边界测试及ruff通过；无raw题目/答案记录入Git、无新依赖/GPU、无CPU冒充模型收益。
