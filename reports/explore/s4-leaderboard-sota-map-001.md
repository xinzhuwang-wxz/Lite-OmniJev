# SOTA 榜单映射:Lite-OmniJev 相关公开基准 × 当前状态

> 2026-10-17 生成 | 副手 SOTA冲刺1 写入 | 数字来源:项目内实测 + 三轮调研(reports/explore/s4-frontier-research-00{1,2,3}.md)
> 协议警告:我方"同协议"指同一数据+同一题型+同一评测格式;不同协议数字不可直接比较

## 一、项目直接相关的公开榜单

### 1. CLEVRER(视频因果推理,核心差距所在)

| 维度 | 详情 |
|---|---|
| **协议** | 4 类题:descriptive / explanatory / predictive / counterfactual;官方测试集需 EvalAI 提交 |
| **指标** | per-option & per-question accuracy |
| **SOTA 符号执行档** | VRDP ~96.0% descriptive(NeurIPS 2021,Wang et al.;二手转述,原文未直接核实);CRCG ~96%+(frontier-research-001 引用);NS-DR 88.1%(ICLR 2020 原文确认) |
| **SOTA 纯直答档** | 70-77%(文献;frontier-research-001);OneJev-27B 同协议 94.2%(项目内实测 reports/s4-* 系列) |
| **我们 D3P-600** | dev 切片 82%(123/150);video500 全集 75.6%(单遍) |
| **同协议教师** | OneJev-27B 94.2%(video dev500,reports/s4-* 系列) |
| **测试集通道** | EvalAI 提交(需账号);validation 可自测 |
| **差距来源** | 18pp 教师差距中 15pp 在 enter 类(106/150=71%),本质=时序/事件感知 |

**关键判断**:纯直答路线文献天花板 70-77;我们 82 已超。教师 94.2 需 27B 规模或符号结构。级联+RL 是超越直答天花板的路径。

**⚠执行器+物理 oracle 终判(SOTA冲刺2,reports/explore/s4-ccd-oracle-001.md)**:CLEVRER 四类题真值世界 CHECK = **D 100 / E 100 / P 97.3 / C 97.1**(per-option)——全部超公开榜首。口径注意:此为 per-option accuracy on validation;公开榜数字多为 per-question on test(EvalAI 提交),两者不可直接混比。执行器+物理=我们自研,非借用 NS-DR/VRDP。

## 口径注记
- 我方所有数字=validation 集,per-option;教师 OneJev-27B 同协议
- 公开榜 CLEVRER test 集数字需 EvalAI 提交,未提交
- MVBench/MMAU 数字来自 S0 阶段,协议各异

### 2. MVBench(多模态视频理解,S0 基准)

| 维度 | 详情 |
|---|---|
| **协议** | 20 任务多选;已有项目 S0 实测 |
| **我们** | S0 阶段已通过(见 reports/s0-stage-acceptance-001.json);具体分数见 S0 证据 |
| **榜单 SOTA** | 未核实(非本轮焦点) |

### 3. MMAU(音频理解)

| 维度 | 详情 |
|---|---|
| **协议** | 音频理解多任务(原方案涉及;S0 ADR0006 有 MVBench 基线) |
| **我们** | audio500 封存 88.4%(校准后);audio dev500 79.6% |
| **榜单 SOTA** | 未核实 |

### 4. JevBench(项目自定义外部榜)

| 维度 | 详情 |
|---|---|
| **协议** | 231 决策(Choice/Score/yes-no);两遍一致性 |
| **我们 D3P-600** | 76.62%(冻结基座 76.19%,+0.43pp;macro −2.46pp 披露) |
| **教师** | 同协议 195 题子集已测(reports/ 里);全集未测 |

### 5. Decision Index(项目自定义)

| 维度 | 详情 |
|---|---|
| **协议** | 1530 文件/5.90GB;准入 27/44 eval-only |
| **状态** | 准入验证完成,评测待最终模型 |

## 二、当前最佳成绩汇总(D3P-600,封存校准后)

| 模态 | 封存 6000 题 | ECE(校准后) | 教师同协议 |
|---|---|---|---|
| text | 99.7% | .0014 ✅ | — |
| screenshot | 97.8% | .0108 ✅ | — |
| image | 92.7% | .0202 ✅ | — |
| audio | 88.4% | .0310 ⚠️ | — |
| **video** | **72.6%** | **.0321** | 94.2%(差 21.6pp) |
| crossmodal | 79.9% | .0528 ❌ | — |

## 三、达到并超过 SOTA 的路径(按证据强度排序)

| 路线 | 预期落点 | 证据 | 状态 |
|---|---|---|---|
| **级联+RL 形式化器** | video 级联 **≥124/150(82.7%)** 起步,上界 oracle **94.67%** | 执行器 v4 已证超教师;RL F1 三连涨 | 🔄 RL 场景生成中 |
| 执行器 v5(EXIT_REL 4 错) | oracle → **97%+**(超符号档下沿) | EXIT_REL 4 错未动,fable 有 dump 待定刀 | 📋 已设计 |
| N3 异源仲裁 | 神经 94.2 + 符号 90+ + 仲裁 70% → **~96** | round-3 预估;异源红利已证 | 📋 已设计 |
| D3P 全池蒸馏 | 直答 +1-2 → ~84 | 曲线 flatten(150→600 仅 +1) | 📋 已设计 |

## 四、核心瓶颈诊断

**18pp 教师差距的精确分解**(SOTA冲刺2 诊断,reports/explore/s4-sota-takeover-diagnosis-001.md):
- **enter 类 15pp**(106/150=71%):可归约为逐帧对象存在性 = 静态感知 → **静态 F1 已 .97,缺口在事件/时序的执行逻辑,非感知**
- 其余 3pp:explanatory/predictive/counterfactual(需物理推理)

**行动结论**:执行器 v4 已证"完美场景+正确执行"可达 94.67%。剩余瓶颈 = VLM 场景抽取的事件谓词质量(exiters F1 .091/collisions .263/moving recall .357)——RL 形式化器的精确训练目标。
