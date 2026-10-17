# 对照矩阵评测器规格(comparator-matrix-spec)

状态:设计冻结,待实现。2026-10-17 晚,SOTA冲刺3。
上游:ADR0025(判定榜与四对照)、`plan/STATUS.md` 接续点。

## 问题

ADR0025 要求四对照(OneJev-4B、Valen-2B、Jev-Omni-12B、OneJev-27B)"同协议同题我方实测"。
现役 s1_dev_eval 是 Gemma 专用(chat template + gemma4_unified processor + answer-token 表),
对照模型两族用不了:OneJev-4B/27B = qwen3_5;Valen-2B = Qwen3.5 底座。
Jev-Omni = gemma4_unified,可直接跑(夜间链已在跑,产出交叉验证锚点)。

## 协议定义(判定口径)

"同协议"冻结为:**同题(契约 jsonl 逐行同 id/同选项顺序/同字母映射)、同判定规则
(答案字母 arglogit,letter token 概率)、同遍数(passes=2)、同媒体字节(sha 钉死)**。
提示框架允许按模型族用其**官方 chat template**(各模型最佳原生形态;S0 教师协议先例)。
偏差必须逐项写进报告 `protocol_notes`。

## 实现要点(scripts/comparator_eval.py)

1. 复用 `public-bench-samples-001/*.jsonl`(Sample 契约)与 media/,零新数据。
2. 每族一个 adapter:
   - `gemma4_unified` → 直接走 s1_dev_eval(不重写;本脚本仅处理非 Gemma 族)。
   - `qwen3_5_omni`(OneJev)→ AutoProcessor + 其音频/视频路径;letter-token 用其 tokenizer 编码。
   - `qwen3.5+valen`(Valen)→ 其 repo 的转换后加载路径(S0 有先例:s0-public-baseline-valen-001)。
3. 打分:对答案字母集合 {A.., U?} 做 next-token logit 读出,取 argmax;与 s1_dev_eval 的
   read_probabilities 语义对齐(确定性,temperature=1)。
4. 记录/报告格式与 s1_dev_eval 兼容(metrics 字段同名),多出 `protocol_notes` 与
   `harness: comparator-eval-v1`。
5. 交叉验证:Jev-Omni-12B 在 MMAU 上跑本器,与夜间链 s1_dev_eval 的教师同协议数对比;
   差异 >1 分即视为实现 bug,不得出数。

## 验收门(CPU 先行)

- 单测:mock tokenizer/model 下 letter 打分与 records 写出行为正确(ruff + mypy strict 全绿)。
- GPU:Jev-Omni MMAU 交叉验证通过后再跑 OneJev-4B MMAU;全族通过后才可上判定表。
- 报告须含:模型 revision、代码版本、数据 sha、耗时、退出码(失败也留证)。

## 依赖与顺序

1. 夜间链产出教师 s1_dev_eval 同协议数(锚点)。
2. 本器实现 + 单测(不占 GPU)。
3. Jev-Omni 交叉验证(≈30 min GPU)。
4. 用户决策后:全对照矩阵烧卡(MMAU ≈0.5-1h/模型;MVBench ≈8h/模型;qwen3_5 27B 更慢)。
5. Valen 权重下载仍被盒上外网阻塞,网络恢复先补权重。
