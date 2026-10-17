# STATUS

更新:2026-10-18 01:5x。**负责人:交付链冲刺(主会话,用户全权)。模型定版不再训练;主线=③产品化→④量化→⑤HF发布→⑥性能→⑦终局验收。Decision Index 公共子集本窗口测(用户确认)。**

## ⚡ 接续点(第一动作)

1. **③ serving 复验进行中**(盒 screen):fp32 平面键 merged 工件物化(`s6-flat`,log
   `/root/autodl-tmp/s6-release-serving-001/logs/flat-005.log`,末行 FLAT_EXIT:0 即成)→ vLLM
   0.26 serve(SERVED_NAME=lite-omnijev-e4b-d3p600,FP32 已准入配方)→ `run-compare-vllm.sh`
   parity(<1e-3,reference 已就绪 `s6-serving-hf-reference-d3p600-001.json`,reference-ready
   100 行两遍精确可重复)→ 网关 → `s0_gateway_parity compare-gateway` → Mac 浏览器
   `browser_s0_cpu --base-url`(SSH 隧道)。发射命令全在 `serve/deploy/launch-vllm.sh`。
2. **④ 量化并行**:llama.cpp 编译中(盒 `s6-llamacpp`,build.log 末行 LLAMACPP_BUILD_EXIT)。
   榜口径已收口(ADR0028):主表=JevBench+MMAU(路由形态)+Decision Index 公共子集,MVBench 参照。
3. flat 工件验收后:release-002 入 `model/merged-fp32-flat/`(或按 SHA256SUMS 重排),填 ⑥ 数据,
   开 ④ 转换,Decision Index 打分(官方 runner=apolinario/decision-index@f8552a1,http 引擎直打
   `/v1/systemone`,数据在 quarantine/decision-index-public0.3,27/44 已准入)。

## 定版与 serving 形态(本轮实测,证据在盒 s6-release-serving-001/artifacts)

- **模型定版 = release-002 / D3P-600(767be09b)**,用户确认不再训练,Bet B 暂停不续。
- **发现①**:bf16 merge 舍入 1.46e-2,构造性过不了 <1e-3 parity 门;fp32 重合并 1.53e-05。
- **发现②**:HF 把 Gemma4 KV 共享层(24–41,k_norm/k_proj/v_proj)按共享存储折叠成 2076 键
  compact 布局,vLLM 0.26 按平面 2130 键拒载;共享规则=同注意力型最后一个非共享层(滑窗→22,全→23),
  两引擎实现一致;平面文件里这些名字是死重。
- **发现③**:vLLM 0.26 LoRA 内核只支持 fp16/bf16——native fp32 LoRA 不可行。
- **serving 身份 = merged-fp32 + 平面键物化**(共享层死名用运行时真源层值填充,两引擎一致);
  merged/merged-fp32(compact)保留用于 HF 分发/量化,模型卡如实标注。资产:
  `model/merged-fp32/`(31.76G,sha 86676e1e…)、`s6-release-serving-001/merged-fp32-flat/`(物化中)。
- hf-reference 已过:基座+767be09b adapter,fp32,100 案例两遍精确可重复(230s)。

## 现役资产(已核验)

- 盒工作树 `/root/autodl-tmp/s6-release-serving-001/repo`(= main 367e008,git archive 同步);
  serve env `envs/serve`(硬链接克隆+av/soundfile,清华镜像;PyPI 直连龟速)。
- 部署脚本入仓 `serve/deploy/`(launch-vllm.sh / launch-gateway.sh / make_parity_records.py / README)。
- release-002 增 `model/merged-fp32/`;report `s6-merged-fp32-report.json`。
- HF:git@hf.co:Chinzhu/Lite-OmniJev,SSH 认证已通(Chinzhu);git-xet 未装,上传前补。
- GitHub:xinzhuwang-wxz/Lite-OmniJev,简介+demo+榜单引流;**LICENSE 由 Apache-2.0 改 CC-BY-NC-SA-4.0(用户明确)**。

## 证伪/勿重试(累计)

- 前清单 9 条 + RL 级联 + G1 原子门 + PTR 门不变。
- **新增**:bf16 merged 直接 serving(舍入 1.46e-2);compact 键工件喂 vLLM 0.26(拒载);
  vLLM fp32+LoRA(内核断言)。全部有 log/repro。

## 已闭(勿重跑)

- MVBench/MMAU/JevBench 实测与差距表(reports/s4-gap-table-001.md);对照公开数抓取;CCD-for-decisions。

## 待我决策

- 暂无阻塞项;⑤ 上传前给一次模型卡终稿确认;量化掉分若超 1pp 门槛,报告后待裁决。
