# S4 条件跨视频排序曝光核对 022

真实 CPU exit0，52.531秒；完整 inventory isolated，86份已知 update-bearing reports，unknown0。固定 shared matched1200 池 append0，该 data SHA 只有一份 current，路径转向本次实际 runtime data.jsonl；原路径和历史绑定保留。

- 独立 video_pair_objective_invocations 首段0→10/paused，实际120 draw，不续训。原 continuous240、matched240、rank64 CE120分别原样保留，不相加累计步数。
- 021 全部 all_bound_input_sha256逐项保持；原两条 continuous 各两份 immutable snapshots及rank64 CE一份snapshot全等。本次另增一份 named snapshot；raw run.json仅当次观察，不永久绑定。
- rank64 initializer62920299…仍是独立候选，数学来源f2db；failed parity、control、独立policy、own baseline f05a及CE control11e140原样保留。parity_validated=false、parent_quality_inherited=false；不继承f2db数值质量。
- 新 config f38c6672…相对实际rank64 CE e019…仅新增五个 video_pair 参数，weight0.1；原global100/sample1200/accum12/seed42/rank64alpha128/CE/帧数/image预算及checkpoint10保持。
- 复用原matched canonical/source/replay validator；调用实际7b75f9f的 load_video_pairs 重建300对，核官方 gold、exact nonmedia/program、opposite labels、distinctmedia/family、canonical TRAIN ledger及shuffle offsets0/2/4。池容量1200不代表本次已曝光1200。
- 新 prepared371bb26d…、sidecarc1ba3471…、root first10 decision6b7ecc32…、source archive49b11d24…、5 source files、launcher0ab1a071…均绑定。完整step10 checkpoint的 model/optimizer/RNG/config/assets/order哈希已核；无resume授权。
- 本核对未执行GPU/训练/评测，未显示私有样本内容；完整隔离使用既有aggregate audit。本证据只核对TRAIN曝光，不代表质量门槛通过。

022 report SHA：960a6b661f6414a514fa3274979b3a979669544979632251f52838f1f882edcb。
021 report SHA：24b2bcb85e7067c3759cba3f007782c46b174695fd77399da15bc468cfb3a35e。
step10 checkpoint metadata SHA：546a765b13493398f2440e81c018ac34fd2b2d6c095724700695a79967a6d3a3。
runner SHA：429f6b4eb051b043347aad9d463e8a80c19594b722812c68b25332b9510126b3。
完整绑定见同名JSON原字节副本。

实际命令使用 Python /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python；cwd与PYTHONPATH均为 /root/autodl-tmp/s4-video-pair-code-7b75f9f，helper /root/autodl-tmp/s4-rank64-exposure-code-001/reconcile-video-objective-exposure.py。
CLI --video-pair-objective --continuous-base-exposure /root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-021/report.json；root=/root/autodl-tmp/s1-text-lora-001，legacy-root=/root/autodl-tmp/lite-omnijev，output=artifacts/s4-pilot-exposure-022。

首次入口缺PYTHONPATH，在导入data前exit1，未创建output；保留 /root/autodl-tmp/s4-pilot-exposure-022.log。修正环境后只运行一次完整审计，日志 /root/autodl-tmp/s4-pilot-exposure-022-retry-001.log，exit0。没有重跑GPU或改动训练输出。

验证：28项相关CPU tests通过；lint/format通过。真实新源 CPU identity/300pairs sidecar绑定26项通过（无inventory），随后本次完整审计通过。报告后fresh验证021全绑定全等、五旧snapshots全等、四分支独立计数、共享池唯一current、fullcheckpoint和5sourcefiles通过。
