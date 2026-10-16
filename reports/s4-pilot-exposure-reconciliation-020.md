# S4 continuous-matched 第二段曝光核对 020

真实CPU exit0，35.934秒，完整inventory isolated；80份已知update-bearing报告，unknown0。
artifacts/s4-pilot-exposure-020首次成功；固定1200池append0，current TRAIN仍唯一continuous-matched runtime data.jsonl。

- matched第二段10→20/paused/exit0，新增120 draw，累计240；区间[0,120)+[120,240)，没有相加10+20累计步数。
- 原continuous仍独立记录240 draw及两份immutable snapshots，与019完全相等；matched另有两份immutable invocation记录。
- 原f2db init、source235ad68三文件、global100/sample1200/data/config固定；真正resume checkpoint step-0010 identity与019原记录完全一致。
- fullcheckpoint step-0020 model/optimizer/RNG/config/assets/data/order SHA全部核验；20个loss中前10个与首段精确相等。
- 019全部all_bound_input_sha256逐项完全保留，native/specialist/textmatched旧计数与失败恢复链保留；两profile rawrun继续只当次观察，不永久绑定。
- 固定300pairs/600video+600replay为池容量，当前只记matched实际240 draw；官方source/pairmetadata/actualshuffle/replay校验复用既有严格profile。
- 未运行GPU、未重跑训练、未读私有样本内容；只通过既有aggregate audit取得完整隔离汇总。

020 report SHA：eb7502b939da6339d4944421cfafae297e2251101f88c9327a4466f9226cc81f。
019 report SHA：a72b12182192a75600afd58a3ef67d8d3238388a70e4017dfbae38e92d9f2a88。
step-0020 checkpoint metadata SHA：22bf37b9339342f4f38abf88324445de41ed284f607afd65f8f712accb4310af。
step-0020 adapter SHA：0689ce206fa2a8cb1b3f8582f6d3d23dfdbab8be792b9f66cf2792bdcc40a087。
完整绑定链及两profile记录见同名JSON副本。

helper仍/root/autodl-tmp/s4-exposure019-code-0d7dec8/reconcile-pilot-exposure.py，未改代码；
Python /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python；
cwd/PYTHONPATH /root/autodl-tmp/s4-continuous-code-235ad68。
CLI --continuous-matched-pool，--continuous-launch-report=reports/s4-continuous-matched-pool-launch-002.json，
--continuous-base-exposure=artifacts/s4-pilot-exposure-019/report.json，output=artifacts/s4-pilot-exposure-020；
root=/root/autodl-tmp/s1-text-lora-001，legacy-root=/root/autodl-tmp/lite-omnijev。
日志 /root/autodl-tmp/s4-pilot-exposure-020.log；实际exit0。

复用未改helper已通过的29项相关CPU测试与lint/format；本次真实审计、019全绑定全等、两profile旧记录保留、loss前缀全等、resume checkpoint全等、独立draw计数及唯一current均通过。
这是曝光与隔离证据，不是质量晋级或后续训练授权。
