# S4 continuous-matched 首段曝光核对 019

真实CPU exit0，33.323秒，完整inventory isolated；78份已知更新报告，unknown0。
artifacts/s4-pilot-exposure-019首次成功，new fixed1200池只追加一次，唯一current TRAIN为continuous-matched runtime data.jsonl。

- 新profile首段0→10/paused/exit0，实际draw区间[0,120)，checkpoint step-0010完整校验。
- continuous_matched_actual_global_draws_seen=120与原continuous actual_global_draws_seen=240分别记录；原continuous_invocations两份immutable snapshots与018完全相等。
- 固定池容量为300pairs/600原官方视频+600原固定replay，共1200；首段仅曝光120 draw，不把容量当全部曝光。
- 复用官方source、原gold与paired identity校验，核300pairs非媒体prefix/question/原program相等、opposite官方gold、不同media/family及source/canonical metadata；100个固定optimizer blocks offset0/2/4的pairs与原600replay绑定。
- 原固定replay canonical完全一致，components原09f5…字节保持；TRAIN1200 ledger精确绑定，未新增teacher标签或loss。
- parent仍原f2db，global100/sample1200/config/source3files固定；首段resume_from=null，full AdamW/RNG checkpoint与完整order identity强绑定。
- 不可变named final/runtime snapshot绑定，raw run.json仅当次观察；原profile与新profile rawrun均不写成可覆写的永久绑定。
- 018全部all_bound_input_sha256逐项保留，包括旧failed/recovered链；native120、specialist120、原textmatched120各自保留。
- 未操作GPU、未重跑训练；私有内容只经现有aggregate inventory audit回汇总。

019 report SHA：a72b12182192a75600afd58a3ef67d8d3238388a70e4017dfbae38e92d9f2a88。
prepared report SHA：2c51e16cb88ab91dfb45dca84a2325b371ad259dce522a01fa50e042090cfa87。
data SHA：17cb9a0066639dafd1feb21e3819cc38a367c86e78ce2a2907a5047d6cb437aa。
config SHA：d8e9e5b0fc8427b32f39165c58486bfd93123fea1a584b5c0f8508efcb389414。
step-0010 checkpoint metadata SHA：af393ae549c544961bc2cb67be2bd3c972ed039d6bbed63d550d68de1b759733。
adapter SHA：f6e9db3ffe88644a932e85eb6832918a42657fdb7c87628bcb7d4e09024e0fe2。

helper /root/autodl-tmp/s4-exposure019-code-0d7dec8/reconcile-pilot-exposure.py；
Python /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python，
cwd/PYTHONPATH /root/autodl-tmp/s4-continuous-code-235ad68。
CLI --continuous-matched-pool --continuous-base-exposure 指向exposure-018/report.json；
root=/root/autodl-tmp/s1-text-lora-001，legacy-root=/root/autodl-tmp/lite-omnijev，output=artifacts/s4-pilot-exposure-019。
日志 /root/autodl-tmp/s4-pilot-exposure-019.log；实际exit0。

复用0d7dec8已通过29项相关CPU tests与lint/format；helper/输入未变。本次真实审计、018全绑定全等、旧snapshots全等、唯一current、独立profile计数和完整checkpoint/source/snapshot均通过；完整报告同名JSON副本。
后继必须base019、同一原f2db/config/data/source并恢复step-0010，记录matched全局20步，不把段1adapter作为新init。本报告不是质量通过或自动跑满100步授权。
