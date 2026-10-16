# S4 连续固定池第二段曝光核对 015

真实CPU核对通过：exit0，29.510秒，完整inventory审计isolated。
69份已知update-bearing报告，unknown0；fixed pool追加0项，current TRAIN仍只有1项。

- 原始runtime paused/exit0、累计20步；本次10→20只新增10步/120 draw。
- 确定完成区间[0,120)+[120,240)，累计240 draw。没有相加10+20，也没有将池容量1200计为已曝光。
- 本次从原同目录step-0010完整checkpoint恢复；恢复metadata与014保存的checkpoint identity全部相等。
- 原f2db init、global config100/sample1200、source三文件、数据SHA、全局shuffle顺序不变。
- 014所有all_bound_input_sha256条目在015逐项完全保留；两段共20个loss，前10个与首段完全相等。
- 新step-0020完整checkpoint及adapter SHA实际核验；raw run.json只当次核验，继续不作为永久旧绑定。
- 014绑定的recovered002保留；其original_failed_launch引用failed001。额外实际SHA核验failed001未变，见下方。此failed001为recovered002内的间接绑定，未声称其已单独列入014/015 all_bound。
- 无GPU操作、未重跑训练；同期只有root只读DEV50，无updates。私有池只经现有内部aggregate audit处理。

015 report SHA：debb448fab4cfe4affe3e578c53f6bc98f79ac2f89f32a44bdfe4c8ed50b9e5e。
上一014 report SHA：2e3d89bbd4be8585057a5e197e3caadeaa5e4159f6cca51700bb2500fc12ac2c。
本次launch002 SHA：44e7da379f0ffc0156e5da3aab4695ee7b038cbfc5ddf847d880e479c65bf5d5。
旧recovered002 SHA：83240d5ddc515f7f2bb6fb33c46a3d4f7128399b116198908668533f200fb124。
旧failed001 SHA：193b25fea89480ab312fad7ebc236014507283450a8a31af84374864c489f9a8。

step-0020 checkpoint metadata SHA：fcd021f97ab7a462f83429a4ecc36e7cac64caa45379ab0d7aff3739a29baf89。
step-0020 adapter SHA：bf59de53785e223210be27fadd89f4fb8c239426c10febbe0d691c718e84825f。
三tensor/config/order与全部旧绑定详见同名JSON。

执行环境：Python /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python；
cwd/PYTHONPATH /root/autodl-tmp/s4-continuous-code-235ad68；
helper /root/autodl-tmp/s4-exposure015-code-001/reconcile-pilot-exposure.py。
CLI使用 --continuous-main-pool --continuous-launch-report 指向launch002，
--continuous-base-exposure 指向artifacts/s4-pilot-exposure-014/report.json；
root=/root/autodl-tmp/s1-text-lora-001，legacy-root=/root/autodl-tmp/lite-omnijev，
output-dir=/root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-015。
日志 /root/autodl-tmp/s4-pilot-exposure-015.log；实际exit0。新增可选CLI base参数只校验其与launcher绑定路径一致，不改变resume或数据身份。

验证：16个continuous曝光CPU测试通过，helper lint/format通过；本次真实CLI、完整审计、旧绑定全等、loss前缀全等、resume identity全等、旧failed001 SHA核验均exit0。
报告只证明已知保存更新输入的曝光和隔离，不是S4质量通过或后续80步授权。
