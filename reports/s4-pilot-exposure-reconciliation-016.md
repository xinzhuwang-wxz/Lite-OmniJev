# S4 native event-condition 单段曝光核对 016

真实CPU核对完成：exit0，29.349秒，完整inventory审计isolated。
发现71份已知update-bearing报告，unknown0；只追加native新单段，唯一current TRAIN为其runtime data.jsonl。

- native是独立原CE/10步/120 draw/新AdamW/f2db初始化，不从continuous checkpoint续训，不新增loss。
- 60视频题与已准入full8299 TRAIN原题逐字段canonical相等，原官方gold保持不变；原60 replay也逐字段相等。
- canonical120条TRAIN ledger核验；每个实际seed42 optimizer block为6video+2cross+4old，视频3yes/3no。
- 官方TRAIN questions、annotations、README、来源池、程序metadata、源helper和训练archive SHA绑定；annotation成员逐项SHA核验。
- proxy只作为sampling metadata，不执行为新gold、不把camera状态称作模型可见性证明。
- prepared report SHA401fb424…、data SHAfa3a16c9…、root选定f2db、final/runtime/config及adapter强绑定。
- 015全部all_bound_input_sha256逐项完全保留；continuous两段immutable snapshot/记录与015完全一致，仍只记continuous实际240 draw，而不是池容量1200。native另记实际120 draw。
- continuous共享rawrun只核当次观察，不重新写成永久绑定。旧failed001新增直接SHA绑定，recovered002与旧链保留。
- 未运行GPU、未重跑训练、未读私有样本内容；inventory只经现有内部aggregate audit返回汇总。

016 report SHA：0164a2cdf0db9381cc76ae7411bd99f459789f4ae057c1cdfbf89d9276870aae。
上一015 report SHA：debb448fab4cfe4affe3e578c53f6bc98f79ac2f89f32a44bdfe4c8ed50b9e5e。
prepared report SHA：401fb4248c079c37e25cff1827b9470b3f3105ee8a3182997aee3c4de0fc7696。
data SHA：fa3a16c9b0b2343fcc69fae21e7505e421c610aa5dbc6993d096498ba6789a72。
failed001 SHA：193b25fea89480ab312fad7ebc236014507283450a8a31af84374864c489f9a8。

原始工件 /root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-016；完整报告同名JSON副本。
helper /root/autodl-tmp/s4-exposure016-code-001/reconcile-pilot-exposure.py；
Python /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python；
cwd/PYTHONPATH /root/autodl-tmp/s4-claim-pair-code-001。
CLI --root=/root/autodl-tmp/s1-text-lora-001、--legacy-root=/root/autodl-tmp/lite-omnijev、
--output-dir为exposure-016，--native-event-condition。
实际日志 /root/autodl-tmp/s4-pilot-exposure-016.log，exit0。

验证：98个曝光回归通过（9个新增native边界测试），helper及新测试lint/format通过；
实际完整审计、015全部绑定全等、两份snapshot全等、唯一current与actual draw计数均通过。
本证据只证明已知保存训练输入的曝光和隔离，不是native质量晋级或S4通过。
