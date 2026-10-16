# 工作父f2db原图像DEV500单遍

**410/500=82.0%，与S3image-stable持平；不是S4质量通过。** 单遍完整500题，exit0，wall207.675秒（限420秒），图像1120预算。没有训练/参数调整、其它模态或真实calibration/holdout读取。

| 指标 | S3稳定模型 | 工作父f2db |
|---|---:|---:|
| 正确/总数 | 410/500 | 410/500 |
| ECE15 | .1487930716 | .1505677310 |
| Brier | .3225937334 | .3214903806 |
| coverage@5% | .68 | .68 |

同500个id、target和选项token集合核验通过，1纠正/1退化。S3原报告双遍一致，本次按授权只一遍。原数据SHA `38efa78efe62ebdd8523c7372b8c0ac87a9030a18d429db488f44e5aad08e7de`，沿用S3同官方模型/template/tokenizer、原媒体根和1120配置。历史S3逐题记录无tensorhash，不宣称逐张量配对。

固定100节点85/100，S3同85，elapsed54.070秒；固定250节点210/250，S3同211，elapsed110.847秒；475节点391/475，elapsed196.523秒。每25题记录progress，末批约9.5秒。没有据前缀改阈值/提前停止或对DEV拟合。

运行前nvidia-smi无compute app；实际基座8文件（含16GB权重）与f2db权重SHA核验通过。固定tar SHA `b8c5b3184ea2d09e6bb1184c7065c27fd6d354022dcf0e37169e6e390f1d6516`；eval/template/readout/trainer/multimodal/token helper逐文件与tar匹配。父训练receipt SHA `0e29da0e72cba7bcf232593f3babb681f7581af794927398ab6a1b0c47d5b309`，S3协议receipt SHA `2539b353fb2f4e499d3609ad7b9a366f467ec381fe46d2b3bac0295ea4405690`。模型、数据、命令、源文件与环境hash详见JSON/receipt。

入口 `/root/autodl-tmp/evaluate-working-parent-image500-001.py`（本地同名helper SHA `54a55635e92ea373a9a5a268b81dea503d214987318b537a45964d12ced30bc3`），解释器 `/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python`，静态source `/root/autodl-tmp/s4-monitored-eval-code-001`（74e96e2）。由现有segment001-video500 wrapper派生，只改为原image协议，并追加实际文件绑定；ruff/format/py_compile及真实GPU全500验证通过。

远端完整工件保留于`/root/autodl-tmp/s1-text-lora-001/artifacts/s4-working-parent-image500-001`；逐题数据不进Git。remote evaluator原报告SHA `5c3ceab645197ed7527454df976740278c3ac5c68a7ed0fa97c60d4ee1401d98`，导出的本地JSON补充明确`stage_gate=false`与源报告SHA；receipt补充已实际观察的进度节点/配对汇总，并保留原receipt SHA。成本null。image单项没有显著整体退化，也不能替代最强允许教师比率/六模态/封存/外部榜门槛。
