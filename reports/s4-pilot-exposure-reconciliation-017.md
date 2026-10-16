# S4 video-specialist 单段曝光核对 017

成功工件为 artifacts/s4-pilot-exposure-017-retry-001：真实CPU exit0，29.627秒，完整inventory审计isolated；73份已知更新报告，unknown0。
原 artifacts/s4-pilot-exposure-017 拒绝receipt保留，同步为本目录017-failed-001.json。
首次在0.0007秒、inventory前因分支仍比较015 pin而拒绝；现已按016 pin修复并加回归，未重跑GPU或覆盖失败证据。

- specialist是独立原CE/10步/accum12/f2db初始化，不继承continuous优化器，不新增loss。
- 60条存储样本与native原官方60视频题canonical完全相等、ledger精确绑定；无replay、无新增family。
- seed42固定顺序循环两遍，真实训练120 draw/60 unique family，每题恰两次。存储样本数和实际曝光数分开记录。
- 复用已有native official-source校验函数，没有复制另一套官方来源审计；prepared/data/config/ledger/helper/训练archive SHA强绑定。
- 016全部all_bound_input_sha256逐项原样保留，包括旧failed001/recovered002链；continuous两段immutable snapshots完全相等，仍只记其actual240 draw。
- 原native actual120保留；specialist另记actual120/60storage/newfamilies0，未把池容量1200当已曝光。
- 唯一current TRAIN为specialist runtime data.jsonl，旧native与continuous转为history。
- 未运行GPU，未读私有样本内容；完整inventory只由现有aggregate audit返回汇总。资源cap360秒是root训练的资源校准，不改变10步或质量门槛。

成功report SHA：5b3d3a26bfad2f35dc85e3f166a5d77f3c0996469fa6cbd0cbe962aa98b7e8b6。
上一016 report SHA：0164a2cdf0db9381cc76ae7411bd99f459789f4ae057c1cdfbf89d9276870aae。
prepared report SHA：af13cfecc2ca1a27985c5fe25528b0b12578e2c5ddc66d517a86e14f77fc1942。
data SHA：11d453d3521646df888c7899b6a0a67ad4a5606d956251c96d69b41e73b0cdbe。
config SHA：a03ca621bfbec82fea12322ce2a70a95c24ec128c5d1d9a0749dcad9a977ec88。
ledger SHA：2b9932604d76c2ffdd8aa15501c7a3670afb4f3bf3ec8e345bdd98ebc10a1f23。
训练adapter SHA：dfd74a267d23b41c890e8f29976758fe7d8b71f00bb77b0f31784b46f549fd87。

Python /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python；
cwd/PYTHONPATH /root/autodl-tmp/s4-claim-pair-code-001；
helper /root/autodl-tmp/s4-exposure017-code-001/reconcile-pilot-exposure.py。
首CLI --video-specialist 输出原017 rejected/exit1；修复后通过同一prepare入口，
root=/root/autodl-tmp/s1-text-lora-001、legacy=/root/autodl-tmp/lite-omnijev，
output=artifacts/s4-pilot-exposure-017-retry-001，video_specialist=True，真实exit0。

验证：34个相关曝光CPU测试通过（specialist9/native9/continuous16），helper/new tests lint与format通过；
真实完整审计、016全绑定全等、两份snapshot全等、unique current与storage/draw/new-family计数均通过。
这不是质量晋级或S4通过证明；后继准备须使用成功retry目录及上述report SHA。
