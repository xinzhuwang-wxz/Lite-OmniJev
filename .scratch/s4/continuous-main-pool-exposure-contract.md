# 连续固定池曝光核对接口

CPU实现，尚未对live训练跑014；训练终态后调用：

```sh
python reconcile-pilot-exposure.py --root /root/autodl-tmp/s1-text-lora-001 \
  --legacy-root /root/autodl-tmp/lite-omnijev \
  --output-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-014 \
  --continuous-main-pool \
  --continuous-launch-report /root/autodl-tmp/s1-text-lora-001/reports/s4-continuous-main-pool-launch-001.json
```

launch报告保持raw训练终态status/exit_code，selected_init_adapter为原f2db identity，
parent_selection=root_selected_f2db_balanced_retention_and_full_video_evidence。
exposure_binding字段：

- base_exposure_report、prepared_report、config_file、runtime_data、final_report、runtime_report_snapshot：均为绝对path与sha256对象。
- runtime_data必须是同一训练输出目录data.jsonl，不能用prepared train.jsonl代替。
- runtime_report_path：共享训练输出目录run.json绝对路径。
- source_files：至少trainer.py、checkpoint.py、train_s1.py的path/sha256；后继invocation固定相同列表。
- start_step/end_step：累计全局步数，首段0/10，后继10/20；一次最多新增10步。
- resume_from：首段null，后继必须是上一paused记录的同一完整checkpoint目录。

named final和runtime snapshot每段独立不可变。snapshot必须复制raw run.json字节；final只允许scripts.train_s1额外增加verified_train=1200的allocation_ledger_check，其余字段与raw完全相等。
永久all_bound_input_sha256只绑定snapshot，不绑定未来会覆写的rawrun；rawrun当次前后仍须匹配snapshot。

第一段从完整013 SHA5872a667…起，后续将上一014/015 report作为base；保留所有旧绑定，禁止跳段、重新初始化优化器或改global config100/sample1200。
paused>0被发现为update-bearing，既有legacy pilot终态默认仍不接受paused。
completed仅允许累计100步，paused只能<100步。failed partial steps>0同样记账；缺最终adapter/checkpoint允许保留失败曝光，不能拿failed继续exact-resume。

prepared report/data/config固定SHA，canonical1200 TRAIN样本、原f2db权重/config、runtime/final/snapshot、完整checkpoint config/model/assets/data/order与model/optimizer/RNG三tensorSHA均强绑定。
只追加一次current1200池；每段actual exposure按非重叠完成optimizer block区间记，10→20只有新增120draw、累计240，不相加10+20。
failed可能已读取下一未完成block，明确标partial_step_draws_unknown及最多额外12draw上界；不冒充精确已读数量。

新分支保持旧reports全glob、所有artifacts/run.json、legacy training/run.json和base持久JSON历史发现；未知update-bearing数据SHA直接拒绝。
每次只在终态后执行一次完整inventory审计，只回汇总，私有池内容不向执行者展示。

CPU验证：16新测试覆盖暂停边界、累计步数、两段递推/单池、optimizer/RNG篡改、原parent/global config、未知paused分支、旧snapshot篡改、failed无adapter/checkpoint、CLI ledger摘要兼容。
合成递推测试mock隔离审计汇总以专测编排，不替代随后真实014审计。既有reconcile测试回归全部通过。
