# S5 校准前置：028 fresh UID inventory审计

真实固定broker执行exit0，19.055620秒，outcome=isolated，ordinary_identity_boundary=true，categories=[]。本记录是独立普通身份边界receipt，补齐root-only028曝光审计不能代替的S5前置证据；没有运行校准producer或GPU。

原始manifest：/root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-028/inventory-manifest.json。
Source SHA807003d0dfc3443f69f3d992566c62d60e7e7824db2655e1347e8bb9d546d6b8，执行前与028 report相符，执行后再次全等。原source/private路径及文件没有修改；trusted root supervisor只建立新的只读快照供固定worker审计，agent未读取私有样本或raw日志。

Public receipt：/root/autodl-tmp/s1-text-lora-001/reports/s5-calibration-inventory-001.json。
完整SHAe14f7dd48d07bffe779ea1998e0e17bdb044a70438255d930173aa5241aa6ea0。
Translated manifest SHA3e71c3b11e8b9d029d3488cb14eb9689aea4999aada693976addc6b5862362ca。
Code fingerprint SHA38e8ab7d41a076f8057734ce95e7f5df42eba4c934b21337bf90f897e3cc77eb。
Config fingerprint SHA42f9a0a512c6e115de41350f8cbab6a0b1c6103e3689151a8e0ff07263c6a5bb。

实际cwd/immutable source=/root/autodl-tmp/s5-calibration-code-daeaa7f。命令：

```bash
/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python -m scripts.private_gold_job \
  --job inventory-audit \
  --inventory /root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-028/inventory-manifest.json \
  --sha256 807003d0dfc3443f69f3d992566c62d60e7e7824db2655e1347e8bb9d546d6b8 \
  --work-root /var/tmp/omnijev-gold-jobs-001 \
  --report /root/autodl-tmp/s1-text-lora-001/reports/s5-calibration-inventory-001.json \
  --worker-uid 64101 --probe-uid 64102 --timeout 300
```

64101/64102在passwd/group及permanent lease预检未占用；本次真实作业已永久占用两份lease。原work-root及既有leases保留，不得重用本次身份。没有ACL/运行时权限放宽，fork固定inventory路径在现有0700 runtime约束下成功。

Fresh后验验证：按实际新源PublicReceipt契约解析；outcome/exit/boundary/categories/source SHA均满足，code_bytes指纹与receipt一致，两个lease实际存在。仅复制public aggregate receipt入Git；私有run树和日志保留原处未读。本receipt不声称namespace/network隔离、gold来源完整性或质量门槛通过，具体边界字段原样保留在JSON。
