# api 发布目录模板（不可运行/不可发布）

此目录只冻结包布局与缺口。manifest中pending项没有伪造大小/hash；把真实产物加入对应路径并核SHA后，才可改为available。manifest校验只证明文件完整性，不代替S4/S5/S6/FINAL。

`bindings.template.json`保留未冻结字段为null；它的文件hash仅证明模板未改，不表示模型或校准已绑定。复制时先改release_version并生成真实bindings文件，再替换manifest中的模板条目。不要用f2db等工作候选填作最终模型。

支持矩阵、目标设备与启动入口仍pending。此模板不创建Android应用、不选择新引擎、不含权重/数据/密钥。
