# Decision Index 显式artifact revision映射补丁准备 001

范围：只准备可审查patch bundle；没有安装补丁、修改materialized source/work、复制Git metadata、创建伪commit、运行上游rebuild、使用GPU、加载pickle或改变许可准入。基线为主树29a2c36；既有闭包事实见 `reports/explore/decision-index-closure-001.md`。

## 选择与调用面

固定上游revision：`f8552a13b643d39f61185f5951bb5fed16bedf51`。
原archive SHA256：`12719836807976e25e9ff5b5dee742679577b48508cbd601f7c57be9672f4082`。
原 `decision_index/suite/build/layout.py` SHA256：`21d36d35cf8c5369cadbd2f509e1dc508b6dae3ec3d408447643895ffade38fb`。
原 `acquire.py` SHA256：`94bc9997f9c95ae33ed9d2d00e6c715bd1408b69ecb64de9e31e06317b761c16`。

只有layout.py的 `git_revision(repo)` 需要派生修改。它被以下八个数据repo调用：

- `adapters_scored.py`：searchless_chess、habermas_machine。
- `adapters_creative.py`：pop909cl。
- `normalize_text.py`：isarcasm、sata、contractnli、nli4ct、esci。

保留原调用签名和返回字符串。不会调用Git作为fallback，因此即使materialize置于父仓库869ee95之下，也不会继承父HEAD。无需为了searchless_chess创建一个虚假的checkout：它的实际输入bag位于downloads，revision记录来自已验证原真实Git根。

## 独立helper接口

`scripts/decision_index_provenance.py`只在一个全新目录写：

1. `revision-map.json`：每个精确materialized repo绝对路径对应原真实Git root、HEAD/pinned revision及相关source/materialized文件SHA。
2. `layout.py`：明确标记派生版，内嵌上述映射完整SHA；原layout不被修改。
3. `receipt.json`：同时记录original_source_sha256、derived_source_sha256、revision_map_sha256、archive来源边界和工具/耗时。

helper先验证preparation.json、input manifest SHA及固定archive实际SHA；从固定SHA的acquire.py只做AST literal读取GIT/HTTP常量，不执行该模块。每个原repo必须拥有真实独立.git目录，`--show-toplevel`精确等于其root，实际HEAD等于acquisition pin且存在commit object。Git环境覆盖被移除，关闭提示、lazy fetch及网络协议；不会fetch或checkout。

原始材料和materialized副本均核验SHA。正常Git-root材料与pinned HTTP材料分别标识；habermas parquet和searchless bag不能冒称原Git working-tree内的文件。Git HEAD验证不等于clean working-tree证明，报告明确以材料哈希独立绑定，不宣称clean checkout。

派生 `git_revision` 必须显式获得 `DECISION_INDEX_PROVENANCE_MAP`；逐次验证map SHA、精确repo键和相关materialized文件SHA。缺环境、未知repo、symlink路径或字节变化均拒绝；不返回None、不调用Git、不静默继承父仓库。source自身只宣称固定tar archive来源，不宣称上游checkout。

## 后续CPU bundle生成命令

在包含新helper的已验证源码快照中，由主会话执行：

```sh
python -m scripts.decision_index_provenance \
  --prepared-root /root/autodl-tmp/s1-text-lora-001/artifacts/s4-index-work-001 \
  --original-root /root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3 \
  --input-manifest /root/autodl-tmp/s1-text-lora-001/artifacts/s4-index-inputs-001/manifest.json \
  --output-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s4-index-provenance-bundle-001
```

输出目录已存在则拒绝。当前未执行该远端命令，因此本报告不伪报真实bundle SHA或可重建状态。source_patch_is_original_upstream=false，ready=false。

安装与重建另行安排。安装前必须复核receipt中的原layout SHA和派生layout SHA、map SHA，保留原archive/原layout证据；采用派生layout后，在同一进程环境显式设置其map绝对路径。不能将patched recipe报告为原版未修改。

## 验证与剩余条件

针对性CPU验证包含真实临时Git repo的root/HEAD检查、没有自身.git的子目录拒绝（即使Git自身会发现父HEAD）、generated lookup无Git回退、未知repo/映射/材料/链接拒绝、缺失searchless目录不创建metadata、原layout不变及真实固定layout补丁seam检查。测试只执行自行生成的查找函数，没有执行上游模块或构建。

剩余：真实远端bundle生成与独立核验、派生patch安装、精确tiktoken缓存和真正网络隔离、完整依赖/输入闭包、明确已有许可准入、官方独立rebuild/verify。本patch只解决来源revision陷阱，不构成这些事项的通过证据。

本地最终验证：`pytest tests/test_decision_index_provenance.py tests/test_decision_index_offline.py -q`共27项通过；其中完整fixture bundle使用8个真实临时Git根，验证原source/work不变、未创建materialized.git、map/派生source SHA以及700/600权限。fixture不是实际来源准入证据。ruff check/format、helper mypy与CLI help通过。
