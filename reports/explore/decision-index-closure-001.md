# Decision Index 离线闭包只读审计 001

日期：2026-10-16。仅源码、文件名、顶层 `question_set` 元数据、Git root/HEAD 与字节哈希；未运行上游、未加载 pickle、未保留或解析题目/答案行、未下载缓存、未作许可准入判断。ready=false。

实测工具：Python 3.11.17；git version 2.34.1；tiktoken 0.14.0。
审计工具SHA256：`314231739f08e676228474e5f7bad1f420aa372ef99b92e75e93505eaaaef12f`。

## 实际位置

- materialized：`/root/autodl-tmp/s1-text-lora-001/artifacts/s4-index-work-001/{source,work,preparation.json}`
- input receipts：`/root/autodl-tmp/s1-text-lora-001/artifacts/s4-index-inputs-001/{manifest.json,report.json}`
- original quarantine：`/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3`

## 1. tiktoken：确实缺失精确缓存

materialized source 的 `decision_index/suite/build/adapters_retrieval.py:110` 与 `adapters_added.py:215` 都调用 `get_encoding("cl100k_base")`。
已安装 `/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/lib/python3.11/site-packages/tiktoken_ext/openai_public.py:75` 的 `cl100k_base()` AST 提取结果：

- URL：`https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken`
- expected SHA256：`223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`
- SHA1(URL)缓存文件名：`9b5ad71b2ce5302211f9c61530b329a4922fc6a4`
- `openai_public.py` SHA256：`954392738e60d0fb6dca1dad80872efc47c8e2733babecbbf0a23970ed66c2cb`
- `tiktoken/load.py` SHA256：`43c3cfb29dc48a74e70e0c85bb01cfd96a3de069a98de8778ccab7f72a2836cc`

实际不存在：`work/cache/tiktoken/9b5ad71b2ce5302211f9c61530b329a4922fc6a4` 与远端 `/tmp/data-gym-cache/9b5ad71b2ce5302211f9c61530b329a4922fc6a4`。另外只读 find `/tmp`、`/root/autodl-tmp`、`/root/.cache` 未发现该key或cl100k命名文件；本机 `/tmp/data-gym-cache`、`/Users/bamboo/.cache`、`/Users/bamboo/Library/Caches/data-gym-cache` 的有界扫描也无命中。不是对未扫描磁盘的全局不存在断言。

`load.py:35` 的 `read_file_cached` 依次采用 `TIKTOKEN_CACHE_DIR`、`DATA_GYM_CACHE_DIR`、临时目录/data-gym-cache；缓存缺失或哈希不符时会走 `read_file(URL)`。因此 HF offline环境变量不构成tiktoken网络隔离。
最小后续修复：只取得上述一个公开缓存文件，验证完整SHA256后以指定key放入 `work/cache/tiktoken/`，重建显式设置该 `TIKTOKEN_CACHE_DIR`，另施加真正网络隔离。本轮没有下载或写入远端。

## 2. ForecastBench：当前36个动态引用已闭合，latest symlink不是当前阻塞

源码 `decision_index/suite/build/adapters_scored.py` 的 `forecastbench()` 枚举 `datasets/resolution_sets/*_resolution_set.json`，从顶层 `question_set` 拼接 `datasets/question_sets/<name>`。
审计只捕获这个顶层字符串；嵌套题目和resolution值不构造对象、不保留。

- resolution文件数：36；实际所需question_set缺失集合：`[]`。
- `latest-llm.json`被引用：false。
- 原symlink：`latest-llm.json -> 2026-09-13-llm.json`；目标真实文件存在。
- 集合摘要SHA256：`1409db80effda12a0e790d29b815eb76287e1d802566d5adb2d1f3bf7b77e838`。
- 摘要算法：排序36个 `(resolution_file basename, question_set basename)` 二元组，`json.dumps(..., separators=(",", ":"))` UTF-8字节的SHA256。

因此不应为该未使用symlink改输入映射。未来若固定recipe/输入revision变动出现引用，再对允许根内的真实目标hash验证后复制为普通文件并明确记录派生来源，不能盲目跟随链接。

实际引用集合（仅文件名元数据）：

- `2024-07-21_resolution_set.json` → `2024-07-21-llm.json`
- `2025-03-02_resolution_set.json` → `2025-03-02-llm.json`
- `2025-03-16_resolution_set.json` → `2025-03-16-llm.json`
- `2025-03-30_resolution_set.json` → `2025-03-30-llm.json`
- `2025-04-13_resolution_set.json` → `2025-04-13-llm.json`
- `2025-04-27_resolution_set.json` → `2025-04-27-llm.json`
- `2025-05-11_resolution_set.json` → `2025-05-11-llm.json`
- `2025-05-25_resolution_set.json` → `2025-05-25-llm.json`
- `2025-06-08_resolution_set.json` → `2025-06-08-llm.json`
- `2025-06-22_resolution_set.json` → `2025-06-22-llm.json`
- `2025-08-03_resolution_set.json` → `2025-08-03-llm.json`
- `2025-08-17_resolution_set.json` → `2025-08-17-llm.json`
- `2025-08-31_resolution_set.json` → `2025-08-31-llm.json`
- `2025-10-26_resolution_set.json` → `2025-10-26-llm.json`
- `2025-11-09_resolution_set.json` → `2025-11-09-llm.json`
- `2025-11-23_resolution_set.json` → `2025-11-23-llm.json`
- `2025-12-07_resolution_set.json` → `2025-12-07-llm.json`
- `2025-12-21_resolution_set.json` → `2025-12-21-llm.json`
- `2026-01-04_resolution_set.json` → `2026-01-04-llm.json`
- `2026-01-18_resolution_set.json` → `2026-01-18-llm.json`
- `2026-02-01_resolution_set.json` → `2026-02-01-llm.json`
- `2026-02-15_resolution_set.json` → `2026-02-15-llm.json`
- `2026-03-01_resolution_set.json` → `2026-03-01-llm.json`
- `2026-03-15_resolution_set.json` → `2026-03-15-llm.json`
- `2026-03-29_resolution_set.json` → `2026-03-29-llm.json`
- `2026-04-12_resolution_set.json` → `2026-04-12-llm.json`
- `2026-04-26_resolution_set.json` → `2026-04-26-llm.json`
- `2026-05-10_resolution_set.json` → `2026-05-10-llm.json`
- `2026-05-24_resolution_set.json` → `2026-05-24-llm.json`
- `2026-06-07_resolution_set.json` → `2026-06-07-llm.json`
- `2026-06-21_resolution_set.json` → `2026-06-21-llm.json`
- `2026-07-05_resolution_set.json` → `2026-07-05-llm.json`
- `2026-07-19_resolution_set.json` → `2026-07-19-llm.json`
- `2026-08-02_resolution_set.json` → `2026-08-02-llm.json`
- `2026-08-16_resolution_set.json` → `2026-08-16-llm.json`
- `2026-08-30_resolution_set.json` → `2026-08-30-llm.json`

## 3. git_revision：七处父仓库继承，一处目录缺失

`layout.py:35`直接执行 `git rev-parse HEAD`，未验证Git根。调用点：`adapters_creative.py:80`；`adapters_scored.py:52,117`；`normalize_text.py:23,57,87,107,315`。

materialized七个目录无独立.git，实测Git向上发现 `/root/autodl-tmp/s1-text-lora-001`，返回父HEAD `869ee9592ed079d1b522f75ca6e994ada0c9fcfe`，不是数据来源revision。`searchless_chess`没有材料selector，整个materialized目录缺失，`git -C`直接失败。source源码包也会发现这个父仓库，不能把它当上游checkout。

以下八个original quarantine已分别验证 `--show-toplevel` 等于各自绝对root，且真实HEAD精确等于固定 `acquire.py:GIT` pin：

| Repo | 已验证真实Git root | HEAD / expected pin |
| --- | --- | --- |
| searchless_chess | `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3/repos/searchless_chess` | `90ae0e6b121673fc3079aaeffa047580bb600c0a` |
| habermas_machine | `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3/repos/habermas_machine` | `7923b71966c14136077c78d7841bc9e1a182dfe0` |
| pop909cl | `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3/repos/pop909cl` | `be9094392903c471a930519e1c0bacf8b6be5d62` |
| isarcasm | `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3/repos/isarcasm` | `dfc708b53bde1bb571abfb5692f63231c2232195` |
| sata | `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3/repos/sata` | `371dd0c18fe75a96fbbcf2d1507ceeaf0d5263c5` |
| contractnli | `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3/repos/contractnli` | `eced6528dd3c1d14d73f9a87df8f7bdbc03126f9` |
| nli4ct | `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3/repos/nli4ct` | `7f32fa6c7db43e577f22e9fc6c28eef1cc6d223e` |
| esci | `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3/repos/esci` | `7916cdf6ab75a462e77f20ab40428a10923998d5` |

最小后续方案有两个，均不得伪造.git或commit：

1. 优先显式artifact provenance映射：把上述真实root/HEAD、source文件哈希、materialized文件哈希绑定到独立receipt；在派生working source里精确hash-bound修改 `git_revision`，仅允许已登记的绝对materialized repo键，返回其已验证source commit，未知key/哈希不符直接报错。必须标明artifact来源，不能声称materialized目录是clean checkout。searchless的空材料路径也有真实source provenance记录，避免chdir到不存在目录。
2. 若保留原Git调用，复制这些已验证original checkout的真实Git metadata到独立work对应root，再检查work的 `--show-toplevel` 必须恰为该root且HEAD必须等于pin。复制前核对HEAD/refs/objects的实际哈希闭包、链接和alternates/commondir等外部依赖；不能创建一个冒充原checkout的伪commit。metadata copying不等于复制完整clean working tree，须如实标记部分materialized内容。

固定上游source以 `revision=f8552a13b643d39f61185f5951bb5fed16bedf51`、tar SHA256 `12719836807976e25e9ff5b5dee742679577b48508cbd601f7c57be9672f4082`证明archive内容；本次未证明它是Git checkout，报告应明确archive来源。不得用父HEAD或None冒充已验证revision。

## 可复执行的只读审计命令

从本工作树执行；通过stdin运行自有审计器，不在远端落工具文件：

```sh
ssh -o BatchMode=yes -o ControlPath=/tmp/omnijev-ssh-root-final -p48926 root@region-42.seetacloud.com \
  '/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python - --site-packages /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/lib/python3.11/site-packages' \
  < .scratch/s4/audit-index-offline-closure.py
```

stdout仅元数据/哈希JSON。stdin模式工具自身 `__file__`不可哈希，故该字段为null；本报告记录了本地精确工具SHA。工具没有运行get_encoding()、加载上游模块或触发下载。

边界：此闭环仅证明上述三项的当前状态；不能作为完整动态文件闭包、包依赖、网络沙箱、许可或重建成功证明。后续修复另行安排。
