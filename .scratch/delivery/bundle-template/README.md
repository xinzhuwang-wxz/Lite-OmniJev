# 三档发布包布局模板

这是目录/绑定/缺口模板，不是已发布模型或可启动包。API、single-card、edge各有独立manifest，复用现有release-inventory校验器。`available`只用于两份实际模板文件；模型、校准、量化、启动/回滚与验收文件均`pending`，大小/hash不编造。

克隆Git后空目录可能不存在；按manifest重建实际目录，复制到**新的**工作目录（存在即拒绝）：

```sh
python - <<'PY'
from pathlib import Path
import json, shutil
source = Path('.scratch/delivery/bundle-template')
target = Path('.scratch/delivery/bundle-draft')
shutil.copytree(source, target)
for manifest in target.glob('*/manifest.json'):
    for entry in json.loads(manifest.read_text())['files']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('invalid package-relative path')
        (manifest.parent / relative).parent.mkdir(parents=True, exist_ok=True)
PY
```

对三档逐一运行，**当前每档应exit1 / inventory_incomplete**：

```sh
PYTHONPATH=. python .scratch/delivery/release-readiness/check_inventory.py \
  --manifest .scratch/delivery/bundle-template/api/manifest.json \
  --root .scratch/delivery/bundle-template/api
PYTHONPATH=. python .scratch/delivery/release-readiness/check_inventory.py \
  --manifest .scratch/delivery/bundle-template/single-card/manifest.json \
  --root .scratch/delivery/bundle-template/single-card
PYTHONPATH=. python .scratch/delivery/release-readiness/check_inventory.py \
  --manifest .scratch/delivery/bundle-template/edge/manifest.json \
  --root .scratch/delivery/bundle-template/edge
```

真实产物到位后替换`bindings.template.json`为版本化bindings，写明最终decision weights/base/template/token-map/按模态×原语calibration/runtime/config/目标设备/200题协议/验收receipt SHA；保持model、calibration、gateway一同打版本和回滚。每个实际文件必须改为available并填准确`size`和`actual_sha256`。目录型模型需要逐文件清单，不能仅用目录存在代替权重与projector哈希。不可把模板校验或工作候选当最终模型通过；即使清单全available，checker仍只证明文件完整性。

edge支持目标、运行时ABI、视觉/音频/完整legal logits与实测设备表在最终版本冻结。模板不强制新增Android APK/Gradle或iOS签名；不能因此略过真正所支持目标设备的离线、内存、时延及模态验收。
