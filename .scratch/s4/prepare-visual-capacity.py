import json
from pathlib import Path
root=Path('/root/autodl-tmp/s1-text-lora-001/artifacts/s4-gold-inventory-001')
config={'inventory':{'path':str(root/'with-clevrer-gold-manifest-001.json'),'sha256':'3bff87d13b41244636dacccf61114db363f26544bc759f9884c920c632482ea7'},'quarantine':{'path':str(root/'quarantined-reserves/exclusions.json'),'sha256':'8901483c170a7b1b1d04cf339eb4fbc6fc9e9722b8c2beb9cde0dc56b437ba7f'},'plan':{'path':'/root/autodl-tmp/lite-omnijev/source-audit/stage3-source-audit/gui-odyssey/app-split-train-media-plan-under-19_5gb.json','sha256':'661a56214ee442ae0c694b21c41b374fc02dad0ea1a962bb8d9638b3d393c0c4'},'annotation_root':'/root/autodl-tmp/s0-gui-dev-001/artifacts/annotations','media_root':'/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/gui-odyssey-source-audit/media/app_split_train_under19_5gb'}
with (root/'visual-capacity-config-001.json').open('x') as h: json.dump(config,h)
