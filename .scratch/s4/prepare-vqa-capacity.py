import json,os
from pathlib import Path
from scripts.vqa2_gold_capacity import PINS
root=Path('/root/autodl-tmp/s1-text-lora-001/artifacts/s4-gold-inventory-001')
source=Path('/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine')
paths={'inventory':root/'with-audio-gold-manifest-001.json','quarantine':root/'quarantined-reserves/exclusions.json','coco_metadata':source/'stage2-sources/coco2017/annotations/captions_train2017.json','aok_train':source/'stage2-sources/aokvqa/extracted/aokvqa_v1p0_train.json','train_annotations':source/'vqa2-annotations-20261015/v2_Annotations_Train_mscoco.zip','train_questions':source/'vqa2-annotations-20261015/v2_Questions_Train_mscoco.zip'}
config={k:{'path':str(v),'sha256':PINS[k]} for k,v in paths.items()}
config['media_roots']=[str(source/'stage2-sources/aokvqa/coco-ccby-publicdomain-images/train')]
fd=os.open(root/'vqa-capacity-config-001.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as h:json.dump(config,h,sort_keys=True)
