import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
base = Path("/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine")
meta = base / "stage2-sources/coco2017/annotations/captions_train2017.json"
d = json.loads(meta.read_text())
assert (
    next(x for x in d["licenses"] if x["id"] == 4)["url"].rstrip("/")
    == "http://creativecommons.org/licenses/by/2.0"
)
aok = base / "stage2-sources/aokvqa/extracted/aokvqa_v1p0_train.json"
# Exclude the entire original AOK source, not only train: preserves all existing image pools.
excluded = {x["image_id"] for x in json.loads(aok.read_text())}
images = {x["id"]: x for x in d["images"] if x["license"] == 4 and x["id"] not in excluded}
p = base / "vqa2-annotations-20261015/v2_Annotations_Train_mscoco.zip"
with zipfile.ZipFile(p) as z:
    anns = json.loads(z.read("v2_mscoco_train2014_annotations.json"))["annotations"]

counts = Counter()
families = set()
candidates = []
colors = {
    "black",
    "white",
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
}
for a in anns:
    if a["image_id"] not in images:
        continue
    answers = {" ".join(x["answer"].casefold().split()).rstrip(".?!") for x in a["answers"]}
    if len(a["answers"]) != 10 or len(answers) != 1:
        continue
    answer = next(iter(answers))
    category = (
        "yes_no"
        if answer in {"yes", "no"}
        else "color"
        if answer in colors
        else "integer"
        if answer in {str(i) for i in range(21)}
        else "other"
    )
    counts[category] += 1
    families.add(a["image_id"])
    if category != "other":
        candidates.append(
            {"image_id": a["image_id"], "question_id": a["question_id"], "category": category}
        )
report = {
    "scope": "new-family feasibility only; no training admission or media download",
    "approved_license": "CC-BY-2.0",
    "coco_split": "train2017",
    "vqa_split": "train2014",
    "excluded_all_aok_train_families": len(excluded),
    "eligible_new_images": len(images),
    "unanimous_questions": dict(counts),
    "unanimous_families": len(families),
    "restricted_candidate_families": len({r["image_id"] for r in candidates}),
    "input_sha256": {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in [meta, aok, p]},
}
(root / "reports/s2-new-family-feasibility-001.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report))
