"""Private Clotho candidate audit; export aggregates, never sample contents."""
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from data.schema import Sample
from scripts.gold_inventory import Manifest, ComponentRecord, bound_lines, extract
from scripts.s1_text_pools import canonical_family_hash
from scripts.s0_audio_dev import load_license_plan, load_metadata, load_train_groups, candidate_rows, FIXED_ARCHIVE_ALIASES, decode_wav_full

start = time.monotonic()
root = Path('/root/autodl-tmp/s1-text-lora-001')
source = Path('/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/clotho-aqa/6473207')
manifest_path = root / 'artifacts/s4-gold-inventory-001/with-clevrer-gold-manifest-001.json'
manifest_sha = '3bff87d13b41244636dacccf61114db363f26544bc759f9884c920c632482ea7'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_sha
manifest = Manifest.model_validate_json(manifest_path.read_bytes())
family_pools = defaultdict(set); media_pools = defaultdict(set); reserves=[]; source_candidates={}
for artifact in manifest.samples:
    for line in bound_lines(artifact):
        sample = Sample.model_validate_json(line)
        extract(sample, artifact)
        if artifact.exposure == 'current' and sample.meta.source_dataset == 'Clotho-AQA' and sample.meta.pool in {'calibration','holdout'}: reserves.append(sample)
        family_pools[canonical_family_hash(sample.meta.family_id)].add(sample.meta.pool)
        for media in sample.prefix.media:
            media_pools[media.sha256].add(sample.meta.pool)
for artifact in manifest.component_ledgers:
    for line in bound_lines(artifact):
        record = ComponentRecord.model_validate_json(line)
        family_pools[record.audio.family_hash].add(record.pool)
        media_pools[record.audio.media_sha256].add(record.pool)
quarantine_path = root / 'artifacts/s4-gold-inventory-001/quarantined-reserves/exclusions.json'
assert hashlib.sha256(quarantine_path.read_bytes()).hexdigest() == '8901483c170a7b1b1d04cf339eb4fbc6fc9e9722b8c2beb9cde0dc56b437ba7f'
quarantine = json.loads(quarantine_path.read_bytes())
meta_sha = '7505ba93c291d344b31e5c33e872b95d84255852cff93278bbdf549402acc1f5'
license_path = root / 'reports/clotho-aqa-license-plan.json'
approved = load_license_plan(license_path, metadata_sha256=meta_sha)
metadata = load_metadata(source / 'clotho_aqa_metadata.csv', metadata_sha256=meta_sha, approved=approved)
pins = {'train':'237c692e21e015a3e9b4391edd716b1387d7eb2d068d7ecdb5a6d6ead7e7a992','val':'c69b32095d70740cee49d66dc33690d0267cf7bc821d71d834725f9d24c0a491','test':'6b72ea4722e2a4dde53465f9a0c165c901b16a2fb4ef16356d37891f71496737'}
counts = Counter(); families = defaultdict(set); seen=set(); decoded={}; reasons=Counter(); unallocated_counts=Counter()
for split, sha in pins.items():
    groups = load_train_groups(source / f'clotho_aqa_{split}.csv', train_sha256=sha)
    candidates, rejected, headers = candidate_rows(groups=groups, metadata=metadata, audio_root=source/'approved-audio-002', aliases=FIXED_ARCHIVE_ALIASES)
    reasons.update(rejected)
    for item in candidates:
        family = canonical_family_hash(item['sound_id']); media = item['header']['sha256']
        source_candidates[(family,item['question'])] = item
        identity = (family, item['question'])
        if identity in seen:
            continue
        seen.add(identity)
        if family in quarantine['families'] or media in quarantine['media']:
            reasons['quarantined'] += 1; continue
        pools = family_pools[family] | media_pools[media]
        if len(pools) > 1:
            reasons['historical_pool_conflict'] += 1; continue
        pool = next(iter(pools)) if pools else 'unallocated'
        if pool in {'train','dev'}:
            counts[pool] += 1; continue
        if media not in decoded:
            decoded[media] = decode_wav_full(item['media_path'])
        counts[pool] += 1; families[pool].add(family)
        if pool == 'unallocated': unallocated_counts[family] += 1
summary=Counter()
for sample in reserves:
    candidate=source_candidates.get((canonical_family_hash(sample.meta.family_id),sample.questions[0].instruction))
    summary['current_reserve_items']+=1
    if candidate is None:
        summary['missing_original_question']+=1
        continue
    summary['original_media_hash_matches']+=int(sample.prefix.media[0].sha256 == candidate['header']['sha256'])
    summary['original_answer_matches']+=int(sample.questions[0].target.value == candidate['answer'])
print(json.dumps({'scope':'current reserve vs original source binding','counts':dict(summary)}))
