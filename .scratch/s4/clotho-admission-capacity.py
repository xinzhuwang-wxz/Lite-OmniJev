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
family_pools = defaultdict(set); media_pools = defaultdict(set)
for artifact in manifest.samples:
    for line in bound_lines(artifact):
        sample = Sample.model_validate_json(line)
        extract(sample, artifact)
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
reachable = 1
for count in unallocated_counts.values(): reachable |= reachable << count
report = {'scope':'Clotho original-question admission capacity, no pool changes or final gold claim','inventory_sha256':manifest_sha,'source_metadata_sha256':meta_sha,'source_question_sha256':pins,'license_plan_sha256':hashlib.sha256(license_path.read_bytes()).hexdigest(),'distinct_original_questions_by_pool':dict(counts),'families_by_pool':{k:len(v) for k,v in families.items()},'rejected':dict(reasons),'fully_decoded_available_media':len(decoded),'unallocated_family_size_histogram':dict(Counter(unallocated_counts.values())),'can_extend_original_holdout_to_500_without_splitting_family':bool((reachable >> (500-counts['holdout'])) & 1),'elapsed_seconds':round(time.monotonic()-start,3),'cost_usd':None,'exit_code':0,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out = root/'reports/s4-clotho-admission-capacity-002.json'
with out.open('x') as h: json.dump(report,h,indent=2)
print(json.dumps(report))
