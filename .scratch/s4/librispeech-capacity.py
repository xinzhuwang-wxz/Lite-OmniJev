"""Inspect candidate public archives privately, exporting counts and hashes only."""
import hashlib
import json
import tarfile
import time
from collections import Counter
from pathlib import Path

start = time.monotonic()
root = Path('/root/autodl-tmp/lite-omnijev/source-audit/librispeech-s4-candidate-001/mirror-002')
expected = {'dev-clean': '42e2234ba48799c1f50f24a7926300a1', 'test-clean': '32fa31d27d2e1cad72775fee3f4849a9'}
report = {'scope': 'candidate_source_capacity_only_no_gold_admission', 'source': 'https://www.openslr.org/12/', 'license_declared': 'CC-BY-4.0', 'splits': {}, 'pool_allocation': 'not_performed', 'historical_isolation': 'not_validated', 'exit_code': 0}
identities = {}
for split, md5 in expected.items():
    path = root / (split + '.tar.gz')
    with path.open('rb') as h:
        assert hashlib.file_digest(h, 'md5').hexdigest() == md5
    with path.open('rb') as h:
        sha = hashlib.file_digest(h, 'sha256').hexdigest()
    transcripts = {}; chapters = {}; audio = {}; speakers = set(); books = set()
    counts = Counter()
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.name.endswith('/CHAPTERS.TXT'):
                h = archive.extractfile(member); assert h is not None
                for line in h.read().decode().splitlines():
                    if not line.strip() or line.startswith(';'):
                        continue
                    fields = [v.strip() for v in line.split('|')]
                    chapters[fields[0]] = fields[5]
            elif member.name.endswith('.trans.txt'):
                h = archive.extractfile(member); assert h is not None
                for line in h.read().decode().splitlines():
                    ident, text = line.split(' ', 1)
                    assert ident not in transcripts
                    transcripts[ident] = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
            elif member.name.endswith('.flac'):
                h = archive.extractfile(member); assert h is not None
                payload = h.read()
                assert payload[:4] == b'fLaC' and payload[4] & 127 == 0
                assert int.from_bytes(payload[5:8], 'big') == 34
                stream = int.from_bytes(payload[18:26], 'big')
                rate = stream >> 44
                channels = ((stream >> 41) & 7) + 1
                duration = (stream & ((1 << 36) - 1)) / rate
                ident = Path(member.name).stem
                assert ident not in audio
                audio[ident] = (hashlib.sha256(payload).hexdigest(), duration, rate, channels)
    assert set(audio) == set(transcripts)
    for ident, (_, duration, rate, channels) in audio.items():
        speaker, chapter, _ = ident.split('-')
        speakers.add(speaker); books.add(chapters[chapter])
        counts['utterances'] += 1
        counts['within_30s_16khz_mono'] += int(0 < duration <= 30 and rate == 16000 and channels == 1)
    identities[split] = {'speakers': speakers, 'books': books, 'media': {v[0] for v in audio.values()}, 'text': set(transcripts.values())}
    report['splits'][split] = {'archive_sha256': sha, 'official_md5_verified': True, **dict(counts), 'speakers': len(speakers), 'books': len(books), 'chapters': len({i.split('-')[1] for i in audio}), 'unique_media': len(identities[split]['media']), 'unique_transcripts': len(identities[split]['text'])}
report['cross_split_overlap'] = {key: len(identities['dev-clean'][key] & identities['test-clean'][key]) for key in ('speakers', 'books', 'media', 'text')}
report['elapsed_seconds'] = round(time.monotonic() - start, 3)
report['cost_usd'] = None
out = root / 'capacity-001.json'
with out.open('x') as h:
    json.dump(report, h, indent=2)
print(json.dumps(report))
