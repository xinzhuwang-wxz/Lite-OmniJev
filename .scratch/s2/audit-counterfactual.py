import hashlib
import json
import random
import subprocess
from pathlib import Path

root = Path('/root/autodl-tmp/s1-text-lora-001')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

results = []
for name in ['s2-dev-eval-image-001', 's2-dev-eval-screenshot-001', 's2-probe-v3-eval-image-001', 's2-probe-v3-eval-screenshot-001', 's2-probe-zeroshot-image-001']:
    report_path = root / 'reports' / (name + '.json')
    report = json.loads(report_path.read_text())
    command = report['command']
    data_path = root / command[command.index('--data') + 1]
    sidecar_path = root / command[command.index('--output-dir') + 1] / 'records.jsonl'
    assert digest(data_path) == report['data_sha256']
    assert digest(sidecar_path) == report['sidecar_sha256']
    samples = {s['id']: s for s in map(json.loads, data_path.read_text().splitlines())}
    records = list(map(json.loads, sidecar_path.read_text().splitlines()))
    originals = {(r['pass'], r['id']): r for r in records if r['variant'] == 'original'}
    rates = []
    for pass_index in range(report['passes']):
        matches = []
        for record in records:
            if record['variant'] != 'counterfactual' or record['pass'] != pass_index:
                continue
            sample = samples[record['id']]
            options = sample['questions'][0]['options']
            indices = list(range(len(options)))
            seed = int.from_bytes(hashlib.sha256(sample['id'].encode()).digest()[:8], 'big')
            random.Random(seed).shuffle(indices)
            permuted = [options[i] for i in indices]
            original = originals[(pass_index, sample['id'])]
            def content(letter, values):
                return 'U' if letter == 'U' else values[ord(letter)-ord('A')]
            assert content(original['target'], options) == content(record['target'], permuted)
            matches.append(content(original['prediction'], options) == content(record['prediction'], permuted))
        assert len(matches) == report['samples']
        rates.append(sum(matches)/len(matches))
    results.append({'source_report': str(report_path), 'source_report_sha256': digest(report_path), 'data_sha256': digest(data_path), 'sidecar_sha256': digest(sidecar_path), 'samples': report['samples'], 'passes': report['passes'], 'old_invalid_letter_match_rate': report['metrics']['counterfactual_content_match_rate'], 'corrected_content_match_rates': rates})
output = {'status': 'completed', 'exit_code': 0, 'method': 'CPU rescore of hash-verified original GPU prediction sidecars; deterministic permutation and target-content correspondence verified; original files unchanged', 'code_revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(), 'audit_script_sha256': digest(Path(__file__)), 'results': results}
(root/'reports/s2-counterfactual-rescore-001.json').write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps(output, indent=2))
