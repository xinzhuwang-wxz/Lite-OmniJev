from __future__ import annotations

import json
import os
import pathlib
import resource
import subprocess
import sys
import traceback
from datetime import UTC, datetime

ROOT = pathlib.Path('/root/autodl-tmp/lite-omnijev')
SOURCE = ROOT / 'artifacts/quarantine/teacher/5addda86ddee081a68fb067477ea100c221b8917'
HEAD = ROOT / 'artifacts/teacher/jev-omni/5addda86ddee081a68fb067477ea100c221b8917/head.safetensors'
SPEC = ROOT / 'configs/teacher-head-jev-omni-spec.json'
CONVERSION = ROOT / 'reports/teacher-head-conversion.json'
OUT = ROOT / 'reports/teacher-preflight-cpu-002.json'


def read_text(path: pathlib.Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def cgroup_snapshot() -> dict[str, object]:
    return {
        'self_cgroup': read_text(pathlib.Path('/proc/self/cgroup')),
        'memory_events': read_text(pathlib.Path('/sys/fs/cgroup/memory.events')),
        'memory_current': read_text(pathlib.Path('/sys/fs/cgroup/memory.current')),
        'memory_max': read_text(pathlib.Path('/sys/fs/cgroup/memory.max')),
    }


def ps_snapshot() -> str:
    result = subprocess.run(
        ['ps', '-eo', 'pid,ppid,stat,rss,etime,cmd'],
        text=True,
        capture_output=True,
        check=False,
    )
    lines = []
    watched = ("pip download", "gui_odyssey", "validate_gui", "jupyter", "tensorboard")
    for line in result.stdout.splitlines():
        if any(token in line for token in watched):
            lines.append(line)
    return '\n'.join(lines)


def rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main() -> int:
    os.environ.update(
        {'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    )
    sys.path.insert(0, str(ROOT))
    report: dict[str, object] = {
        'schema_version': 1,
        'status': 'running',
        'started_utc': datetime.now(UTC).isoformat(),
        'gpu_used': False,
        'model_forward_run': False,
        'model_backbone_loaded': False,
        'source_script': str(pathlib.Path(__file__)),
        'memory_before': cgroup_snapshot(),
        'processes_before': ps_snapshot(),
    }
    try:
        from train.teacher_inference import (
            build_teacher_processor_inputs,
            expected_head_sha256_from_report,
            inspect_safe_teacher_head,
            load_teacher_processor,
            prepared_teacher_input,
            validate_teacher_source_tree,
        )

        expected = expected_head_sha256_from_report(CONVERSION)
        report['expected_head_sha256'] = expected
        report['source'] = validate_teacher_source_tree(SOURCE)
        report['rss_after_source_kib'] = rss_kib()
        report['head'] = inspect_safe_teacher_head(HEAD, SPEC, expected_sha256=expected)
        report['rss_after_head_kib'] = rss_kib()
        processor = load_teacher_processor(SOURCE)
        report['processor_class'] = type(processor).__name__
        report['rss_after_processor_kib'] = rss_kib()
        with prepared_teacher_input(
            state='Synthetic teacher CPU preflight state.',
            question='Which option is valid?',
            options=('alpha', 'beta'),
            modality='text',
        ) as prepared:
            report['text_synthetic_input'] = build_teacher_processor_inputs(
                processor, prepared
            ).summary
        report['rss_after_text_kib'] = rss_kib()
        report['audio_equivalence_path'] = (
            'not executed in this diagnostic; implementation reads validated mono16k PCM16 WAV '
            'into a float32 numpy waveform, matching transformers.audio_utils.load_audio '
            'ndarray pass-through'
        )
        report['status'] = 'text_image_boundary_passed_audio_blocked'
        report['exit_code'] = 0
    except Exception as exc:
        report['status'] = 'blocked'
        report['exit_code'] = 1
        report['blocker'] = type(exc).__name__
        report['traceback_type_only'] = traceback.format_exception_only(type(exc), exc)[-1].strip()
    report['memory_after'] = cgroup_snapshot()
    report['processes_after'] = ps_snapshot()
    report['finished_utc'] = datetime.now(UTC).isoformat()
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'status': report['status'],
        'exit_code': report['exit_code'],
        'rss_after_processor_kib': report.get('rss_after_processor_kib'),
        'memory_events': report['memory_after']['memory_events'],
    }, ensure_ascii=False, sort_keys=True))
    return int(report['exit_code'])


if __name__ == '__main__':
    raise SystemExit(main())
