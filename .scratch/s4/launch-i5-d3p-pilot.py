"""I5->D3P pilot launcher: self-check the chain, print the full command set.

CPU only. Chain: teacher cache (OneJev-27B, GPU by main) -> D3P student row
conversion (CPU) -> 240s student slice (GPU by main) -> report-gated DEV50 and
video150 evaluation. Soft targets are the teacher's conditional legal
distribution — the existing single-token readout protocol, not a U product
feature (U-centric arms are cancelled per ADR0024).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCAL = REPO / "artifacts/s4-ensemble-oracle-001"
PYTHON = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"
ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
DATA = ROOT / "artifacts/s4-ensemble-oracle-001"
MODEL_ROOT = "/root/autodl-tmp/lite-omnijev/artifacts/models"
E4B = f"{MODEL_ROOT}/google__gemma-4-E4B-it/ee0ef6023621cff504d75826d4e04895a5af4a2"
ASSET_MANIFEST = "/root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json"
INIT_ADAPTER = (
    "/root/autodl-tmp/s1-text-lora-001/artifacts/s4-program-stratified-pool-train-001/"
    "adapter-invocation-step-0100"
)
INIT_SHA = "e46550143b28f8798484515f0abe21996fe92ad241625a4cf6da15dc093277fd"
SOURCE_LEDGER = (
    "/root/autodl-tmp/s1-text-lora-001/artifacts/s3-pools-video-001/run-a/allocation-ledger.jsonl"
)
MEDIA_ROOT = "/root/autodl-tmp/s3-media-candidates-001/video/media"
VIDEO150 = ROOT / "artifacts/s3-stable-probe-001/video-dev150.jsonl"
SLICE_SHA = "6b224886ac62a36d8b67d53158dd896b9f3c4b39bdd74c4551a0cba8d571c94a"
OUT = ROOT / "artifacts/s6-d3p-student-pilot-001"

STUDENT_CONFIG = {
    "seed": 42,
    "sample_count": 150,
    "steps": 40,
    "accumulation": 1,
    "max_length": 4096,
    "learning_rate": 2e-6,
    "rank": 16,
    "alpha": 32,
    "dtype": "bfloat16",
    "modality": "video",
    "video_num_frames": 16,
    "media_roots": [MEDIA_ROOT],
    "expected_data_version": "clevrer-train-video-noul-s3-pools-v1+d3p-teacher-soft-v1",
    "checkpoint_every": 5,
    "preparation_mode": "lazy",
    "preparation_cache_bytes": 134217728,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def self_test() -> dict:
    results = {}
    # 1. slice rule determinism from the raw pool
    pool_rows = [
        json.loads(line)
        for line in (LOCAL / "clevrer-train-pool.jsonl").read_text().splitlines()
        if line.strip()
    ]
    videos = [r for r in pool_rows if r["meta"]["modality"] == ["video"]]
    ranked = sorted(videos, key=lambda r: hashlib.sha256(("i5:d3p:" + r["id"]).encode()).hexdigest())
    selected = ranked[:150]
    results["slice_rule_reproduced"] = [r["id"] for r in selected] == [
        json.loads(line)["id"] for line in (LOCAL / "i5-slice-150.jsonl").read_text().splitlines()
    ]
    results["slice_sha_ok"] = sha256_file(LOCAL / "i5-slice-150.jsonl") == SLICE_SHA
    # 2. converter on a synthetic teacher record fixture
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fixture = {
            "id": selected[0]["id"],
            "pass_index": 1,
            "conditional_without_u": {"yes": 0.91, "no": 0.09},
            "p_U": 1e-5,
            "prediction": "yes",
        }
        (tmp_path / "teacher.jsonl").write_text(json.dumps(fixture) + "\n")
        (tmp_path / "slice.jsonl").write_text(json.dumps(selected[0]) + "\n")
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                str(Path(__file__).with_name("build_d3p_student_rows.py")),
                "--slice",
                str(tmp_path / "slice.jsonl"),
                "--teacher-records",
                str(tmp_path / "teacher.jsonl"),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--ledger",
                str(tmp_path / "out-ledger.jsonl"),
            ],
            capture_output=True,
            text=True,
        )
        results["converter_exit"] = proc.returncode
        if proc.returncode == 0:
            from data.schema import Sample

            row = json.loads((tmp_path / "out.jsonl").read_text().splitlines()[0])
            sample = Sample.model_validate(row)
            results["converter_row_valid"] = sample.questions[0].target.kind == "soft"
            results["converter_probs"] = sample.questions[0].target.probs
    # 3. student config validity
    from train.trainer import LoopConfig

    LoopConfig.model_validate(STUDENT_CONFIG)
    results["student_config_valid"] = True
    return results


def commands() -> dict:
    teacher = (
        f"cd /root/autodl-tmp/lite-omnijev && timeout --signal=INT --kill-after=30s 1800 "
        f"{PYTHON} {DATA}/run_teacher_cache.py"
        f" --data {DATA}/i5-slice-150.jsonl"
        f" --media-root {MEDIA_ROOT}"
        f" --model-root {MODEL_ROOT}"
        f" --report {OUT}/teacher-cache/report.json"
        f" --records {OUT}/teacher-cache/records.jsonl"
    )
    convert = (
        f"{PYTHON} {DATA}/build_d3p_student_rows.py"
        f" --slice {DATA}/i5-slice-150.jsonl"
        f" --teacher-records {OUT}/teacher-cache/records.jsonl"
        f" --output {OUT}/d3p-student.jsonl"
        f" --ledger {OUT}/d3p-student-ledger.jsonl"
    )
    train = (
        f"timeout --signal=INT --kill-after=20s 240 {PYTHON} -m scripts.train_s1 run"
        f" --config {OUT}/config.json"
        f" --model-dir {E4B}"
        f" --asset-manifest {ASSET_MANIFEST}"
        f" --data {OUT}/d3p-student.jsonl"
        f" --allocation-ledger {SOURCE_LEDGER}"
        f" --allocation-ledger {OUT}/d3p-student-ledger.jsonl"
        f" --output-dir {OUT}"
        f" --init-adapter {INIT_ADAPTER}"
        f" --init-adapter-sha256 {INIT_SHA}"
        f" --report {OUT}/report.json"
    )
    gate = (
        f"{PYTHON} -c \"import json;r=json.load(open('{OUT}/report.json'));"
        "assert r.get('status')=='completed' and r.get('exit_code')==0;"
        "print(r['adapter_dir'],r['adapter_sha256'])\""
    )
    video150 = (
        f"timeout --signal=INT --kill-after=20s 420 {PYTHON} -m scripts.s1_dev_eval"
        f" --model-dir {E4B}"
        f" --adapter {OUT}/adapter"
        f" --data {VIDEO150}"
        " --passes 2 --video-frames 16"
        f" --output-dir {OUT}/video150/eval --report {OUT}/video150/report.json"
        f" --media-root {MEDIA_ROOT}"
    )
    dev50 = (
        f"{PYTHON} -c \"import importlib.util,json,pathlib\n"
        "report=json.load(open('" + str(OUT) + "/report.json'))\n"
        "assert report.get('status')=='completed'\n"
        "runner=pathlib.Path('.scratch/s4/evaluate-video-motion-coverage.py')\n"
        "spec=importlib.util.spec_from_file_location('motion',runner)\n"
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n"
        "module.ADAPTER=pathlib.Path(report['adapter_dir'])\n"
        "module.ADAPTER_SHA=report['adapter_sha256']\n"
        "import sys;sys.argv=[str(runner),'--output','" + str(OUT / "dev50") + "','--execute','--policy','uniform']\n"
        "raise SystemExit(module.main())\""
    )
    return {
        "1_teacher_cache": teacher,
        "2_convert_cpu": convert,
        "3_student_train": train,
        "4_gate": gate,
        "5_video150": video150,
        "6_dev50": dev50,
        "config_write": f"mkdir -p {OUT} && cat > {OUT}/config.json <<'CFG'\n{json.dumps(STUDENT_CONFIG, indent=2)}\nCFG",
    }


def main() -> int:
    results = self_test()
    print(json.dumps(results, indent=2))
    if not all(
        [
            results["slice_rule_reproduced"],
            results["slice_sha_ok"],
            results["converter_exit"] == 0,
            results.get("converter_row_valid"),
            results["student_config_valid"],
        ]
    ):
        raise SystemExit("self-test failed")
    print("\n=== I5->D3P command chain (main schedules; steps 1/3/5/6 are GPU) ===")
    for name, command in commands().items():
        print(f"\n# {name}\n{command}")
    (Path(__file__).resolve().parent / "launch-i5-d3p-pilot.selftest.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
