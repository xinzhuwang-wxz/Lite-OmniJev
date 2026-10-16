"""Generate and self-check the two B1 pilot arms; print final commands for main.

Self-contained CPU tool. No GPU is started here: the script only validates
configs, reconciles ledgers against data rows, and prints the exact commands
for main to schedule inside the agreed GPU window.

Arm 1  b1-ce  : one-hot grid, 126 rows, 40 steps.
Arm 1w b1-wsft: WSFT grid, 292 rows, 60 steps (primary).

Arm 2 (r1-coupled twin loss) is NOT implemented by decision; the only delta vs
arm 1w would be: (a) a trainer-side loss extension combining, per batch,
L = alpha*CE(wsft targets incl. U) + beta*sum(<p_left,p_right> over the 43 twin
pairs, both forwards through the same LoRA) + beta3*sum(1-P_consistent(triple)
over 6 triples), closed form per .scratch/s4/r1_lattice.py; (b) batch sampling
by twin group so both members of a pair are in the same step; (c) config fields
for alpha/beta/beta3 (defaults 1.0/0.3/0.1). Everything else identical.

Remote bindings verified 2026-10-16 (paths exist, sha256 pinned below).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PYTHON = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"
REMOTE_ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
DATA_DIR = REMOTE_ROOT / "artifacts/s4-ensemble-oracle-001"
MODEL_DIR = (
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/"
    "ee0ef6023621cff504d758262d4e04895a5af4a2"
)
ASSET_MANIFEST = "/root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json"
INIT_ADAPTER = (
    "/root/autodl-tmp/s1-text-lora-001/artifacts/s4-program-stratified-pool-train-001/"
    "adapter-invocation-step-0100"
)
INIT_ADAPTER_SHA = "e46550143b28f8798484515f0abe21996fe92ad241625a4cf6da15dc093277fd"
INIT_ADAPTER_CONFIG_SHA = "3b19c886d51a9715e2db026818df0f050b8183bfa0c5367079a5208ebb28e62f"
SOURCE_LEDGER = (
    "/root/autodl-tmp/s1-text-lora-001/artifacts/s3-pools-video-001/run-a/"
    "allocation-ledger.jsonl"
)
SOURCE_LEDGER_SHA = "3049be487ca4884273397c5f6c9e17e934887d9a96ad3997d5a4561afe159e92"
VIDEO_MEDIA_ROOT = "/root/autodl-tmp/s3-media-candidates-001/video/media"
VIDEO150 = REMOTE_ROOT / "artifacts/s3-stable-probe-001/video-dev150.jsonl"
VIDEO150_SHA = "e9b91fedecf2ccf7dc30e837ac11bcc4b233624d1f6c16812a40b1319c15b080"
BUDGET_SECONDS = 240
SEED = 42

PINNED = {
    "b1-grid.jsonl": "41ebacc452cbd90ff4ffb5e2c5c2cc17d4f3b8a0ca950aebbead46b44e745fe9",
    "b1-wsft-grid.jsonl": "a6a7c30caef0edd1f19174c4ef7c6554060bed851d04334972036672e232b51e",
    "b1-grid-ledger.jsonl": "c011e67353ebddd1ac8c4b9b2de5750254628b98eedd5ca48bd568838594307d",
    "b1-wsft-ledger.jsonl": "bb922bd45b9cde91772be47791eb4206cf1419b29a2b63a2648ca10743371006",
}

ARMS = [
    {
        "tag": "s4-b1-ce-pilot-001",
        "data": "b1-grid.jsonl",
        "ledger": "b1-grid-ledger.jsonl",
        "sample_count": 126,
        "steps": 40,
    },
    {
        "tag": "s4-b1-wsft-pilot-001",
        "data": "b1-wsft-grid.jsonl",
        "ledger": "b1-wsft-ledger.jsonl",
        "sample_count": 292,
        "steps": 60,
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arm_config(arm: dict) -> dict:
    return {
        "seed": SEED,
        "sample_count": arm["sample_count"],
        "steps": arm["steps"],
        "accumulation": 1,
        "max_length": 4096,
        "learning_rate": 2e-6,
        "rank": 16,
        "alpha": 32,
        "dtype": "bfloat16",
        "modality": "video",
        "video_num_frames": 16,
        "media_roots": [VIDEO_MEDIA_ROOT],
        "expected_data_version": "clevrer-train-video-b1-grid-v1",
        "checkpoint_every": 5,
        "preparation_mode": "lazy",
        "preparation_cache_bytes": 134217728,
    }


def validate_config(config: dict) -> None:
    try:
        from train.trainer import LoopConfig
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from train.trainer import LoopConfig
    LoopConfig.model_validate(config)


def reconcile(data_path: Path, ledger_path: Path, expected_rows: int) -> dict:
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{data_path.name}: duplicate ids")
    if len(rows) != expected_rows:
        raise SystemExit(f"{data_path.name}: {len(rows)} rows != {expected_rows}")
    ledger = {}
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        item_id, pool = entry.get("item_id"), entry.get("pool")
        if not isinstance(item_id, str) or not isinstance(pool, str):
            raise SystemExit(f"{ledger_path.name}: ledger row lacks item_id/pool")
        ledger[item_id] = pool
    missing = [i for i in ids if i not in ledger]
    wrong = [i for i in ids if ledger.get(i) not in (None, "train")]
    if missing or wrong:
        raise SystemExit(f"{data_path.name}: ledger reconcile failed ({len(missing)} missing, {len(wrong)} non-train)")
    versions = {row["meta"]["data_version"] for row in rows}
    kinds = {}
    for row in rows:
        kind = row["questions"][0]["target"]["kind"]
        kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "rows": len(rows),
        "ledger_rows": len(ledger),
        "all_train": True,
        "data_versions": sorted(versions),
        "target_kinds": kinds,
        "sha256": sha256_file(data_path),
        "ledger_sha256": sha256_file(ledger_path),
    }


def train_command(arm: dict) -> list[str]:
    out = REMOTE_ROOT / "artifacts" / arm["tag"]
    return [
        "timeout", "--signal=INT", "--kill-after=20s", str(BUDGET_SECONDS),
        PYTHON, "-m", "scripts.train_s1", "run",
        "--config", str(out / "config.json"),
        "--model-dir", MODEL_DIR,
        "--asset-manifest", ASSET_MANIFEST,
        "--data", str(DATA_DIR / arm["data"]),
        "--allocation-ledger", SOURCE_LEDGER,
        "--allocation-ledger", str(DATA_DIR / arm["ledger"]),
        "--output-dir", str(out),
        "--init-adapter", INIT_ADAPTER,
        "--init-adapter-sha256", INIT_ADAPTER_SHA,
        "--report", str(out / "report.json"),
    ]


def eval_commands(arm: dict) -> list[list[str]]:
    out = REMOTE_ROOT / "artifacts" / arm["tag"]
    adapter = str(out / "adapter")
    # Gate on the training report; the fresh adapter sha replaces the init sha.
    gate = [
        PYTHON, "-c",
        "import json,sys\n"
        f"r=json.load(open({str(out / 'report.json')!r}))\n"
        "assert r.get('status')=='completed' and r.get('exit_code')==0, r.get('status')\n"
        "print(r['adapter_dir'], r['adapter_sha256'])",
    ]
    video150 = [
        "timeout", "--signal=INT", "--kill-after=20s", "420",
        PYTHON, "-m", "scripts.s1_dev_eval",
        "--model-dir", MODEL_DIR,
        "--adapter", adapter,
        "--data", str(VIDEO150),
        "--passes", "2",
        "--video-frames", "16",
        "--output-dir", str(out / "video150/eval"),
        "--report", str(out / "video150/report.json"),
        "--media-root", VIDEO_MEDIA_ROOT,
    ]
    # DEV50 via the proven motion-coverage runner with module-level adapter override;
    # adapter sha is read from this arm's training report at run time.
    dev50 = [
        PYTHON, "-c",
        "import importlib.util,json,sys\n"
        "from pathlib import Path\n"
        f"report=json.load(open({str(out / 'report.json')!r}))\n"
        "assert report.get('status')=='completed', report.get('status')\n"
        "runner=Path('.scratch/s4/evaluate-video-motion-coverage.py')\n"
        "spec=importlib.util.spec_from_file_location('motion',runner)\n"
        "module=importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module.ADAPTER=Path(report['adapter_dir'])\n"
        "module.ADAPTER_SHA=report['adapter_sha256']\n"
        f"sys.argv=[str(runner),'--output',str({str(out / 'dev50')!r}),'--execute','--policy','uniform']\n"
        "raise SystemExit(module.main())",
    ]
    return [gate, video150, dev50]


def main() -> int:
    local_dir = Path(__file__).resolve().parents[2] / "artifacts/s4-ensemble-oracle-001"
    summary = {"arms": {}, "checks": []}
    for arm in ARMS:
        data_local = local_dir / arm["data"]
        ledger_local = local_dir / arm["ledger"]
        if data_local.exists():
            stats = reconcile(data_local, ledger_local, arm["sample_count"])
            for name in (arm["data"], arm["ledger"]):
                actual = sha256_file(local_dir / name)
                if actual != PINNED[name]:
                    raise SystemExit(f"{name}: sha {actual} != pinned {PINNED[name]}")
            summary["arms"][arm["tag"]] = stats
        else:
            summary["arms"][arm["tag"]] = {
                "local_files": "absent; remote bindings only",
                "data_sha256_pinned": PINNED[arm["data"]],
                "ledger_sha256_pinned": PINNED[arm["ledger"]],
            }
        validate_config(arm_config(arm))
        summary["checks"].append(f"{arm['tag']}: LoopConfig-valid, ledger reconciled, seed {SEED}")
    blob = json.dumps(summary, indent=2, ensure_ascii=False)
    print(blob)
    print("\n=== final commands (run by main on the GPU box, cwd = repo root) ===")
    for arm in ARMS:
        out = REMOTE_ROOT / "artifacts" / arm["tag"]
        config = arm_config(arm)
        print(f"\n# arm {arm['tag']} — write config then train then auto-eval")
        print(f"mkdir -p {out} && cat > {out}/config.json <<'CFG'\n{json.dumps(config, indent=2)}\nCFG")
        print(" ".join(train_command(arm)))
        for command in eval_commands(arm):
            print(" ".join(command))
    (Path(__file__).resolve().parent / "launch-b1-wsft-pilot.selftest.json").write_text(blob + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
