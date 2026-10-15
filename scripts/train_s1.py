"""S1 text decision-LoRA training entry over the frozen pool; ledger-enforced."""

import argparse
import json
from pathlib import Path

from scripts.assets import write_json
from train.trainer import LoopConfig, run


def load_ledger_pools(path: Path) -> dict[str, str]:
    pools: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            item_id = row.get("item_id")
            pool = row.get("pool")
            if not isinstance(item_id, str) or not isinstance(pool, str):
                raise ValueError(f"invalid ledger row at line {line_no}")
            if item_id in pools and pools[item_id] != pool:
                raise ValueError(f"ledger maps {item_id} to multiple pools")
            pools[item_id] = pool
    return pools


def verify_data_is_train_pool(data_path: Path, ledger_paths: list[Path]) -> dict[str, int]:
    """Reject any sample the frozen allocations did not place in a train pool."""
    pools: dict[str, str] = {}
    for ledger_path in ledger_paths:
        for item_id, pool in load_ledger_pools(ledger_path).items():
            prior = pools.setdefault(item_id, pool)
            if prior != pool:
                raise ValueError(f"ledgers disagree about {item_id}: {prior} vs {pool}")
    data_ids = [
        json.loads(line)["id"]
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(set(data_ids)) != len(data_ids):
        raise ValueError("training data contains duplicate ids")

    def ledger_pool(item_id: str) -> str | None:
        """Deterministic variant ids resolve to their parent's frozen pool."""
        if item_id in pools:
            return pools[item_id]
        if item_id.endswith("-cf1") and item_id[:-4] in pools:
            return pools[item_id[:-4]]
        return None

    reserved = [
        item_id for item_id in data_ids if ledger_pool(item_id) in {"holdout", "calibration"}
    ]
    unlisted = [item_id for item_id in data_ids if ledger_pool(item_id) is None]
    dev_pooled = [item_id for item_id in data_ids if ledger_pool(item_id) == "dev"]
    if reserved or unlisted or dev_pooled:
        raise ValueError(
            f"ledger violations: {len(reserved)} reserved, {len(dev_pooled)} dev, "
            f"{len(unlisted)} unlisted"
        )
    return {"ledger_rows": len(pools), "verified_train": len(data_ids)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--allocation-ledger",
        type=Path,
        action="append",
        default=None,
        help="allocation ledger(s) binding sample ids to frozen pools; repeatable",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args(argv)
    try:
        config = LoopConfig.model_validate_json(args.config.read_text())
        ledger_check: dict[str, int] | None = None
        if args.allocation_ledger:
            ledger_check = verify_data_is_train_pool(args.data, args.allocation_ledger)
        report = run(
            config,
            args.model_dir.resolve(),
            args.asset_manifest,
            args.output_dir,
            execute=args.action == "run",
            data_path=args.data,
            resume_from=args.resume_from,
            scope=(
                "S1 text decision-LoRA over frozen pool; training-pool-only with "
                "ledger enforcement; not quality-gate evidence by itself"
            ),
        )
        if ledger_check is not None:
            report["allocation_ledger_check"] = ledger_check
    except Exception as exc:
        report = {"status": "blocked", "exit_code": 1, "error": f"{type(exc).__name__}: {exc}"}
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
