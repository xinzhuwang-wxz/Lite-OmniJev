"""Continuous exposure uses cumulative steps and immutable checkpoint snapshots."""

import copy
import json
import random

import pytest
from test_gold_inventory import sample
from test_reconcile_crossmodal_retention import module

from train.checkpoint import stable_hash
from train.trainer import REFERENCE_FIELDS, LoopConfig


def fixed_config():
    return {
        "seed": 42,
        "steps": 100,
        "sample_count": 1200,
        "accumulation": 12,
        "rank": 16,
        "alpha": 32,
        "video_num_frames": 16,
        "image_max_soft_tokens": 1120,
        "modality": "mixed",
        "media_roots": ["/media"],
        "expected_data_versions": ["text"],
        "checkpoint_every": 10,
    }


def runtime_config(config, parent):
    result = LoopConfig.model_validate(config).model_dump()
    for field in REFERENCE_FIELDS:
        result.pop(field, None)
    result.pop("brier_weight", None)
    if result.get("preparation_mode") == "eager":
        result.pop("preparation_mode")
        result.pop("preparation_cache_bytes")
    result["init_adapter"] = parent
    return result


def final_report(code, start=0, end=10, status="paused"):
    parent = {
        "path": "/f2db",
        "weights_sha256": code.CONTINUOUS_PARENT_SHA,
        "config_sha256": "a" * 64,
    }
    return {
        "status": status,
        "exit_code": 0 if status in ("paused", "completed") else 1,
        "steps_completed": end,
        "init_adapter": parent,
        "config": runtime_config(fixed_config(), parent),
        "model_revision": "revision",
        "data_sha256": code.CONTINUOUS_DATA_SHA,
        "asset_hashes": {"model": "b" * 64},
        "invocation": {
            "steps_requested": 10,
            "start_step": start,
            "steps_completed": end - start,
            "global_steps_target": 100,
            "next_draw": 12 * end,
        },
    }


def test_paused_discovery_only_does_not_relax_legacy_terminal_pilots():
    code = module()
    r = final_report(code)
    with pytest.raises(ValueError, match="nonterminal_report"):
        code.terminal_report(r)
    code.terminal_report(r, allow_paused=True)
    r["exit_code"] = 1
    with pytest.raises(ValueError, match="inconsistent_final_status"):
        code.terminal_report(r, allow_paused=True)


@pytest.mark.parametrize(
    "start,end,status",
    [(0, 10, "paused"), (10, 20, "paused"), (90, 100, "completed"), (10, 13, "failed")],
)
def test_actual_seen_intervals_never_sum_cumulative_steps(start, end, status):
    code = module()
    r = final_report(code, start, end, status)
    result = code.continuous_interval(r, start, end)
    assert result["actual_draws_this_invocation"] == 12 * (end - start)
    assert result["actual_global_draws_seen"] == 12 * end
    assert result["actual_draw_interval"] == [12 * start, 12 * end]
    assert result["global_pool_draw_capacity"] == 1200


@pytest.mark.parametrize(
    "start,end,status",
    [(10, 30, "paused"), (0, 10, "completed"), (90, 100, "paused"), (10, 9, "failed")],
)
def test_out_of_order_overbudget_and_wrong_terminal_rejected(start, end, status):
    code = module()
    with pytest.raises(ValueError):
        code.continuous_interval(final_report(code, start, end, status), start, end)


def test_fixed_global_config_and_original_parent_are_required():
    code = module()
    r = final_report(code)
    prep = {
        "status": "data_ready_only",
        "parent": None,
        "training_authorized": False,
        "inventory_isolation": "isolated",
    }
    code.continuous_fixed_identity(prep, fixed_config(), r)
    wrong = copy.deepcopy(r)
    wrong["init_adapter"]["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="continuous_original_f2db_parent"):
        code.continuous_fixed_identity(prep, fixed_config(), wrong)
    wrong = copy.deepcopy(r)
    wrong["config"]["steps"] = 10
    with pytest.raises(ValueError, match="continuous_runtime_config_identity"):
        code.continuous_fixed_identity(prep, fixed_config(), wrong)


def checkpoint(code, root, report, step):
    root.mkdir(parents=True)
    hashes = {}
    for n in ("model.safetensors", "optimizer.safetensors", "rng.safetensors"):
        (root / n).write_bytes(n.encode())
        hashes[n] = code.sha(root / n)
    order = list(range(1200))
    random.Random(42).shuffle(order)
    payload = {
        "schema_version": 1,
        "step": step,
        "file_sha256": hashes,
        "config": report["config"],
        "config_sha256": stable_hash(report["config"]),
        "model_revision": report["model_revision"],
        "data_sha256": report["data_sha256"],
        "asset_hashes": report["asset_hashes"],
        "asset_hashes_sha256": stable_hash(report["asset_hashes"]),
        "order": order,
        "order_sha256": stable_hash(order),
        "losses": [1.0] * step,
    }
    (root / "checkpoint.json").write_text(json.dumps(payload))


def test_checkpoint_binds_full_optimizer_rng_config_data_and_order(tmp_path):
    code = module()
    r = final_report(code)
    path = tmp_path / "step-0010"
    checkpoint(code, path, r, 10)
    bindings = {}
    result = code.continuous_checkpoint(path, r, 10, bindings)
    assert result["step"] == 10 and len(bindings) == 4
    changed = copy.deepcopy(r)
    changed["config"]["steps"] = 10
    with pytest.raises(ValueError, match="checkpoint config"):
        code.continuous_checkpoint(path, changed, 10, {})
    (path / "optimizer.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="tensor file hash"):
        code.continuous_checkpoint(path, r, 10, {})


def integration_fixture(tmp_path, monkeypatch):
    code = module()
    from types import SimpleNamespace

    monkeypatch.setattr(
        code,
        "audit_manifest",
        lambda _: SimpleNamespace(
            outcome="isolated", model_dump=lambda **_: {"outcome": "isolated"}
        ),
    )
    root = tmp_path / "root"
    legacy = tmp_path / "legacy"
    (root / "reports").mkdir(parents=True)
    prepdir = root / "artifacts/prep"
    prepdir.mkdir(parents=True)
    data = prepdir / "train.jsonl"
    data.write_text(
        "".join(json.dumps(sample(f"draw{i}", pool="train")) + "\n" for i in range(1200))
    )
    monkeypatch.setattr(code, "CONTINUOUS_DATA_SHA", code.sha(data))
    (prepdir / "components.jsonl").write_text("")
    config = prepdir / "config-ce.json"
    config.write_text(json.dumps(fixed_config()))
    monkeypatch.setattr(code, "CONTINUOUS_CONFIG_SHA", code.sha(config))
    prep = prepdir / "report.json"
    prep.write_text(
        json.dumps(
            {
                "status": "data_ready_only",
                "parent": None,
                "training_authorized": False,
                "inventory_isolation": "isolated",
                "output_sha256": {
                    n: code.sha(prepdir / n)
                    for n in ("train.jsonl", "components.jsonl", "config-ce.json")
                },
            }
        )
    )
    monkeypatch.setattr(code, "CONTINUOUS_PREP_SHA", code.sha(prep))
    parentdir = tmp_path / "parent"
    parentdir.mkdir()
    (parentdir / "adapter_model.safetensors").write_bytes(b"parent")
    (parentdir / "adapter_config.json").write_text("{}")
    monkeypatch.setattr(
        code, "CONTINUOUS_PARENT_SHA", code.sha(parentdir / "adapter_model.safetensors")
    )
    base_dir = root / "artifacts/base013"
    base_dir.mkdir()
    # Another historical TRAIN input allows unknown-branch discovery to retain all history.
    old = root / "old.jsonl"
    old.write_text(json.dumps(sample("old", pool="train")) + "\n")
    inv = {
        "schema_version": "gold-inventory-manifest-v1",
        "component_ledgers": [],
        "samples": [
            {
                "path": str(old),
                "sha256": code.sha(old),
                "expected_pool": "train",
                "exposure": "current",
                "cohorts_by_data_version": {"text": "text"},
            }
        ],
    }
    (base_dir / "inventory-manifest.json").write_text(json.dumps(inv))
    base = base_dir / "report.json"
    base.write_text(
        json.dumps(
            {
                "status": "known_pilot_exposures_reconciled",
                "exit_code": 0,
                "unknown_update_bearing_inputs": 0,
                "all_bound_input_sha256": {},
                "output_sha256": {
                    "inventory-manifest.json": code.sha(base_dir / "inventory-manifest.json")
                },
            }
        )
    )
    monkeypatch.setattr(code, "CONTINUOUS_BASE_SHA", code.sha(base))
    index = root / "index.json"
    index.write_text(json.dumps({"status": "indexed", "unresolved": [], "files": []}))
    source_files = []
    for name in ("trainer.py", "checkpoint.py", "train_s1.py"):
        source = root / name
        source.write_text("# immutable trainer snapshot\n")
        source_files.append(code.bind(source))
    run = root / "artifacts/train"
    run.mkdir()
    runtime_data = run / "data.jsonl"
    runtime_data.write_bytes(data.read_bytes())

    def invocation(start, end, basepath):
        r = final_report(code, start, end)
        r["init_adapter"].update(
            path=str(parentdir), config_sha256=code.sha(parentdir / "adapter_config.json")
        )
        r["config"] = runtime_config(fixed_config(), r["init_adapter"])
        cp = run / "checkpoints" / f"step-{end:04d}"
        checkpoint(code, cp, r, end)
        r["checkpoint"] = str(cp)
        ad = run / f"adapter-invocation-step-{end:04d}"
        ad.mkdir()
        (ad / "adapter_model.safetensors").write_bytes(str(end).encode())
        (ad / "adapter_config.json").write_text("{}")
        r.update(adapter_dir=str(ad), adapter_sha256=code.sha(ad / "adapter_model.safetensors"))
        if start:
            r["resumed_from"] = str(run / "checkpoints" / f"step-{start:04d}")
        raw = run / "run.json"
        raw.write_text(json.dumps(r))
        final = root / f"reports/training-{end:04d}.json"
        final.write_bytes(raw.read_bytes())
        snapshot = run / f"run-invocation-{end:04d}.json"
        snapshot.write_bytes(raw.read_bytes())
        launch = root / f"reports/launch-{end:04d}.json"
        eb = {
            k: code.bind(p)
            for k, p in [
                ("base_exposure_report", basepath),
                ("prepared_report", prep),
                ("config_file", config),
                ("runtime_data", runtime_data),
                ("final_report", final),
                ("runtime_report_snapshot", snapshot),
            ]
        }
        eb.update(
            runtime_report_path=str(raw),
            source_files=source_files,
            start_step=start,
            end_step=end,
            resume_from=r.get("resumed_from"),
        )
        launch.write_text(
            json.dumps(
                {
                    "status": "paused",
                    "exit_code": 0,
                    "exposure_binding": eb,
                    "selected_init_adapter": r["init_adapter"],
                    "parent_selection": (
                        "root_selected_f2db_balanced_retention_and_full_video_evidence"
                    ),
                }
            )
        )
        return launch

    return code, root, legacy, index, base, invocation


def test_two_invocations_append_pool_once_and_immutable_snapshots_survive_resume(
    tmp_path, monkeypatch
):
    code, root, legacy, index, base, invoke = integration_fixture(tmp_path, monkeypatch)
    out1 = root / "artifacts/exposure014"
    first = code.prepare_continuous(root, legacy, out1, index, invoke(0, 10, base))
    assert first["actual_global_draws_seen"] == 120 and len(first["appended_artifacts"]) == 1
    assert str(root / "artifacts/train/run.json") not in first["all_bound_input_sha256"]
    out2 = root / "artifacts/exposure015"
    second = code.prepare_continuous(
        root, legacy, out2, index, invoke(10, 20, out1 / "report.json")
    )
    assert second["actual_global_draws_seen"] == 240 and second["appended_artifacts"] == []
    assert len(second["continuous_invocations"]) == 2
    inventory = json.loads((out2 / "inventory-manifest.json").read_text())
    assert len([a for a in inventory["samples"] if a["exposure"] == "current"]) == 1
    assert len(inventory["samples"]) == 2


def test_unknown_paused_branch_is_discovered_and_rejected(tmp_path, monkeypatch):
    code, root, legacy, index, base, invoke = integration_fixture(tmp_path, monkeypatch)
    launch = invoke(0, 10, base)
    (root / "reports/unknown-paused.json").write_text(
        json.dumps(
            {"status": "paused", "exit_code": 0, "steps_completed": 3, "data_sha256": "f" * 64}
        )
    )
    with pytest.raises(ValueError, match="unknown_update_bearing_input"):
        code.prepare_continuous(root, legacy, root / "artifacts/rejected", index, launch)


def test_mutated_prior_snapshot_rejected_before_next_inventory(tmp_path, monkeypatch):
    code, root, legacy, index, base, invoke = integration_fixture(tmp_path, monkeypatch)
    out = root / "artifacts/exposure014"
    first = code.prepare_continuous(root, legacy, out, index, invoke(0, 10, base))
    launch = invoke(10, 20, out / "report.json")
    snapshot = first["continuous_invocations"][0]["runtime_report_snapshot"]["path"]
    from pathlib import Path

    Path(snapshot).write_text("{}")
    with pytest.raises(ValueError, match="continuous_prior_mutation"):
        code.prepare_continuous(root, legacy, root / "artifacts/rejected015", index, launch)


def test_failed_partial_steps_remain_exposure_without_adapter_or_checkpoint(tmp_path, monkeypatch):
    code, root, legacy, index, base, invoke = integration_fixture(tmp_path, monkeypatch)
    launch_path = invoke(0, 3, base)
    launch = json.loads(launch_path.read_text())
    eb = launch["exposure_binding"]
    from pathlib import Path

    final_path = Path(eb["final_report"]["path"])
    final = json.loads(final_path.read_text())
    final.update(status="failed", exit_code=1)
    for k in ("adapter_dir", "adapter_sha256", "checkpoint", "invocation"):
        final.pop(k, None)
    for path in (
        final_path,
        Path(eb["runtime_report_snapshot"]["path"]),
        Path(eb["runtime_report_path"]),
    ):
        path.write_text(json.dumps(final))
    for key in ("final_report", "runtime_report_snapshot"):
        eb[key] = code.bind(Path(eb[key]["path"]))
    launch.update(status="failed", exit_code=1)
    launch_path.write_text(json.dumps(launch))
    report = code.prepare_continuous(root, legacy, root / "artifacts/failed014", index, launch_path)
    invocation = report["continuous_invocations"][0]
    assert invocation["actual_global_draws_seen"] == 36
    assert invocation["partial_step_draws_unknown"]
    assert invocation["possible_global_draws_seen_upper_bound"] == 48
    assert invocation["checkpoint"] is None and invocation["adapter"] is None
    assert len(report["appended_artifacts"]) == 1


def test_cli_only_ledger_summary_can_extend_final_without_changing_runtime(tmp_path, monkeypatch):
    from pathlib import Path

    code, root, legacy, index, base, invoke = integration_fixture(tmp_path, monkeypatch)
    launch_path = invoke(0, 10, base)
    launch = json.loads(launch_path.read_text())
    final_path = Path(launch["exposure_binding"]["final_report"]["path"])
    final = json.loads(final_path.read_text())
    final["allocation_ledger_check"] = {"verified_train": 1200, "ledger_rows": 1200}
    final_path.write_text(json.dumps(final))
    launch["exposure_binding"]["final_report"] = code.bind(final_path)
    launch_path.write_text(json.dumps(launch))
    report = code.prepare_continuous(
        root, legacy, root / "artifacts/exposure014", index, launch_path
    )
    assert report["actual_global_draws_seen"] == 120
