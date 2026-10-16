"""Derivative provenance preserves failed numeric parity and forbids inherited quality."""

import json

import pytest
from test_reconcile_crossmodal_retention import module


def fixture(tmp_path, monkeypatch):
    code = module()
    root = tmp_path / "root"
    adapter = root / "artifacts/s4-rank64-initializer-001"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"expanded")
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 64, "lora_alpha": 128}))
    source = root / "original"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"original")
    (source / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    monkeypatch.setattr(code, "RANK64_WEIGHTS_SHA", code.sha(adapter / "adapter_model.safetensors"))
    monkeypatch.setattr(code, "RANK64_CONFIG_SHA", code.sha(adapter / "adapter_config.json"))
    monkeypatch.setattr(
        code, "CONTINUOUS_PARENT_SHA", code.sha(source / "adapter_model.safetensors")
    )
    expansion = {
        "status": "converted_research_initializer",
        "exit_code": 0,
        "new_rank": 64,
        "new_alpha": 128,
        "old_rank": 16,
        "old_alpha": 32,
        "source_weights_sha256": code.CONTINUOUS_PARENT_SHA,
        "training_authorized": False,
        "production_replacement": False,
        "source_path": str(source),
        "source_config_sha256": code.sha(source / "adapter_config.json"),
        "output_sha256": {
            "adapter_model.safetensors": code.RANK64_WEIGHTS_SHA,
            "adapter_config.json": code.RANK64_CONFIG_SHA,
        },
    }
    common = {
        "expanded_weights_sha256": code.RANK64_WEIGHTS_SHA,
        "expanded_config_sha256": code.RANK64_CONFIG_SHA,
        "source_weights_sha256": code.CONTINUOUS_PARENT_SHA,
        "parity_validated": False,
    }
    parity = {
        **common,
        "status": "failed",
        "exit_code": 1,
        "max_joint_probability_difference": 0.031,
        "parity_tolerance": 0.001,
    }
    control = {
        **common,
        "status": "completed",
        "exit_code": 0,
        "same_process": True,
        "source_repeat_bit_identical_logits": True,
    }
    pins = {}
    paths = []
    for relative, payload in zip(
        code.RANK64_EVIDENCE_SHA, (expansion, parity, control), strict=True
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        pins[relative] = code.sha(path)
        paths.append(path)
    monkeypatch.setattr(code, "RANK64_EVIDENCE_SHA", pins)
    policy = tmp_path / "policy.md"
    policy.write_text("Independent candidate; parity remains failed.")
    monkeypatch.setattr(code, "RANK64_POLICY_SHA", code.sha(policy))
    archive = root.parent / "omnijev-rank64-code-b97edc0.tar"
    archive.write_bytes(b"code")
    monkeypatch.setattr(code, "RANK64_SOURCE_ARCHIVE_SHA", code.sha(archive))
    return code, root, policy, paths


def test_identity_binding_never_authorizes_training_or_inherits_parent_quality(
    tmp_path, monkeypatch
):
    code, root, policy, _ = fixture(tmp_path, monkeypatch)
    bindings, identity = code.rank64_derivative_bindings(root, policy)
    assert len(bindings) == 9
    assert identity["rank"] == 64 and identity["alpha"] == 128
    assert identity["numerical_parity_validated"] is False
    assert identity["parent_quality_inheritance"] is False
    assert identity["training_authorized"] is False and identity["own_baseline_required"] is True
    assert identity["same_train_data_sha256"] == code.MATCHED_CONTINUOUS_DATA_SHA


@pytest.mark.parametrize("bad", ["policy", "weights", "failed-parity-relabel"])
def test_independent_policy_and_real_failed_probe_cannot_be_relabelled(tmp_path, monkeypatch, bad):
    code, root, policy, paths = fixture(tmp_path, monkeypatch)
    if bad == "policy":
        policy.write_text("parity passed")
    elif bad == "weights":
        (root / "artifacts/s4-rank64-initializer-001/adapter_model.safetensors").write_bytes(
            b"other"
        )
    else:
        path = paths[1]
        p = json.loads(path.read_text())
        p["parity_validated"] = True
        path.write_text(json.dumps(p))
        pins = dict(code.RANK64_EVIDENCE_SHA)
        pins[str(path.relative_to(root))] = code.sha(path)
        monkeypatch.setattr(code, "RANK64_EVIDENCE_SHA", pins)
    with pytest.raises(ValueError):
        code.rank64_derivative_bindings(root, policy)


def test_rank64_profile_uses_derived_config_and_an_independent_history_key():
    code = module()
    profile = code.continuous_profile(rank64=True)
    assert profile["name"] == "s4-rank64-matched-pilot"
    assert profile["history_key"] == "rank64_invocations"
    assert (
        profile["config_sha"] == "e01916029545aea72be7e03dc2cbb9a5b35530c26c4a71711ec1bd4518c6f871"
    )
    assert profile["prep_sha"] == code.MATCHED_CONTINUOUS_PREP_SHA
    assert profile["data_sha"] == code.MATCHED_CONTINUOUS_DATA_SHA
    assert profile["base_sha"] == "eb7502b939da6339d4944421cfafae297e2251101f88c9327a4466f9226cc81f"
    assert profile["history_key"] != code.continuous_profile(True)["history_key"]


def test_rank64_budget_refuses_old_rank16_or_other_initializer():
    from test_reconcile_continuous_main_pool import final_report, fixed_config

    code = module()
    report = final_report(code)
    prep = {
        "status": "data_ready_only",
        "parent_model": None,
        "training_authorized": False,
        "inventory_isolation": "isolated",
    }
    with pytest.raises(ValueError, match="continuous_frozen_config"):
        code.continuous_fixed_identity(prep, fixed_config(), report, rank64=True)
    config = {**fixed_config(), "rank": 64, "alpha": 128}
    with pytest.raises(ValueError, match="continuous_original_f2db_parent"):
        code.continuous_fixed_identity(prep, config, report, rank64=True)
