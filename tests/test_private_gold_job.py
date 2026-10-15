"""Portable boundary validation; real root/UID probes run on Linux separately."""

import json
import os
import signal
import time
from pathlib import Path

import pytest

from scripts.gold_inventory import Receipt, digest, publish
from scripts.private_gold_job import (
    JobRejected,
    bounded_read,
    main,
    safe_path,
    snapshot,
    validate_receipt,
    wait_child,
)


def test_no_symlinks_or_traversal(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.write_bytes(b"secret")
    link = tmp_path / "link"
    link.symlink_to(original)
    for path in (link, tmp_path / ".." / "secret", Path("relative")):
        with pytest.raises(JobRejected):
            safe_path(path)
    with pytest.raises(JobRejected):
        bounded_read(original, 2)


def test_snapshot_hash_mismatch_never_copies(tmp_path: Path) -> None:
    source = tmp_path / "manifest"
    source.write_bytes(b"secret")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    with pytest.raises(JobRejected):
        snapshot(source, "0" * 64, inputs, os.getgid())
    assert list(inputs.iterdir()) == []


def test_receipt_strict_binding_and_exit(tmp_path: Path) -> None:
    path = tmp_path / "receipt"
    receipt = Receipt(outcome="rejected", categories=["invalid_sample"], manifest_sha256="0" * 64)
    path.write_text(receipt.model_dump_json())
    assert validate_receipt(path, "0" * 64, 1) == receipt
    for expected, code in (("1" * 64, 1), ("0" * 64, 0)):
        with pytest.raises(JobRejected):
            validate_receipt(path, expected, code)
    payload = receipt.model_dump()
    payload["private_label"] = "secret"
    path.write_text(json.dumps(payload))
    with pytest.raises(JobRejected) as error:
        validate_receipt(path, "0" * 64, 1)
    assert error.value.category == "invalid_receipt"


def test_missing_and_malformed_receipt(tmp_path: Path) -> None:
    path = tmp_path / "receipt"
    with pytest.raises(JobRejected) as error:
        validate_receipt(path, "0" * 64, 1)
    assert error.value.category == "missing_receipt"
    path.write_bytes(b"not-json secret")
    with pytest.raises(JobRejected) as error:
        validate_receipt(path, "0" * 64, 1)
    assert error.value.category == "invalid_receipt"


def test_publish_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "receipt"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        publish(Receipt(outcome="rejected"), path)
    assert path.read_bytes() == b"existing"


def test_cli_redacts_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--private-secret", "/private/secret"]) == 1
    output = capsys.readouterr()
    assert "secret" not in output.out + output.err
    assert json.loads(output.out)["outcome"] == "rejected"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork required")
def test_timeout_reaps_process_group() -> None:
    # Test-only safe sleeping child; production registry accepts no callbacks/code.
    pid = os.fork()
    if pid == 0:
        os.setsid()
        time.sleep(20)
        os._exit(0)
    code, timed_out = wait_child(pid, 0.05)
    assert timed_out
    assert code in (-signal.SIGTERM, -signal.SIGKILL)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def test_digest_stable() -> None:
    assert len(digest(b"inventory")) == 64


def test_snapshot_rewrites_only_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "chown", lambda *args: None)
    original = tmp_path / "private-label.jsonl"
    original.write_bytes(b"private sample payload\n")
    payload = {
        "schema_version": "gold-inventory-manifest-v1",
        "samples": [
            {
                "path": str(original),
                "sha256": digest(original.read_bytes()),
                "cohort": "text",
                "exposure": "current",
                "expected_pool": "holdout",
            }
        ],
        "component_ledgers": [],
    }
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(payload))
    before = source.read_bytes()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    translated, originals = snapshot(source, digest(before), inputs, os.getgid())
    rewritten = json.loads(translated.read_bytes())
    assert source.read_bytes() == before
    assert originals == [source, original]
    assert rewritten["samples"][0]["sha256"] == payload["samples"][0]["sha256"]
    assert Path(rewritten["samples"][0]["path"]).read_bytes() == original.read_bytes()
    assert "private-label" not in translated.read_text()
    assert "private sample" not in translated.read_text()


def test_public_contract_has_no_private_receipt_fields() -> None:
    from scripts.private_gold_job import PublicReceipt

    receipt = PublicReceipt(
        categories=["invalid_receipt"],
        source_manifest_sha256="0" * 64,
        translated_manifest_sha256="1" * 64,
        code_sha256="2" * 64,
        config_sha256="3" * 64,
        nonce_sha256="4" * 64,
        duration_seconds=1.0,
        exit_code=1,
        ordinary_identity_boundary=False,
    )
    payload = receipt.model_dump()
    assert payload["namespace_enforced"] is False
    assert payload["network_isolation"] == "not_enforced"
    assert payload["gold_provenance"] == "not_validated"
    assert "verified_artifact_sha256" not in payload
    with pytest.raises(ValueError):
        PublicReceipt.model_validate({**payload, "stdout": "private label"})


@pytest.mark.parametrize("field", ["Uid", "Gid", "Groups"])
def test_live_credentials_include_all_ids(field: str) -> None:
    from scripts.private_gold_job import process_credentials

    rows = {"Uid": "1 2 3 4", "Gid": "5 6 7 8", "Groups": "9 10"}
    rows[field] += " 62041" if field == "Groups" else ""
    if field != "Groups":
        rows[field] = "1 2 62041 4"
    status = "\n".join(f"{key}: {value}" for key, value in rows.items())
    assert 62041 in process_credentials(status)
    with pytest.raises(JobRejected):
        process_credentials("Uid: 0 0 0 0\nGid: 0 0 0 0")


def test_proc_collision_and_unexpected_missing_fail_closed(tmp_path: Path) -> None:
    from scripts.private_gold_job import check_process_ids

    process = tmp_path / "1234"
    process.mkdir()
    status = process / "status"
    status.write_text("Uid: 1 2 3 4\nGid: 5 6 7 8\nGroups: 62041")
    with pytest.raises(JobRejected):
        check_process_ids({62041}, tmp_path)
    check_process_ids({62042}, tmp_path)
    status.unlink()
    with pytest.raises(JobRejected):
        check_process_ids({62042}, tmp_path)


def test_prior_unleased_run_ownership_rejected(tmp_path: Path) -> None:
    from scripts.private_gold_job import check_prior_ownership

    (tmp_path / "old-run").mkdir()
    with pytest.raises(JobRejected):
        check_prior_ownership(tmp_path, {os.getuid()})


@pytest.mark.parametrize("fail", [False, True])
def test_private_run_revoked_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    from scripts.private_gold_job import private_run

    changes = []
    monkeypatch.setattr(os, "chown", lambda path, uid, gid: changes.append((uid, gid)))
    run = tmp_path / "run"
    try:
        with private_run(run, 62041):
            run.chmod(0o750)
            (run / "private").write_text("retained")
            if fail:
                raise RuntimeError("synthetic")
    except RuntimeError:
        assert fail
    assert run.stat().st_mode & 0o777 == 0o700
    assert changes[-1] == (0, 0)
    assert (run / "private").read_text() == "retained"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork required")
def test_concurrent_identity_leases_are_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.private_gold_job as job

    monkeypatch.setattr(job, "check_process_ids", lambda ids: None)
    monkeypatch.setattr(job, "check_prior_ownership", lambda root, ids: None)
    original_stat = Path.stat

    def root_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        info = original_stat(path, follow_symlinks=follow_symlinks)
        if path.name == ".identity-leases":
            values = list(info)
            values[4] = 0
            return os.stat_result(values)
        return info

    monkeypatch.setattr(Path, "stat", root_stat)
    children = []
    for _ in range(2):
        pid = os.fork()
        if pid == 0:
            try:
                job.reserve_ids(tmp_path, 62041, 62042)
            except JobRejected:
                os._exit(1)
            os._exit(0)
        children.append(pid)
    codes = sorted(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) for pid in children)
    assert codes == [0, 1]
    with pytest.raises(JobRejected):
        job.reserve_ids(tmp_path, 62041, 62042)
