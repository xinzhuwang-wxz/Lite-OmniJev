"""Observable asset preparation failures use fake network transport, never model weights."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.assets import main, model_directory, plan_assets, prepare_assets
from scripts.check_tokens import main as token_main


@pytest.fixture
def manifest():
    content = b"asset"
    return {
        "schema_version": 1,
        "models": [
            {
                "repo_id": "example/model",
                "license": "apache-2.0",
                "revision": "a" * 40,
                "files": [
                    {
                        "path": name,
                        "size": len(content),
                        "kind": kind,
                        "upstream_sha256": hashlib.sha256(content).hexdigest(),
                    }
                    for name, kind in [
                        ("tokenizer.json", "metadata"),
                        ("model.safetensors", "weights"),
                        ("config.json", "metadata"),
                        ("tokenizer_config.json", "metadata"),
                    ]
                ],
            }
        ],
    }


def fake_download(**kwargs):
    path = Path(kwargs["local_dir"]) / kwargs["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"asset")
    return str(path)


def test_metadata_only_never_downloads_weights(tmp_path, manifest):
    download = Mock(side_effect=fake_download)
    report = prepare_assets(manifest, tmp_path, metadata_only=True, download=download)
    assert report["status"] == "metadata_verified"
    assert report["weights_ready"] is False
    assert download.call_count == 3
    assert {call.kwargs["filename"] for call in download.call_args_list} == {
        "tokenizer.json",
        "config.json",
        "tokenizer_config.json",
    }
    assert download.call_args.kwargs["revision"] == "a" * 40
    assert manifest["models"][0]["files"][0]["actual_sha256"]


def test_verified_download_is_reusable_offline(tmp_path, manifest):
    download = Mock(side_effect=fake_download)
    prepare_assets(manifest, tmp_path, metadata_only=False, download=download)
    report = prepare_assets(manifest, tmp_path, metadata_only=False)
    assert report["weights_ready"] is True
    assert report["files_verified"] == 4
    assert download.call_count == 4


def test_low_disk_does_not_start_download(tmp_path, manifest, monkeypatch):
    monkeypatch.setattr("scripts.assets.shutil.disk_usage", lambda _: SimpleNamespace(free=1))
    download = Mock()
    with pytest.raises(ValueError, match="Insufficient disk"):
        prepare_assets(manifest, tmp_path, metadata_only=True, download=download)
    download.assert_not_called()


def test_corrupt_download_is_rejected(tmp_path, manifest):
    def corrupt(**kwargs):
        path = Path(fake_download(**kwargs))
        path.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare_assets(manifest, tmp_path, metadata_only=True, download=corrupt)


def test_missing_file_fails_verification(tmp_path, manifest):
    with pytest.raises(ValueError, match="Missing file"):
        prepare_assets(manifest, tmp_path, metadata_only=True)


def test_corrupt_existing_asset_is_not_overwritten(tmp_path, manifest):
    prepare_assets(manifest, tmp_path, metadata_only=True, download=fake_download)
    path = model_directory(tmp_path, manifest["models"][0]) / "tokenizer.json"
    path.write_bytes(b"wrong")
    download = Mock()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare_assets(manifest, tmp_path, metadata_only=True, download=download)
    download.assert_not_called()


def test_plan_pins_one_metadata_request_and_excludes_unsafe_files():
    api = Mock()
    api.model_info.return_value = SimpleNamespace(
        sha="a" * 40,
        card_data={"license": "apache-2.0"},
        gated=False,
        siblings=[
            SimpleNamespace(rfilename=name, size=5, lfs=None, blob_id="b" * 40)
            for name in [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors",
                "pytorch_model.bin",
                "modeling_model.py",
                "../escape.json",
            ]
        ],
    )
    result = plan_assets(
        {"models": {"e4b": {"repo_id": "example/model", "accepted_licenses": ["apache-2.0"]}}},
        ["e4b"],
        api,
    )
    assert api.model_info.call_count == 1
    assert result["models"][0]["revision"] == "a" * 40
    assert len(result["models"][0]["files"]) == 4


def test_unknown_license_blocks_plan():
    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha="a" * 40, card_data={})
    with pytest.raises(ValueError, match="License not reviewed"):
        plan_assets({"models": {"e4b": {"repo_id": "example/model"}}}, ["e4b"], api)


def test_cli_records_blocker_and_failure_exit(tmp_path, manifest):
    path, report = tmp_path / "manifest.json", tmp_path / "report.json"
    path.write_text(json.dumps(manifest))
    code = main(
        [
            "--report",
            str(report),
            "verify",
            "--manifest",
            str(path),
            "--root",
            str(tmp_path / "models"),
        ]
    )
    assert code == 1
    assert json.loads(report.read_text())["status"] == "blocked"
    assert json.loads(path.read_text())["status"] == "blocked"


def test_token_probe_missing_files_never_passes(tmp_path):
    report = tmp_path / "probe.json"
    assert token_main(["--model-dir", str(tmp_path), "--output", str(report)]) == 1
    assert json.loads(report.read_text())["status"] == "blocked"


@pytest.mark.parametrize("filename", ["../config.json", "model.py", "pytorch_model.bin"])
def test_manifest_rejects_unapproved_paths(tmp_path, manifest, filename):
    manifest["models"][0]["files"][0]["path"] = filename
    with pytest.raises(ValueError, match="Unsafe file"):
        prepare_assets(manifest, tmp_path, metadata_only=True)


def test_metadata_kind_cannot_disguise_weights(tmp_path, manifest):
    manifest["models"][0]["files"][1]["kind"] = "metadata"
    with pytest.raises(ValueError, match="Incorrect file kind"):
        prepare_assets(manifest, tmp_path, metadata_only=True)


def make_tokenizer_fixture(tmp_path, *, missing_answer=None, supports_thinking=True):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    from scripts.check_tokens import ANSWERS

    vocabulary = {"[UNK]": 0, **{token: index + 1 for index, token in enumerate(ANSWERS)}}
    if missing_answer:
        del vocabulary[missing_answer]
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"think_token": "<think>", "unk_token": "[UNK]"})
    )
    template = "{% if enable_thinking %}<think> {% endif %}" if supports_thinking else ""
    (tmp_path / "chat_template.jinja").write_text(
        template + "user {{ messages[0]['content'] }} model "
    )


def test_token_probe_success_records_all_mapping_and_template_evidence(tmp_path):
    make_tokenizer_fixture(tmp_path)
    output = tmp_path / "probe.json"
    assert token_main(["--model-dir", str(tmp_path), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["answer_count"] == 33
    assert len(set(report["answer_token_ids"].values())) == 33
    assert report["thinking_disabled"] is True
    assert report["shared_prefix_tokens"] > 0
    assert report["template_sha256"]


def test_template_without_provable_thinking_control_blocks(tmp_path):
    make_tokenizer_fixture(tmp_path, supports_thinking=False)
    output = tmp_path / "probe.json"
    assert token_main(["--model-dir", str(tmp_path), "--output", str(output)]) == 1
    assert "thinking disablement" in json.loads(output.read_text())["blocker"]


def test_unknown_token_cannot_count_as_single_answer_token(tmp_path):
    make_tokenizer_fixture(tmp_path, missing_answer="yes")
    output = tmp_path / "probe.json"
    assert token_main(["--model-dir", str(tmp_path), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert any("unknown" in failure for failure in report["failures"])


def test_full_verify_cannot_claim_ready_for_metadata_only_inventory(tmp_path, manifest):
    manifest["models"][0]["files"] = [manifest["models"][0]["files"][0]]
    with pytest.raises(ValueError, match="incomplete"):
        prepare_assets(manifest, tmp_path, metadata_only=False, download=fake_download)


def test_missing_safetensors_shard_is_rejected_before_download(tmp_path, manifest):
    manifest["models"][0]["files"][1]["path"] = "model-00001-of-00002.safetensors"
    with pytest.raises(ValueError, match="incomplete"):
        prepare_assets(manifest, tmp_path, metadata_only=False, download=fake_download)


def test_onejev_teacher_name_is_not_confused_with_prohibited_jev_weights():
    api = Mock()
    api.model_info.side_effect = RuntimeError("network-reached")
    with pytest.raises(RuntimeError, match="network-reached"):
        plan_assets({"models": {"teacher": {"repo_id": "OmniJev/OneJev-27B"}}}, ["teacher"], api)


@pytest.mark.parametrize("repo", ["autotrust/JEV-27B", "fork/JEV-27B-VL"])
def test_prohibited_jev_names_still_block_without_network(repo):
    api = Mock()
    with pytest.raises(ValueError, match="prohibited"):
        plan_assets({"models": {"teacher": {"repo_id": repo}}}, ["teacher"], api)
    api.model_info.assert_not_called()
