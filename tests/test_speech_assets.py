"""Speech asset ledgers keep readiness and blockers explicit."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_manifest() -> dict:
    return json.loads(Path("configs/speech-assets.json").read_text())


def test_selected_tts_payload_stays_below_one_gb():
    manifest = load_manifest()
    selected = [item for item in manifest["tts_candidates"] if item["selected"]]
    assert len(selected) == 1
    assert selected[0]["selected_payload_bytes"] < 1_000_000_000
    assert selected[0]["staging"]["remote_status"] == "downloaded_hash_verified"
    assert selected[0]["canary"]["status"] == "passed"
    assert selected[0]["canary"]["type"] == "text_to_misaki_g2p_to_tokenizer_to_onnx_cpu_wav"
    assert selected[0]["canary"]["runtime"]["python"].startswith("3.11.")
    assert selected[0]["canary"]["sample_rate_hz"] == 24_000
    assert selected[0]["canary"]["case_count"] == 4
    assert len(selected[0]["canary"]["wav_sha256s"]) == 4
    assert selected[0]["fixed_token_canary"]["not_full_text_tts"] is True


def test_common_voice_audio_is_not_claimed_ready_without_api_terms():
    manifest = load_manifest()
    selected = [item for item in manifest["common_voice_candidates"] if item["selected"]]
    assert len(selected) == 1
    candidate = selected[0]
    assert candidate["file"]["size_bytes"] < 1_000_000_000
    assert candidate["license"]["declared"] == "Creative Commons Zero v1.0 Universal (CC0-1.0)"
    assert candidate["status"] == "metadata_selected_download_blocked_by_api_key_and_terms"
    assert manifest["common_voice_public_alternative_check"]["status"].startswith("no_legitimate")
    assert "terms acceptance" in manifest["preparation_decision"]["blocked"][0]


def test_text_g2p_canary_report_records_rendered_wav_evidence():
    report = json.loads(Path("reports/speech-kokoro-text-g2p-cpu-canary.json").read_text())
    assert report["status"] == "passed"
    assert (
        report["quality_claim"]
        == "none; this is an execution and text-to-token coverage canary only"
    )
    assert report["runtime"]["python"].startswith("3.11.")
    assert report["runtime"]["cgroup"]["cpu_max"] == "50000 100000"
    assert report["case_count"] == 4
    assert {case["voice_id"] for case in report["cases"]} == {"af_heart", "am_puck"}
    assert len({case["sentence_id"] for case in report["cases"]}) == 2
    for case in report["cases"]:
        assert case["finite_waveform"] is True
        assert case["duration_seconds"] > 0
        assert case["sample_rate_hz"] == 24_000
        assert case["channels"] == 1
        assert case["wav_sha256"]
        assert case["unsupported_phoneme_characters"] == []
        assert case["token_count_without_padding"] <= 510
        assert case["style_index"] < case["voice_shape"][0]


def test_text_g2p_canary_config_matches_report():
    config = json.loads(Path("configs/speech-tts-canary.json").read_text())
    report = json.loads(Path("reports/speech-kokoro-text-g2p-cpu-canary.json").read_text())
    assert config["model"]["sha256"] == report["model_sha256"]
    assert config["tokenizer"]["sha256"] == report["tokenizer_sha256"]
    assert config["runtime"]["python_version_required"] == "3.11"
    assert config["runtime"]["cpu_lock"] == "/tmp/omnijev-cpu-heavy.lock"
    assert report["dependency_versions"]["misaki"] == config["g2p"]["version"]


def import_script(path: str) -> None:
    spec = importlib.util.spec_from_file_location("script_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_helpers_are_import_safe_without_optional_runtime_dependencies():
    import_script("scripts/kokoro_onnx_canary.py")
    import_script("scripts/kokoro_text_g2p_canary.py")
    import_script("scripts/mdc_download_dataset.py")


def test_cli_help_paths_do_not_need_optional_runtime_dependencies():
    for script in [
        "scripts/kokoro_onnx_canary.py",
        "scripts/kokoro_text_g2p_canary.py",
        "scripts/mdc_download_dataset.py",
    ]:
        completed = subprocess.run(
            [sys.executable, script, "--help"], check=True, capture_output=True, text=True
        )
        assert "usage:" in completed.stdout
