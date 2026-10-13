"""Speech asset ledger keeps readiness and blockers explicit."""

import json
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
    assert selected[0]["canary"]["sample_rate_hz"] == 24_000
    assert selected[0]["canary"]["wav_sha256"]


def test_common_voice_audio_is_not_claimed_ready_without_api_terms():
    manifest = load_manifest()
    selected = [item for item in manifest["common_voice_candidates"] if item["selected"]]
    assert len(selected) == 1
    candidate = selected[0]
    assert candidate["file"]["size_bytes"] < 1_000_000_000
    assert candidate["license"]["declared"] == "Creative Commons Zero v1.0 Universal (CC0-1.0)"
    assert candidate["status"] == "metadata_selected_download_blocked_by_api_key_and_terms"
    assert manifest["common_voice_public_alternative_check"]["status"].startswith("no_legitimate")


def test_canary_report_records_rendered_wav_evidence():
    report = json.loads(Path("reports/speech-kokoro-cpu-wav-canary.json").read_text())
    assert report["status"] == "passed"
    assert report["quality_claim"] == "none; this is an execution canary only"
    assert report["finite_waveform"] is True
    assert report["duration_seconds"] > 0
    assert report["sample_rate_hz"] == 24_000
    assert report["channels"] == 1
    assert report["wav_sha256"]
