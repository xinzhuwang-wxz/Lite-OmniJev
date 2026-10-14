"""CPU comparison controls, without importing vLLM or claiming native acceptance."""

import base64
import io
import json
import wave
from types import SimpleNamespace

import pytest

from scripts.s0_vllm_processor import canonical_mm_data, compare_native, main


class NativeFields(dict):
    def get_data(self, **_kwargs):
        return self


def test_canonical_audio_decodes_exact_pcm_samples():
    np = pytest.importorskip("numpy")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x00\x80\xff\x7f")
    data = canonical_mm_data(
        [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(buffer.getvalue()).decode(),
                    "format": "wav",
                },
            }
        ],
        {},
    )
    waveform, rate = data["audio"]
    assert rate == 16000
    assert waveform.dtype == np.float32
    assert waveform.tolist() == [-1.0, 32767 / 32768]


def test_compares_expanded_ids_image_key_translation_and_audio_unpadding():
    torch = pytest.importorskip("torch")
    image = torch.ones(1, 2, 3)
    features = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])
    prepared = SimpleNamespace(
        inputs={
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": image,
            "image_position_ids": torch.zeros(1, 2, 2, dtype=torch.int64),
            "input_features": features,
            "input_features_mask": torch.tensor([[True, False]]),
        },
        http_media_io_kwargs={},
    )
    result = {
        "prompt_token_ids": [1, 2],
        "mm_kwargs": NativeFields(
            pixel_values=image,
            pixel_position_ids=prepared.inputs["image_position_ids"],
            input_features_padded=features[:, :1],
            input_features_mask=torch.tensor([[True]]),
        ),
        "mm_placeholders": {"image": [SimpleNamespace(offset=1, length=1)]},
    }
    assert compare_native(prepared, result)["passed"]
    result["prompt_token_ids"] = [1, 9]
    assert not compare_native(prepared, result)["passed"]
    result["prompt_token_ids"] = [1, 2]
    result["mm_kwargs"]["input_features_padded"] = features[:, :1] + 0.01
    assert not compare_native(prepared, result)["passed"]


def test_video_timestamp_and_position_translation():
    torch = pytest.importorskip("torch")
    inputs = {
        "input_ids": torch.tensor([[1, 2]]),
        "pixel_values_videos": torch.ones(2, 3, 4),
        "video_position_ids": torch.zeros(2, 3, 2, dtype=torch.int64),
        "num_frames_per_video": torch.tensor([2]),
    }
    prepared = SimpleNamespace(
        inputs=inputs, http_media_io_kwargs={"video": {"fps": 2.0, "frames_indices": [0, 4]}}
    )
    result = {
        "prompt_token_ids": [1, 2],
        "mm_kwargs": NativeFields(
            pixel_values_videos=inputs["pixel_values_videos"],
            pixel_position_ids_videos=inputs["video_position_ids"],
            video_frame_counts=inputs["num_frames_per_video"],
            video_timestamps=[[0.0, 2.0]],
        ),
        "mm_placeholders": {},
    }
    assert compare_native(prepared, result)["passed"]
    result["mm_kwargs"]["video_timestamps"] = [[0.0, 1.0]]
    assert not compare_native(prepared, result)["passed"]


def test_single_video_batch_translation_keeps_exact_pixel_equality():
    torch = pytest.importorskip("torch")
    pixels = torch.ones(1, 2, 3, 4)
    positions = torch.zeros(1, 2, 3, 2, dtype=torch.int64)
    prepared = SimpleNamespace(
        inputs={
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values_videos": pixels,
            "video_position_ids": positions,
        },
        http_media_io_kwargs={"video": {"fps": 2.0, "frames_indices": [0, 4]}},
    )
    native = NativeFields(
        pixel_values_videos=pixels[0].clone(),
        pixel_position_ids_videos=positions[0],
        video_frame_counts=torch.tensor([2]),
        video_timestamps=[[0.0, 2.0]],
    )
    result = {"prompt_token_ids": [1, 2], "mm_kwargs": native, "mm_placeholders": {}}
    assert compare_native(prepared, result)["passed"]
    native["pixel_values_videos"][0, 0, 0] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    assert not compare_native(prepared, result)["passed"]


def test_explicit_offline_environment_required_and_errors_sanitized(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    output = tmp_path / "report.json"
    args = [
        "--manifest",
        str(tmp_path / "PRIVATE"),
        "--asset-root",
        str(tmp_path),
        "--model-dir",
        str(tmp_path),
        "--hf-reference",
        str(tmp_path / "PRIVATE"),
        "--output",
        str(output),
    ]
    assert main(args) == 1
    report = json.loads(output.read_text())
    assert report["failure_stage"] == "before_native_import"
    assert report["weights_loaded"] is False
    assert "PRIVATE" not in output.read_text()
    with pytest.raises(SystemExit):
        main(args)


@pytest.mark.parametrize("native_count", [None, 1, 2])
def test_video_frame_count_stays_required_when_hf_version_omits_count(native_count):
    torch = pytest.importorskip("torch")
    prepared = SimpleNamespace(
        inputs={
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values_videos": torch.ones(2, 3, 4),
            "video_position_ids": torch.zeros(2, 3, 2, dtype=torch.int64),
        },
        http_media_io_kwargs={"video": {"fps": 2.0, "frames_indices": [0, 4]}},
    )
    native = NativeFields(
        pixel_values_videos=prepared.inputs["pixel_values_videos"],
        pixel_position_ids_videos=prepared.inputs["video_position_ids"],
        video_timestamps=[[0.0, 2.0]],
    )
    if native_count is not None:
        native["video_frame_counts"] = torch.tensor([native_count])
    result = {"prompt_token_ids": [1, 2], "mm_kwargs": native, "mm_placeholders": {}}
    assert compare_native(prepared, result)["passed"] is (native_count == 2)
