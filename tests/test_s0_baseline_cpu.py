import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from scripts import s0_baseline_cpu as baseline


def _content_for(path: str) -> bytes:
    return path.encode()


def _git_blob_sha1(data: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(data)}\0".encode())
    digest.update(data)
    return digest.hexdigest()


def asset_file(path: str, *, kind: str = "metadata") -> dict[str, Any]:
    data = _content_for(path)
    return {
        "path": path,
        "bytes": len(data),
        "required": True,
        "kind": kind,
        "lfs_sha256": "a" * 64 if kind == "weights" else None,
        "git_blob_sha1": _git_blob_sha1(data) if kind == "metadata" else None,
    }


def model(model_id: str, repo: str, revision: str = "1" * 40) -> dict[str, Any]:
    return {
        "id": model_id,
        "display_name": model_id,
        "repo_id": repo,
        "revision": revision,
        "role": "test-role",
        "license": "apache-2.0",
        "files": [
            asset_file("config.json"),
            asset_file("tokenizer.json"),
            asset_file("tokenizer_config.json"),
            asset_file("chat_template.jinja"),
            asset_file("model.safetensors", kind="weights"),
        ],
    }


def manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "models": [
            model("gemma4-e4b-it", "google/gemma-4-E4B-it", "e" * 40),
            model("gemma4-e2b-it", "google/gemma-4-E2B-it", "d" * 40),
            model("jev-omni-12b", "akhilaaa3/Jev-Omni", "c" * 40),
            model("onejev-4b", "OmniJev/OneJev-4B", "b" * 40),
            model("onejev-27b", "OmniJev/OneJev-27B", "a" * 40),
            model("valen-preview-0923", "Valen-Team/Valen-Preview-0923", "9" * 40),
            model("qwen3.5-2b-for-valen", "Qwen/Qwen3.5-2B", "8" * 40),
        ],
    }


def write_manifest(tmp_path: Path, value: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "assets.json"
    path.write_text(json.dumps(value or manifest()))
    return path


class FakeTokenizer:
    all_special_ids = [0]
    unk_token_id = 0
    init_kwargs = {"think_token": "<think>"}

    def __init__(self, *, split_u: bool = False, supports_thinking: bool = True) -> None:
        tokens = [
            *"ABCDEFGHIJKLMNOPQRST",
            "U",
            "yes",
            "no",
            *[str(i) for i in range(10)],
            "<think>",
        ]
        self.vocab = {token: index + 1 for index, token in enumerate(dict.fromkeys(tokens))}
        self.split_u = split_u
        self.supports_thinking = supports_thinking

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if text in self.vocab:
            if text == "U" and self.split_u:
                return [self.vocab["U"], self.vocab["A"]]
            return [self.vocab[text]]
        return [100 + index for index, _ in enumerate(text.split() or [text])]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        reverse = {value: key for key, value in self.vocab.items()}
        return str(reverse.get(ids[0], "?"))

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str | list[int]:
        rendered = (
            ("<think> " if enable_thinking and self.supports_thinking else "")
            + "user "
            + str(messages[0]["content"])
            + " model"
        )
        return self.encode(rendered) if tokenize else rendered


class FakeTensor:
    dtype = "torch.int64"

    def __init__(self, data: bytes = b"abc") -> None:
        self.data = data

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> Any:
        return SimpleNamespace(
            shape=(1, 3), dtype=SimpleNamespace(kind="i"), tobytes=lambda: self.data
        )


class FakeProcessor:
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str | list[int]:
        return "processor prompt"

    def __call__(self, **kwargs: Any) -> dict[str, FakeTensor]:
        return {"input_ids": FakeTensor(b"ids"), "pixel_values": FakeTensor(b"pixels")}


def make_model_dirs(root: Path, value: dict[str, Any]) -> None:
    for entry in value["models"]:
        directory = root / str(entry["repo_id"]).replace("/", "__") / str(entry["revision"])
        directory.mkdir(parents=True)
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ):
            (directory / name).write_bytes(_content_for(name))


def test_validate_expands_five_core_plus_valen_base_pair(tmp_path: Path) -> None:
    path = write_manifest(tmp_path)
    report = baseline.validate_report(path, model_root=tmp_path / "models")
    assert report["exit_code"] == 0
    assert report["subject_count"] == 6
    subjects = {subject["id"]: subject for subject in report["subjects"]}
    assert (
        subjects["valen-2b-qwen3.5-base"]["adapter"]["repo_id"] == "Valen-Team/Valen-Preview-0923"
    )
    assert subjects["valen-2b-qwen3.5-base"]["base"]["repo_id"] == "Qwen/Qwen3.5-2B"
    assert report["gpu_used"] is False
    assert report["weights_loaded"] is False
    assert report["quality_admitted"] is False


def test_missing_required_model_rejected_before_probe(tmp_path: Path) -> None:
    value = manifest()
    value["models"] = value["models"][:-1]
    path = write_manifest(tmp_path, value)
    with pytest.raises(ValueError, match="missing required"):
        baseline.validate_report(path)


def test_run_records_blocked_missing_local_metadata_without_weight_claim(tmp_path: Path) -> None:
    path = write_manifest(tmp_path)
    report = baseline.run_report(path, tmp_path / "missing")
    assert report["status"] == "partial"
    assert report["exit_code"] == 1
    assert report["summary"]["blocked"] == 6
    assert all(
        subject["blocker"] == "local_model_directory_missing" for subject in report["subjects"]
    )
    assert report["weights_loaded"] is False
    assert report["quality_admitted"] is False


def test_run_probes_tokenizer_processor_and_real_metadata_hashes(tmp_path: Path) -> None:
    value = manifest()
    path = write_manifest(tmp_path, value)
    root = tmp_path / "models"
    make_model_dirs(root, value)
    report = baseline.run_report(
        path,
        root,
        tokenizer_factory=lambda _: FakeTokenizer(),
        processor_factory=lambda _: FakeProcessor(),
    )
    assert report["status"] == "partial"
    assert report["summary"]["partial"] == 6
    first = report["subjects"][0]
    assert first["local_metadata_sha256"]
    assert first["tokenizer"]["answer_readout"]["status"] == "passed"
    assert first["tokenizer"]["thinking"]["status"] == "proved_disabled"
    assert first["processor"]["class"].endswith("FakeProcessor")
    assert first["published_teacher_readout_verified"] is False


def test_non_single_token_answer_marks_subject_failed(tmp_path: Path) -> None:
    value = manifest()
    path = write_manifest(tmp_path, value)
    root = tmp_path / "models"
    make_model_dirs(root, value)
    report = baseline.run_report(
        path,
        root,
        tokenizer_factory=lambda _: FakeTokenizer(split_u=True),
        processor_factory=lambda _: FakeProcessor(),
    )
    assert report["status"] == "failed"
    assert report["exit_code"] == 1
    assert report["subjects"][0]["tokenizer"]["answer_readout"]["status"] == "failed"
    assert any(
        "U: token_count=2" in item
        for item in report["subjects"][0]["tokenizer"]["answer_readout"]["failures"]
    )


def test_tampered_required_metadata_rejected_before_tokenizer_factory(tmp_path: Path) -> None:
    value = manifest()
    path = write_manifest(tmp_path, value)
    root = tmp_path / "models"
    make_model_dirs(root, value)
    for entry in value["models"]:
        bad = (
            root / str(entry["repo_id"]).replace("/", "__") / str(entry["revision"]) / "config.json"
        )
        bad.write_bytes(b"tampered")
    calls: list[str] = []

    def factory(_: Path) -> FakeTokenizer:
        calls.append("tokenizer")
        return FakeTokenizer()

    report = baseline.run_report(
        path,
        root,
        tokenizer_factory=factory,
        processor_factory=lambda _: FakeProcessor(),
    )
    assert calls == []
    assert report["exit_code"] == 1
    assert report["subjects"][0]["blocker"] == "local_metadata_invalid"


def test_valen_adapter_directory_is_required_and_hash_checked(tmp_path: Path) -> None:
    value = manifest()
    path = write_manifest(tmp_path, value)
    root = tmp_path / "models"
    make_model_dirs(root, value)
    adapter = value["models"][5]
    adapter_dir = root / str(adapter["repo_id"]).replace("/", "__") / str(adapter["revision"])
    for child in adapter_dir.iterdir():
        child.unlink()
    adapter_dir.rmdir()
    report = baseline.run_report(
        path,
        root,
        tokenizer_factory=lambda _: FakeTokenizer(),
        processor_factory=lambda _: FakeProcessor(),
    )
    valen = next(
        subject for subject in report["subjects"] if subject["id"] == "valen-2b-qwen3.5-base"
    )
    assert report["exit_code"] == 1
    assert valen["blocker"] == "adapter_metadata_invalid"


def test_thinking_not_proved_is_partial_and_nonzero(tmp_path: Path) -> None:
    value = manifest()
    path = write_manifest(tmp_path, value)
    root = tmp_path / "models"
    make_model_dirs(root, value)
    report = baseline.run_report(
        path,
        root,
        tokenizer_factory=lambda _: FakeTokenizer(supports_thinking=False),
        processor_factory=lambda _: FakeProcessor(),
    )
    assert report["status"] == "partial"
    assert report["exit_code"] == 1
    assert report["subjects"][0]["tokenizer"]["thinking"]["status"] == "not_proved"


def test_public_image_probe_uses_hash_checked_fixture_and_never_outputs_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeImage:
        width = 1
        height = 1

        def __enter__(self) -> "FakeImage":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
            return False

        def load(self) -> None:
            return None

        def convert(self, mode: str) -> Any:
            return SimpleNamespace(mode=mode)

    fake_image_module = SimpleNamespace(open=lambda _path: FakeImage())
    real_import = __import__("importlib").import_module

    def fake_import(name: str) -> Any:
        if name == "PIL.Image":
            return fake_image_module
        return real_import(name)

    monkeypatch.setattr("scripts.s0_baseline_cpu.importlib.import_module", fake_import)
    value = manifest()
    path = write_manifest(tmp_path, value)
    root = tmp_path / "models"
    make_model_dirs(root, value)
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    image_path = asset_root / "image.png"
    image_path.write_bytes(b"not a real image; fake decoder owns this test")
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    public = tmp_path / "public.json"
    base_meta = {
        "source_dataset": "unit",
        "source_revision": "r1",
        "source_sha256": "1" * 64,
        "license": "internal",
        "usage": "research",
        "pool": "dev",
        "family_id": "f1",
        "data_version": "v1",
    }
    public.write_text(
        json.dumps(
            [
                {
                    "id": "text-1",
                    "category": "text",
                    "prefix": {"state": "state", "media": []},
                    "question": {
                        "qid": "q",
                        "type": "choice",
                        "instruction": "pick",
                        "options": ["x", "y"],
                    },
                    "meta": {**base_meta, "modality": ["text"]},
                    "video_frames": 16,
                },
                {
                    "id": "image-1",
                    "category": "image",
                    "prefix": {
                        "state": "state",
                        "media": [{"type": "image", "uri": "image.png", "sha256": image_sha}],
                    },
                    "question": {
                        "qid": "q",
                        "type": "choice",
                        "instruction": "pick",
                        "options": ["x", "y"],
                    },
                    "meta": {**base_meta, "modality": ["image"]},
                    "video_frames": 16,
                },
            ]
        )
    )
    report = baseline.run_report(
        path,
        root,
        public_manifest=public,
        public_asset_root=asset_root,
        allow_partial_public_manifest=True,
        tokenizer_factory=lambda _: FakeTokenizer(),
        processor_factory=lambda _: FakeProcessor(),
    )
    image = report["subjects"][0]["public_inputs"]["image"]
    assert image["status"] == "passed"
    encoded = json.dumps(image)
    assert "image.png" not in encoded
    assert image_sha in encoded


def test_image_header_pixel_limit_rejects_before_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HugeImage:
        width = 40_000_001
        height = 1
        loaded = False

        def __enter__(self) -> "HugeImage":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
            return False

        def load(self) -> None:
            HugeImage.loaded = True

    image_path = tmp_path / "huge.png"
    image_path.write_bytes(b"huge")
    monkeypatch.setattr(
        "scripts.s0_baseline_cpu.importlib.import_module",
        lambda name: (
            SimpleNamespace(open=lambda _path: HugeImage())
            if name == "PIL.Image"
            else __import__(name)
        ),
    )
    case = SimpleNamespace(
        id="case",
        category="image",
        prefix=SimpleNamespace(
            state="state",
            media=[
                SimpleNamespace(
                    type="image", uri="huge.png", sha256=hashlib.sha256(b"huge").hexdigest()
                )
            ],
        ),
        question=SimpleNamespace(qid="q", type="choice", instruction="pick", options=["x", "y"]),
    )
    with pytest.raises(ValueError, match="pixel limit"):
        baseline.probe_image_case(FakeProcessor(), case, tmp_path)
    assert HugeImage.loaded is False


def test_cli_failure_leaves_report_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    assert (
        baseline.main(
            [
                "validate",
                "--asset-manifest",
                str(tmp_path / "missing.json"),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["gpu_used"] is False
    with pytest.raises(SystemExit):
        baseline.main(
            [
                "validate",
                "--asset-manifest",
                str(tmp_path / "missing.json"),
                "--output",
                str(output),
            ]
        )
