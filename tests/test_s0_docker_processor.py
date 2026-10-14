"""Resource and partial-transfer controls; not Docker/native processor acceptance."""

import http.client
from types import SimpleNamespace

import pytest

from scripts.s0_docker_processor import docker_run, read_prefix, require_budget


def test_partial_read_preserves_only_contiguous_prefix():
    class Response:
        calls = 0

        def read(self, width):
            self.calls += 1
            if self.calls == 1:
                return b"ab"
            raise http.client.IncompleteRead(b"cd", 2)

    assert read_prefix(Response(), 6) == b"abcd"


def test_empty_transfer_error_is_not_retried_as_progress():
    class Response:
        def read(self, width):
            raise TimeoutError("controlled empty timeout")

    with pytest.raises(TimeoutError):
        read_prefix(Response(), 10)


def test_budget_includes_proposed_extra_bytes(tmp_path, monkeypatch):
    import scripts.s0_docker_processor as runner

    monkeypatch.setattr(runner, "DISK_LIMIT", 10)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    (tmp_path / "existing").write_bytes(b"1234")
    require_budget(tmp_path, 6)
    with pytest.raises(ValueError, match="budget"):
        require_budget(tmp_path, 7)


def test_existing_container_is_never_reused_or_stopped(tmp_path, monkeypatch):
    import scripts.s0_docker_processor as runner

    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    with pytest.raises(ValueError, match="already_exists"):
        docker_run(
            tmp_path, image="fake", name="lite-omnijev-test", command=["python"], offline=True
        )
    assert calls == [["docker", "container", "inspect", "lite-omnijev-test"]]
