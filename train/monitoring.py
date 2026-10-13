"""Optional SwanLab telemetry for real training runs."""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

PROJECT = "Lite-OmniJev"
WORKSPACE = "maxen"
MODE_ENV = "LITE_OMNIJEV_SWANLAB_MODE"
ENABLE_ENV = "LITE_OMNIJEV_SWANLAB"
FAILURE_ENV = "LITE_OMNIJEV_SWANLAB_FAILURE"
LOGDIR_ENV = "LITE_OMNIJEV_SWANLAB_LOGDIR"
OFFICIAL_API_KEY_ENV = "SWANLAB_API_KEY"
VALID_MODES = {"online", "offline", "local", "disabled"}
VALID_FAILURE_POLICIES = {"warn", "fail"}


class Telemetry(Protocol):
    def log(self, metrics: dict[str, int | float], step: int | None = None) -> None: ...

    def finish(self, report: dict[str, Any]) -> None: ...


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() not in {"", "0", "false", "no", "off"}


@dataclass(frozen=True)
class SwanLabConfig:
    mode: str
    failure_policy: str
    logdir: Path

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> SwanLabConfig:
        explicit_mode = env.get(MODE_ENV)
        enabled = _truthy(env.get(ENABLE_ENV)) or explicit_mode is not None
        mode = (explicit_mode or ("offline" if enabled else "disabled")).lower()
        failure_policy = env.get(FAILURE_ENV, "fail").lower()
        logdir = Path(env.get(LOGDIR_ENV, "artifacts/monitoring/swanlab"))
        if mode not in VALID_MODES:
            raise ValueError(f"{MODE_ENV} must be one of {sorted(VALID_MODES)}")
        if failure_policy not in VALID_FAILURE_POLICIES:
            raise ValueError(f"{FAILURE_ENV} must be one of {sorted(VALID_FAILURE_POLICIES)}")
        return cls(mode=mode, failure_policy=failure_policy, logdir=logdir)

    def describe(self) -> dict[str, str | bool]:
        return {
            "provider": "swanlab",
            "enabled": self.enabled,
            "mode": self.mode,
            "project": PROJECT,
            "workspace": WORKSPACE,
            "failure_policy": self.failure_policy,
            "logdir": str(self.logdir),
        }

    def blockers(self, env: Mapping[str, str] = os.environ) -> list[str]:
        if not self.enabled:
            return []
        blockers = []
        if importlib.util.find_spec("swanlab") is None:
            blockers.append("monitoring: SwanLab SDK is not installed")
        if self.mode == "online" and not env.get(OFFICIAL_API_KEY_ENV):
            blockers.append(f"monitoring: {OFFICIAL_API_KEY_ENV} is required for online mode")
        return blockers


class DisabledTelemetry:
    def log(self, metrics: dict[str, int | float], step: int | None = None) -> None:
        del metrics, step

    def finish(self, report: dict[str, Any]) -> None:
        del report


class SwanLabTelemetry:
    def __init__(self, config: SwanLabConfig, run: Any):
        self.config = config
        self.run = run
        self.error: str | None = None

    def _handle_failure(self, exc: Exception) -> None:
        self.error = f"{type(exc).__name__}: {exc}"
        if self.config.failure_policy == "fail":
            raise exc

    def log(self, metrics: dict[str, int | float], step: int | None = None) -> None:
        try:
            self.run.log(metrics, step=step)
        except Exception as exc:
            self._handle_failure(exc)

    def finish(self, report: dict[str, Any]) -> None:
        try:
            module = importlib.import_module("swanlab")
            module.finish()
        except Exception as exc:
            self._handle_failure(exc)
        if self.error is not None:
            report["monitoring_error"] = self.error


def start_swanlab(config: SwanLabConfig, report: dict[str, Any]) -> Telemetry:
    if not config.enabled:
        return DisabledTelemetry()
    try:
        module = importlib.import_module("swanlab")
        run = module.init(
            project=PROJECT,
            workspace=WORKSPACE,
            name=f"p0-smoke-{report['code_revision'][:12]}",
            description=report["scope"],
            tags=["p0", "smoke", "text-only"],
            config={
                "code_revision": report["code_revision"],
                "model_revision": report["model_revision"],
                "training": report["config"],
                "resume_supported": report["resume_supported"],
            },
            log_dir=str(config.logdir),
            mode=config.mode,
            public=False,
        )
        return SwanLabTelemetry(config, run)
    except Exception as exc:
        if config.failure_policy == "fail":
            raise
        report["monitoring_error"] = f"{type(exc).__name__}: {exc}"
        return DisabledTelemetry()
