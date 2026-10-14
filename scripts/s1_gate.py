"""组装S1门槛测量值;判定交给eval.gate;不做质量声称。

CLI: python -m scripts.s1_gate --smoke-run R --smoke-selfacc R --dev-30k R --dev-10k R
    --output metrics.json [--baseline-onejev4 X --baseline-e4b Y]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, TypeGuard

from scripts.assets import write_json

# dev500 OneJev4 文本零样本基线（reports/s0-baseline-summary.md，text 行 96.6%）。
ONEJEV4_TEXT_DEV500 = 0.966
# dev500 E4B 文本零样本基线（reports/s0-baseline-summary.md，text 行 93.6%）。
E4B_TEXT_DEV500 = 0.936
# S1 dev 评测协议：500 题、两遍完全一致。
DEV_SAMPLES = 500
DEV_PASSES = 2


class GateInputError(ValueError):
    """输入报告缺失或不符合 S1 评测契约。"""


def _finite_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_report(path: Path, label: str) -> dict[str, Any]:
    """读取一份 run/eval 报告，要求 status == "passed" 且 exit_code == 0。"""
    if not path.is_file():
        raise GateInputError(f"{label} report not found: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateInputError(f"{label} report is not readable JSON: {path} ({exc})") from exc
    if not isinstance(report, dict):
        raise GateInputError(f"{label} report is not a JSON object: {path}")
    # 训练 run 终态为 "completed"（train.trainer），评测报告为 "passed"（scripts.s1_dev_eval）。
    if report.get("status") not in {"passed", "completed"} or report.get("exit_code") != 0:
        raise GateInputError(
            f"{label} report did not pass: {path} "
            f"(status={report.get('status')!r}, exit_code={report.get('exit_code')!r})"
        )
    return report


def _metric(report: dict[str, Any], key: str, label: str, path: Path) -> float:
    metrics = report.get("metrics")
    value = metrics.get(key) if isinstance(metrics, dict) else None
    if not _finite_number(value):
        raise GateInputError(f"{label} report metrics.{key} is missing or not finite: {path}")
    return float(value)


def load_dev_report(path: Path, label: str) -> tuple[float, float]:
    """dev500 两遍评测报告；强制协议字段后返回 (product_accuracy, ece15)。"""
    report = load_report(path, label)
    if report.get("repeat_identical") is not True:
        raise GateInputError(f"{label} report repeat_identical is not true: {path}")
    if report.get("samples") != DEV_SAMPLES or report.get("passes") != DEV_PASSES:
        raise GateInputError(
            f"{label} report is off the dev500 two-pass protocol: {path} "
            f"(samples={report.get('samples')!r}, passes={report.get('passes')!r})"
        )
    return _metric(report, "product_accuracy", label, path), _metric(report, "ece15", label, path)


def loss_decreasing(losses: object) -> float:
    """口径：首/尾至多20步(不重叠半段)均值比较，尾均值 < 首均值为 1.0。

    300步硬目标CE的逐步loss噪声大（单步可在0与3之间跳动），成对严格下降必然为0，
    违背该门"优化有效"的本意；窗口均值仍能区分正常收敛与发散/不学习。
    任何非有限值或不足两个损失值为 0.0。2026-10-15 负责人口径决定，依据记录于票 s1:01。
    """
    if not isinstance(losses, list) or len(losses) < 2:
        return 0.0
    if any(not _finite_number(value) for value in losses):
        return 0.0
    window = min(20, len(losses) // 2)
    first = math.fsum(losses[:window]) / window
    last = math.fsum(losses[-window:]) / window
    return 1.0 if last < first else 0.0


def assemble_metrics(
    *,
    smoke_run_path: Path,
    smoke_selfacc_path: Path,
    dev_30k_path: Path,
    dev_10k_path: Path,
    baseline_onejev4: float,
    baseline_e4b: float,
) -> dict[str, float]:
    """从四份报告组装 configs/gates.json S1 节对齐的扁平测量值。"""
    smoke_run = load_report(smoke_run_path, "smoke run")
    smoke_selfacc = load_report(smoke_selfacc_path, "smoke selfacc")
    acc_30k, ece_30k = load_dev_report(dev_30k_path, "dev-30k")
    acc_10k, _ = load_dev_report(dev_10k_path, "dev-10k")

    config = smoke_run.get("config")
    steps = config.get("steps") if isinstance(config, dict) else None
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise GateInputError(
            f"smoke run report config.steps is missing or not an integer: {smoke_run_path}"
        )
    accuracy = _metric(smoke_selfacc, "product_accuracy", "smoke selfacc", smoke_selfacc_path)

    return {
        "smoke_steps": float(steps),
        "smoke_accuracy": accuracy,
        "loss_decreasing": loss_decreasing(smoke_run.get("losses")),
        "text_vs_onejev_delta": acc_30k - baseline_onejev4,
        "text_vs_base_delta": acc_30k - baseline_e4b,
        "text_ece": ece_30k,
        "30k_vs_10k_delta": acc_30k - acc_10k,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-run", type=Path, required=True, help="smoke 训练报告")
    parser.add_argument("--smoke-selfacc", type=Path, required=True, help="smoke 自测报告")
    parser.add_argument("--dev-30k", type=Path, required=True, help="30k adapter dev500 评测报告")
    parser.add_argument("--dev-10k", type=Path, required=True, help="10k adapter dev500 评测报告")
    parser.add_argument("--output", type=Path, required=True, help="输出的 S1 metrics JSON")
    parser.add_argument("--baseline-onejev4", type=float, default=ONEJEV4_TEXT_DEV500)
    parser.add_argument("--baseline-e4b", type=float, default=E4B_TEXT_DEV500)
    args = parser.parse_args(argv)
    try:
        metrics = assemble_metrics(
            smoke_run_path=args.smoke_run,
            smoke_selfacc_path=args.smoke_selfacc,
            dev_30k_path=args.dev_30k,
            dev_10k_path=args.dev_10k,
            baseline_onejev4=args.baseline_onejev4,
            baseline_e4b=args.baseline_e4b,
        )
    except GateInputError as exc:
        print(f"invalid S1 gate input: {exc}", file=sys.stderr)
        return 1
    write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
