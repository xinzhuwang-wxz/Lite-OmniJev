"""组装S2门槛测量值;判定交给eval.gate;不做质量声称。

CLI: python -m scripts.s2_gate --image R --screenshot R --text R --smoke-run R
    --output metrics.json [--baseline-onejev4-image X --baseline-onejev4-screenshot Y
    --baseline-s1-text Z]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, TypeGuard

from scripts.assets import write_json

# dev500 OneJev4 图像零样本基线（reports/s0-baseline-summary.md，image 行 83.6%）。
ONEJEV4_IMAGE_DEV500 = 0.836
# dev500 OneJev4 截图零样本基线（reports/s0-baseline-summary.md，screenshot 行 66.2%）。
ONEJEV4_SCREENSHOT_DEV500 = 0.662
# S1 30k adapter dev500 文本成绩（reports/s1-dev-eval-30k-001.json），text 回归参照。
S1_TEXT_DEV500_ADAPTER = 0.996
# S2 dev 评测协议：500 题、两遍完全一致。
DEV_SAMPLES = 500
DEV_PASSES = 2


FROZEN_DEV_SHA256 = {
    "image": "38efa78efe62ebdd8523c7372b8c0ac87a9030a18d429db488f44e5aad08e7de",
    "screenshot": "19c9d2c3652cb6554b490573754cb02ee0367d89a1161e570b9ee01e8b7689a2",
    "text": "8c7809e8b07673904b71bc4676a49a5e31abb22ee7126f2dc09e8fde45b4e070",
}


class GateInputError(ValueError):
    """输入报告缺失或不符合 S2 评测契约。"""


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
    # 训练 run 终态为 "completed"（train.trainer），评测报告为 "passed"（scripts.s2_dev_eval）。
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


def load_dev_report(path: Path, label: str) -> dict[str, Any]:
    """dev500 两遍评测报告；强制协议字段后返回报告供取 metrics。"""
    report = load_report(path, label)
    if report.get("repeat_identical") is not True:
        raise GateInputError(f"{label} report repeat_identical is not true: {path}")
    if report.get("samples") != DEV_SAMPLES or report.get("passes") != DEV_PASSES:
        raise GateInputError(
            f"{label} report is off the dev500 two-pass protocol: {path} "
            f"(samples={report.get('samples')!r}, passes={report.get('passes')!r})"
        )
    return report


def image_budget(report: dict[str, Any]) -> int:
    """Legacy frozen reports used official 280; new profiles must be explicit and effective."""
    if "research_input_profile" in report:
        raise GateInputError("research input overrides are not formal stage evidence")
    profile = report.get("image_max_soft_tokens")
    if profile is None:
        return 280
    if not isinstance(profile, dict):
        raise GateInputError("invalid image budget profile")
    effective = profile.get("effective")
    requested = profile.get("requested")
    if type(effective) is not int or effective not in (70, 140, 280, 560, 1120):
        raise GateInputError("invalid effective image budget")
    if requested is not None and (type(requested) is not int or requested != effective):
        raise GateInputError("requested image budget was not applied")
    return effective


def assemble_metrics(
    *,
    image_path: Path,
    screenshot_path: Path,
    text_path: Path,
    smoke_run_path: Path,
    baseline_onejev4_image: float,
    baseline_onejev4_screenshot: float,
    baseline_s1_text: float,
) -> dict[str, float]:
    """从四份报告组装 configs/gates.json S2 节对齐的扁平测量值。

    前四个键对应 S2 门；其余为非门诊断键，随 metrics 一起落盘供报告引用
    （eval.gate 只遍历阈值文件的键，metrics 里多出的诊断键不影响判定）。
    """
    image = load_dev_report(image_path, "image dev")
    screenshot = load_dev_report(screenshot_path, "screenshot dev")
    text = load_dev_report(text_path, "text dev")
    smoke_run = load_report(smoke_run_path, "smoke run")

    reports = {"image": image, "screenshot": screenshot, "text": text}
    identities = {(r.get("model_revision"), r.get("adapter_sha256")) for r in reports.values()}
    if len(identities) != 1 or any(
        not isinstance(model, str)
        or not re.fullmatch(r"[a-f0-9]{40}", model)
        or not isinstance(adapter, str)
        or not re.fullmatch(r"[a-f0-9]{64}", adapter)
        for model, adapter in identities
    ):
        raise GateInputError("all S2 reports must identify the same model revision and adapter")
    for name, report in reports.items():
        if report.get("data_sha256") != FROZEN_DEV_SHA256[name]:
            raise GateInputError(f"{name} must use its frozen development dataset")
    if image_budget(image) != image_budget(screenshot):
        raise GateInputError("image and screenshot must use the same deployed image budget")

    config = smoke_run.get("config")
    steps = config.get("steps") if isinstance(config, dict) else None
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise GateInputError(
            f"smoke run report config.steps is missing or not an integer: {smoke_run_path}"
        )

    acc_image = _metric(image, "product_accuracy", "image dev", image_path)
    acc_screenshot = _metric(screenshot, "product_accuracy", "screenshot dev", screenshot_path)
    acc_text = _metric(text, "product_accuracy", "text dev", text_path)
    ece_image = _metric(image, "ece15", "image dev", image_path)
    ece_screenshot = _metric(screenshot, "ece15", "screenshot dev", screenshot_path)
    cf_image = _metric(image, "counterfactual_content_match_rate", "image dev", image_path)
    cf_screenshot = _metric(
        screenshot,
        "counterfactual_content_match_rate",
        "screenshot dev",
        screenshot_path,
    )

    return {
        "image_vs_onejev_delta": acc_image - baseline_onejev4_image,
        "screenshot_vs_onejev_delta": acc_screenshot - baseline_onejev4_screenshot,
        "text_regression": baseline_s1_text - acc_text,
        # cf 口径：image 与 screenshot 两个模态都须各自达标；取 min 使
        # counterfactual_flip_rate 的单阈值判定等价于逐模态 ≥ 阈值。
        "counterfactual_flip_rate": min(cf_image, cf_screenshot),
        "image_accuracy": acc_image,
        "screenshot_accuracy": acc_screenshot,
        "text_accuracy": acc_text,
        "image_ece15": ece_image,
        "screenshot_ece15": ece_screenshot,
        "image_cf_match": cf_image,
        "screenshot_cf_match": cf_screenshot,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="image dev500 评测报告")
    parser.add_argument("--screenshot", type=Path, required=True, help="screenshot dev500 评测报告")
    parser.add_argument("--text", type=Path, required=True, help="text dev500 评测报告")
    parser.add_argument("--smoke-run", type=Path, required=True, help="smoke 训练报告")
    parser.add_argument("--output", type=Path, required=True, help="输出的 S2 metrics JSON")
    parser.add_argument("--baseline-onejev4-image", type=float, default=ONEJEV4_IMAGE_DEV500)
    parser.add_argument(
        "--baseline-onejev4-screenshot", type=float, default=ONEJEV4_SCREENSHOT_DEV500
    )
    parser.add_argument("--baseline-s1-text", type=float, default=S1_TEXT_DEV500_ADAPTER)
    args = parser.parse_args(argv)
    try:
        metrics = assemble_metrics(
            image_path=args.image,
            screenshot_path=args.screenshot,
            text_path=args.text,
            smoke_run_path=args.smoke_run,
            baseline_onejev4_image=args.baseline_onejev4_image,
            baseline_onejev4_screenshot=args.baseline_onejev4_screenshot,
            baseline_s1_text=args.baseline_s1_text,
        )
    except GateInputError as exc:
        print(f"invalid S2 gate input: {exc}", file=sys.stderr)
        return 1
    write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
