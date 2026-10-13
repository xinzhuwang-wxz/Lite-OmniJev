# P0 SwanLab training telemetry

SwanLab is optional for the P0 training smoke. A run uses project `Lite-OmniJev` and workspace `maxen`; do not replace them with example projects. The integration only records metrics computed by `scripts.train_smoke`: preflight example counts, max prompt tokens, real optimizer-step loss, learning rate, gradient norm, and completed step count.

## Official SDK behavior checked

The Python SDK documents `swanlab.init(project=..., workspace=..., mode=...)`, `swanlab.log(...)`, and `swanlab.finish()` for ending a run. Its modes are `online`, `offline`, `local`, and `disabled`; `online` uploads to the cloud, `offline` records locally, `local` stays local, and `disabled` neither uploads nor records ([swanlab.init docs](https://docs.swanlab.cn/en/api/py-init.html)). The settings docs list `api_key`, `mode`, and `log_dir` settings, with modes matching the init API ([swanlab.Settings docs](https://docs.swanlab.cn/en/api/py-settings.html)). The FAQ documents `SWANLAB_API_KEY` as the environment variable login path and notes that `mode="disabled"` prevents experiment creation and data logging ([SwanLab FAQ](https://docs.swanlab.cn/en/guide_cloud/experiment_track/FAQ.html)).

## Install

Telemetry is a separate optional dependency:

```bash
uv sync --frozen --extra training --extra monitoring --group dev
```

Without `--extra monitoring`, training and preflight still run. If telemetry is enabled while the SDK is missing, preflight reports a structured blocker before training starts.

## Modes

Telemetry is disabled by default and performs no SwanLab import or cloud-capable setup:

```bash
uv run python -m scripts.train_smoke preflight ...
```

Enable local/offline telemetry explicitly. When `LITE_OMNIJEV_SWANLAB=1` is set without a mode, the integration defaults to `offline`:

```bash
LITE_OMNIJEV_SWANLAB=1 \
uv run python -m scripts.train_smoke run ...
```

Set a mode deliberately when needed:

```bash
LITE_OMNIJEV_SWANLAB_MODE=offline uv run python -m scripts.train_smoke run ...
LITE_OMNIJEV_SWANLAB_MODE=local uv run python -m scripts.train_smoke run ...
LITE_OMNIJEV_SWANLAB_MODE=disabled uv run python -m scripts.train_smoke run ...
```

Online mode requires the official API key environment variable and refuses to start without it:

```bash
export SWANLAB_API_KEY=...
LITE_OMNIJEV_SWANLAB_MODE=online uv run python -m scripts.train_smoke run ...
```

Do not write API keys to reports, configs, commits, or shell history snippets. The report records only the mode, project, workspace, failure policy, and log directory.

## Failure policy

The default failure policy is `fail`: if SwanLab initialization or metric logging fails after telemetry has been enabled, the training report is marked failed and `swanlab.finish()` is still attempted. This keeps telemetry failures visible during GPU evidence runs.

For exploratory runs where training should continue despite telemetry issues, set:

```bash
LITE_OMNIJEV_SWANLAB_FAILURE=warn
```

Warn mode records `monitoring_error` in the report and continues with training. Use it only when the training artifact matters more than live monitoring.

Telemetry errors are sanitized before they enter `run.json` or the top-level training report. Reports include the exception class and a stable operation label such as `ValueError: SwanLab telemetry init failed`; they do not copy SDK exception strings, URLs, API keys, or request bodies.

## Logged fields

The integration never logs fabricated demonstration metrics. Values come from the current training process:

- `validation/examples`: number of prepared examples after template/token checks.
- `validation/max_prompt_tokens`: maximum prompt length after rendering with the official chat template.
- `train/loss`: mean cross-entropy over the real accumulated examples in the optimizer step.
- `train/learning_rate`: configured optimizer learning rate.
- `train/grad_norm`: clipped LoRA gradient norm for the step.
- `train/steps_completed`: completed optimizer steps.

Successful runs still write the local `run.json` report. SwanLab is additional evidence, not a replacement for the report, saved adapter, hashes, and CUDA/GPU validation required by the P0 training runbook.
