"""CPU-only official video processor budget probe on fixed public development items."""

import hashlib
import json
import platform
import time
from pathlib import Path

import transformers

from scripts import s0_development_baselines as baseline

root = Path("/root/autodl-tmp/s1-text-lora-001")
base = Path(
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2"
)
reference = json.loads((root / "reports/s3-stable-video-screen-001.json").read_text())
args = reference["command"]
data = Path(args[args.index("--data") + 1])
media = Path(args[args.index("--media-root") + 1])
assert baseline.sha256_file(data) == reference["data_sha256"]
samples = sorted(
    baseline.load_samples(data),
    key=lambda s: hashlib.sha256(("video-budget:42:" + s.id).encode()).hexdigest(),
)[:6]
assert all(s.meta.pool == "dev" for s in samples)
processor = baseline.default_processor_factory(base)
source = Path(transformers.__file__).parent / "models/gemma4/video_processing_gemma4.py"
result = {
    "scope": "CPU official processor inputs only; no model forward or quality claim",
    "gpu_used": False,
    "model_revision": base.name,
    "data_sha256": baseline.sha256_file(data),
    "default_video_soft_tokens": processor.video_processor.max_soft_tokens,
    "transformers": transformers.__version__,
    "processor_code_sha256": baseline.sha256_file(source),
    "python": platform.python_version(),
    "seed": 42,
    "video_frames": 16,
    "rows": [],
    "cost": None,
    "runner_sha256": baseline.sha256_file(Path(__file__)),
}
for budget in (70, 140, 280):

    class Processor:
        def __init__(self, value):
            self.budget = value

        def __getattr__(self, name):
            return getattr(processor, name)

        def __call__(self, **kwargs):
            kwargs["videos_kwargs"] = {
                **kwargs.get("videos_kwargs", {}),
                "max_soft_tokens": self.budget,
            }
            return processor(**kwargs)

    for sample in samples:
        start = time.monotonic()
        p = baseline.prepare_sample(sample, Processor(budget), media)
        result["rows"].append(
            {
                "sample_id_sha256": hashlib.sha256(sample.id.encode()).hexdigest(),
                "budget": budget,
                "input_tokens": p.inputs["input_ids"].shape[-1],
                "inputs": {
                    k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                    for k, v in p.inputs.items()
                    if hasattr(v, "shape")
                },
                "prepare_seconds": time.monotonic() - start,
                "answer_token_ids": p.answer_token_ids,
            }
        )
result["exit_code"] = 0
path = root / "reports/s4-video-budget-cpu-001.json"
with path.open("x") as h:
    json.dump(result, h, indent=2)
print(
    json.dumps(
        {
            "default_video_soft_tokens": result["default_video_soft_tokens"],
            "ranges": {
                b: [
                    min(r["input_tokens"] for r in result["rows"] if r["budget"] == b),
                    max(r["input_tokens"] for r in result["rows"] if r["budget"] == b),
                ]
                for b in (70, 140, 280)
            },
            "gpu_used": False,
        }
    )
)
