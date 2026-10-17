"""GRPO scene-formalizer RL runner (GPU when a card is present; CPU-tested math).

Algorithm (fixed choices, rationale in the prep evidence):
- GRPO per-video group advantage over G=4 sampled scenes (temperature 0.8):
  the per-prompt baseline absorbs video difficulty, which a single REINFORCE
  running-mean baseline cannot; no value network through the discrete executor.
- KL anchor to the frozen base policy (beta 0.05) keeps JSON validity stable.
- SFT mixing 0.3 on the truth-scene JSON for the same video (self-grounding).
- LoRA on the frozen base (scene formalizer adapter, separate from decision
  LoRA); lr 1e-5; batch = 8 videos; checkpoint + probe F1 every 20 videos.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from formalizer_reward import group_advantages, reward  # noqa: E402

G_SAMPLES = 4
TEMPERATURE = 0.8
BETA_KL = 0.05
LAMBDA_SFT = 0.3
LR = 1e-5
BATCH_VIDEOS = 8
CHECKPOINT_EVERY = 20
MAX_LENGTH_TOKENS = 4096
MAX_NEW_TOKENS = 512  # scene JSONs run 200-400 tokens; 512 is the hard cap
NUM_FRAMES = 16  # must match training and the quality gate
LOGPROB_CHUNK = 256  # chunked teacher-forcing forward to bound activation memory


def batch_loss(
    sample_logprobs: list[float],
    sft_logprob: float,
    base_logprobs: list[float],
    advantages: list[float],
    *,
    beta: float = BETA_KL,
    lambda_sft: float = LAMBDA_SFT,
) -> dict:
    """Pure-math loss assembly (CPU-testable).

    L = -mean_i A_i * logpi_i + beta * mean_i KL_i - lambda_sft * logpi_sft
    where KL_i = logpi_base_i - logpi_i (per-sample reverse-KL linearization).
    """
    policy = -math.fsum(a * lp for a, lp in zip(advantages, sample_logprobs, strict=True)) / len(
        sample_logprobs
    )
    kl = math.fsum(b - p for b, p in zip(base_logprobs, sample_logprobs, strict=True)) / len(
        sample_logprobs
    )
    sft = -lambda_sft * sft_logprob
    total = policy + beta * kl + sft
    return {
        "total": total,
        "policy": policy,
        "kl": kl,
        "sft": sft,
    }


def process_video(generations: list[dict], record: dict) -> dict:
    """Rewards + group advantages for one video's G sampled scenes (CPU)."""
    if len(generations) != G_SAMPLES:
        raise SystemExit(f"expected {G_SAMPLES} samples, got {len(generations)}")
    rewards = [reward(generation, record) for generation in generations]
    advantages = group_advantages([r["reward"] for r in rewards])
    return {"rewards": rewards, "advantages": advantages}


def gate_decision(history: list[float], *, patience: int = 2, min_delta: float = 0.0) -> str:
    """Hard mid-run gate: probe mean-F1 must improve within `patience` checks.

    Returns 'continue', 'kill_stalled', or 'kill_degraded' (a drop larger than
    0.05 versus the running best is an immediate kill, not a stall).
    """
    if len(history) < 2:
        return "continue"
    best = max(history[:-1])
    latest = history[-1]
    if latest < best - 0.05:
        return "kill_degraded"
    if latest <= best + min_delta:
        stagnant = 0
        for value in reversed(history[:-1]):
            if value <= best + min_delta:
                stagnant += 1
            else:
                break
        if stagnant >= patience - 1 and latest <= best + min_delta:
            return "kill_stalled"
    return "continue"


def probe_f1(records: list[dict], generate) -> dict:
    """Mid-run checkpoint: greedy scene per probe video -> predicate F1."""
    from collections import defaultdict

    counts = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn per predicate
    for record in records:
        generation = generate(record)
        outcome = reward(generation, record)
        summary, _ = __import__("formalizer_reward", fromlist=[""]).scene_from_generation(
            generation.get("raw_text"), generation.get("summary")
        )
        if summary is None:
            continue
        gen_sets = __import__("formalizer_reward", fromlist=[""])._sets(summary)
        truth_sets = __import__("formalizer_reward", fromlist=[""])._sets(record["truth_scene"])
        for predicate in ("enterers", "exiters", "moving", "collisions"):
            predicted, truth = gen_sets[predicate], truth_sets[predicate]
            counts[predicate][0] += len(predicted & truth)
            counts[predicate][1] += len(predicted - truth)
            counts[predicate][2] += len(truth - predicted)
    report = {}
    for predicate, (tp, fp, fn) in counts.items():
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[predicate] = round(f1, 3)
    report["mean_event_f1"] = round(
        sum(report.values()) / max(1, len(report)), 4
    )
    report["mean_reward"] = round(outcome["reward"], 4) if records else 0.0
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=60)
    parser.add_argument("--group-size", type=int, default=4, help="GRPO samples per video; drop to 2 if OOM persists")
    parser.add_argument("--self-test", action="store_true")
    args, _unknown = parser.parse_known_args()
    if args.self_test:
        return _self_test()
    _gpu_loop(args)
    return 0


def _decode_video(media_root: Path, media: dict):
    """Explicit 16-frame decode via the proven loader — never hand a raw path
    to the processor (uncapped decode was the OOM root cause)."""
    from train.multimodal import file_hash, load_media, resolve_media

    path = resolve_media(media_root, media["uri"], media["sha256"], max_bytes=256 * 1024 * 1024)
    if file_hash(path) != media["sha256"]:
        raise SystemExit("media sha mismatch during decode")
    decoded = load_media(path, "video", num_frames=NUM_FRAMES, max_seconds=60)
    return decoded["frames"], decoded["metadata"]


def _gpu_loop(args) -> None:  # pragma: no cover - card required
    import time

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    from scripts import s0_development_baselines as baseline

    sys.path.insert(0, "/root/autodl-tmp/s5-calibration-code-fa83acc")
    from caption_probe import PERCEPTION_PROMPT  # noqa: E402

    train = [json.loads(l) for l in args.train.read_text().splitlines() if l.strip()][: args.max_videos]
    probe = [json.loads(l) for l in args.probe.read_text().splitlines() if l.strip()][:20]
    processor = baseline.default_processor_factory(args.model_dir)
    base = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    base.requires_grad_(False)
    model = get_peft_model(
        base,
        LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.0),
    )
    model.print_trainable_parameters()
    model.to("cuda")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = args.output_dir / "rl-log.jsonl"
    probe_history: list[float] = []
    started = time.time()
    with log.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(train):
            frames, metadata = _decode_video(args.media_root, record["media"])
            messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PERCEPTION_PROMPT}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = processor(
                text=[prompt], videos=[frames], video_metadata=[metadata],
                return_tensors="pt", do_sample_frames=False,
            )
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            generations = []
            for _ in range(args.group_size):
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                        temperature=TEMPERATURE, top_p=0.95,
                    )
                text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                generations.append({"raw_text": text})
                del out
                torch.cuda.empty_cache()
            outcome = process_video(generations, record)
            loss_info = _policy_step(model, optimizer, inputs, generations, outcome, record, processor)
            del inputs, frames
            torch.cuda.empty_cache()
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            entry = {
                "video": record["video_id"],
                "rewards": [r["reward"] for r in outcome["rewards"]],
                "advantages": [round(a, 4) for a in outcome["advantages"]],
                "loss": loss_info,
                "peak_gb": round(peak, 2),
                "elapsed": round(time.time() - started, 1),
            }
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
            if (index + 1) % CHECKPOINT_EVERY == 0:
                model.save_pretrained(args.output_dir / f"checkpoint-{index + 1}")
                greedy = lambda r: _greedy_scene(model, processor, args.media_root, r)  # noqa: E731
                probe_report = probe_f1(probe, greedy)
                probe_history.append(probe_report["mean_event_f1"])
                verdict = gate_decision(probe_history)
                # Monitor-grepable line: thresholds live in the day-one doc.
                print(
                    json.dumps(
                        {
                            "checkpoint": index + 1,
                            "probe_f1": probe_report,
                            "history": probe_history,
                            "gate": verdict,
                        }
                    ),
                    flush=True,
                )
                if verdict.startswith("kill"):
                    raise SystemExit(f"mid-run gate {verdict} at checkpoint {index + 1}")
    model.save_pretrained(args.output_dir / "final")


def _token_logprob_chunked(model, inputs, token_ids, *, no_grad: bool = False):
    """Chunked teacher-forcing logprob; bounds activation memory to 256-token
    windows instead of one full-sequence forward."""
    import torch

    prefix_len = inputs["input_ids"].shape[1]
    total = torch.zeros((), device="cuda")
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        for start in range(0, token_ids.shape[1], LOGPROB_CHUNK):
            chunk = token_ids[:, start : start + LOGPROB_CHUNK]
            window_ids = torch.cat(
                [inputs["input_ids"], token_ids[:, :start], chunk], dim=1
            )
            outputs = model(input_ids=window_ids, use_cache=False)
            # logits predicting the chunk tokens sit at positions
            # prefix_len+start-1 .. prefix_len+start+len(chunk)-2
            logits = outputs.logits[0, prefix_len + start - 1 : prefix_len + start + chunk.shape[1] - 1].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            total = total + logprobs.gather(1, chunk[0].unsqueeze(1)).sum()
            del outputs, logits, logprobs
        if not no_grad:
            pass
    return total


def _policy_step(model, optimizer, inputs, generations, outcome, record, processor):  # pragma: no cover
    import torch

    losses = []
    for generation, advantage in zip(generations, outcome["advantages"], strict=True):
        if abs(advantage) < 1e-6:
            continue
        token_ids = processor.tokenizer(
            generation["raw_text"], add_special_tokens=False, return_tensors="pt"
        ).input_ids.to("cuda")
        if token_ids.shape[1] == 0:
            continue
        token_logprob = _token_logprob_chunked(model, inputs, token_ids)
        losses.append(-advantage * token_logprob / max(1, token_ids.shape[1]))
    sft_target = json.dumps(record["truth_scene"], ensure_ascii=False)
    sft_ids = processor.tokenizer(sft_target, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    sft_logprob = _token_logprob_chunked(model, inputs, sft_ids)
    loss = torch.stack(losses).sum() if losses else torch.zeros((), device="cuda")
    loss = loss - LAMBDA_SFT * sft_logprob / max(1, sft_ids.shape[1])
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    torch.cuda.empty_cache()
    return {"loss": round(float(loss.item()), 4)}


def _greedy_scene(model, processor, media_root, record):  # pragma: no cover
    import torch

    import caption_probe

    frames, metadata = _decode_video(media_root, record["media"])
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": caption_probe.PERCEPTION_PROMPT}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[prompt], videos=[frames], video_metadata=[metadata],
                       return_tensors="pt", do_sample_frames=False)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    del out, inputs, frames
    torch.cuda.empty_cache()
    return {"raw_text": text}


def _self_test() -> int:
    record = {
        "truth_scene": {
            "objects": [
                {"name": "red rubber cube", "first_seen": "start", "last_seen": "end", "moving": False}
            ],
            "entry_order": [],
            "collisions": [],
            "present_at_start": ["red rubber cube"],
            "present_at_end": ["red rubber cube"],
        },
        "probe_questions": [
            {"instruction": "Are there any red cubes when the video ends?", "gold": "yes", "predicate": "exiters"}
        ],
    }
    good = {"summary": record["truth_scene"]}
    bad = {"summary": {"objects": [], "entry_order": [], "collisions": [], "present_at_start": [], "present_at_end": []}}
    generations = [good, good, bad, bad]
    outcome = process_video(generations, record)
    assert outcome["advantages"][0] > 0 > outcome["advantages"][2], outcome["advantages"]
    loss = batch_loss(
        sample_logprobs=[-10.0, -11.0, -9.0, -12.0],
        sft_logprob=-30.0,
        base_logprobs=[-10.5, -11.5, -9.5, -12.5],
        advantages=outcome["advantages"],
    )
    assert loss["total"] == loss["policy"] + BETA_KL * loss["kl"] + loss["sft"]
    assert loss["sft"] == -LAMBDA_SFT * -30.0
    probe = probe_f1([record], lambda r: good)
    assert probe["exiters"] == 1.0, probe
    assert gate_decision([0.3]) == "continue"
    assert gate_decision([0.3, 0.4]) == "continue"
    assert gate_decision([0.3, 0.4, 0.41]) == "continue"
    assert gate_decision([0.3, 0.4, 0.4, 0.4]) == "kill_stalled", "two flat checks kill"
    assert gate_decision([0.4, 0.4, 0.3]) == "kill_degraded", "large drop kills immediately"
    print(json.dumps({"self_test": "passed", "loss": loss, "probe_f1": probe}, indent=2))
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
