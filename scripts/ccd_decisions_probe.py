"""CCD-for-decisions probe: family-routed structured execution on JevBench public 231.

Zero-training intervention under the ADR0024 structured-execution lane: weak
families are routed to decomposed forwards (atom extraction -> short decision;
per-candidate support verification -> aggregate) while every other family passes
the pinned D3P-600 direct-answer baseline through unchanged. Gates are
pre-registered and never moved: weak-family pooled accuracy >= 0.80 and
full-board micro accuracy >= 0.84. Routing may only be revised twice after the
first full run; every revision must be declared via --routing-revision.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from data.jevbench import JevBenchTask, deterministic_state_text
from data.schema import Prefix, Question
from eval.jevbench import argmax_label, score_predictions
from scripts.assets import write_json
from scripts.s0_public_baselines import (
    MODEL_SPECS,
    SEED,
    ChatTokenizer,
    PreparedQuestion,
    adapter_evidence,
    append_record_jsonl,
    attach_public_adapter,
    build_prepared_question,
    configure_torch_runtime,
    default_tokenizer_factory,
    ensure_cuda_available,
    ensure_output,
    environment_summary,
    load_json,
    load_public_source,
    model_evidence,
    now_iso,
    readout_from_logprobs,
    sha256_bytes,
    sha256_file,
    sidecar_evidence,
    source_identity,
    torch_dtype,
)
from train.readout import read_probabilities
from train.template import check_answer_tokens, render_prompt

PROBE_VERSION = "s4-ccd-decisions-probe-v1"
WEAK_FAMILIES: tuple[str, ...] = (
    "temporal_numeric",
    "long_policy",
    "multi_hop",
    "ambiguous",
    "tradeoff",
    "probability",
    "judge_hard",
)
ATOMS_FAMILIES = frozenset({"temporal_numeric"})
VERIFY_FAMILIES = frozenset(
    {"long_policy", "multi_hop", "ambiguous", "tradeoff", "probability", "judge_hard"}
)
GATE_WEAK_POOLED = 0.80
GATE_FULL_MICRO = 0.84
MAX_ROUTING_REVISIONS = 2
DEFAULT_MAX_NEW_TOKENS = 384
VERDICT_TOKENS: tuple[str, str] = ("yes", "no")

EXTRACTION_PROMPT = (
    "Extract the decision-relevant facts from the record below.\n"
    "\n"
    "Output ONLY one JSON object, no markdown fence, no prose. Keys and values:\n"
    '"dates": list of short strings with dates or date-times from the record;\n'
    '"durations": list of short strings with durations or validity periods;\n'
    '"quantities": list of short strings with counts, sums, thresholds or percentages;\n'
    '"timezones": list of short strings with timezone or clock-change facts;\n'
    '"deadlines": list of short strings with deadlines, cutoffs or expiry rules;\n'
    '"rules": list of short strings with explicit rules about precedence, month-end,'
    " rounding, weekends, holidays or counting.\n"
    "\n"
    "Copy values from the record; include every fact needed to answer the question"
    " below; use [] when a key has nothing applicable.\n"
    "\n"
    "Question that the atoms must support: {instruction}\n"
    "\n"
    "Record:\n"
    "{state}"
)

VERIFICATION_PROMPT = (
    "You verify one candidate answer against a record.\n"
    "\n"
    "Record:\n"
    "{state}\n"
    "\n"
    "Question: {instruction}\n"
    'Candidate answer "{label}": {description}\n'
    "\n"
    "Task: based only on the record above, decide whether this candidate is the"
    ' correct answer to the question. Reply with exactly one token: "yes" when the'
    ' record supports this candidate being the correct answer; "no" when it is'
    " wrong, unsupported, incomplete, or another candidate fits better."
)

AGGREGATION_RULE = (
    "per-candidate independent verification forward; support = P(yes | not U) from"
    " read_probabilities over (yes, no) with U; final distribution = support renormalized"
    " linearly across candidates; prediction = official argmax_label semantics;"
    " all-support-zero or extraction-parse failures fall back to the direct baseline"
)


class ProbeError(RuntimeError):
    """Raised when probe evidence is not admissible."""


class ProbeTokenizer(ChatTokenizer, Protocol):
    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str: ...


@dataclass
class UsageStats:
    forwards: int = 0
    generated_tokens: int = 0
    wall_seconds: float = 0.0


@dataclass
class ProbeOutcome:
    strategy: Literal["passthrough", "atoms", "verify"]
    effective: Literal["passthrough", "atoms", "verify"]
    probabilities: dict[str, float]
    prediction: str
    fallback: str | None
    forwards: int
    generated_tokens: int
    wall_seconds: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineBundle:
    report_path: Path
    report_sha256: str
    adapter_sha256: str
    records_by_id: dict[str, dict[str, Any]]
    score: dict[str, Any]
    accuracy: float


class ProbeModel:
    """Adapter-bound backbone offering last-position logprobs and greedy generation."""

    def __init__(
        self, model: Any, tokenizer: ProbeTokenizer, torch: Any, stats: UsageStats
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.stats = stats

    def read_logprobs(self, prompt: str, token_ids: Mapping[str, int]) -> dict[int, float]:
        started = time.perf_counter()
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            raise ProbeError("probe prompt encoded to zero tokens")
        input_ids = self.torch.tensor([ids], device="cuda", dtype=self.torch.long)
        attention = self.torch.ones_like(input_ids)
        with self.torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
            logits = output.logits[0, -1].float()
            logprobs = self.torch.nn.functional.log_softmax(logits, dim=-1)
            result = {
                int(token_id): float(logprobs[int(token_id)].item())
                for token_id in token_ids.values()
            }
        self.stats.forwards += 1
        self.stats.wall_seconds += time.perf_counter() - started
        return result

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int]:
        started = time.perf_counter()
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            raise ProbeError("probe prompt encoded to zero tokens")
        input_ids = self.torch.tensor([ids], device="cuda", dtype=self.torch.long)
        attention = self.torch.ones_like(input_ids)
        with self.torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            new_ids = [int(value) for value in sequences[0][input_ids.shape[1] :]]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        self.stats.forwards += 1
        self.stats.generated_tokens += len(new_ids)
        self.stats.wall_seconds += time.perf_counter() - started
        return text, len(new_ids)


def chat_render(tokenizer: ProbeTokenizer, user_text: str) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ProbeError("official chat template must return a nonempty string")
    return rendered


def verify_generation_boundary(
    tokenizer: ProbeTokenizer, prompt: str, token_ids: Mapping[str, int], context: str
) -> None:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    for token, token_id in token_ids.items():
        if tokenizer.encode(prompt + token, add_special_tokens=False) != [*prompt_ids, token_id]:
            raise ProbeError(f"token {token!r} changes at generation boundary ({context})")


def parse_atoms_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    candidate = fence.group(1).strip() if fence else stripped
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def candidate_texts(task: JevBenchTask) -> list[tuple[str, str]]:
    criteria: Any = task.question["criteria"]
    if task.primitive == "choice":
        if not isinstance(criteria, dict):
            raise ProbeError(f"choice criteria must be an object: {task.id}")
        return [(label, str(criteria[label])) for label in task.labels]
    if task.primitive == "score":
        if not isinstance(criteria, list):
            raise ProbeError(f"score criteria must be a list: {task.id}")
        return [
            (label, str(description))
            for label, description in zip(task.labels, criteria, strict=True)
        ]
    if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
        raise ProbeError(f"noul criteria must have true/false meanings: {task.id}")
    return [
        ("yes", str(criteria["true"])),
        ("no", str(criteria["false"])),
    ]


def load_baseline(
    report_path: Path, tasks: Sequence[JevBenchTask], adapter_sha256: str
) -> BaselineBundle:
    data = load_json(report_path)
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise ProbeError(f"baseline report lacks records: {report_path}")
    records_by_id: dict[str, dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict) or row.get("pass_index") != 1:
            continue
        records_by_id[str(row["id"])] = row
    if set(records_by_id) != {task.id for task in tasks}:
        raise ProbeError("baseline records do not cover the 231 public tasks")
    adapter = data.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("weights_sha256") != adapter_sha256:
        raise ProbeError("baseline report is not bound to the requested adapter sha")
    rows = [
        {"id": task_id, "probabilities": record["source_probabilities"]}
        for task_id, record in records_by_id.items()
    ]
    score = score_predictions(tasks, rows)
    published = data.get("score_pass_1")
    if not isinstance(published, dict) or not math.isclose(
        float(score["accuracy"]), float(published["accuracy"]), abs_tol=1e-12
    ):
        raise ProbeError("baseline rescore does not reproduce the published accuracy")
    return BaselineBundle(
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        adapter_sha256=adapter_sha256,
        records_by_id=records_by_id,
        score=score,
        accuracy=float(score["accuracy"]),
    )


def passthrough_outcome(
    baseline_record: Mapping[str, Any],
    attempted: Literal["passthrough", "atoms", "verify"],
    fallback: str | None,
    details: Mapping[str, Any],
) -> ProbeOutcome:
    probabilities = {
        str(label): float(value) for label, value in baseline_record["source_probabilities"].items()
    }
    return ProbeOutcome(
        strategy=attempted,
        effective="passthrough",
        probabilities=probabilities,
        prediction=str(baseline_record["prediction_source"]),
        fallback=fallback,
        forwards=0,
        generated_tokens=0,
        wall_seconds=0.0,
        details=dict(details),
    )


def run_atoms_strategy(
    runner: ProbeModel,
    tokenizer: ProbeTokenizer,
    item: PreparedQuestion,
    max_new_tokens: int,
    baseline_record: Mapping[str, Any],
) -> ProbeOutcome:
    started = time.perf_counter()
    state_text = deterministic_state_text(item.task.state)
    instruction = str(item.task.question["instructions"])
    extraction_user = EXTRACTION_PROMPT.format(instruction=instruction, state=state_text)
    extraction_text, generated = runner.generate(
        chat_render(tokenizer, extraction_user), max_new_tokens
    )
    parsed = parse_atoms_json(extraction_text)
    if parsed is None:
        return passthrough_outcome(
            baseline_record,
            "atoms",
            "extraction_parse_failed",
            {
                "extraction_text": extraction_text[:2000],
                "extraction_generated_tokens": generated,
            },
        )
    atoms_text = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    prompt = render_prompt(tokenizer, Prefix(state=atoms_text, media=[]), item.question)
    verify_generation_boundary(tokenizer, prompt, item.answer_token_ids, f"atoms:{item.task.id}")
    logprobs = runner.read_logprobs(prompt, item.answer_token_ids)
    readout = readout_from_logprobs(item, logprobs)
    wall = time.perf_counter() - started
    return ProbeOutcome(
        strategy="atoms",
        effective="atoms",
        probabilities={str(k): float(v) for k, v in readout["source_probabilities"].items()},
        prediction=str(readout["prediction_source"]),
        fallback=None,
        forwards=2,
        generated_tokens=generated,
        wall_seconds=wall,
        details={
            "atoms": parsed,
            "extraction_sha256": sha256_bytes(extraction_text.encode()),
            "extraction_generated_tokens": generated,
            "p_U": float(readout["p_U"]),
            "raw_logprobs": readout["raw_logprobs"],
            "conditional_without_u_project": readout["conditional_without_u_project"],
        },
    )


def run_verify_strategy(
    runner: ProbeModel,
    tokenizer: ProbeTokenizer,
    item: PreparedQuestion,
    verdict_ids: Mapping[str, int],
    baseline_record: Mapping[str, Any],
) -> ProbeOutcome:
    started = time.perf_counter()
    state_text = deterministic_state_text(item.task.state)
    instruction = str(item.task.question["instructions"])
    supports: dict[str, float] = {}
    p_insufficient: dict[str, float] = {}
    for label, description in candidate_texts(item.task):
        user = VERIFICATION_PROMPT.format(
            state=state_text, instruction=instruction, label=label, description=description
        )
        logprobs = runner.read_logprobs(chat_render(tokenizer, user), verdict_ids)
        logits = {token: logprobs[verdict_ids[token]] for token in (*VERDICT_TOKENS, "U")}
        readout = read_probabilities(logits, VERDICT_TOKENS)
        supports[label] = float(readout.probabilities["yes"])
        p_insufficient[label] = float(readout.p_insufficient)
    total = math.fsum(supports.values())
    if total <= 0.0:
        return passthrough_outcome(
            baseline_record,
            "verify",
            "support_underflow",
            {"supports": supports, "p_insufficient": p_insufficient},
        )
    probabilities = {label: value / total for label, value in supports.items()}
    wall = time.perf_counter() - started
    return ProbeOutcome(
        strategy="verify",
        effective="verify",
        probabilities=probabilities,
        prediction=argmax_label(probabilities),
        fallback=None,
        forwards=len(supports),
        generated_tokens=0,
        wall_seconds=wall,
        details={"supports": supports, "p_insufficient": p_insufficient},
    )


def strategy_for(family: str, selected: Sequence[str]) -> Literal["passthrough", "atoms", "verify"]:
    if family in ATOMS_FAMILIES and family in selected:
        return "atoms"
    if family in VERIFY_FAMILIES and family in selected:
        return "verify"
    return "passthrough"


def select_tasks(
    tasks: Sequence[JevBenchTask], families: Sequence[str], limit: int | None
) -> list[JevBenchTask]:
    buckets = [[task for task in tasks if task.family == family] for family in families]
    if limit is None:
        return [task for bucket in buckets for task in bucket]
    chosen: list[JevBenchTask] = []
    index = 0
    while len(chosen) < limit and any(buckets):
        bucket = buckets[index % len(buckets)]
        index += 1
        if bucket:
            chosen.append(bucket.pop(0))
    return chosen


def parity_task_ids(tasks: Sequence[JevBenchTask]) -> list[str]:
    chosen: list[str] = []
    for family in ("fact", "long_policy"):
        match = next((task for task in tasks if task.family == family), None)
        if match is not None:
            chosen.append(match.id)
    return chosen


def parity_check(
    runner: ProbeModel,
    prepared: Mapping[str, PreparedQuestion],
    baseline: BaselineBundle,
    task_ids: Sequence[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        item = prepared[task_id]
        record = baseline.records_by_id[task_id]
        observed_ids = runner.read_logprobs(item.prompt, item.answer_token_ids)
        observed = {
            token: observed_ids[token_id] for token, token_id in item.answer_token_ids.items()
        }
        expected = {str(token): float(value) for token, value in record["raw_logprobs"].items()}
        if set(observed) != set(expected):
            raise ProbeError(f"parity token set mismatch for {task_id}")
        max_diff = max(abs(observed[key] - expected[key]) for key in observed)
        prediction = item.task.project_to_source[
            readout_from_logprobs(item, observed_ids)["prediction_project"]
        ]
        rows.append(
            {
                "id": task_id,
                "max_abs_logprob_diff": max_diff,
                "prediction_matches_baseline": prediction == record["prediction_source"],
            }
        )
        if max_diff > 0.5 or prediction != record["prediction_source"]:
            raise ProbeError(
                f"parity forward diverges from baseline record for {task_id} "
                f"(max_abs_logprob_diff={max_diff})"
            )
    return {
        "checked_ids": list(task_ids),
        "rows": rows,
        "passed": True,
        "note": "direct-answer forward reproduced the pinned baseline records exactly",
    }


def build_report(
    *,
    args: argparse.Namespace,
    baseline: BaselineBundle,
    probe_score: dict[str, Any],
    family_rows: Sequence[dict[str, Any]],
    records_path: Path,
    stats: UsageStats,
    parity: dict[str, Any],
    started_at: str,
    started: float,
    model_meta: dict[str, Any],
    adapter: dict[str, str],
    intervened: int,
) -> dict[str, Any]:
    weak_correct = sum(
        int(probe_score["per_family"][family]["correct"])
        for family in WEAK_FAMILIES
        if family in probe_score["per_family"]
    )
    weak_n = sum(
        int(probe_score["per_family"][family]["n_scorable"])
        for family in WEAK_FAMILIES
        if family in probe_score["per_family"]
    )
    weak_observed = weak_correct / weak_n if weak_n else 0.0
    full_observed = float(probe_score["accuracy"])
    weak_passed = weak_observed >= GATE_WEAK_POOLED
    full_passed = full_observed >= GATE_FULL_MICRO
    if weak_passed and full_passed:
        verdict = "strong-signal"
    elif weak_passed or full_passed:
        verdict = "partial"
    else:
        verdict = "falsified"
    smoke = args.limit is not None
    return {
        "schema": "lite-omnijev.s4_ccd_decisions_probe.report.v1",
        "version": PROBE_VERSION,
        "action": "run",
        "status": "completed",
        "exit_code": 0,
        "smoke": smoke,
        "verdict": verdict if not smoke else "smoke-only-not-adjudicated",
        "started_at": started_at,
        "finished_at": now_iso(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "environment": environment_summary(),
        "source": {
            **source_identity(),
            "probe_code_sha256": sha256_file(Path(__file__).resolve()),
        },
        "command": {"argv": sys.argv, "cwd": str(Path.cwd())},
        "cost": {"new_paid_resources": None, "gpu_started_by_this_cli": False},
        "model": model_meta,
        "adapter": adapter,
        "baseline": {
            "report_path": str(baseline.report_path),
            "report_sha256": baseline.report_sha256,
            "adapter_weights_sha256": baseline.adapter_sha256,
            "accuracy": baseline.accuracy,
            "macro_accuracy": baseline.score["macro_accuracy"],
            "per_family": baseline.score["per_family"],
        },
        "routing": {
            "weak_families": list(WEAK_FAMILIES),
            "atoms_families": sorted(ATOMS_FAMILIES),
            "verify_families": sorted(VERIFY_FAMILIES),
            "selected_families": list(args.families),
            "limit": args.limit,
            "routing_revision": args.routing_revision,
            "routing_note": args.routing_note,
            "routing_revisions_remaining": MAX_ROUTING_REVISIONS - args.routing_revision,
            "aggregation_rule": AGGREGATION_RULE,
            "prompt_template_sha256": {
                "extraction": sha256_bytes(EXTRACTION_PROMPT.encode()),
                "verification": sha256_bytes(VERIFICATION_PROMPT.encode()),
            },
        },
        "precision": {"dtype": args.dtype, "seed": SEED, "tf32_allowed": False},
        "parity": parity,
        "probe_score": {
            "accuracy": probe_score["accuracy"],
            "macro_accuracy": probe_score["macro_accuracy"],
            "per_family": probe_score["per_family"],
            "paraphrase_consistency": probe_score["paraphrase_consistency"],
        },
        "weak_pool": {
            "families": list(WEAK_FAMILIES),
            "n": weak_n,
            "correct": weak_correct,
            "observed": weak_observed,
        },
        "family_comparison": family_rows,
        "gates": {
            "weak_pooled_accuracy": {
                "gate": GATE_WEAK_POOLED,
                "observed": weak_observed,
                "passed": weak_passed,
                "preregistered": True,
            },
            "full_micro_accuracy": {
                "gate": GATE_FULL_MICRO,
                "observed": full_observed,
                "passed": full_passed,
                "preregistered": True,
            },
        },
        "usage": {
            "forwards": stats.forwards,
            "generated_tokens": stats.generated_tokens,
            "forward_wall_seconds": round(stats.wall_seconds, 6),
            "tasks_intervened": intervened,
        },
        "records_evidence": sidecar_evidence(records_path),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = now_iso()
    if args.routing_revision > MAX_ROUTING_REVISIONS:
        raise ProbeError(
            f"routing revision {args.routing_revision} exceeds the "
            f"{MAX_ROUTING_REVISIONS}-revision exploration budget"
        )
    if args.routing_revision > 0 and not args.routing_note:
        raise ProbeError("routing revisions require an explicit --routing-note")
    if args.model_id != "gemma4-e4b-it" or args.adapter_dir is None:
        raise ProbeError("the probe pins gemma4-e4b-it plus the D3P-600 adapter")
    tasks = load_public_source(args.source_root, args.source_config)
    baseline = load_baseline(args.baseline_report, tasks, args.adapter_weights_sha256)
    model_meta, directory = model_evidence(
        args.asset_manifest, args.model_root, args.model_id, require_weights=True
    )
    adapter = adapter_evidence(args.adapter_dir, args.adapter_weights_sha256, model_meta, directory)
    if adapter is None:
        raise ProbeError("adapter binding failed")
    tokenizer = cast(ProbeTokenizer, default_tokenizer_factory(directory))
    prepared = {task.id: build_prepared_question(task, tokenizer) for task in tasks}

    torch = ensure_cuda_available()
    configure_torch_runtime(torch)
    dtype = torch_dtype(torch, args.dtype)
    transformers = importlib.import_module("transformers")
    class_name = MODEL_SPECS[args.model_id]["transformers_class"]
    model_cls = getattr(transformers, class_name)
    model = model_cls.from_pretrained(
        str(directory),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    model = attach_public_adapter(model, adapter)
    model.eval()
    model.to("cuda")
    stats = UsageStats()
    runner = ProbeModel(model, tokenizer, torch, stats)
    try:
        verdict_question = Question.model_validate(
            {"qid": "probe-verdict", "type": "noul", "instruction": "probe verdict readout"}
        )
        verdict_ids = check_answer_tokens(tokenizer, verdict_question)
        if tuple(verdict_ids) != ("yes", "no", "U"):
            raise ProbeError("verdict token order changed")
        selected = select_tasks(tasks, args.families, args.limit)
        selected_ids = {task.id for task in selected}
        if selected:
            first = selected[0]
            sample_user = VERIFICATION_PROMPT.format(
                state=deterministic_state_text(first.state),
                instruction=str(first.question["instructions"]),
                label="probe",
                description="probe",
            )
            verify_generation_boundary(
                tokenizer, chat_render(tokenizer, sample_user), verdict_ids, "verdict-tokens"
            )
        parity = parity_check(runner, prepared, baseline, parity_task_ids(tasks))

        records: list[dict[str, Any]] = []
        direct_results = baseline.score["results"]
        for task in tasks:
            strategy = strategy_for(task.family, args.families)
            baseline_record = baseline.records_by_id[task.id]
            item = prepared[task.id]
            if strategy == "atoms" and task.id in selected_ids:
                outcome = run_atoms_strategy(
                    runner, tokenizer, item, args.max_new_tokens, baseline_record
                )
            elif strategy == "verify" and task.id in selected_ids:
                outcome = run_verify_strategy(runner, tokenizer, item, verdict_ids, baseline_record)
            else:
                outcome = passthrough_outcome(baseline_record, "passthrough", None, {})
            record = {
                "id": task.id,
                "family": task.family,
                "split": task.split_name,
                "primitive": task.primitive,
                "selected": task.id in selected_ids,
                "strategy": outcome.strategy,
                "effective": outcome.effective,
                "fallback": outcome.fallback,
                "n_forwards": outcome.forwards,
                "generated_tokens": outcome.generated_tokens,
                "wall_seconds": round(outcome.wall_seconds, 6),
                "expected_present_for_scoring_only": str(task.expected),
                "direct_prediction": str(baseline_record["prediction_source"]),
                "direct_correct": bool(direct_results[task.id]["correct"]),
                "direct_probabilities": baseline_record["source_probabilities"],
                "probe_prediction": argmax_label(outcome.probabilities),
                "probe_correct": argmax_label(outcome.probabilities) == str(task.expected),
                "probe_probabilities": outcome.probabilities,
                "details": outcome.details,
            }
            records.append(record)
            append_record_jsonl(args.records, record)

        probe_rows = [
            {"id": row["id"], "probabilities": row["probe_probabilities"]} for row in records
        ]
        probe_score = score_predictions(tasks, probe_rows)
        probe_results = probe_score["results"]
        family_rows: list[dict[str, Any]] = []
        for family in sorted({task.family for task in tasks}):
            family_tasks = [task for task in tasks if task.family == family]
            forwards = sum(record["n_forwards"] for record in records if record["family"] == family)
            wall = math.fsum(
                record["wall_seconds"] for record in records if record["family"] == family
            )
            direct_correct = sum(1 for task in family_tasks if direct_results[task.id]["correct"])
            probe_correct = sum(1 for task in family_tasks if probe_results[task.id]["correct"])
            strategies = sorted(
                {record["strategy"] for record in records if record["family"] == family}
            )
            family_rows.append(
                {
                    "family": family,
                    "n": len(family_tasks),
                    "strategy": strategies,
                    "direct_correct": direct_correct,
                    "probe_correct": probe_correct,
                    "delta_correct": probe_correct - direct_correct,
                    "forwards": forwards,
                    "wall_seconds": round(wall, 6),
                }
            )
        intervened = sum(1 for record in records if record["effective"] != "passthrough")
        return build_report(
            args=args,
            baseline=baseline,
            probe_score=probe_score,
            family_rows=family_rows,
            records_path=args.records,
            stats=stats,
            parity=parity,
            started_at=started_at,
            started=started,
            model_meta=model_meta,
            adapter=adapter,
            intervened=intervened,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-config", type=Path, default=Path("configs/jevbench-public-assets.json")
    )
    parser.add_argument(
        "--asset-manifest", type=Path, default=Path("configs/project-model-assets.json")
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-id", default="gemma4-e4b-it")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--adapter-weights-sha256", required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--families", default=",".join(WEAK_FAMILIES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--routing-revision", type=int, default=0)
    parser.add_argument("--routing-note", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.families = [family.strip() for family in str(args.families).split(",") if family.strip()]
    if not args.families:
        raise SystemExit("--families must name at least one family")
    known = set(WEAK_FAMILIES)
    if not set(args.families) <= known:
        raise SystemExit(f"--families must be a subset of {sorted(known)}")
    ensure_output(args.report)
    ensure_output(args.records)
    started = time.perf_counter()
    started_at = now_iso()
    try:
        report = run_probe(args)
    except Exception as exc:
        report = {
            "schema": "lite-omnijev.s4_ccd_decisions_probe.report.v1",
            "version": PROBE_VERSION,
            "action": "run",
            "status": "blocked",
            "exit_code": 2,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "started_at": started_at,
            "finished_at": now_iso(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "environment": environment_summary(),
            "source": {
                **source_identity(),
                "probe_code_sha256": sha256_file(Path(__file__).resolve()),
            },
            "command": {"argv": sys.argv, "cwd": str(Path.cwd())},
            "records_evidence": sidecar_evidence(args.records),
        }
        write_json(args.report, report)
        return 2
    write_json(args.report, report)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
