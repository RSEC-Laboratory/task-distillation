"""Official GEPA 0.1.4 with local reflection and a shared attempted-call cap.

This is an access-constrained baseline configuration, not an unmodified
reproduction of a published GEPA result. The remote task-pack teacher is never
called here. Both the solver and GEPA's actual reflection prompt use LocalModel.
Only generated optimization/selection examples reach GEPA.

API audited: https://github.com/gepa-ai/gepa/tree/v0.1.4
Missing/incompatible dependencies and non-budget optimizer errors fail explicitly.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import time
from dataclasses import asdict
from typing import Any

from .experiment import (
    CallLedger, LocalModel, ModelCallError, SearchConfig, common_contract, ensure_local, score_output,
    normalized_input_hash, TaskAdapter, TaskScoringError, _ScoringAbort,
    optimizer_splits, validate_optimizer_pack,
)
from .schema import content_hash

GEPA_VERSION = "0.1.4"


class GEPACompilationError(RuntimeError):
    """Failed baseline execution, retaining attempted calls for an error manifest."""

    def __init__(self, message: str, calls: list[dict]):
        super().__init__(message)
        self.calls = [dict(row) for row in calls]


class _BudgetExhausted(RuntimeError):
    pass


def _load_gepa():
    """Import the pinned official implementation, never an imitation fallback."""
    try:
        installed = importlib.metadata.version("gepa")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(
            "gepa_local requires the optional official dependency gepa==0.1.4; "
            "no fallback optimizer is used"
        ) from exc
    if installed != GEPA_VERSION:
        raise ImportError(
            f"gepa_local requires gepa=={GEPA_VERSION}, found {installed}; "
            "pin and audit the dependency before running this baseline"
        )
    package = importlib.import_module("gepa")
    module = importlib.import_module("gepa.adapters.default_adapter.default_adapter")
    return package, module.DefaultAdapter, module.EvaluationResult


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _messages(prompt: str | list[dict]) -> tuple[str, str]:
    if isinstance(prompt, str):
        return "", prompt
    systems, others = [], []
    for message in prompt:
        text = message.get("content")
        if not isinstance(text, str):
            raise TypeError("GEPA local reflection accepts text messages only")
        (systems if message.get("role") == "system" else others).append(text)
    return "\n\n".join(systems), "\n\n".join(others)


def optimize_gepa(
    pack: dict, model: LocalModel, config: SearchConfig, *, prepared_splits: dict | None = None,
    task_adapter: TaskAdapter | None = None,
) -> dict:
    """Return a frozen bundle and all solver/reflection attempted-call records.

    A budget stop selects only a candidate with a completed selection-set
    evaluation. It is marked budget_stopped, never completed. Other optimizer
    failures raise GEPACompilationError with calls, rather than returning a raw
    or partially evaluated prompt. Empty reflection schema means free-form text
    and must omit structured-output constraints in the local runtime adapter.
    """
    validate_optimizer_pack(pack, task_adapter)
    config.validate()
    ensure_local(model)
    if pack["provenance"]["kind"] == "fixture" and model.identity["mode"] != "fixture":
        raise ValueError("Fixture task packs cannot be used for live empirical runs")
    package, DefaultAdapter, EvaluationResult = _load_gepa()
    splits = optimizer_splits(pack, config, prepared_splits, task_adapter)
    train_cases, val_cases = splits["optimization"], splits["selection"]
    if config.max_calls < len(val_cases):
        raise ValueError("GEPA call budget must cover one complete selection evaluation")
    base = task_adapter.contract if task_adapter else common_contract(pack)
    output_schema = task_adapter.output_schema if task_adapter else pack["output_schema"]
    seed_text = pack["teacher_instruction"]

    def instruction(fragment: str) -> str:
        if not isinstance(fragment, str) or not fragment.strip():
            raise ValueError("GEPA returned an empty or non-text instruction")
        return base + "\n\n" + fragment

    if len(instruction(seed_text)) > config.max_instruction_chars:
        raise ValueError("GEPA seed exceeds the common instruction length cap")
    ledger = CallLedger(model, config.max_calls, task_adapter=task_adapter)
    case_by_id = {case["id"]: case for case in train_cases + val_cases}
    reference_texts = ({case_id: task_adapter.answer(case) for case_id, case in case_by_id.items()}
                       if task_adapter else {})
    train_ids = {case["id"] for case in train_cases}
    val_ids = {case["id"] for case in val_cases}
    reflection_size = min(3, len(train_cases))
    completed_validation: list[dict] = []
    rejections: list[dict] = []
    state = {"budget_stop_requested": False}
    batch_scores: dict[str, dict] = {}

    def reserve(count: int = 1) -> None:
        if len(ledger.rows) + count > config.max_calls:
            raise _BudgetExhausted("Shared solver/reflection call budget exhausted")

    class TaskCallable:
        batch: list[dict]
        position: int
        fragment: str
        candidate_id: str
        reject_reason: str | None

        def __call__(self, messages: list[dict]) -> str:
            if self.position >= len(self.batch):
                raise RuntimeError("GEPA adapter invoked an unexpected task example")
            data = self.batch[self.position]
            self.position += 1
            case_id = data["additional_context"]["case_id"]
            if case_id not in case_by_id:
                raise ValueError("GEPA attempted to access a non-optimization example")
            case = case_by_id[case_id]
            system, user = _messages(messages)
            if user != case["input"] or system != self.fragment:
                raise RuntimeError("GEPA DefaultAdapter message contract changed")
            if self.reject_reason is not None:
                rejections.append({"candidate_id": self.candidate_id,
                                   "case_id": case_id, "reason": self.reject_reason})
                return ""
            reserve()
            phase = "gepa_optimization" if case_id in train_ids else "gepa_selection"
            try:
                row = ledger.ask(instruction(self.fragment), case, output_schema,
                                 phase, self.candidate_id)
            except TaskScoringError as exc:
                ledger.rows[-1]["role"] = "solver"
                raise _ScoringAbort(exc) from exc
            row["role"] = "solver"
            if task_adapter is not None:
                batch_scores[case_id] = row
            return row["output"] if isinstance(row["output"], str) else ""

    task_callable = TaskCallable()

    def evaluator(data: dict, response: str):
        case = case_by_id[data["additional_context"]["case_id"]]
        if task_callable.reject_reason is not None:
            return EvaluationResult(0.0, task_callable.reject_reason, None)
        scored = (batch_scores[case["id"]] if task_adapter is not None
                  else score_output(response, case["reference"], output_schema))
        feedback = {
            "reference_kind": "teacher_generated_unverified",
            "format_valid": scored["format_valid"],
            "reference_agreement": scored["correct"],
            "expected_json": case.get("reference"),
            "parsed_output": scored.get("parsed"),
            "parse_error": scored.get("parse_error"),
            "instruction": "Improve reusable instructions; do not copy this example.",
        }
        if task_adapter is not None:
            feedback.pop("expected_json")
            feedback.pop("reference_agreement")
            feedback.update(scorer_id=task_adapter.scorer_id, score=scored["score"],
                            task_passed=scored["correct"],
                            expected_text=reference_texts[case["id"]],
                            native=scored.get("native"), feedback=scored.get("feedback"),
                            model_error=scored.get("error"))
        return EvaluationResult(scored["score"] if task_adapter else float(scored["correct"]),
                                json.dumps(feedback, ensure_ascii=False), None)

    class TrackedAdapter(DefaultAdapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            if set(candidate) != {"system_prompt"}:
                raise ValueError("GEPA candidate must contain only system_prompt")
            fragment = candidate["system_prompt"]
            full = instruction(fragment)
            candidate_id = "gepa_" + _text_hash(fragment)[:16]
            for data in batch:
                case_id = data.get("additional_context", {}).get("case_id")
                if case_id not in case_by_id or data["input"] != case_by_id[case_id]["input"]:
                    raise ValueError("GEPA attempted to access an unknown input")
            too_long = len(full) > config.max_instruction_chars
            if not too_long:
                # Refuse an incomplete batch so partially scored candidates cannot win.
                reserve(len(batch))
            task_callable.batch = list(batch)
            batch_scores.clear()
            task_callable.position = 0
            task_callable.fragment = fragment
            task_callable.candidate_id = candidate_id
            task_callable.reject_reason = (
                "Candidate rejected before inference: common instruction length cap exceeded"
                if too_long else None
            )
            evaluated = super().evaluate(batch, candidate, capture_traces=capture_traces)
            if task_callable.position != len(batch):
                raise RuntimeError("GEPA DefaultAdapter did not evaluate the complete batch")
            ids = [data["additional_context"]["case_id"] for data in batch]
            if not too_long and len(ids) == len(val_ids) and set(ids) == val_ids:
                scores = list(evaluated.scores)
                if len(scores) != len(ids):
                    raise RuntimeError("GEPA returned incomplete selection scores")
                completed_validation.append({
                    "fragment": fragment, "candidate_id": candidate_id,
                    "score": sum(scores) / len(scores),
                    "scores": scores, "case_ids": ids,
                })
            return evaluated

    adapter = TrackedAdapter(model=task_callable, evaluator=evaluator)

    def reflect(prompt: str | list[dict]) -> str:
        reserve()
        system, user = _messages(prompt)
        row = {"call_index": len(ledger.rows), "phase": "gepa_reflection",
               "role": "optimizer", "case_id": None, "candidate_id": None,
               "input_hash": _text_hash(user), "input_tokens": None, "output_tokens": None,
               "latency_seconds": 0.0, "error": None, "format_valid": None,
               "correct": None, "output": None}
        ledger.rows.append(row)
        started = time.perf_counter()
        try:
            response = model.generate(
                "Improve reusable instructions locally. The following contract governs "
                "the task solver, not the format of this reflection reply:\n"
                "<fixed_task_contract>\n" + base + "\n</fixed_task_contract>\n"
                "Return new instructions in the format requested below. "
                "Do not answer a task example." + ("\n\n" + system if system else ""),
                user, {},
            )
            row.update(input_tokens=response.input_tokens,
                       output_tokens=response.output_tokens, output=response.text)
            if not isinstance(response.text, str) or not response.text.strip():
                raise ValueError("Local GEPA reflection returned empty text")
            return response.text
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, ModelCallError):
                row.update(input_tokens=exc.input_tokens, output_tokens=exc.output_tokens)
            raise
        finally:
            row["latency_seconds"] = time.perf_counter() - started

    def stop_at_shared_budget(_state) -> bool:
        # Native GEPA can check its metric cap only between batches. Reserve a
        # conservative full next iteration: old/new minibatches, reflection, val.
        remaining = config.max_calls - len(ledger.rows)
        needed = 2 * reflection_size + 1 + len(val_cases)
        stop = remaining < needed
        state["budget_stop_requested"] |= stop
        return stop

    def prepared(rows: list[dict]) -> list[dict]:
        return [{"input": row["input"],
                 "answer": (reference_texts[row["id"]] if task_adapter
                            else json.dumps(row["reference"], ensure_ascii=False)),
                 "additional_context": {"case_id": row["id"]}}
                for row in rows]

    result = None
    hard_budget_stop = False
    try:
        result = package.optimize(
            seed_candidate={"system_prompt": seed_text},
            trainset=prepared(train_cases), valset=prepared(val_cases),
            adapter=adapter, reflection_lm=reflect,
            reflection_minibatch_size=reflection_size,
            max_metric_calls=config.max_calls,
            stop_callbacks=stop_at_shared_budget,
            candidate_selection_strategy="pareto", use_merge=False,
            val_evaluation_policy="full_eval",
            cache_evaluation=False, track_best_outputs=False,
            display_progress_bar=False, use_wandb=False, use_mlflow=False,
            seed=config.seed, raise_on_exception=True,
        )
    except _ScoringAbort as exc:
        exc.error.calls = [dict(row) for row in ledger.rows]
        raise exc.error from exc
    except TaskScoringError as exc:
        exc.calls = [dict(row) for row in ledger.rows]
        raise
    except _BudgetExhausted:
        hard_budget_stop = True
    except Exception as exc:
        raise GEPACompilationError(
            f"Official GEPA failed; no optimizer fallback was used: {type(exc).__name__}: {exc}",
            ledger.rows,
        ) from exc

    if not completed_validation:
        raise GEPACompilationError(
            "GEPA produced no fully evaluated selection candidate; no bundle frozen",
            ledger.rows,
        )
    if hard_budget_stop:
        chosen = max(completed_validation, key=lambda item: item["score"])
        selection_source = "best_completed_validation_before_hard_budget_stop"
    else:
        best = getattr(result, "best_candidate", None)
        if not isinstance(best, dict) or set(best) != {"system_prompt"}:
            raise GEPACompilationError("Official GEPA returned an invalid best candidate", ledger.rows)
        matching = [item for item in completed_validation
                    if item["fragment"] == best["system_prompt"]]
        if not matching:
            raise GEPACompilationError(
                "GEPA best candidate has no complete common-contract selection evaluation",
                ledger.rows,
            )
        chosen = matching[-1]
        selection_source = "official_gepa_result"

    unique: dict[str, dict] = {}
    for item in completed_validation:
        full = instruction(item["fragment"])
        unique.setdefault(item["candidate_id"], {
            "id": item["candidate_id"], "instruction": full,
            "instruction_hash": _text_hash(full),
        })
    selected = unique[chosen["candidate_id"]]
    budget_stopped = hard_budget_stop or state["budget_stop_requested"]
    failed = any(row["error"] is not None for row in ledger.rows)
    compile_status = "budget_stopped" if budget_stopped else ("partial" if failed else "completed")
    protocol = {
        "implementation": "official_gepa_package",
        "gepa_version": GEPA_VERSION,
        "baseline_label": "GEPA with synthetic data and local student reflection",
        "adaptations": ["local frozen student is reflection model",
                        "immutable common task/output contract",
                        "shared attempted-call cap",
                        "teacher-generated unverified optimization references"],
        "published_result_reproduction": False,
        "remote_teacher_calls": 0,
        "benchmark_test_access": False,
        "synthetic_holdout_access": False,
        "metric_call_budget": config.max_calls,
        "native_metric_calls": getattr(result, "total_metric_calls", None),
        "shared_attempted_call_budget": config.max_calls,
        "solver_calls": sum(row["role"] == "solver" for row in ledger.rows),
        "reflection_calls": sum(row["role"] == "optimizer" for row in ledger.rows),
        "candidate_selection": "pareto", "use_merge": False,
        "reflection_minibatch_size": reflection_size,
        "validation_policy": "full_eval", "cache_evaluation": False,
        "selection_source": selection_source,
        "optimizer_returned": result is not None,
        "budget_stopped": budget_stopped,
        "unused_call_budget": config.max_calls - len(ledger.rows),
        "candidate_rejections": rejections,
    }
    bundle = {
        "version": 1, "task_id": pack["task_id"], "method": "gepa_local",
        "mode": model.identity["mode"], "task_pack_hash": content_hash(pack),
        "description_hash": _text_hash(pack["description"]),
        "output_schema": output_schema, "config": asdict(config),
        "student_identity": dict(model.identity), "selected": selected,
        "candidates": list(unique.values()),
        "candidate_scores": [
            {"candidate_id": item["candidate_id"],
             "selection_score" if task_adapter else "selection_exact_match": item["score"]}
            for item in completed_validation
        ],
        "split_ids": {name: [case["id"] for case in rows] for name, rows in splits.items()},
        "generated_input_hashes": [normalized_input_hash(case["input"]) for case in pack["cases"]],
        "synthetic_holdout_hashes": {case["id"]: content_hash(case)
                                     for case in splits["synthetic_holdout"]},
        "usage": ledger.summary(), "compile_status": compile_status,
        "novelty_status": "not_established", "semantic_equivalence": "not_proven",
        "search_reference_kind": "teacher_generated_unverified",
        "optimizer_protocol": protocol,
    }
    if task_adapter is not None:
        bundle["task_adapter"] = task_adapter.metadata()
        protocol["task_scorer_id"] = task_adapter.scorer_id
        protocol["adaptations"][-1] = "generated practice cases scored by the declared local task verifier"
    bundle["bundle_hash"] = content_hash(bundle)
    return {"bundle": bundle, "calls": ledger.rows}
