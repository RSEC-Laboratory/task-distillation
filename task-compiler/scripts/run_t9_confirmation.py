"""Run the frozen T9 patched-CEGIS confirmation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from taskcompiler.config import RULES_TRACK, RunConfig
from taskcompiler.run import execute_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "protocols"
    / "patched_cegis_confirmation_2026-07-27.json"
)
MANIFEST_PATH = PROJECT_ROOT / "runs" / "t9_confirmation.json"


def _load_environment() -> None:
    """Load project credentials."""
    for path in [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / "task-compiler" / ".env",
    ]:
        if path.exists():
            load_dotenv(path)
            return


def _base(task_id: int) -> RunConfig:
    """Build shared confirmation settings.

    Args:
        task_id: Locked confirmation task number.
    """
    return RunConfig(
        track=RULES_TRACK,
        dataset="synthetic_policies",
        baseline="raw_task",
        run_label=f"t9_raw_task{task_id:03d}",
        benchmark_split="validation_confirmation",
        benchmark_task_ids=[task_id],
        num_examples=27,
        seed=20260727,
        open_final_evaluation=True,
        compilation_failure_fallback="raw_task",
        teacher_model="meta-llama/llama-3.1-70b-instruct",
        teacher_backend="openrouter",
        teacher_selection="lineage_matched",
        teacher_structured_outputs=True,
        teacher_timeout_seconds=180,
        student_model="llama3.2:3b",
        student_backend="ollama",
        ollama_base_url="http://192.168.2.115:11434",
        ollama_timeout_seconds=120,
        student_context_tokens=8192,
        student_thinking=False,
        student_seed=0,
        max_answer_tokens=16,
        deployment_include_task_spec=False,
        token_budget=1800,
        max_repair_rounds=3,
        max_teacher_repair_rounds=3,
        max_counterexample_patches=3,
        max_protected_regressions=0,
        retriever_model="sentence-transformers/all-MiniLM-L6-v2",
        top_k_rules=6,
        routing_opaque_labels=False,
    )


def _summary(run_dir: Path) -> dict:
    """Read comparable metrics from one run.

    Args:
        run_dir: Completed run directory.
    """
    payload = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    metrics = payload["task_metrics"]
    return {
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "correct": round(metrics["accuracy"] * metrics["num_final_examples"]),
        "num_examples": metrics["num_final_examples"],
        "compilation_failure_rate": metrics["compilation_failure_rate"],
        "teacher_calls": metrics["teacher_calls"],
        "student_calls": metrics["student_calls"],
        "remote_student_calls": 0,
    }


def _run_task(task_id: int) -> dict:
    """Run all three frozen task conditions.

    Args:
        task_id: Locked confirmation task number.
    """
    base = _base(task_id)
    raw_dir = execute_run(base)
    record = {
        "task_id": task_id,
        "conditions": {"raw_task": _summary(raw_dir)},
    }
    method = replace(
        base,
        baseline="teacher_validated_routing",
        run_label=f"t9_patched_cegis_task{task_id:03d}",
        teacher_refine_with_audit=True,
        teacher_build_pass_threshold=1.0,
        teacher_audit_pass_threshold=1.0,
    )
    method_dir = execute_run(method)
    record["conditions"]["patched_cegis"] = _summary(method_dir)
    compilation = json.loads(
        (
            method_dir
            / "artifacts"
            / f"policy-{task_id:03d}.teacher.compilation.json"
        ).read_text(encoding="utf-8")
    )
    record["source_verification"] = {
        "build": compilation["final_build_pass_rate"],
        "audit": compilation["final_audit_pass_rate"],
        "repairs": len(compilation["repairs"]),
        "patches": sum(
            item["rule_name"].startswith("source-counterexample:")
            for item in compilation["repairs"]
        ),
        "teacher_calls": compilation["teacher_calls"],
    }
    initial_path = (
        method_dir
        / "artifacts"
        / f"policy-{task_id:03d}.teacher.initial.dsl"
    )
    one_shot = replace(
        base,
        baseline="teacher_one_shot",
        run_label=f"t9_one_shot_task{task_id:03d}",
        teacher_program_path=str(initial_path),
        max_teacher_repair_rounds=0,
        max_counterexample_patches=0,
    )
    one_shot_dir = execute_run(one_shot)
    record["conditions"]["frozen_teacher_one_shot"] = _summary(one_shot_dir)
    return record


def _counts(
    manifest: dict,
    task_ids: list[int],
    *,
    minimum: int,
) -> dict:
    """Count completed task-level gate outcomes.

    Args:
        manifest: Current confirmation records.
        task_ids: Tasks in the active stage.
        minimum: Required exact and positive task counts.
    """
    records = [
        manifest["tasks"][str(task_id)]
        for task_id in task_ids
        if str(task_id) in manifest["tasks"]
    ]
    complete = [row for row in records if "conditions" in row]
    exact = sum(
        row["source_verification"]["build"] == 1.0
        and row["source_verification"]["audit"] == 1.0
        for row in complete
    )
    positive_raw = sum(
        row["conditions"]["patched_cegis"]["correct"]
        > row["conditions"]["raw_task"]["correct"]
        for row in complete
    )
    positive_one_shot = sum(
        row["conditions"]["patched_cegis"]["correct"]
        > row["conditions"]["frozen_teacher_one_shot"]["correct"]
        for row in complete
    )
    completed = len(records)
    remaining = len(task_ids) - completed
    futile = (
        exact + remaining < minimum
        or positive_raw + remaining < minimum
        or positive_one_shot + remaining < minimum
    )
    passed = (
        completed == len(task_ids)
        and exact >= minimum
        and positive_raw >= minimum
        and positive_one_shot >= minimum
    )
    return {
        "completed_tasks": completed,
        "exactly_verified_tasks": exact,
        "positive_tasks_vs_raw": positive_raw,
        "positive_tasks_vs_one_shot": positive_one_shot,
        "remaining_tasks": remaining,
        "minimum_required": minimum,
        "futile": futile,
        "passed": passed,
    }


def _save(manifest: dict) -> None:
    """Persist confirmation progress.

    Args:
        manifest: Current task records.
    """
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Run one locked confirmation stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[1, 2], type=int, required=True)
    args = parser.parse_args()
    _load_environment()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    stage_task_ids = protocol[f"stage_{args.stage}_task_ids"]
    manifest = {
        "protocol": str(PROTOCOL_PATH.relative_to(PROJECT_ROOT)),
        "tasks": {},
    }
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if args.stage == 2:
        stage_1_gate = manifest.get("stage_1_gate") or {}
        if not stage_1_gate.get("passed"):
            raise RuntimeError("T9 stage 1 has not passed")
        gate_task_ids = [
            *protocol["stage_1_task_ids"],
            *protocol["stage_2_task_ids"],
        ]
        minimum = protocol["full_llama_gate"][
            "positive_tasks_vs_each_control_minimum"
        ]
    else:
        gate_task_ids = stage_task_ids
        minimum = protocol["stage_1_gate"][
            "positive_tasks_vs_raw_minimum"
        ]
    for task_id in stage_task_ids:
        if str(task_id) in manifest["tasks"]:
            continue
        print(f"starting T9 task {task_id:03d}", flush=True)
        try:
            record = _run_task(task_id)
        except Exception as exc:
            record = {
                "task_id": task_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        manifest["tasks"][str(task_id)] = record
        gate = _counts(
            manifest,
            gate_task_ids,
            minimum=minimum,
        )
        manifest[f"stage_{args.stage}_gate"] = gate
        _save(manifest)
        print(json.dumps(record), flush=True)
        print(json.dumps(gate), flush=True)
        if gate["futile"]:
            print(
                f"stopping T9 stage {args.stage} for futility",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
