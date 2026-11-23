"""Run the prospectively specified original-pipeline/native-GEPA comparison.

Optimization receives only serialized public DTOs. All instructions are locked
on disk before the evaluation function can read the separate final-case file.
No existing run directory is overwritten and no failed attempt is retried.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/protocols/matched_original/development_20261005_v1.json"


def now() -> str:
    return datetime.fromtimestamp(time.time(), timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def source_inventory() -> dict[str, str]:
    tracked = subprocess.check_output(
        ["git", "ls-files", "src", "experiments/drivers/matched_original.py",
         "experiments/drivers/run_matched_original.py", "experiments/drivers/report_matched_original.py",
         "experiments/protocols/matched_original/development_20261005_v1.json"],
        cwd=ROOT, text=True, encoding="utf-8",
    ).splitlines()
    required = {"experiments/drivers/matched_original.py", "experiments/drivers/run_matched_original.py",
                "experiments/drivers/report_matched_original.py",
                "experiments/protocols/matched_original/development_20261005_v1.json"}
    if not required.issubset(tracked):
        raise ValueError("Commit the complete implementation and protocol before live inference")
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--", *tracked],
                                    cwd=ROOT, encoding="utf-8")
    if dirty:
        raise ValueError("Experiment sources differ from the committed snapshot")
    return {name: digest(ROOT / name) for name in tracked}


def verify_sources(inventory: dict[str, str]) -> None:
    for name, expected in inventory.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")


def freeze_native_package(output: Path) -> dict[str, str]:
    distribution = importlib.metadata.distribution("gepa")
    if distribution.version != "0.1.4":
        raise ValueError("Native GEPA version differs from protocol")
    hashes = {}
    for relative in distribution.files or []:
        name = str(relative).replace("\\", "/")
        if not (name.endswith(".py") or name.endswith("/METADATA") or name.endswith("/RECORD")):
            continue
        original = Path(distribution.locate_file(relative))
        target = output / "dependencies" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        hashes[name] = digest(original)
    if not hashes or "gepa/core/engine.py" not in hashes:
        raise ValueError("Official native GEPA source snapshot is incomplete")
    write_json(output / "native_package_snapshot.json", {"distribution": "gepa", "version": distribution.version,
               "files_sha256": hashes})
    return hashes


def verify_native_package(hashes: dict[str, str]) -> None:
    distribution = importlib.metadata.distribution("gepa")
    if distribution.version != "0.1.4":
        raise ValueError("Native GEPA distribution version changed")
    for name, expected in hashes.items():
        if digest(Path(distribution.locate_file(name))) != expected:
            raise ValueError(f"Frozen native GEPA distribution changed: {name}")


def lock_prepared(output: Path, paths: list[Path], expected_count: int) -> dict:
    if len(paths) != expected_count or len(set(paths)) != expected_count:
        raise ValueError("All task/model/algorithm artifacts must exist before final opening")
    lock = {"locked_at": now(), "expected_artifacts": expected_count,
            "artifacts": {str(path.relative_to(output)): digest(path) for path in paths}}
    write_json(output / "prepared_lock.json", lock)
    return lock


def verify_prepared_lock(output: Path, expected_count: int) -> dict:
    path = output / "prepared_lock.json"
    if not path.is_file():
        raise ValueError("Final evaluation requires the global instruction lock")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["expected_artifacts"] != expected_count or len(lock["artifacts"]) != expected_count:
        raise ValueError("Incomplete prepared-artifact lock")
    for name, expected in lock["artifacts"].items():
        artifact = output / name
        if artifact.resolve().parent != (output / "prepared").resolve():
            raise ValueError("Prepared artifact must remain in the prepared folder")
        if digest(artifact) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    return lock


def open_final_cases(output: Path, expected_count: int) -> dict:
    verify_prepared_lock(output, expected_count)
    seal = json.loads((output / "case_commitment.json").read_text(encoding="utf-8"))
    final_path = output / "sealed/final_cases.json"
    if digest(final_path) != seal["final_cases_sha256"]:
        raise ValueError("Final-case commitment changed")
    write_json(output / "final_opening.json", {"opened_at": now(),
               "prepared_lock_sha256": digest(output / "prepared_lock.json"),
               "final_cases_sha256": seal["final_cases_sha256"]})
    return json.loads(final_path.read_text(encoding="utf-8"))


def prepare_data(output: Path, protocol: dict) -> None:
    from taskcompiler.rules_track.synthetic_policies import make_policy_task
    public, final = [], {}
    for number in protocol["task_ids"]:
        task = make_policy_task(number)
        def rows(cases, split):
            return [{"case_id": f"{task.task_id}-{split}-{index:02d}",
                     "input_record": case.input_record,
                     "expected_label": case.expected_label}
                    for index, case in enumerate(cases)]
        public.append({"task_id": task.task_id, "spec": task.task_spec.to_dict(),
                       "build_cases": rows(task.build_cases, "build"),
                       "audit_cases": rows(task.audit_cases, "refinement")})
        final[task.task_id] = rows(task.final_cases, "final")
    write_json(output / "public_tasks.json", public)
    write_json(output / "sealed/final_cases.json", final)
    write_json(output / "case_commitment.json", {
        "committed_at": now(), "public_tasks_sha256": digest(output / "public_tasks.json"),
        "final_cases_sha256": digest(output / "sealed/final_cases.json"),
        "optimizer_access": "Only public_tasks.json DTOs; final inputs/labels and oracle AST excluded"})


def run(output: Path, protocol_path: Path = PROTOCOL, *, prepare_only: bool = False) -> None:
    from matched_original import (PublicTask, PublicCase, Limits, EventLedger,
                                 ModelSession, PreparedMethod, compile_full,
                                 compile_gepa, evaluate_prepared, make_router)
    from taskcompiler.procedural.local_model import OllamaLocal

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol_path.resolve() != PROTOCOL.resolve():
        raise ValueError("This version supports only its frozen protocol")
    if output.exists() or output.is_symlink():
        raise ValueError("Choose a fresh output directory; completed and failed evidence is retained")
    if not output.resolve().is_relative_to((ROOT / "evaluation/results/matched_gepa").resolve()):
        raise ValueError("Output must be inside reports/matched_original")
    if shutil.disk_usage(ROOT).free < protocol["stop_rules"]["minimum_free_workspace_bytes"]:
        raise ValueError("Workspace free-space stop rule failed")
    sources = source_inventory()
    output.mkdir(parents=True)
    started = time.monotonic()
    # This declaration precedes initialization, sampling or any model inference.
    write_json(output / "resource_scope.json", {
        "declared_at": now(), "prospective": True,
        "scope": "All local preparation and final evaluation; model identity queries included in wall time.",
        "costs": "Every model attempt, token usage where available and model latency; per-cell wall time includes Full routing.",
        "limitations": "No energy measurement or exact GPU utilization attribution; equal total compute is not claimed.",
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {key: importlib.metadata.version(key)
                     for key in ("gepa", "torch", "sentence-transformers", "psutil")}})
    shutil.copyfile(protocol_path, output / "protocol.json")
    write_json(output / "source_snapshot.json", {"frozen_at": now(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": sources})
    for name in sources:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    package_hashes = freeze_native_package(output)
    prepare_data(output, protocol)
    if prepare_only:
        write_json(output / "prepare_only.json", {"completed_at": now(), "model_inference_calls": 0})
        return

    limits = Limits(solver_calls=protocol["optimization_solver_call_ceiling_per_task_arm"],
                    teacher_calls=protocol["optimization_teacher_call_ceiling_per_task_arm"],
                    teacher_tokens=protocol["teacher_max_output_tokens"],
                    max_instruction_words=protocol["max_instruction_words"],
                    top_k=protocol["full"]["router_top_k"], seed=protocol["seed"])
    models = {}
    for name in [protocol["teacher_model"], *protocol["solver_models"]]:
        models[name] = OllamaLocal(name, endpoint=protocol["endpoint"], seed=protocol["seed"],
                     max_output_tokens=(protocol["teacher_max_output_tokens"]
                                        if name == protocol["teacher_model"]
                                        else protocol["student_max_output_tokens"]),
                     context_tokens=protocol["context_tokens"],
                     timeout=protocol["model_timeout_seconds"])
        # The reused transport's generic identity describes schema decoding.
        # Here it is determined by each actual request: DSL JSON versus text.
        models[name].identity["structured_decoding"] = "per_request_output_schema"
        if models[name].identity["digest"] != protocol["model_digests"][name]:
            raise ValueError("Installed checkpoint digest differs from prospective protocol")
    write_json(output / "model_identities.json", {name: model.identity for name, model in models.items()})
    server_version = models[protocol["teacher_model"]]._request("/api/version")
    if server_version["version"] != protocol["ollama_version"]:
        raise ValueError("Ollama runtime version differs from prospective protocol")
    write_json(output / "backend_identity.json", {
        "server_version": server_version,
        "models": {name: model._request("/api/show", {"model": name}) for name, model in models.items()}})
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    snapshot = (Path(os.environ["USERPROFILE"]) / ".cache/huggingface/hub" /
                "models--sentence-transformers--all-MiniLM-L6-v2/snapshots" /
                protocol["full"]["router_revision"])
    router = make_router(snapshot)
    write_json(output / "router_identity.json", {
        "identity": router.identity,
        "files_sha256": {str(path.relative_to(snapshot)): digest(path)
                         for path in sorted(snapshot.rglob("*")) if path.is_file()}})
    tasks = [PublicTask.from_dict(row) for row in json.loads((output / "public_tasks.json").read_text(encoding="utf-8"))]
    artifacts, prepared = [], {}
    backend_student_failures = attempted_student = 0

    def check_stop():
        verify_sources(sources)
        verify_native_package(package_hashes)
        if time.monotonic() - started > protocol["stop_rules"]["wall_limit_seconds"]:
            raise RuntimeError("Prospective wall-time stop rule reached")
        if shutil.disk_usage(ROOT).free < protocol["stop_rules"]["minimum_free_workspace_bytes"]:
            raise RuntimeError("Prospective free-space stop rule reached")
        if attempted_student and backend_student_failures / attempted_student > .05:
            raise RuntimeError("Prospective student backend-error stop rule reached")

    try:
        for model_index, model_name in enumerate(protocol["solver_models"]):
            for task_index, task in enumerate(tasks):
                order = ["full", "gepa"] if (model_index + task_index) % 2 == 0 else ["gepa", "full"]
                for arm in order:
                    check_stop()
                    cell = f"{model_name.replace(':', '_')}-{task.task_id}-{arm}"
                    ledger = EventLedger(output / "calls" / f"{cell}.jsonl",
                                         context={"cell": cell, "task_id": task.task_id,
                                                  "solver_model": model_name, "arm": arm})
                    solver = ModelSession(models[model_name], ledger, "solver_optimization", limits.solver_calls)
                    teacher = ModelSession(models[protocol["teacher_model"]], ledger, "teacher_optimization", limits.teacher_calls)
                    cell_started = time.monotonic()
                    result = (compile_full(task, solver, teacher, router, limits)
                              if arm == "full" else compile_gepa(task, solver, teacher, limits))
                    path = output / "prepared" / f"{cell}.json"
                    write_json(path, {"task_id": task.task_id, "solver_model": model_name,
                               "arm": arm, "prepared_at": now(), "method": result.to_dict(),
                               "preparation_wall_seconds": time.monotonic() - cell_started,
                               "call_summary": ledger.summary()})
                    artifacts.append(path)
                    prepared[(model_name, task.task_id, arm)] = path
                    # A failure report stays in the ledger. Infrastructure errors are
                    # distinct from truncated outputs and ordinary compilation failure.
                    attempted_student += solver.calls
                    backend_student_failures += solver.backend_failures
                    print(f"PREPARED {cell} solver={solver.calls} teacher={teacher.calls}", flush=True)
        check_stop()
        expected = len(tasks) * len(protocol["solver_models"]) * 2
        lock_prepared(output, artifacts, expected)
        final_data = open_final_cases(output, expected)
        for model_index, model_name in enumerate(protocol["solver_models"]):
            for task_index, task in enumerate(tasks):
                check_stop()
                final_cases = [PublicCase(**row) for row in final_data[task.task_id]]
                arms = ["raw", "full", "gepa"]
                if (model_index + task_index) % 2:
                    arms.reverse()
                for arm in arms:
                    check_stop()
                    verify_prepared_lock(output, expected)
                    cell = f"{model_name.replace(':', '_')}-{task.task_id}-{arm}"
                    method = (PreparedMethod.raw(task) if arm == "raw" else PreparedMethod.from_dict(
                        json.loads(prepared[(model_name, task.task_id, arm)].read_text(encoding="utf-8"))["method"]))
                    ledger = EventLedger(output / "calls" / f"{cell}-final.jsonl",
                                         context={"cell": cell, "task_id": task.task_id,
                                                  "solver_model": model_name, "arm": arm})
                    solver = ModelSession(models[model_name], ledger, "solver_final", len(final_cases))
                    cell_started = time.monotonic()
                    rows = evaluate_prepared(task, method, final_cases, solver, router)
                    write_json(output / "final" / f"{cell}.json", {
                        "task_id": task.task_id, "solver_model": model_name, "arm": arm,
                        "completed_at": now(), "rows": rows,
                        "final_wall_seconds": time.monotonic() - cell_started,
                        "call_summary": ledger.summary()})
                    attempted_student += solver.calls
                    backend_student_failures += solver.backend_failures
                    print(f"FINAL {cell} correct={sum(row['correct'] for row in rows)}/{len(rows)}", flush=True)
                # The teacher-produced program is an additional computational
                # control. The oracle program never enters this execution path.
                from taskcompiler.dsl.parser import parse_program
                from taskcompiler.dsl.interpreter import run as execute_dsl
                full = json.loads(prepared[(model_name, task.task_id, "full")].read_text(encoding="utf-8"))["method"]
                source = full.get("dsl_source")
                program = parse_program(source, task_spec=task.spec) if source else None
                direct_rows = []
                for case in final_cases:
                    predicted = execute_dsl(program, case.input_record).output_label if program else "__NO_VERIFIED_PROGRAM__"
                    direct_rows.append({"case_id": case.case_id, "expected_label": case.expected_label,
                                        "predicted_label": predicted, "correct": predicted == case.expected_label})
                write_json(output / "final" / f"{model_name.replace(':', '_')}-{task.task_id}-direct_dsl.json", {
                    "task_id": task.task_id, "solver_model": model_name, "arm": "direct_dsl",
                    "completed_at": now(), "rows": direct_rows, "program_available": bool(program),
                    "additional_model_calls": 0})
        verify_sources(sources)
        verify_prepared_lock(output, expected)
        write_json(output / "completion.json", {"status": "complete", "completed_at": now(),
                   "elapsed_seconds": time.monotonic() - started,
                   "total_student_attempts": attempted_student,
                   "backend_student_failures": backend_student_failures,
                   "prepared_lock_sha256": digest(output / "prepared_lock.json")})
    except BaseException as exc:
        write_json(output / "interruption.json", {"status": "interrupted", "recorded_at": now(),
                   "exception_type": type(exc).__name__, "reason": str(exc),
                   "elapsed_seconds": time.monotonic() - started,
                   "final_opened": (output / "final_opening.json").exists(),
                   "automatic_retry": False})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), prepare_only=args.prepare_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
