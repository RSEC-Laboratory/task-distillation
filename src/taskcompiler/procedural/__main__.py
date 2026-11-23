"""Command-line entry point for reproducible academic experiments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .experiment import METHODS, OPTIMIZER_METHODS, SearchConfig, candidates_for, compile_instructions, evaluate_frozen
from .local_model import OllamaLocal
from .prompts import build_teacher_request
from .schema import content_hash, load_task_pack, split_cases, strict_json_loads, validate_task_pack


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def new_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def provenance() -> dict:
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    packages = {}
    for name in ("taskcompiler", "gepa", "dspy", "litellm", "optuna", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "python": sys.version, "platform": platform.platform(), "packages": packages}


def config_from(args) -> SearchConfig:
    return SearchConfig(seed=args.seed, candidates=args.candidates, shortlist=args.shortlist,
                        max_calls=args.max_calls, max_instruction_chars=args.max_instruction_chars)


def model_from(args) -> OllamaLocal:
    return OllamaLocal(args.model, endpoint=args.endpoint, seed=args.student_seed,
                       max_output_tokens=args.max_output_tokens, context_tokens=args.context_tokens,
                       timeout=args.timeout)


def prepare(args) -> None:
    # This is the sole remote-teacher entry point. There is no benchmark, private
    # input, student model or failure-feedback argument to this operation.
    description = args.description.read_text(encoding="utf-8").strip()
    system, prompt = build_teacher_request(description, args.task_id, args.num_cases,
                                           args.candidates, args.seed)
    new_directory(args.out)
    write_json(args.out / "teacher_request.json", {"system": system, "prompt": prompt})
    manifest = {"status": "running", "mode": "live", "provenance": provenance(),
                "teacher_access": "public_description_and_fixed_protocol_only",
                "teacher_usage": {"logical_calls": 0, "input_tokens": None,
                                  "output_tokens": None, "latency_seconds": None,
                                  "transport_retries": 0},
                "novelty_status": "not_established"}
    write_json(args.out / "manifest.json", manifest)
    try:
        from taskcompiler.teacher import TeacherClient
        teacher = TeacherClient(args.teacher_model, backend=args.teacher_backend, max_retries=0)
        manifest["teacher_identity"] = dict(teacher.identity)
        manifest["teacher_usage"]["logical_calls"] = 1
        response = teacher.call(system, prompt, max_tokens=args.teacher_max_tokens, seed=args.seed,
                                response_format={"type": "json_object"})
        manifest["teacher_identity"] = dict(teacher.identity)
        # The legacy client turns absent OpenRouter usage into zero. A generated
        # task-pack request necessarily consumes tokens; preserve that ambiguity.
        input_tokens = response.input_tokens if response.input_tokens and response.input_tokens > 0 else None
        output_tokens = response.output_tokens if response.output_tokens and response.output_tokens > 0 else None
        manifest["teacher_usage"].update(input_tokens=input_tokens, output_tokens=output_tokens,
                                         token_usage_complete=input_tokens is not None and output_tokens is not None,
                                         latency_seconds=response.latency_seconds,
                                         accounting="final_response_only; legacy zero counts treated as unknown")
        manifest["teacher_finish_reason"] = getattr(response, "finish_reason", None)
        (args.out / "teacher_response.txt").write_text(response.text, encoding="utf-8")
        if manifest["teacher_finish_reason"] in {"length", "max_tokens"}:
            raise ValueError("Teacher output reached its token cap")
        text = response.text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = "\n".join(text.splitlines()[1:-1])
        pack = strict_json_loads(text)
        if not isinstance(pack, dict):
            raise ValueError("Teacher task pack must be a JSON object")
        # Input metadata is authoritative. The teacher proposes semantic content,
        # not a replacement user request. Preserve its echoed metadata for audit.
        manifest["teacher_reported_metadata"] = {key: pack.get(key) for key in ("task_id", "description")}
        manifest["input_metadata_assigned_by_runner"] = True
        pack["task_id"] = args.task_id
        pack["description"] = description
        # Provenance comes from the actual invocation, never the model's claims.
        import hashlib
        pack["provenance"] = {"kind": "teacher_generated", "teacher_model": response.model,
                              "generation_seed": args.seed,
                              "description_sha256": hashlib.sha256(description.encode()).hexdigest()}
        validate_task_pack(pack)
        if len(pack["cases"]) != args.num_cases or len(pack["procedures"]) != args.candidates:
            raise ValueError("Teacher did not produce the locked number of cases/candidates")
        write_json(args.out / "task_pack.json", pack)
        manifest.update(status="completed", task_pack_hash=content_hash(pack))
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.out / "manifest.json", manifest)


def plan(args) -> None:
    pack = load_task_pack(args.pack)
    config = config_from(args)
    config.validate()
    splits = split_cases(pack["cases"], config.seed)
    methods = {}
    for method in METHODS:
        variants = candidates_for(pack, method, config)
        calls = len(variants) * len(splits["optimization"]) + config.shortlist * len(splits["selection"]) if method.endswith("_search") else 0
        methods[method] = {"candidate_count": len(variants), "required_compilation_calls": calls}
        if calls > config.max_calls:
            raise ValueError(f"{method} needs {calls} calls, exceeding budget {config.max_calls}")
    new_directory(args.out)
    write_json(args.out / "plan.json", {"mode": "dry_run", "status": "validated",
               "empirical_results": None, "novelty_status": "not_established",
               "task_pack_hash": content_hash(pack), "provenance": provenance(),
               "splits": {k: [c["id"] for c in v] for k, v in splits.items()}, "methods": methods})


def compile_run(args) -> None:
    pack = load_task_pack(args.pack)
    if pack["provenance"]["kind"] == "fixture":
        raise ValueError("A fixture task pack cannot be used as a live academic experiment")
    config = config_from(args)
    config.validate()
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Each method must occur once; duplicate arms would overwrite costs")
    new_directory(args.out)
    write_json(args.out / "task_pack.json", pack)
    manifest = {"status": "running", "mode": "live", "provenance": provenance(),
                "task_pack_hash": content_hash(pack), "novelty_status": "not_established",
                "empirical_effectiveness": "not_established", "methods": {}}
    generation_manifest = args.pack.parent / "manifest.json"
    manifest["shared_teacher_usage"] = None
    if generation_manifest.exists():
        generation = strict_json_loads(generation_manifest.read_text(encoding="utf-8"))
        if generation.get("task_pack_hash") == content_hash(pack):
            manifest["shared_teacher_usage"] = generation.get("teacher_usage")
            manifest["generation_manifest_hash"] = content_hash(generation)
    write_json(args.out / "manifest.json", manifest)
    try:
        model = model_from(args)
        order = list(args.methods)
        random.Random(config.seed).shuffle(order)
        manifest["method_order"] = order
        manifest["student_identity"] = dict(model.identity)
        for method in order:
            try:
                if method == "gepa_local":
                    from .gepa import optimize_gepa
                    result = optimize_gepa(pack, model, config)
                elif method == "mipro_local":
                    from .mipro import optimize_mipro
                    result = optimize_mipro(pack, model, config)
                else:
                    result = compile_instructions(pack, model, method, config)
            except Exception as exc:
                manifest["methods"][method] = "failed"
                if hasattr(exc, "calls"):
                    write_json(args.out / "compilation" / method / "calls.json", exc.calls)
                write_json(args.out / "compilation" / method / "error.json",
                           {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                write_json(args.out / "manifest.json", manifest)
                continue
            write_json(args.out / "compilation" / method / "bundle.json", result["bundle"])
            write_json(args.out / "compilation" / method / "calls.json", result["calls"])
            manifest["methods"][method] = result["bundle"]["compile_status"]
            write_json(args.out / "manifest.json", manifest)
        manifest["status"] = "partial" if any(v != "completed" for v in manifest["methods"].values()) else "completed"
        if "failed" in manifest["methods"].values():
            raise RuntimeError("One or more methods failed; independent arms and error artifacts were retained")
    except Exception as exc:
        manifest.update(status="partial" if manifest["methods"] else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.out / "manifest.json", manifest)


def evaluate(args) -> None:
    # This subcommand is separate: no teacher or instruction optimizer is created.
    bundle = strict_json_loads(args.bundle.read_text(encoding="utf-8"))
    if args.synthetic_pack:
        pack = load_task_pack(args.synthetic_pack)
        if content_hash(pack) != bundle["task_pack_hash"]:
            raise ValueError("Synthetic evaluation task pack differs from the frozen compilation")
        evaluation = {"task_id": pack["task_id"], "reference_kind": "teacher_generated_unverified",
                      "source": "jointly generated task pack; not independent task accuracy",
                      "cases": split_cases(pack["cases"], bundle["config"]["seed"])["synthetic_holdout"]}
    else:
        evaluation = strict_json_loads(args.evaluation.read_text(encoding="utf-8"))
    if evaluation.get("task_id") != bundle["task_id"]:
        raise ValueError("External dataset and frozen instruction have different task ids")
    if not args.synthetic_pack and (evaluation.get("reference_kind") != "independent" or not evaluation.get("source")):
        raise ValueError("External evaluation requires declared independent references and a source")
    new_directory(args.out)
    write_json(args.out / "bundle.json", bundle)
    write_json(args.out / "evaluation.json", evaluation)
    manifest = {"status": "running", "provenance": provenance(), "mode": "live",
                "evaluation_hash": content_hash(evaluation), "source": evaluation["source"],
                "bundle_hash": bundle["bundle_hash"], "evaluation_role": "exploratory",
                "task_id": bundle["task_id"], "method": bundle["method"],
                "compilation_status": bundle["compile_status"],
                "student_identity": bundle["student_identity"],
                "expected_case_count": len(evaluation["cases"])}
    write_json(args.out / "manifest.json", manifest)
    try:
        result = evaluate_frozen(bundle, model_from(args), evaluation["cases"],
                                 reference_kind=evaluation["reference_kind"], max_calls=args.max_calls)
        result["evaluation_hash"] = content_hash(evaluation)
        write_json(args.out / "predictions.json", result.pop("predictions"))
        write_json(args.out / "metrics.json", result)
        manifest["status"] = result["status"]
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.out / "manifest.json", manifest)


def summarize(args) -> None:
    from .reporting import build_report, write_report
    report = build_report(args.evaluations, args.methods,
                          bootstrap_seed=args.bootstrap_seed,
                          bootstrap_samples=args.bootstrap_samples)
    write_report(report, args.out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    teacher = commands.add_parser("prepare", help="Generate a task pack from a public description")
    teacher.add_argument("--description", type=Path, required=True)
    teacher.add_argument("--task-id", required=True)
    teacher.add_argument("--teacher-model", required=True)
    teacher.add_argument("--teacher-backend", choices=["anthropic", "openrouter"], default="anthropic")
    teacher.add_argument("--teacher-max-tokens", type=int, default=16000)
    teacher.add_argument("--num-cases", type=int, default=30)
    teacher.add_argument("--candidates", type=int, default=4)
    teacher.add_argument("--seed", type=int, default=0)
    teacher.set_defaults(func=prepare)
    for command, function in (("plan", plan), ("compile", compile_run), ("evaluate", evaluate)):
        sub = commands.add_parser(command)
        sub.set_defaults(func=function)
        sub.add_argument("--max-calls", type=int, default=200)
        if command in {"plan", "compile"}:
            sub.add_argument("--pack", type=Path, required=True)
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--candidates", type=int, default=4)
            sub.add_argument("--shortlist", type=int, default=2)
            sub.add_argument("--max-instruction-chars", type=int, default=12000)
        if command == "compile":
            sub.add_argument("--methods", nargs="+", choices=[*METHODS, *OPTIMIZER_METHODS], default=list(METHODS))
        if command == "evaluate":
            sub.add_argument("--bundle", type=Path, required=True)
            source = sub.add_mutually_exclusive_group(required=True)
            source.add_argument("--evaluation", type=Path)
            source.add_argument("--synthetic-pack", type=Path)
        if command in {"compile", "evaluate"}:
            sub.add_argument("--model", required=True)
            sub.add_argument("--endpoint", default="http://127.0.0.1:11434")
            sub.add_argument("--student-seed", type=int, default=0)
            sub.add_argument("--max-output-tokens", type=int, default=512)
            sub.add_argument("--context-tokens", type=int, default=8192)
            sub.add_argument("--timeout", type=float, default=120)
        sub.add_argument("--out", type=Path, required=True)
    teacher.add_argument("--out", type=Path, required=True)
    report = commands.add_parser("summarize", help="Audit and aggregate independent evaluation artifacts")
    report.add_argument("--evaluations", type=Path, nargs="+", required=True)
    report.add_argument("--methods", nargs="+", choices=[*METHODS, *OPTIMIZER_METHODS], required=True)
    report.add_argument("--bootstrap-seed", type=int, default=0)
    report.add_argument("--bootstrap-samples", type=int, default=2000)
    report.add_argument("--out", type=Path, required=True)
    report.set_defaults(func=summarize)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
