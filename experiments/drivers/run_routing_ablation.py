"""Evaluate routing widths with frozen artifacts and a frozen student."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from taskcompiler.config import RUNS_DIR, load_config
from taskcompiler.route import CertifiedInstructionRouter
from taskcompiler.rules_track.synthetic_policies import (
    _deployment_description,
    _evaluate_cases,
    _opaque_label_codebook,
    _student,
    make_policy_task,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top-k", default="0,1,2,4")
    parser.add_argument("--num-examples", type=int)
    parser.add_argument(
        "--current-compact-description",
        action="store_true",
    )
    parser.add_argument("--opaque-labels", action="store_true")
    args = parser.parse_args()

    config = load_config(args.run_dir / "config.yaml")
    widths = [
        int(value.strip())
        for value in args.top_k.split(",")
        if value.strip()
    ]
    if any(width < 0 for width in widths):
        raise ValueError("top-k values must be non-negative")
    student = _student(config)
    router = CertifiedInstructionRouter(
        config.retriever_model,
        retriever_revision=config.retriever_revision,
    )
    artifacts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.run_dir / "artifacts").glob("policy-*.json"))
    ]
    results = []
    for width in widths:
        total_correct = 0
        total_examples = 0
        total_input_tokens = 0
        total_output_tokens = 0
        predictions = []
        task_metrics = []
        for artifact in artifacts:
            task_number = int(artifact["task_id"].split("-")[1])
            task = make_policy_task(task_number)
            label_encoder, label_decoder = (
                _opaque_label_codebook(task)
                if args.opaque_labels
                else (None, None)
            )
            count = args.num_examples or config.num_examples
            cases = task.final_cases[:count]
            correct, rows, input_tokens, output_tokens = _evaluate_cases(
                task,
                cases,
                student,
                artifact["rendered_bundle"],
                instruction_router=router,
                top_k_rules=width,
                task_description=(
                    _deployment_description(
                        task,
                        False,
                        list(label_decoder)
                        if label_decoder
                        else None,
                    )
                    if args.current_compact_description
                    else artifact.get("deployment_description")
                ),
                label_encoder=label_encoder,
                label_decoder=label_decoder,
            )
            total_correct += correct
            total_examples += len(cases)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            predictions.extend(rows)
            task_metrics.append(
                {
                    "task_id": task.task_id,
                    "accuracy": correct / len(cases),
                    "num_examples": len(cases),
                }
            )
        results.append(
            {
                "top_k": width,
                "accuracy": total_correct / total_examples,
                "correct": total_correct,
                "num_examples": total_examples,
                "student_input_tokens": total_input_tokens,
                "student_output_tokens": total_output_tokens,
                "coverage_rate": sum(
                    row["routing_certificate"]["coverage_closed"]
                    for row in predictions
                )
                / len(predictions),
                "closure_add_rate": sum(
                    bool(
                        row["routing_certificate"][
                            "closure_added_rule_indices"
                        ]
                    )
                    for row in predictions
                )
                / len(predictions),
                "mean_instruction_count": sum(
                    row["routed_instruction_count"] for row in predictions
                )
                / len(predictions),
                "task_metrics": task_metrics,
            }
        )

    report = {
        "study": "routing_width_development_ablation",
        "source_run": str(args.run_dir.resolve()),
        "created_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "student_identity": student.identity,
        "router_identity": router.identity,
        "current_compact_description": args.current_compact_description,
        "opaque_labels": args.opaque_labels,
        "results": results,
    }
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = RUNS_DIR / f"routing_ablation_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"routing ablation: {path}")


if __name__ == "__main__":
    main()
