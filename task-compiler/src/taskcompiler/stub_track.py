"""Exercise the orchestration with a small hardcoded track."""

from __future__ import annotations

from taskcompiler.config import RunConfig
from taskcompiler.metrics import RunRecorder

EXAMPLES = [
    ("unauthorized write outside maintenance window", "manipulation"),
    ("authorized read of a diagnostic register", "not_manipulation"),
    ("authorized write during approved maintenance window", "not_manipulation"),
    ("unauthorized read of a diagnostic register", "not_manipulation"),
]


def _rule(text: str) -> str:
    if "write" in text and "unauthorized" in text and "maintenance" not in text:
        return "manipulation"
    return "not_manipulation"


def execute(config: RunConfig, recorder: RunRecorder) -> dict:
    correct = 0
    with recorder.stage("stub_classification"):
        for text, gold in EXAMPLES:
            prediction = _rule(text)
            recorder.log(f"example={text!r} prediction={prediction} gold={gold}")
            if prediction == gold:
                correct += 1

    accuracy = correct / len(EXAMPLES)
    metrics = {"accuracy": accuracy, "num_examples": len(EXAMPLES)}
    recorder.set_task_metrics(metrics)
    return metrics
