"""Run the frozen T14 Gemma confirmation."""

from dataclasses import replace
from pathlib import Path

import run_t10_gemma_replication as confirmation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
confirmation.PROTOCOL_PATH = (
    PROJECT_ROOT
    / "experiments" / "protocols"
    / "gemma_generalized_confirmation_2026-07-27.json"
)
confirmation.MANIFEST_PATH = (
    PROJECT_ROOT / "evaluation/results/confirmation/gemma_confirmation/manifest.json"
)
confirmation.RUN_PREFIX = "t14"
_shared_base = confirmation._base


def _base(task_id: int):
    """Use syntax recovery and generalized source corrections."""
    return replace(
        _shared_base(task_id),
        max_teacher_syntax_repair_rounds=2,
        max_counterexample_patches=4,
    )


confirmation._base = _base


if __name__ == "__main__":
    confirmation.main()
