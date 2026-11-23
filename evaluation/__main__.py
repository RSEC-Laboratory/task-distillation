"""Execute integrity checks, program replay, and saved-prediction evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from evaluation.layout import PROJECT_ROOT, prepare_records, verify_manifest


def compare_reference(actual: dict, expected: dict) -> None:
    """Require identical results, allowing optional upstream validation coverage."""
    comparable = deepcopy(actual)
    coverage = comparable["reserved_nlp"]["source_instance_validation"]
    if coverage["expected_tasks"] != 20 or not 0 <= coverage["tasks_checked"] <= 20:
        raise ValueError("Unexpected upstream source-validation inventory.")
    comparable["reserved_nlp"]["source_instance_validation"] = expected["reserved_nlp"]["source_instance_validation"]
    if comparable != expected:
        raise ValueError("Reproduced results differ from the retained reference.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluation/results/reproduced")
    parser.add_argument("--verify-only", action="store_true", help="Check release hashes without scoring.")
    args = parser.parse_args()
    count = verify_manifest()
    print(f"Verified {count} project files.", flush=True)
    if args.verify_only:
        return
    output = args.output.resolve()
    if output == PROJECT_ROOT or any(
        output.is_relative_to(PROJECT_ROOT / directory)
        for directory in ("src", "tests", "experiments", "data", "provenance", "evaluation/records")
    ):
        parser.error("Choose an output directory separate from code and retained inputs.")
    prepare_records()
    output.mkdir(parents=True, exist_ok=True)
    commands = (
        ["evaluation.core", "--output", str(output / "core.json")],
        ["evaluation.reproduce", "--output", str(output)],
    )
    for module, *arguments in commands:
        subprocess.run([sys.executable, "-B", "-S", "-m", module, *arguments], cwd=PROJECT_ROOT, check=True)
    expected = json.loads((PROJECT_ROOT / "evaluation/results/reference.json").read_text())
    actual = json.loads((output / "results.json").read_text())
    compare_reference(actual, expected)
    print(f"All checks passed. Results: {output}")


if __name__ == "__main__":
    main()
