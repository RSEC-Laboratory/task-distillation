"""Load ContractNLI examples into the project's data structures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from taskcompiler.config import DATA_DIR

LABEL_MAP = {
    "Entailment": "entailment",
    "Contradiction": "contradiction",
    "NotMentioned": "not_mentioned",
}


@dataclass
class ContractNliExample:
    document_id: str
    hypothesis_id: str
    hypothesis_text: str
    contract_text: str
    label: str


def load_split(split: str, data_dir: Path = DATA_DIR / "contract_nli") -> list[ContractNliExample]:
    path = data_dir / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_data.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data["labels"]

    examples: list[ContractNliExample] = []
    for document in data["documents"]:
        annotation_sets = document.get("annotation_sets", [])
        if not annotation_sets:
            continue
        annotations = annotation_sets[0]["annotations"]
        for hypothesis_id, annotation in annotations.items():
            if hypothesis_id not in labels:
                continue
            examples.append(
                ContractNliExample(
                    document_id=str(document["id"]),
                    hypothesis_id=hypothesis_id,
                    hypothesis_text=labels[hypothesis_id]["hypothesis"],
                    contract_text=document["text"],
                    label=LABEL_MAP[annotation["choice"]],
                )
            )
    return examples


def filter_by_hypothesis(examples: list[ContractNliExample], hypothesis_id: str) -> list[ContractNliExample]:
    return [e for e in examples if e.hypothesis_id == hypothesis_id]
