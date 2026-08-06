"""Store per-rule provenance and results as linked JSONL records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RuleRecord:
    rule_name: str
    source_clause: str | None
    build_tests_passed: int
    build_tests_total: int
    audit_tests_passed: int = 0
    audit_tests_total: int = 0
    mutation_adequacy: float | None = None
    obligation_ids: list[str] = field(default_factory=list)
    student_failures_before: list[str] = field(default_factory=list)
    student_failures_after: list[str] = field(default_factory=list)
    repairs_applied: list[str] = field(default_factory=list)
    translation_roundtrip_ok: bool | None = None


class EvidenceGraph:
    def __init__(self, path: Path):
        self.path = path
        self._records: list[RuleRecord] = []

    def add(self, record: RuleRecord) -> None:
        self._records.append(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def find(self, rule_name: str) -> RuleRecord | None:
        for record in self._records:
            if record.rule_name == rule_name:
                return record
        return None

    @staticmethod
    def load(path: Path) -> list[RuleRecord]:
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(RuleRecord(**json.loads(line)))
        return records
