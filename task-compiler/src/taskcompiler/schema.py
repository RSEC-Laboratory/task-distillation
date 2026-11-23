"""Define task specifications and three-valued logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Boolean3(str, Enum):
    """Represent a fact as true, false, or unknown."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"

    def __and__(self, other: "Boolean3") -> "Boolean3":
        if self is Boolean3.FALSE or other is Boolean3.FALSE:
            return Boolean3.FALSE
        if self is Boolean3.UNKNOWN or other is Boolean3.UNKNOWN:
            return Boolean3.UNKNOWN
        return Boolean3.TRUE

    def __or__(self, other: "Boolean3") -> "Boolean3":
        if self is Boolean3.TRUE or other is Boolean3.TRUE:
            return Boolean3.TRUE
        if self is Boolean3.UNKNOWN or other is Boolean3.UNKNOWN:
            return Boolean3.UNKNOWN
        return Boolean3.FALSE

    def negate(self) -> "Boolean3":
        if self is Boolean3.UNKNOWN:
            return Boolean3.UNKNOWN
        return Boolean3.FALSE if self is Boolean3.TRUE else Boolean3.TRUE


@dataclass
class EnumType:
    """A finite, task-declared set of named values, e.g. an operation type."""

    name: str
    values: list[str]


@dataclass
class FieldDecl:
    """Define a record field and its value type."""

    name: str
    type_name: str


@dataclass
class RecordType:
    """A task-declared input record, e.g. one event with several fields."""

    name: str
    fields: list[FieldDecl]


@dataclass
class TaskSpec:
    """Describe the public task shown to the teacher."""

    task_name: str
    description: str
    labels: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    forbidden_shortcuts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    enums: list[EnumType] = field(default_factory=list)
    records: list[RecordType] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "description": self.description,
            "labels": self.labels,
            "requirements": self.requirements,
            "forbidden_shortcuts": self.forbidden_shortcuts,
            "sources": self.sources,
            "enums": [{"name": e.name, "values": e.values} for e in self.enums],
            "records": [
                {"name": r.name, "fields": [{"name": f.name, "type_name": f.type_name} for f in r.fields]}
                for r in self.records
            ],
        }

    @staticmethod
    def from_dict(data: dict) -> "TaskSpec":
        return TaskSpec(
            task_name=data["task_name"],
            description=data["description"],
            labels=list(data.get("labels", [])),
            requirements=list(data.get("requirements", [])),
            forbidden_shortcuts=list(data.get("forbidden_shortcuts", [])),
            sources=list(data.get("sources", [])),
            enums=[EnumType(e["name"], list(e["values"])) for e in data.get("enums", [])],
            records=[
                RecordType(r["name"], [FieldDecl(f["name"], f["type_name"]) for f in r["fields"]])
                for r in data.get("records", [])
            ],
        )
