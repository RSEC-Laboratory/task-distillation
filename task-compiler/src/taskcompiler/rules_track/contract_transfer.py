"""Transfer public obligation knowledge to frozen students."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taskcompiler.config import (
    CONTRACT_ARTIFACT_DELIVERY_MODES,
    CONTRACT_CONTRASTIVE_RENDERED,
    CONTRACT_LINEAGE_RENDERED,
    CONTRACT_RETRIEVAL_AND_CHECKS,
    CONTRACT_RETRIEVAL_ONLY,
    DATA_DIR,
    OPENROUTER_BACKEND,
    RunConfig,
)
from taskcompiler.lineage import DistillationLineage, validate_lineage_config
from taskcompiler.metrics import RunRecorder
from taskcompiler.rules_track.contract_nli import (
    ContractNliExample,
    filter_by_hypothesis,
    load_split,
)
from taskcompiler.statistics import wilson_accuracy
from taskcompiler.student import StudentModel
from taskcompiler.teacher import TeacherClient


PUBLIC_HYPOTHESES = {
    "nda-1": "All Confidential Information shall be expressly identified by the Disclosing Party.",
    "nda-2": "Confidential Information shall only include technical information.",
    "nda-3": "Confidential Information may include verbally conveyed information.",
    "nda-4": "Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.",
    "nda-5": "Receiving Party may share some Confidential Information with some of Receiving Party's employees.",
    "nda-7": "Receiving Party may share some Confidential Information with some third-parties, including consultants, agents and professional advisors.",
    "nda-8": "Receiving Party shall notify Disclosing Party when law, regulation or judicial process requires disclosure.",
    "nda-10": "Receiving Party shall not disclose that Agreement was agreed or negotiated.",
    "nda-11": "Receiving Party shall not reverse engineer objects embodying Disclosing Party's Confidential Information.",
    "nda-12": "Receiving Party may independently develop information similar to Confidential Information.",
    "nda-13": "Receiving Party may acquire information similar to Confidential Information from a third party.",
    "nda-15": "Agreement shall not grant Receiving Party any right to Confidential Information.",
    "nda-16": "Receiving Party shall destroy or return some Confidential Information when Agreement terminates.",
    "nda-17": "Receiving Party may create a copy of some Confidential Information in some circumstances.",
    "nda-18": "Receiving Party shall not solicit some of Disclosing Party's representatives.",
    "nda-19": "Some obligations of Agreement may survive termination of Agreement.",
    "nda-20": "Receiving Party may retain some Confidential Information after return or destruction.",
}
ARTIFACT_FIELDS = (
    "hypothesis_id",
    "supporting_cues",
    "contradiction_cues",
    "exception_cues",
    "decision_checks",
)
COMPILED_BASELINE = "contract_compiled_transfer"
LOCAL_BASELINE = "contract_local_retrieval"
SYSTEM_PROMPT = """Compile reusable legal task knowledge for a frozen local
small model. Return JSON only with one top-level key, obligations. Each
obligation must contain hypothesis_id and four non-empty string arrays:
supporting_cues, contradiction_cues, exception_cues, and decision_checks.
Describe short lexical or semantic cues that help retrieve and judge relevant
NDA clauses. Do not invent a contract, decide a benchmark example, or include
an output label. Put exactly two strings of at most 12 words in every array."""
STUDENT_SYSTEM = """Classify the hypothesis from only the retrieved contract
clauses. Return entailment when a clause explicitly supports the complete
hypothesis. Return contradiction only when a clause explicitly conflicts with
it. Return not_mentioned when support and conflict are both absent or
insufficient. Return one label only."""
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]+")
_LABEL = re.compile(
    r"\b(entailment|contradiction|not[_\s-]?mentioned)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ObligationArtifact:
    hypothesis_id: str
    supporting_cues: tuple[str, ...]
    contradiction_cues: tuple[str, ...]
    exception_cues: tuple[str, ...]
    decision_checks: tuple[str, ...]

    def retrieval_text(self) -> str:
        return " ".join(
            (
                PUBLIC_HYPOTHESES[self.hypothesis_id],
                *self.supporting_cues,
                *self.contradiction_cues,
                *self.exception_cues,
            )
        )

    def render(self) -> str:
        return json.dumps(
            {
                field: list(getattr(self, field))
                if field != "hypothesis_id"
                else self.hypothesis_id
                for field in ARTIFACT_FIELDS
            },
            ensure_ascii=False,
            indent=2,
        )

    def render_for_student(self) -> str:
        """Render only decision instructions for the final SLM."""
        return "\n".join(
            f"- {check}" for check in self.decision_checks
        )

    def render_lineage_guidance(self) -> str:
        """Render compact label-conditioned knowledge for a distilled SLM."""
        groups = (
            ("Entailment evidence", self.supporting_cues),
            ("Contradiction evidence", self.contradiction_cues),
            ("Boundary or exception evidence", self.exception_cues),
            ("Questions to resolve", self.decision_checks),
        )
        return "\n".join(
            f"{label}: {'; '.join(values)}"
            for label, values in groups
        )

    def render_contrastive_guidance(self) -> str:
        """Map parent cues to labels with an explicit evidence boundary."""
        return "\n".join(
            (
                "Decision procedure:",
                "1. Require the obligation and its trigger to match the hypothesis.",
                "2. Entailment requires explicit support for that complete claim.",
                "3. Contradiction requires an explicit conflicting duty or permission.",
                "4. Otherwise return not_mentioned, even with related vocabulary.",
                f"Parent support signals: {'; '.join(self.supporting_cues)}",
                f"Parent conflict signals: {'; '.join(self.contradiction_cues)}",
            )
        )


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN.findall(text)]


def chunk_contract(text: str, max_chars: int = 900) -> list[str]:
    """Split local contract text into retrieval units.

    Args:
        text: Private contract text.
        max_chars: Maximum approximate chunk length.
    """
    units = [
        unit.strip()
        for unit in re.split(r"\n+|(?<=[.;])\s+(?=[A-Z0-9])", text)
        if unit.strip()
    ]
    chunks = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = f"{current} {unit}".strip()
    if current:
        chunks.append(current)
    return chunks


def retrieve_clauses(
    contract_text: str,
    query_text: str,
    top_k: int = 5,
) -> list[str]:
    """Rank private contract chunks with a local BM25-style score.

    Args:
        contract_text: Private contract text.
        query_text: Public hypothesis and optional compiled retrieval cues.
        top_k: Number of chunks to return.
    """
    chunks = chunk_contract(contract_text)
    documents = [_tokens(chunk) for chunk in chunks]
    query = Counter(_tokens(query_text))
    if not documents or not query:
        return chunks[:top_k]
    document_frequency = Counter()
    for document in documents:
        document_frequency.update(set(document))
    average_length = sum(map(len, documents)) / len(documents)

    scored = []
    for index, document in enumerate(documents):
        frequencies = Counter(document)
        score = 0.0
        for token, query_count in query.items():
            frequency = frequencies[token]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1 + (len(documents) - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * len(document) / max(average_length, 1)
            )
            score += query_count * inverse_frequency * frequency * 2.2 / denominator
        scored.append((score, -index, chunks[index]))
    return [
        chunk
        for _, _, chunk in sorted(scored, reverse=True)[:top_k]
    ]


def parse_artifact_response(
    text: str,
    expected_ids: list[str],
) -> dict[str, ObligationArtifact]:
    """Normalize provider formatting before strict artifact validation."""
    candidate = text.strip()
    fenced = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        candidate = fenced.group(1)
    elif not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    value = json.loads(candidate)
    obligations = value.get("obligations") if isinstance(value, dict) else None
    if isinstance(obligations, dict):
        normalized = []
        for hypothesis_id, item in obligations.items():
            if not isinstance(item, dict):
                raise ValueError("obligation artifact must be an object")
            prepared = dict(item)
            prepared["hypothesis_id"] = hypothesis_id
            for field in ARTIFACT_FIELDS[1:]:
                entries = prepared.get(field)
                if isinstance(entries, list) and len(entries) > 2:
                    prepared[field] = entries[:2]
            normalized.append(prepared)
        value = {"obligations": normalized}
    return validate_artifacts(value, expected_ids)


def validate_artifacts(
    value: dict[str, Any],
    expected_ids: list[str],
) -> dict[str, ObligationArtifact]:
    """Validate a batch of instance-free teacher artifacts.

    Args:
        value: Parsed teacher JSON.
        expected_ids: Public obligation identifiers requested in this batch.
    """
    if set(value) != {"obligations"} or not isinstance(
        value["obligations"], list
    ):
        raise ValueError("artifact must contain only an obligations array")
    artifacts = {}
    for item in value["obligations"]:
        if not isinstance(item, dict) or set(item) != set(ARTIFACT_FIELDS):
            raise ValueError("obligation artifact has an invalid schema")
        hypothesis_id = item["hypothesis_id"]
        if hypothesis_id not in expected_ids or hypothesis_id in artifacts:
            raise ValueError("obligation artifact has an unexpected identifier")
        fields = {}
        for field in ARTIFACT_FIELDS[1:]:
            entries = item[field]
            if (
                not isinstance(entries, list)
                or len(entries) != 2
                or not all(
                    isinstance(entry, str) and entry.strip()
                    for entry in entries
                )
            ):
                raise ValueError(f"{field} must contain exactly two strings")
            if any(len(entry.split()) > 12 for entry in entries):
                raise ValueError(f"{field} entries must contain at most 12 words")
            fields[field] = tuple(entry.strip() for entry in entries)
        artifacts[hypothesis_id] = ObligationArtifact(
            hypothesis_id=hypothesis_id,
            **fields,
        )
    if set(artifacts) != set(expected_ids):
        raise ValueError("teacher did not return every requested obligation")
    return artifacts


def _compile_artifacts(
    config: RunConfig,
    recorder: RunRecorder,
    hypothesis_ids: list[str],
) -> tuple[dict[str, ObligationArtifact], str]:
    prompt = (
        "Public obligation specifications:\n"
        + "\n".join(
            f"{identifier}: {PUBLIC_HYPOTHESES[identifier]}"
            for identifier in hypothesis_ids
        )
        + "\n\nCompile task knowledge from these definitions only. No contract "
        "text, annotation, or evaluation example is available."
    )
    teacher = TeacherClient(
        config.teacher_model,
        backend=config.teacher_backend,
        base_url=config.openrouter_base_url,
        timeout_seconds=config.teacher_timeout_seconds,
        provider=config.teacher_openrouter_provider,
    )
    recorder.set_model_identity("teacher", teacher.identity)
    response_format = (
        _artifact_response_format(hypothesis_ids)
        if config.teacher_structured_outputs
        else None
    )
    if response_format is None:
        recorder.log(
            "teacher JSON schema enforcement disabled; "
            "validating the prompted JSON response locally"
        )
    response = teacher.call(
        SYSTEM_PROMPT,
        prompt,
        max_tokens=max(600, min(config.token_budget, 8000)),
        response_format=response_format,
        seed=config.seed,
    )
    recorder.set_model_identity("teacher", teacher.identity)
    recorder.add_teacher_tokens(response.input_tokens, response.output_tokens)
    (recorder.run_dir / "artifacts" / "teacher_response.json").write_text(
        json.dumps(
            {
                "text": response.text,
                "model": response.model,
                "backend": response.backend,
                "finish_reason": response.finish_reason,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if response.finish_reason == "length":
        raise RuntimeError(
            "teacher response reached the output-token limit before completion"
        )
    return parse_artifact_response(response.text, hypothesis_ids), prompt


def _artifact_response_format(hypothesis_ids: list[str]) -> dict[str, Any]:
    """Constrain OpenRouter teachers to the obligation artifact schema."""
    cue_array = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 2,
        "maxItems": 2,
    }
    obligation = {
        "type": "object",
        "properties": {
            "hypothesis_id": {
                "type": "string",
                "enum": hypothesis_ids,
            },
            "supporting_cues": cue_array,
            "contradiction_cues": cue_array,
            "exception_cues": cue_array,
            "decision_checks": cue_array,
        },
        "required": list(ARTIFACT_FIELDS),
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "contract_obligation_artifacts",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "obligations": {
                        "type": "array",
                        "items": obligation,
                        "minItems": len(hypothesis_ids),
                        "maxItems": len(hypothesis_ids),
                    }
                },
                "required": ["obligations"],
                "additionalProperties": False,
            },
        },
    }


def _lineage_manifest(
    config: RunConfig,
    lineage: DistillationLineage | None,
) -> dict[str, Any]:
    """Describe how the task-knowledge teacher was selected."""
    return {
        "teacher_selection": config.teacher_selection,
        "teacher_backend": config.teacher_backend,
        "teacher_model": config.teacher_model,
        "teacher_structured_outputs": config.teacher_structured_outputs,
        "teacher_openrouter_provider": config.teacher_openrouter_provider,
        "student_model": config.student_model,
        "lineage": lineage.to_dict() if lineage else None,
        "transfer_target": "frozen_task_artifact",
        "student_weights_updated": False,
    }


def _validate_loaded_lineage_manifest(
    artifact_path: Path,
    config: RunConfig,
    lineage: DistillationLineage | None,
) -> None:
    """Verify provenance before reusing a lineage-matched artifact."""
    if lineage is None:
        return
    manifest_path = artifact_path.with_name(
        "obligation_artifacts.manifest.json"
    )
    if not manifest_path.exists():
        raise ValueError(
            "lineage-matched artifact reuse requires "
            "obligation_artifacts.manifest.json beside the artifact"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _lineage_manifest(config, lineage)
    for field in (
        "teacher_selection",
        "teacher_backend",
        "teacher_model",
        "teacher_openrouter_provider",
        "student_model",
    ):
        if manifest.get(field) != expected[field]:
            raise ValueError(
                f"artifact lineage manifest mismatch for {field}"
            )


def _data_root() -> Path:
    candidates = (
        DATA_DIR / "contract_nli",
        DATA_DIR / "contract-nli" / "contract-nli",
    )
    for candidate in candidates:
        if (candidate / "test.json").exists():
            return candidate
    return candidates[0]


def _load_examples(
    config: RunConfig,
    hypothesis_ids: list[str],
) -> tuple[list[ContractNliExample], str]:
    split = (
        "dev"
        if config.benchmark_split in {"validation", "dev"}
        else config.benchmark_split
    )
    if split not in {"dev", "test"}:
        raise ValueError("ContractNLI split must be validation, dev, or test")
    complete = load_split(split, _data_root())
    selected = []
    for hypothesis_id in hypothesis_ids:
        rows = filter_by_hypothesis(complete, hypothesis_id)
        by_label: dict[str, list[ContractNliExample]] = {}
        for row in rows:
            by_label.setdefault(row.label, []).append(row)
        for label, group in by_label.items():
            random.Random(
                f"{config.seed}:{hypothesis_id}:{label}"
            ).shuffle(group)
        task_rows = []
        labels = sorted(by_label)
        while len(task_rows) < config.num_examples:
            added = False
            for label in labels:
                group = by_label[label]
                if group:
                    task_rows.append(group.pop())
                    added = True
                    if len(task_rows) == config.num_examples:
                        break
            if not added:
                break
        selected.extend(task_rows)
    return selected, split


def _parse_label(text: str) -> str | None:
    matches = list(_LABEL.finditer(text))
    if not matches:
        return None
    value = matches[-1].group(1).casefold()
    return "not_mentioned" if value != "entailment" and value != "contradiction" else value


def execute(config: RunConfig, recorder: RunRecorder, run_dir: Path) -> dict:
    """Evaluate local retrieval with or without compiled task knowledge."""
    lineage = validate_lineage_config(config)
    if config.contract_artifact_delivery not in (
        CONTRACT_ARTIFACT_DELIVERY_MODES
    ):
        raise ValueError(
            "unknown contract artifact delivery mode: "
            f"{config.contract_artifact_delivery}"
        )
    if config.contract_hypothesis_ids:
        unknown = set(config.contract_hypothesis_ids) - set(PUBLIC_HYPOTHESES)
        if unknown:
            raise ValueError(
                f"unknown ContractNLI hypotheses: {sorted(unknown)}"
            )
        hypothesis_ids = list(dict.fromkeys(config.contract_hypothesis_ids))
    else:
        hypothesis_ids = list(PUBLIC_HYPOTHESES)
        random.Random(config.seed).shuffle(hypothesis_ids)
        hypothesis_ids = hypothesis_ids[
            : min(config.num_tasks, len(hypothesis_ids))
        ]
    artifacts = {}
    teacher_prompt = None
    artifact_source = None
    if config.baseline == COMPILED_BASELINE:
        if config.compiled_guidance_path:
            artifact_path = Path(config.compiled_guidance_path)
            with recorder.stage("load_public_obligation_knowledge"):
                _validate_loaded_lineage_manifest(
                    artifact_path,
                    config,
                    lineage,
                )
                stored = json.loads(
                    artifact_path.read_text(encoding="utf-8")
                )
                artifacts = validate_artifacts(
                    {"obligations": list(stored.values())},
                    hypothesis_ids,
                )
            artifact_source = str(artifact_path.resolve())
        else:
            with recorder.stage("compile_public_obligation_knowledge"):
                artifacts, teacher_prompt = _compile_artifacts(
                    config,
                    recorder,
                    hypothesis_ids,
                )
            artifact_source = "generated_for_run"
        serialized = {
            key: json.loads(artifact.render())
            for key, artifact in artifacts.items()
        }
        (run_dir / "artifacts" / "obligation_artifacts.json").write_text(
            json.dumps(serialized, indent=2),
            encoding="utf-8",
        )
        (run_dir / "artifacts" / "obligation_artifacts.manifest.json").write_text(
            json.dumps(_lineage_manifest(config, lineage), indent=2),
            encoding="utf-8",
        )

    with recorder.stage("load_private_contract_evaluation"):
        examples, evaluation_split = _load_examples(config, hypothesis_ids)
    student = StudentModel(
        config.student_model,
        revision=config.student_revision,
        max_new_tokens=config.max_answer_tokens,
        backend=config.student_backend,
        ollama_base_url=config.ollama_base_url,
        openrouter_base_url=config.openrouter_base_url,
        openrouter_timeout_seconds=config.student_openrouter_timeout_seconds,
        openrouter_model_name=config.student_openrouter_model,
        openrouter_provider=config.student_openrouter_provider,
        ollama_timeout_seconds=config.ollama_timeout_seconds,
        seed=config.student_seed,
        context_tokens=config.student_context_tokens,
        enable_thinking=config.student_thinking,
    )
    recorder.set_model_identity("student", student.identity)

    predictions = []
    with recorder.stage("local_contract_evaluation"):
        for index, example in enumerate(examples, start=1):
            artifact = artifacts.get(example.hypothesis_id)
            query = (
                artifact.retrieval_text()
                if artifact
                and config.contract_artifact_delivery
                in {
                    CONTRACT_RETRIEVAL_AND_CHECKS,
                    CONTRACT_RETRIEVAL_ONLY,
                    CONTRACT_LINEAGE_RENDERED,
                    CONTRACT_CONTRASTIVE_RENDERED,
                }
                else example.hypothesis_text
            )
            clauses = retrieve_clauses(example.contract_text, query)
            knowledge = (
                "\n\nSelected task checks:\n"
                f"{artifact.render_for_student()}"
                if artifact
                and config.contract_artifact_delivery
                == CONTRACT_RETRIEVAL_AND_CHECKS
                else ""
            )
            if (
                artifact
                and config.contract_artifact_delivery
                == CONTRACT_LINEAGE_RENDERED
            ):
                knowledge = (
                    "\n\nParent-transferred task knowledge:\n"
                    f"{artifact.render_lineage_guidance()}\n"
                    "These are interpretation aids, not contract facts. "
                    "Return a label only when the retrieved clauses justify it."
                )
            if (
                artifact
                and config.contract_artifact_delivery
                == CONTRACT_CONTRASTIVE_RENDERED
            ):
                knowledge = (
                    "\n\nParent-transferred task knowledge:\n"
                    f"{artifact.render_contrastive_guidance()}"
                )
            prompt = (
                f"Hypothesis: {example.hypothesis_text}\n\n"
                "Locally retrieved contract clauses:\n"
                + "\n\n".join(
                    f"[{chunk_index}] {chunk}"
                    for chunk_index, chunk in enumerate(clauses, start=1)
                )
                + knowledge
            )
            response = student.ask(prompt, system=STUDENT_SYSTEM)
            recorder.add_student_tokens(
                response.input_tokens,
                response.output_tokens,
            )
            predicted = _parse_label(response.text)
            predictions.append(
                {
                    "document_id": example.document_id,
                    "task_id": example.hypothesis_id,
                    "predicted": predicted,
                    "gold": example.label,
                    "correct": predicted == example.label,
                    "raw_output": response.text,
                    "retrieved_clause_hashes": [
                        hashlib.sha256(clause.encode("utf-8")).hexdigest()
                        for clause in clauses
                    ],
                }
            )
            recorder.log(
                f"ContractNLI example {index}/{len(examples)}: "
                f"{'passed' if predicted == example.label else 'failed'}"
            )

    (run_dir / "artifacts" / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in predictions),
        encoding="utf-8",
    )
    correct = sum(row["correct"] for row in predictions)
    interval = wilson_accuracy(correct, len(predictions))
    task_metrics = []
    for task_id in hypothesis_ids:
        task_rows = [
            row for row in predictions
            if row["task_id"] == task_id
        ]
        task_correct = sum(row["correct"] for row in task_rows)
        task_metrics.append(
            {
                "task_id": task_id,
                "accuracy": task_correct / len(task_rows),
                "correct": task_correct,
                "num_examples": len(task_rows),
            }
        )
    metrics = {
        "dataset": "contract_nli",
        "baseline": config.baseline,
        "accuracy": correct / len(predictions),
        "accuracy_wilson_95": {
            "low": interval.low,
            "high": interval.high,
            "descriptive_only": True,
        },
        "correct": correct,
        "num_examples": len(predictions),
        "num_task_groups": len(hypothesis_ids),
        "task_groups": hypothesis_ids,
        "task_metrics": task_metrics,
        "teacher_received_contract_text": False,
        "teacher_received_annotations": False,
        "teacher_prompt_sha256": (
            hashlib.sha256(teacher_prompt.encode("utf-8")).hexdigest()
            if teacher_prompt
            else None
        ),
        "teacher_calls": int(teacher_prompt is not None),
        "teacher_backend": config.teacher_backend,
        "teacher_model": config.teacher_model,
        "teacher_selection": config.teacher_selection,
        "lineage": lineage.to_dict() if lineage else None,
        "student_weights_updated": False,
        "student_inference_location": (
            "hosted_openrouter"
            if student.backend == OPENROUTER_BACKEND
            else "local"
        ),
        "student_inputs_left_device": (
            student.backend == OPENROUTER_BACKEND
        ),
        "on_device_privacy_claim_applicable": (
            student.backend != OPENROUTER_BACKEND
        ),
        "transfer_target": "frozen_task_artifact",
        "artifact_delivery": config.contract_artifact_delivery,
        "artifact_source": artifact_source,
        "private_retrieval": "local_bm25_style",
        "inferential_unit": "contract obligation task group (pilot)",
        "evaluation_split": evaluation_split,
        "development_only": True,
    }
    recorder.set_task_metrics(metrics)
    return metrics
