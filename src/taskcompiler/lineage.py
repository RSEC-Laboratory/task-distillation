"""Document verified lineage for distilled student checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from taskcompiler.config import (
    LINEAGE_MATCHED_TEACHER,
    MANUAL_TEACHER,
    OPENROUTER_TEACHER,
    RunConfig,
)


@dataclass(frozen=True)
class DistillationLineage:
    """Describe a student's base weights and supervision teacher."""

    local_model: str
    canonical_student: str
    base_model: str
    distillation_teacher: str
    openrouter_teacher: str
    evidence_url: str
    expected_ollama_basename: str
    alternate_openrouter_teachers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


LINEAGES = {
    "llama3.2:1b": DistillationLineage(
        local_model="llama3.2:1b",
        canonical_student="meta-llama/llama-3.2-1b-instruct",
        base_model=(
            "pruned Llama 3.1-derived checkpoint; exact pruning source "
            "not isolated in the model card"
        ),
        distillation_teacher=(
            "meta-llama/Llama-3.1-70B-Instruct "
            "(one of two documented logit teachers)"
        ),
        openrouter_teacher="meta-llama/llama-3.1-70b-instruct",
        evidence_url=(
            "https://github.com/meta-llama/llama-models/blob/main/"
            "models/llama3_2/MODEL_CARD.md"
        ),
        expected_ollama_basename="Llama-3.2",
        alternate_openrouter_teachers=(
            "meta-llama/llama-3.1-8b-instruct",
        ),
    ),
    "gemma2:9b": DistillationLineage(
        local_model="gemma2:9b",
        canonical_student="google/gemma-2-9b-it",
        base_model="google/gemma-2-9b",
        distillation_teacher="google/gemma-2-27b",
        openrouter_teacher="google/gemma-2-27b-it",
        evidence_url=(
            "https://developers.googleblog.com/en/"
            "gemma-explained-new-in-gemma-2/"
        ),
        expected_ollama_basename="Gemma2",
    ),
    "deepseek-r1:8b": DistillationLineage(
        local_model="deepseek-r1:8b",
        canonical_student="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        base_model="Qwen/Qwen3-8B",
        distillation_teacher="deepseek-ai/DeepSeek-R1-0528",
        openrouter_teacher="deepseek/deepseek-r1-0528",
        evidence_url=(
            "https://huggingface.co/deepseek-ai/"
            "DeepSeek-R1-0528-Qwen3-8B"
        ),
        expected_ollama_basename="DeepSeek-R1-0528-Qwen3",
    ),
    "deepseek-r1:32b": DistillationLineage(
        local_model="deepseek-r1:32b",
        canonical_student="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        base_model="Qwen/Qwen2.5-32B",
        distillation_teacher="deepseek-ai/DeepSeek-R1",
        openrouter_teacher="deepseek/deepseek-r1",
        evidence_url="https://github.com/deepseek-ai/DeepSeek-R1",
        expected_ollama_basename="DeepSeek-R1-Distill-Qwen",
    ),
    "llama3.2:3b": DistillationLineage(
        local_model="llama3.2:3b",
        canonical_student="meta-llama/Llama-3.2-3B-Instruct",
        base_model=(
            "pruned Llama 3.1-derived checkpoint; exact pruning source "
            "not isolated in the model card"
        ),
        distillation_teacher=(
            "meta-llama/Llama-3.1-70B-Instruct "
            "(one of two documented logit teachers)"
        ),
        openrouter_teacher="meta-llama/llama-3.1-70b-instruct",
        evidence_url=(
            "https://github.com/meta-llama/llama-models/blob/main/"
            "models/llama3_2/MODEL_CARD.md"
        ),
        expected_ollama_basename="Llama-3.2",
        alternate_openrouter_teachers=(
            "meta-llama/llama-3.1-8b-instruct",
        ),
    ),
    "qwen3:4b": DistillationLineage(
        local_model="qwen3:4b",
        canonical_student="Qwen/Qwen3-4B",
        base_model="Qwen/Qwen3-4B-Base",
        distillation_teacher=(
            "Qwen3-235B-A22B or Qwen3-32B during on-policy "
            "strong-to-weak distillation"
        ),
        openrouter_teacher="qwen/qwen3-235b-a22b",
        evidence_url="https://arxiv.org/pdf/2505.09388",
        expected_ollama_basename="Qwen3",
        alternate_openrouter_teachers=("qwen/qwen3-32b",),
    ),
}


def get_lineage(student_model: str) -> DistillationLineage:
    """Return documented lineage for a local student model."""
    normalized = student_model.casefold()
    matches = [
        lineage
        for name, lineage in LINEAGES.items()
        if normalized
        in {
            name.casefold(),
            lineage.local_model.casefold(),
            lineage.canonical_student.casefold(),
        }
    ]
    if not matches:
        supported = ", ".join(sorted(LINEAGES))
        raise ValueError(
            f"no verified distillation lineage for {student_model!r}; "
            f"supported local models: {supported}"
        )
    return matches[0]


def validate_lineage_config(config: RunConfig) -> DistillationLineage | None:
    """Require the documented supervision teacher for a matched run."""
    if config.teacher_selection == MANUAL_TEACHER:
        return None
    if config.teacher_selection != LINEAGE_MATCHED_TEACHER:
        raise ValueError(
            f"unknown teacher selection mode: {config.teacher_selection}"
        )
    lineage = get_lineage(config.student_model)
    if config.teacher_backend != OPENROUTER_TEACHER:
        raise ValueError(
            "lineage-matched runs currently require the OpenRouter teacher backend"
        )
    valid_teachers = (
        lineage.openrouter_teacher,
        *lineage.alternate_openrouter_teachers,
    )
    if config.teacher_model not in valid_teachers:
        raise ValueError(
            "lineage-matched teacher mismatch: "
            f"{config.student_model!r} requires "
            f"one of {valid_teachers!r}, not {config.teacher_model!r}"
        )
    return lineage
