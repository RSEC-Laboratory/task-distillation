"""Experiment configuration: what to run, on what data, with which models."""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_DIR = PROJECT_ROOT / "data"

RULES_TRACK = "rules_track"
GUIDANCE_TRACK = "guidance_track"

DATASETS = {
    RULES_TRACK: ["synthetic_policies", "contract_nli"],
    GUIDANCE_TRACK: [
        "open_ended",
        "bbh",
        "rulearena",
        "superni",
        "humaneval",
        "slr_bench",
    ],
}

TASK_LEVEL_GUIDANCE = "task_level"
QUESTION_LEVEL_GUIDANCE = "question_level"
RAW_TASK = "raw_task"
LEGACY_SPECIFICATION_GUIDANCE = "specification_only_guidance"
FREEFORM_GUIDANCE = "freeform_guidance"
STRUCTURED_GUIDANCE = "structured_guidance"
TEACHER_ANSWER = "teacher_answer"
ZERO_SHOT_COT = "zero_shot_cot"
FEW_SHOT_COT = "few_shot_cot"
TEXTGRAD_PUBLISHED_PROMPT = "textgrad_published_prompt"
TEXTGRAD_TGD = "textgrad_tgd"
GEPA_OPTIMIZER = "gepa"
APE_OPTIMIZER = "ape"
RULEARENA_ZERO_SHOT = "rulearena_zero_shot"
RULEARENA_ONE_SHOT = "rulearena_one_shot"
RULEARENA_COMPILER_MATRIX = "rulearena_compiler_matrix"
SUPERNI_COMPILER_MATRIX = "superni_compiler_matrix"
SLR_COMPILED_GUIDANCE = "slr_compiled_guidance"
SLR_LOCAL_EVIDENCE = "slr_local_evidence"
LINEAGE_COMPACT_GUIDANCE = "lineage_compact_guidance"
LINEAGE_SCRATCHPAD_GUIDANCE = "lineage_scratchpad_guidance"
LINEAGE_VERIFIED_SCRATCHPAD = "lineage_verified_scratchpad"
NONPARENT_VERIFIED_SCRATCHPAD = "nonparent_verified_scratchpad"
GENERIC_VERIFIED_SCRATCHPAD = "generic_verified_scratchpad"

GUIDANCE_BASELINES = (
    STRUCTURED_GUIDANCE,
    FREEFORM_GUIDANCE,
    LEGACY_SPECIFICATION_GUIDANCE,
    RAW_TASK,
    TEACHER_ANSWER,
    ZERO_SHOT_COT,
    FEW_SHOT_COT,
    TEXTGRAD_PUBLISHED_PROMPT,
    TEXTGRAD_TGD,
    GEPA_OPTIMIZER,
    APE_OPTIMIZER,
    RULEARENA_ZERO_SHOT,
    RULEARENA_ONE_SHOT,
    RULEARENA_COMPILER_MATRIX,
    SUPERNI_COMPILER_MATRIX,
    SLR_COMPILED_GUIDANCE,
    SLR_LOCAL_EVIDENCE,
    LINEAGE_COMPACT_GUIDANCE,
    LINEAGE_SCRATCHPAD_GUIDANCE,
    LINEAGE_VERIFIED_SCRATCHPAD,
    NONPARENT_VERIFIED_SCRATCHPAD,
    GENERIC_VERIFIED_SCRATCHPAD,
)
GUIDANCE_MODES = (TASK_LEVEL_GUIDANCE, QUESTION_LEVEL_GUIDANCE)
TRANSFORMERS_BACKEND = "transformers"
OLLAMA_BACKEND = "ollama"
OPENROUTER_BACKEND = "openrouter"
OLLAMA_OPENROUTER_BACKEND = "ollama_then_openrouter"
STUDENT_BACKENDS = (
    TRANSFORMERS_BACKEND,
    OLLAMA_BACKEND,
    OPENROUTER_BACKEND,
    OLLAMA_OPENROUTER_BACKEND,
)
DEFAULT_OLLAMA_BASE_URL = "http://192.168.2.115:11434"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ANTHROPIC_TEACHER = "anthropic"
OPENROUTER_TEACHER = OPENROUTER_BACKEND
TEACHER_BACKENDS = (ANTHROPIC_TEACHER, OPENROUTER_TEACHER)
MANUAL_TEACHER = "manual"
LINEAGE_MATCHED_TEACHER = "lineage_matched"
TEACHER_SELECTION_MODES = (MANUAL_TEACHER, LINEAGE_MATCHED_TEACHER)
CONTRACT_RETRIEVAL_AND_CHECKS = "retrieval_and_checks"
CONTRACT_RETRIEVAL_ONLY = "retrieval_only"
CONTRACT_LINEAGE_RENDERED = "lineage_rendered"
CONTRACT_CONTRASTIVE_RENDERED = "contrastive_rendered"
CONTRACT_ARTIFACT_DELIVERY_MODES = (
    CONTRACT_RETRIEVAL_AND_CHECKS,
    CONTRACT_RETRIEVAL_ONLY,
    CONTRACT_LINEAGE_RENDERED,
    CONTRACT_CONTRASTIVE_RENDERED,
)


@dataclass
class RunConfig:
    track: str
    dataset: str
    baseline: str = "full_method"
    run_label: str = "run"
    num_examples: int = 20
    num_tasks: int = 3
    teacher_model: str = "claude-sonnet-4-5-20250929"
    teacher_backend: str = ANTHROPIC_TEACHER
    teacher_selection: str = MANUAL_TEACHER
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    teacher_openrouter_provider: str | None = None
    teacher_timeout_seconds: float = 180.0
    teacher_structured_outputs: bool = True
    student_model: str = "meta-llama/llama-3.2-3b-instruct"
    student_openrouter_model: str | None = None
    student_revision: str | None = None
    student_backend: str = OPENROUTER_BACKEND
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_timeout_seconds: float = 120.0
    student_openrouter_timeout_seconds: float = 180.0
    student_openrouter_provider: str | None = None
    student_context_tokens: int = 32768
    student_thinking: bool = False
    retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retriever_revision: str | None = None
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_revision: str | None = None
    top_k_rules: int = 6
    deployment_include_task_spec: bool = True
    routing_opaque_labels: bool = True
    token_budget: int = 800
    max_repair_rounds: int = 3
    max_teacher_repair_rounds: int | None = None
    max_teacher_syntax_repair_rounds: int = 0
    max_counterexample_patches: int = 0
    teacher_refine_with_audit: bool = False
    max_protected_regressions: int = 3
    nli_entailment_threshold: float = 0.6
    guidance_mode: str = TASK_LEVEL_GUIDANCE
    task_description: str = (
        "Answer open-ended questions accurately, explain the reasoning needed "
        "to support the conclusion, and state important uncertainty."
    )
    open_questions: list[str] = field(default_factory=list)
    reference_answers: list[str] = field(default_factory=list)
    max_answer_tokens: int = 400
    max_guidance_retries: int = 1
    compiled_guidance_path: str | None = None
    teacher_program_path: str | None = None
    enable_docker_sandbox: bool = False
    docker_image: str = "taskcompiler-humaneval:latest"
    docker_timeout_seconds: float = 5.0
    docker_memory_mb: int = 256
    docker_cpus: float = 0.5
    docker_pids_limit: int = 32
    seed: int = 0
    student_seed: int = 0
    benchmark_task: str = "logical_deduction_five_objects"
    contract_hypothesis_ids: list[str] = field(default_factory=list)
    contract_artifact_delivery: str = CONTRACT_RETRIEVAL_AND_CHECKS
    benchmark_split: str = "validation"
    benchmark_task_ids: list[int] = field(default_factory=list)
    open_final_evaluation: bool = True
    compilation_failure_fallback: str = "abort"
    teacher_build_pass_threshold: float = 1.0
    teacher_audit_pass_threshold: float = 1.0
    benchmark_complexity: int = 0
    decoding_trials: int = 1
    optimizer_metric_budget: int = 150

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "RunConfig":
        known = {f: data[f] for f in RunConfig.__dataclass_fields__ if f in data}
        return RunConfig(**known)


def new_run_dir(config: RunConfig) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = f"{timestamp}_{config.dataset}_{config.run_label}"
    run_dir = RUNS_DIR / name
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config(config: RunConfig, path: Path) -> None:
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))


def load_config(path: Path) -> RunConfig:
    data = yaml.safe_load(path.read_text())
    return RunConfig.from_dict(data)


def list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True)
