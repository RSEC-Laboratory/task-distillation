"""Configure and launch an experiment in the background."""

import subprocess
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from taskcompiler.config import (
    ANTHROPIC_TEACHER,
    DATASETS,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OPENROUTER_BASE_URL,
    FREEFORM_GUIDANCE,
    GEPA_OPTIMIZER,
    FEW_SHOT_COT,
    GUIDANCE_TRACK,
    OLLAMA_BACKEND,
    OLLAMA_OPENROUTER_BACKEND,
    OPENROUTER_BACKEND,
    OPENROUTER_TEACHER,
    QUESTION_LEVEL_GUIDANCE,
    RAW_TASK,
    RULEARENA_ONE_SHOT,
    RULEARENA_ZERO_SHOT,
    RULES_TRACK,
    TRANSFORMERS_BACKEND,
    STRUCTURED_GUIDANCE,
    TASK_LEVEL_GUIDANCE,
    TEACHER_ANSWER,
    TEXTGRAD_PUBLISHED_PROMPT,
    TEXTGRAD_TGD,
    ZERO_SHOT_COT,
    RunConfig,
    save_config,
)

st.title("Configure and launch")

track = st.selectbox("Track", [RULES_TRACK, GUIDANCE_TRACK])
dataset = st.selectbox("Dataset", DATASETS[track])
guidance_baselines = [
    STRUCTURED_GUIDANCE,
    FREEFORM_GUIDANCE,
    RAW_TASK,
    TEACHER_ANSWER,
]
if dataset == "bbh":
    guidance_baselines[2:2] = [
        ZERO_SHOT_COT,
        FEW_SHOT_COT,
        TEXTGRAD_PUBLISHED_PROMPT,
        TEXTGRAD_TGD,
        GEPA_OPTIMIZER,
    ]
baseline = st.selectbox(
    "Baseline",
    (
        [
            "full_method",
            "test_driven",
            GEPA_OPTIMIZER,
            "one_shot_render",
            "direct_dsl",
            "raw_task",
        ]
        if track == RULES_TRACK
        else [RULEARENA_ZERO_SHOT, RULEARENA_ONE_SHOT]
        if dataset == "rulearena"
        else guidance_baselines
    ),
    help=(
        "raw_task sends the frozen student no teacher guidance. teacher_answer "
        "is an upper comparator and bypasses the student."
    ),
)

guidance_mode = TASK_LEVEL_GUIDANCE
benchmark_task = "logical_deduction_five_objects"
benchmark_split = "validation"
benchmark_complexity = 0
task_description = (
    "Answer open-ended questions accurately, explain the reasoning needed to "
    "support the conclusion, and state important uncertainty."
)
open_questions: list[str] = []
reference_answers: list[str] = []
enable_docker_sandbox = False
docker_image = "taskcompiler-humaneval:latest"
docker_timeout_seconds = 5.0
docker_memory_mb = 256
docker_cpus = 0.5
docker_pids_limit = 32

if track == GUIDANCE_TRACK:
    guidance_mode = st.segmented_control(
        "Guidance generation mode",
        [TASK_LEVEL_GUIDANCE, QUESTION_LEVEL_GUIDANCE],
        default=TASK_LEVEL_GUIDANCE,
        format_func=lambda value: {
            TASK_LEVEL_GUIDANCE: "Task-level: compile once and reuse",
            QUESTION_LEVEL_GUIDANCE: "Question-level: generate on demand",
        }[value],
        help=(
            "Task-level guidance is generated before any evaluation question is "
            "shown to the teacher. Question-level guidance is generated separately "
            "for every incoming question and is recorded as online teacher assistance."
        ),
    )
    if dataset == "bbh":
        benchmark_task = st.selectbox(
            "BBH task",
            ["object_counting", "word_sorting"],
            help=(
                "These use TextGrad's published positional 50/100/100 split "
                "from the official BBH files."
            ),
        )
    elif dataset == "rulearena":
        benchmark_task = st.selectbox(
            "RuleArena domain",
            ["airline", "nba", "tax"],
            help="Each domain uses the official public rules, instances, and scorer.",
        )
        benchmark_complexity = st.selectbox(
            "RuleArena complexity",
            [0, 1, 2],
            format_func=lambda value: f"Level {value}",
            help="Higher levels require longer chains of rule application.",
        )

    if baseline in {
        RAW_TASK,
        ZERO_SHOT_COT,
        FEW_SHOT_COT,
        TEXTGRAD_PUBLISHED_PROMPT,
        TEXTGRAD_TGD,
        GEPA_OPTIMIZER,
        TEACHER_ANSWER,
        RULEARENA_ZERO_SHOT,
        RULEARENA_ONE_SHOT,
    }:
        st.caption(
            "The selected baseline does not generate a guidance artifact; the mode "
            "is saved for grouping but does not change its answering path."
        )
    elif guidance_mode == TASK_LEVEL_GUIDANCE:
        st.info("One guidance artifact will be generated and reused for every question.")
    else:
        st.warning(
            "The teacher will see each question and generate an answer-free scaffold. "
            "This is an online condition, not offline task distillation."
        )

    if dataset == "open_ended":
        task_description = st.text_area(
            "Public task specification",
            value=task_description,
            height=110,
            help=(
                "This specification is shown to the teacher and student. Reference "
                "answers are never shown to the teacher."
            ),
        )
        st.subheader("Open-ended questions")
        st.caption(
            "Add one row per question. References are optional and are used only for "
            "descriptive exact-match and token-F1 scoring after answers are generated."
        )
        edited_rows = st.data_editor(
            [
                {
                    "question": "Why can correlation alone not establish causation?",
                    "reference": "",
                }
            ],
            num_rows="dynamic",
            width="stretch",
            key="open_ended_rows",
            column_config={
                "question": st.column_config.TextColumn("Question", required=True),
                "reference": st.column_config.TextColumn("Optional reference answer"),
            },
        )
        records = (
            edited_rows.to_dict(orient="records")
            if hasattr(edited_rows, "to_dict")
            else list(edited_rows)
        )
        for row in records:
            question = str(row.get("question") or "").strip()
            if question:
                open_questions.append(question)
                reference_answers.append(str(row.get("reference") or "").strip())
    elif dataset == "humaneval":
        st.warning(
            "HumanEval executes model-generated Python. It is available only "
            "through the hardened Docker runner; there is no local fallback."
        )
        enable_docker_sandbox = st.checkbox(
            "I explicitly enable isolated Docker execution",
            value=False,
            help=(
                "The runner uses no network or host mounts, a read-only root "
                "filesystem, a non-root user, dropped capabilities, and explicit "
                "CPU, memory, PID, and time limits."
            ),
        )

run_label = st.text_input("Run label", value="run")
if track == RULES_TRACK and dataset == "synthetic_policies":
    benchmark_split = st.selectbox(
        "Locked suite split",
        [
            "development",
            "validation_confirmation",
            "validation",
            "final",
        ],
        format_func=lambda value: {
            "development": "Development (40 tasks)",
            "validation_confirmation": "Confirmation (32 unopened tasks)",
            "validation": "Historical validation (contains 8 opened tasks)",
            "final": "Final (40 unopened tasks)",
        }[value],
        help=(
            "Use development while changing code, confirmation after freezing, "
            "and final only after every setting is fixed."
        ),
    )
    if benchmark_split == "validation":
        st.warning(
            "Eight tasks in this split were opened before the latest code fix. "
            "Do not use it as confirmation evidence for the revised method."
        )
    if benchmark_split == "final":
        st.warning(
            "Opening the final split makes subsequent changes post-hoc. Freeze "
            "the code and analysis settings before launching."
        )
max_examples = (
    164
    if track == GUIDANCE_TRACK and dataset == "humaneval"
    else 100
    if track == GUIDANCE_TRACK
    else 27
    if dataset == "synthetic_policies"
    else 100
)
num_examples = st.slider(
    (
        "Maximum questions to answer"
        if track == GUIDANCE_TRACK
        else "Held-out final examples per task group"
    ),
    min_value=1 if track == GUIDANCE_TRACK else 5,
    max_value=max_examples,
    value=10 if track == GUIDANCE_TRACK else 20,
)
num_tasks = 1
if track == RULES_TRACK:
    max_tasks = 40 if dataset == "synthetic_policies" else 17
    num_tasks = st.slider(
        "Independent task groups (rules_track)",
        min_value=1,
        max_value=max_tasks,
        value=3,
    )

with st.expander("Model and pipeline settings"):
    teacher_backend = st.selectbox(
        "Teacher backend",
        [ANTHROPIC_TEACHER, OPENROUTER_TEACHER],
        format_func=lambda value: {
            ANTHROPIC_TEACHER: "Anthropic",
            OPENROUTER_TEACHER: "OpenRouter",
        }[value],
        help="Anthropic models use the Anthropic API. Other teachers use OpenRouter.",
    )
    teacher_model = st.text_input(
        "Teacher model",
        value=(
            "claude-sonnet-4-5-20250929"
            if teacher_backend == ANTHROPIC_TEACHER
            else "meta-llama/llama-3.1-70b-instruct"
        ),
    )
    student_backend = st.selectbox(
        "Student backend",
        [
            OPENROUTER_BACKEND,
            OLLAMA_OPENROUTER_BACKEND,
            OLLAMA_BACKEND,
            TRANSFORMERS_BACKEND,
        ],
        format_func=lambda value: {
            OLLAMA_OPENROUTER_BACKEND: "Ollama, then OpenRouter",
            OPENROUTER_BACKEND: "OpenRouter",
            TRANSFORMERS_BACKEND: "Local Transformers",
            OLLAMA_BACKEND: "Ollama server",
        }[value],
        key="student_backend",
    )
    openrouter_base_url = st.text_input(
        "OpenRouter API URL",
        value=DEFAULT_OPENROUTER_BASE_URL,
        disabled=(
            teacher_backend != OPENROUTER_TEACHER
            and student_backend
            not in {
                OPENROUTER_BACKEND,
                OLLAMA_OPENROUTER_BACKEND,
            }
        ),
        help="Used by OpenRouter teachers and students.",
    )
    teacher_openrouter_provider = st.text_input(
        "Teacher OpenRouter provider",
        value="",
        disabled=teacher_backend != OPENROUTER_TEACHER,
        help=(
            "Optional provider name for the teacher. Pin it in frozen studies."
        ),
    )
    student_openrouter_provider = st.text_input(
        "Student OpenRouter provider",
        value="",
        disabled=student_backend
        not in {
            OPENROUTER_BACKEND,
            OLLAMA_OPENROUTER_BACKEND,
        },
        help="Pin this provider so paired student conditions use the same stack.",
    )
    if student_backend == OLLAMA_OPENROUTER_BACKEND:
        student_model = st.text_input(
            "Ollama student model",
            value="llama3.2:3b",
        )
        student_openrouter_model = st.text_input(
            "OpenRouter fallback model",
            value="meta-llama/llama-3.2-3b-instruct",
            help="Use the hosted equivalent of the Ollama checkpoint.",
        )
        ollama_base_url = st.text_input(
            "Ollama server URL",
            value=DEFAULT_OLLAMA_BASE_URL,
        )
        ollama_timeout_seconds = st.number_input(
            "Ollama request timeout (seconds)",
            min_value=1.0,
            max_value=1800.0,
            value=120.0,
            step=10.0,
        )
        student_openrouter_timeout_seconds = st.number_input(
            "OpenRouter fallback timeout (seconds)",
            min_value=1.0,
            max_value=1800.0,
            value=180.0,
            step=10.0,
        )
        st.warning(
            "The runner selects one backend before evaluation. If Ollama or "
            "the requested local model is unavailable, the complete run uses "
            "OpenRouter and no longer supports the on-device privacy claim."
        )
    elif student_backend == OLLAMA_BACKEND:
        student_openrouter_model = None
        student_openrouter_timeout_seconds = 180.0
        student_model = st.text_input("Ollama model", value="qwen2.5:1.5b")
        ollama_base_url = st.text_input(
            "Ollama server URL",
            value=DEFAULT_OLLAMA_BASE_URL,
            help="Remote LAN server used for experiment inference.",
        )
        ollama_timeout_seconds = st.number_input(
            "Ollama request timeout (seconds)",
            min_value=1.0,
            max_value=1800.0,
            value=120.0,
            step=10.0,
        )
    elif student_backend == TRANSFORMERS_BACKEND:
        student_openrouter_model = None
        student_openrouter_timeout_seconds = 180.0
        student_model = st.text_input(
            "Student model", value="Qwen/Qwen2.5-1.5B-Instruct"
        )
        ollama_base_url = DEFAULT_OLLAMA_BASE_URL
        ollama_timeout_seconds = 120.0
    else:
        student_openrouter_model = None
        student_model = st.text_input(
            "OpenRouter student model",
            value="meta-llama/llama-3.2-3b-instruct",
        )
        ollama_base_url = DEFAULT_OLLAMA_BASE_URL
        ollama_timeout_seconds = 120.0
        student_openrouter_timeout_seconds = st.number_input(
            "OpenRouter request timeout (seconds)",
            min_value=1.0,
            max_value=1800.0,
            value=180.0,
            step=10.0,
        )
        st.warning(
            "OpenRouter receives benchmark prompts and outputs. Use this setting "
            "for hosted evaluation, not as evidence that user data stays on-device."
        )
    retriever_model = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model = "cross-encoder/nli-deberta-v3-small"
    top_k_rules = 6
    max_repair_rounds = 3
    max_protected_regressions = 3
    if track == RULES_TRACK:
        retriever_model = st.text_input("Retriever model", value=retriever_model)
        nli_model = st.text_input("NLI model", value=nli_model)
        top_k_rules = st.slider(
            "Rules per query (rules_track)", min_value=1, max_value=20, value=6
        )
        max_repair_rounds = st.slider(
            "Max repair rounds (rules_track)", min_value=0, max_value=10, value=3
        )
        max_protected_regressions = st.slider(
            "Protected probe regression cap",
            min_value=0,
            max_value=10,
            value=3,
            help="Maximum previously correct build probes an accepted edit may lose.",
        )
    token_budget = st.slider(
        (
            "Maximum teacher output tokens per guidance artifact"
            if track == GUIDANCE_TRACK
            else "Instruction token budget"
        ),
        min_value=100,
        max_value=3000,
        value=800,
    )
    max_answer_tokens = st.slider(
        "Maximum answer tokens",
        min_value=32,
        max_value=2000,
        value=400,
        disabled=track == RULES_TRACK,
    )
    max_guidance_retries = st.slider(
        "Guidance validation retries",
        min_value=0,
        max_value=3,
        value=1,
        disabled=track == RULES_TRACK,
    )
    if track == GUIDANCE_TRACK and dataset == "humaneval":
        st.markdown("**Docker sandbox settings**")
        docker_image = st.text_input("Docker image", value=docker_image)
        docker_timeout_seconds = st.number_input(
            "Candidate timeout in seconds",
            min_value=0.5,
            max_value=60.0,
            value=5.0,
            step=0.5,
        )
        docker_memory_mb = st.number_input(
            "Container memory limit (MB)",
            min_value=64,
            max_value=2048,
            value=256,
            step=64,
        )
        docker_cpus = st.number_input(
            "Container CPU limit",
            min_value=0.1,
            max_value=4.0,
            value=0.5,
            step=0.1,
        )
        docker_pids_limit = st.number_input(
            "Container PID limit",
            min_value=1,
            max_value=256,
            value=32,
            step=1,
        )
    seed = st.number_input("Random seed", value=0, step=1)
    student_seed = st.number_input(
        "Student decoding seed",
        value=0,
        step=1,
        help="Kept separate from dataset and task selection.",
    )
    student_context_tokens = st.number_input(
        "Student context window",
        min_value=2048,
        max_value=131072,
        value=32768,
        step=2048,
        help="Ollama receives this as num_ctx. RuleArena NBA needs a large window.",
    )
    student_thinking = st.toggle(
        "Enable model thinking",
        value=False,
        help=(
            "Leave off for matched direct-answer baselines. When enabled, "
            "reasoning models may spend output tokens in a separate thinking channel."
        ),
    )
    optimizer_metric_budget = st.number_input(
        "Optimizer metric-call budget",
        min_value=10,
        max_value=2000,
        value=150,
        step=10,
        disabled=baseline != GEPA_OPTIMIZER,
        help="GEPA stops after this many scored candidate-example calls.",
    )

if st.button("Launch run"):
    normalized_teacher = teacher_model.strip().lower()
    if (
        teacher_backend == ANTHROPIC_TEACHER
        and not normalized_teacher.startswith("claude")
    ):
        st.error("The Anthropic backend requires a Claude model identifier.")
        st.stop()
    if (
        teacher_backend == OPENROUTER_TEACHER
        and (
            normalized_teacher.startswith("claude")
            or normalized_teacher.startswith("anthropic/")
        )
    ):
        st.error("Select the Anthropic backend for Claude models.")
        st.stop()
    if (
        student_backend == OPENROUTER_BACKEND
        and "/" not in student_model
    ):
        st.error(
            "Enter an OpenRouter student identifier such as "
            "meta-llama/llama-3.2-3b-instruct."
        )
        st.stop()
    if (
        student_backend == OLLAMA_OPENROUTER_BACKEND
        and (
            not student_openrouter_model
            or "/" not in student_openrouter_model
        )
    ):
        st.error("Enter a provider/model identifier for the OpenRouter fallback.")
        st.stop()
    if track == GUIDANCE_TRACK and dataset == "open_ended" and not open_questions:
        st.error("Add at least one non-empty open-ended question before launching.")
        st.stop()
    if (
        track == GUIDANCE_TRACK
        and dataset == "humaneval"
        and not enable_docker_sandbox
    ):
        st.error("Explicitly enable the Docker sandbox before launching HumanEval.")
        st.stop()

    config = RunConfig(
        track=track,
        dataset=dataset,
        baseline=baseline,
        run_label=run_label,
        num_examples=num_examples,
        num_tasks=num_tasks,
        teacher_model=teacher_model,
        teacher_backend=teacher_backend,
        openrouter_base_url=openrouter_base_url,
        teacher_openrouter_provider=teacher_openrouter_provider or None,
        student_model=student_model,
        student_openrouter_model=student_openrouter_model,
        student_backend=student_backend,
        ollama_base_url=ollama_base_url,
        ollama_timeout_seconds=float(ollama_timeout_seconds),
        student_openrouter_timeout_seconds=float(
            student_openrouter_timeout_seconds
        ),
        student_openrouter_provider=student_openrouter_provider or None,
        retriever_model=retriever_model,
        nli_model=nli_model,
        top_k_rules=top_k_rules,
        token_budget=token_budget,
        max_repair_rounds=max_repair_rounds,
        max_protected_regressions=max_protected_regressions,
        guidance_mode=guidance_mode,
        task_description=task_description,
        open_questions=open_questions,
        reference_answers=reference_answers,
        max_answer_tokens=max_answer_tokens,
        max_guidance_retries=max_guidance_retries,
        enable_docker_sandbox=enable_docker_sandbox,
        docker_image=docker_image,
        docker_timeout_seconds=float(docker_timeout_seconds),
        docker_memory_mb=int(docker_memory_mb),
        docker_cpus=float(docker_cpus),
        docker_pids_limit=int(docker_pids_limit),
        seed=int(seed),
        student_seed=int(student_seed),
        student_context_tokens=int(student_context_tokens),
        student_thinking=student_thinking,
        optimizer_metric_budget=int(optimizer_metric_budget),
        benchmark_task=benchmark_task,
        benchmark_split=benchmark_split,
        benchmark_complexity=int(benchmark_complexity),
    )

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "runs" / "_pending_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    save_config(config, config_path)

    process = subprocess.Popen(
        [sys.executable, "-m", "taskcompiler.run", "--config", str(config_path)],
        cwd=project_root,
    )
    st.session_state["launched_pid"] = process.pid
    st.success(f"Run launched (process id {process.pid}). Check the Monitor page for progress.")
