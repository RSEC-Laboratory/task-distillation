import json
from types import SimpleNamespace

from taskcompiler.config import GEPA_OPTIMIZER, RULES_TRACK, RunConfig
from taskcompiler.guidance_track import gepa_baseline
from taskcompiler.metrics import RunRecorder
from taskcompiler.dsl.interpreter import run
from taskcompiler.rules_track.synthetic_policies import (
    FROZEN_BUNDLE,
    TASK_SPLITS,
    _opaque_label_codebook,
    _passes_teacher_validation,
    _routing_summary,
    execute,
    make_policy_task,
    suite_manifest,
)
from taskcompiler.translate import (
    InstructionBundle,
    InstructionEntry,
    _certified_representation_candidates,
    build_obligation_probes,
    certify_verified_decision_list,
    compile_verified_decision_list,
    compile_verified_decision_tree,
)


def test_teacher_validation_uses_frozen_empirical_thresholds():
    compilation = SimpleNamespace(
        final_build=SimpleNamespace(pass_rate=25 / 27),
        final_audit=SimpleNamespace(pass_rate=26 / 27),
    )

    assert _passes_teacher_validation(
        compilation,
        RunConfig(
            track=RULES_TRACK,
            dataset="synthetic_policies",
            teacher_build_pass_threshold=25 / 27,
            teacher_audit_pass_threshold=25 / 27,
        ),
    )
    assert not _passes_teacher_validation(
        compilation,
        RunConfig(
            track=RULES_TRACK,
            dataset="synthetic_policies",
        ),
    )


def test_routing_summary_ignores_raw_fallback_examples():
    predictions = [
        {
            "routing_certificate": None,
            "routed_instruction_count": 0,
        }
    ]

    assert _routing_summary(
        predictions,
        top_k=6,
        opaque_labels=False,
    ) == {"enabled": False}


def test_policy_splits_are_disjoint_and_deterministic():
    task = make_policy_task(40)
    repeated = make_policy_task(40)
    build = {str(case.input_record) for case in task.build_cases}
    audit = {str(case.input_record) for case in task.audit_cases}
    final = {str(case.input_record) for case in task.final_cases}

    assert task.suite_split == "validation"
    assert len(build | audit | final) == 81
    assert not build & audit
    assert not build & final
    assert not audit & final
    assert task == repeated


def test_policy_expected_outputs_come_from_oracle():
    task = make_policy_task(80)
    inputs = [case.input_record for case in task.build_cases]
    _, probes = build_obligation_probes(
        task.oracle_program,
        inputs,
        "build",
        oracle_program=task.oracle_program,
    )

    assert [probe.expected_label for probe in probes] == [
        run(task.oracle_program, record).output_label for record in inputs
    ]


def test_opaque_codebook_is_reversible_and_includes_abstention():
    task = make_policy_task(0)

    encoder, decoder = _opaque_label_codebook(task)

    assert set(encoder) == {*task.task_spec.labels, "uncertain"}
    assert set(encoder.values()) == set(decoder)
    assert decoder[encoder["uncertain"]] is None
    assert all(
        decoder[encoder[label]] == label
        for label in task.task_spec.labels
    )


def test_manifest_locks_all_tasks_in_split():
    manifest = suite_manifest("final")

    assert len(manifest["tasks"]) == len(TASK_SPLITS["final"])
    assert len(manifest["manifest_sha256"]) == 64


def test_confirmation_split_excludes_opened_validation_tasks():
    opened = {40, 67, 52, 61, 69, 55, 78, 74}
    confirmation = set(TASK_SPLITS["validation_confirmation"])

    assert len(confirmation) == 32
    assert not confirmation & opened
    assert confirmation | opened == set(TASK_SPLITS["validation"])


def test_verified_decision_list_is_complete_and_label_pure():
    task = make_policy_task(0)
    _, build = build_obligation_probes(
        task.oracle_program,
        [case.input_record for case in task.build_cases],
        "build",
        oracle_program=task.oracle_program,
    )
    decision_list = compile_verified_decision_list(task.oracle_program, build)

    assert decision_list is not None
    assert len(decision_list) < 82
    certificate = certify_verified_decision_list(
        task.oracle_program,
        build,
        decision_list,
    )
    assert certificate["exhaustive_coverage"] is True
    assert certificate["label_pure"] is True
    assert certificate["semantically_equivalent"] is True
    assert certificate["finite_domain_size"] == 81
    for case in task.build_cases + task.audit_cases + task.final_cases:
        matches = []
        for line in decision_list[1:]:
            conditions_text, label = line.removeprefix("If ").split(", return ")
            conditions = conditions_text.split(", ")
            if all(
                case.input_record["event"][field] == value
                for field, value in (
                    condition.split("=", 1) for condition in conditions
                )
            ):
                matches.append(label.removesuffix("."))
        expected = case.expected_label or "uncertain"
        assert matches
        assert set(matches) == {expected}


def test_verified_decision_tree_is_disjoint_and_equivalent():
    task = make_policy_task(1)
    _, build = build_obligation_probes(
        task.oracle_program,
        [case.input_record for case in task.build_cases],
        "build",
        oracle_program=task.oracle_program,
    )

    tree = compile_verified_decision_tree(task.oracle_program, build)
    certificate = certify_verified_decision_list(
        task.oracle_program, build, tree
    )

    assert tree[0].startswith("Verified decision tree.")
    assert certificate["finite_domain_size"] == 81
    assert certificate["exhaustive_coverage"] is True
    assert certificate["label_pure"] is True
    assert certificate["semantically_equivalent"] is True


def test_certificate_rejects_a_changed_leaf():
    task = make_policy_task(2)
    _, build = build_obligation_probes(
        task.oracle_program,
        [case.input_record for case in task.build_cases],
        "build",
        oracle_program=task.oracle_program,
    )
    tree = compile_verified_decision_tree(task.oracle_program, build)
    changed = list(tree)
    original_label = changed[1].rsplit(", return ", 1)[1].removesuffix(".")
    wrong_label = next(
        label for label in task.task_spec.labels if label != original_label
    )
    changed[1] = changed[1].rsplit(", return ", 1)[0] + f", return {wrong_label}."

    certificate = certify_verified_decision_list(
        task.oracle_program, build, changed
    )

    assert certificate["semantically_equivalent"] is False


def test_certified_search_candidates_carry_valid_certificates():
    task = make_policy_task(3)
    _, build = build_obligation_probes(
        task.oracle_program,
        [case.input_record for case in task.build_cases],
        "build",
        oracle_program=task.oracle_program,
    )
    bundle = InstructionBundle(
        [InstructionEntry(rule) for rule in task.oracle_program.rules]
    )

    candidates = _certified_representation_candidates(
        bundle, task.oracle_program, build
    )

    assert {candidate.representation_kind for _, candidate in candidates} == {
        "verified_decision_list",
        "verified_decision_tree",
        "verified_decision_tree_reversed",
    }
    assert all(
        candidate.semantic_certificate["semantically_equivalent"]
        for _, candidate in candidates
    )


def test_synthetic_gepa_uses_only_build_rows(monkeypatch, tmp_path):
    seen = []

    class Student:
        identity = {"backend": "fake", "model": "student"}

        def ask(self, _prompt, system=None):
            return SimpleNamespace(
                text="uncertain",
                input_tokens=1,
                output_tokens=1,
            )

    def optimize(
        _config,
        _recorder,
        _run_dir,
        *,
        task_id,
        rows,
        starting_prompt,
        **_kwargs,
    ):
        seen.extend(rows)
        return starting_prompt, {
            "task_model_calls": len(rows),
            "reflection_calls": 0,
            "best_build_accuracy": 0.0,
        }

    monkeypatch.setattr(
        "taskcompiler.rules_track.synthetic_policies._student",
        lambda _config: Student(),
    )
    monkeypatch.setattr(
        "taskcompiler.rules_track.synthetic_policies.TeacherClient",
        lambda _model, **_kwargs: SimpleNamespace(identity={"model": "teacher"}),
    )
    monkeypatch.setattr(gepa_baseline, "optimize_label_prompt", optimize)
    run_dir = tmp_path / "run"
    (run_dir / "artifacts").mkdir(parents=True)
    config = RunConfig(
        track=RULES_TRACK,
        dataset="synthetic_policies",
        baseline=GEPA_OPTIMIZER,
        benchmark_split="development",
        num_tasks=1,
        num_examples=2,
    )

    metrics = execute(config, RunRecorder(run_dir), run_dir)

    assert len(seen) == 27
    assert all("-build-" in row["item_id"] for row in seen)
    assert metrics["num_final_examples"] == 2
    artifact = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_dir / "artifacts").glob("policy-*.json")
    )
    assert artifact["optimized_prompt_prefix"]


def test_teacher_one_shot_records_public_only_compilation(
    monkeypatch, tmp_path
):
    from taskcompiler.dsl.parser import render_program_source
    from taskcompiler.rules_track.synthetic_policies import TEACHER_ONE_SHOT
    from taskcompiler.spec_compiler import SpecificationCompilation
    from taskcompiler.verify import run_unit_tests

    tasks_by_source = {
        task.task_spec.sources[0]: task
        for task in (make_policy_task(number) for number in TASK_SPLITS["development"])
    }

    class Student:
        identity = {"backend": "fake", "model": "student"}

        def ask(self, _prompt, system=None):
            return SimpleNamespace(
                text="uncertain",
                input_tokens=1,
                output_tokens=1,
            )

    class Teacher:
        identity = {"backend": "fake", "model": "parent"}

    def compile_public_spec(
        _teacher,
        task_spec,
        build_cases,
        audit_cases,
        **_kwargs,
    ):
        task = tasks_by_source[task_spec.sources[0]]
        build = run_unit_tests(task.oracle_program, build_cases)
        audit = run_unit_tests(task.oracle_program, audit_cases)
        return SpecificationCompilation(
            initial_source=render_program_source(task.oracle_program),
            initial_program=task.oracle_program,
            final_program=task.oracle_program,
            initial_build=build,
            final_build=build,
            final_audit=audit,
            repairs=[],
            teacher_input_tokens=10,
            teacher_output_tokens=20,
            teacher_calls=1,
        )

    monkeypatch.setattr(
        "taskcompiler.rules_track.synthetic_policies._student",
        lambda _config: Student(),
    )
    monkeypatch.setattr(
        "taskcompiler.rules_track.synthetic_policies.TeacherClient",
        lambda _model, **_kwargs: Teacher(),
    )
    monkeypatch.setattr(
        "taskcompiler.rules_track.synthetic_policies.compile_specification",
        compile_public_spec,
    )
    run_dir = tmp_path / "run"
    (run_dir / "artifacts").mkdir(parents=True)
    config = RunConfig(
        track=RULES_TRACK,
        dataset="synthetic_policies",
        baseline=TEACHER_ONE_SHOT,
        benchmark_split="development",
        num_tasks=1,
        num_examples=2,
        deployment_include_task_spec=False,
    )

    metrics = execute(config, RunRecorder(run_dir), run_dir)

    assert metrics["teacher_calls"] == 1
    assert metrics["teacher_received_final_cases"] is False
    assert metrics["teacher_received_private_deployment_data"] is False
    assert metrics["private_deployment_data_used_during_compilation"] is False
    assert metrics["teacher_compilation"]["tasks_compiled"] == 1
    artifact = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_dir / "artifacts").glob("policy-*.json")
    )
    assert artifact["teacher_compilation"]["enabled"] is True
    assert artifact["teacher_compilation"]["teacher_received_final_cases"] is False
    assert artifact["deployment_includes_full_task_spec"] is False


def test_frozen_bundle_uses_the_selected_task_without_teacher(
    monkeypatch, tmp_path
):
    class Student:
        identity = {"backend": "fake", "model": "student"}

        def ask(self, _prompt, system=None):
            return SimpleNamespace(
                text="uncertain",
                input_tokens=1,
                output_tokens=1,
            )

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "task_id": "policy-000",
                "representation_kind": "verified_decision_list",
                "instructions": [
                    "Verified decision list.",
                    "If risk=TRUE, return review.",
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "taskcompiler.rules_track.synthetic_policies._student",
        lambda _config: Student(),
    )
    run_dir = tmp_path / "run"
    (run_dir / "artifacts").mkdir(parents=True)
    config = RunConfig(
        track=RULES_TRACK,
        dataset="synthetic_policies",
        baseline=FROZEN_BUNDLE,
        benchmark_split="development",
        num_tasks=1,
        num_examples=2,
        seed=0,
        deployment_include_task_spec=False,
        compiled_guidance_path=str(bundle_path),
    )

    metrics = execute(config, RunRecorder(run_dir), run_dir)

    assert metrics["teacher_calls"] == 0
    artifact = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_dir / "artifacts").glob("policy-*.json")
    )
    assert artifact["frozen_bundle"]["task_id"] == "policy-000"
    assert artifact["rendered_bundle"] == [
        "Verified decision list.",
        "If risk=TRUE, return review.",
    ]


def test_artifact_freeze_keeps_final_cases_closed(monkeypatch, tmp_path):
    class Student:
        identity = {"backend": "fake", "model": "student"}

        def ask(self, _prompt, system=None):
            return SimpleNamespace(
                text="uncertain",
                input_tokens=1,
                output_tokens=1,
            )

    monkeypatch.setattr(
        "taskcompiler.rules_track.synthetic_policies._student",
        lambda _config: Student(),
    )
    run_dir = tmp_path / "run"
    (run_dir / "artifacts").mkdir(parents=True)
    config = RunConfig(
        track=RULES_TRACK,
        dataset="synthetic_policies",
        baseline="raw_task",
        benchmark_split="development",
        benchmark_task_ids=[3],
        num_examples=27,
        open_final_evaluation=False,
    )

    metrics = execute(config, RunRecorder(run_dir), run_dir)

    assert metrics["final_evaluation_opened"] is False
    assert metrics["accuracy"] is None
    assert metrics["num_final_examples"] == 0
    assert metrics["student_calls"]["final"] == 0
    assert metrics["task_metrics"][0]["task_id"] == "policy-003"
    predictions = (
        run_dir / "artifacts" / "predictions.jsonl"
    ).read_text(encoding="utf-8")
    assert predictions == ""


def test_failed_certificate_uses_predeclared_raw_fallback(
    monkeypatch, tmp_path
):
    from taskcompiler.dsl.parser import render_program_source
    from taskcompiler.rules_track.synthetic_policies import TEACHER_VALIDATED
    from taskcompiler.spec_compiler import SpecificationCompilation
    from taskcompiler.verify import run_unit_tests

    task = make_policy_task(0)
    source = render_program_source(task.oracle_program).replace(
        "then require label = review",
        "then require label = allow",
        1,
    )
    from taskcompiler.dsl.parser import parse_program

    flawed_program = parse_program(source, task_spec=task.task_spec)
    build = run_unit_tests(flawed_program, task.build_cases)
    audit = run_unit_tests(flawed_program, task.audit_cases)

    class Student:
        identity = {"backend": "fake", "model": "student"}

        def ask(self, _prompt, system=None):
            return SimpleNamespace(
                text="uncertain",
                input_tokens=1,
                output_tokens=1,
            )

    class Teacher:
        identity = {"backend": "fake", "model": "parent"}

    def failed_compilation(*_args, **_kwargs):
        return SpecificationCompilation(
            initial_source=source,
            initial_program=flawed_program,
            final_program=flawed_program,
            initial_build=build,
            final_build=build,
            final_audit=audit,
            repairs=[],
            teacher_input_tokens=0,
            teacher_output_tokens=0,
            teacher_calls=0,
        )

    monkeypatch.setattr(
        "taskcompiler.rules_track.synthetic_policies._student",
        lambda _config: Student(),
    )
    monkeypatch.setattr(
        "taskcompiler.rules_track.synthetic_policies.TeacherClient",
        lambda _model, **_kwargs: Teacher(),
    )
    monkeypatch.setattr(
        "taskcompiler.rules_track.synthetic_policies.compile_specification",
        failed_compilation,
    )
    run_dir = tmp_path / "run"
    (run_dir / "artifacts").mkdir(parents=True)
    config = RunConfig(
        track=RULES_TRACK,
        dataset="synthetic_policies",
        baseline=TEACHER_VALIDATED,
        benchmark_split="development",
        benchmark_task_ids=[0],
        open_final_evaluation=False,
        deployment_include_task_spec=False,
        compilation_failure_fallback="raw_task",
    )

    metrics = execute(config, RunRecorder(run_dir), run_dir)

    assert metrics["compilation_failure_rate"] == 1.0
    assert metrics["student_calls"]["compilation"] == 0
    artifact = json.loads(
        (
            run_dir / "artifacts" / "policy-000.json"
        ).read_text(encoding="utf-8")
    )
    assert artifact["compilation_failure"]["fallback"] == "raw_task"
    assert artifact["deployment_includes_full_task_spec"] is True


def test_failed_syntax_repairs_use_raw_fallback_and_keep_costs(
    monkeypatch, tmp_path
):
    from taskcompiler.dsl.parser import DslSyntaxError
    from taskcompiler.rules_track.synthetic_policies import TEACHER_VALIDATED
    from taskcompiler.verify import RepairAttempt

    class Student:
        identity = {"backend": "fake", "model": "student"}

        def ask(self, _prompt, system=None):
            return SimpleNamespace(
                text="uncertain",
                input_tokens=1,
                output_tokens=1,
            )

    class Teacher:
        identity = {"backend": "fake", "model": "parent"}

    def failed_compilation(*_args, **_kwargs):
        error = DslSyntaxError("expected rule")
        error.failed_source = "invalid program"
        error.syntax_repairs = [
            RepairAttempt(
                rule_name="<syntax-program>",
                original_source="bad",
                proposed_source="invalid program",
                accepted=False,
            )
        ]
        error.teacher_calls = 3
        error.teacher_input_tokens = 30
        error.teacher_output_tokens = 12
        raise error

    monkeypatch.setattr(
        "taskcompiler.rules_track.synthetic_policies._student",
        lambda _config: Student(),
    )
    monkeypatch.setattr(
        "taskcompiler.rules_track.synthetic_policies.TeacherClient",
        lambda _model, **_kwargs: Teacher(),
    )
    monkeypatch.setattr(
        "taskcompiler.rules_track.synthetic_policies.compile_specification",
        failed_compilation,
    )
    run_dir = tmp_path / "run"
    (run_dir / "artifacts").mkdir(parents=True)
    config = RunConfig(
        track=RULES_TRACK,
        dataset="synthetic_policies",
        baseline=TEACHER_VALIDATED,
        benchmark_split="development",
        benchmark_task_ids=[0],
        open_final_evaluation=False,
        deployment_include_task_spec=False,
        compilation_failure_fallback="raw_task",
    )

    metrics = execute(config, RunRecorder(run_dir), run_dir)

    assert metrics["compilation_failure_rate"] == 1.0
    assert metrics["teacher_calls"] == 3
    failure = metrics["compilation_failures"][0]
    assert failure["stage"] == "teacher_syntax"
    assert failure["fallback"] == "raw_task"
    artifact = json.loads(
        (run_dir / "artifacts" / "policy-000.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["deployment_includes_full_task_spec"] is True
    assert artifact["compilation_failure"]["teacher_calls"] == 3
    assert (
        run_dir / "artifacts" / "policy-000.teacher.failed.dsl"
    ).read_text(encoding="utf-8") == "invalid program"
