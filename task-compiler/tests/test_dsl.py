from taskcompiler.dsl.interpreter import find_responsible_rule, run
import pytest

from taskcompiler.dsl.parser import DslSyntaxError, parse_program
from taskcompiler.schema import FieldDecl, RecordType, TaskSpec

PROGRAM_SOURCE = """
predicate StateChanging(event):
    event.operation in {write_coil, write_register}

rule UnauthorizedStateChange:
    when StateChanging(event) and event.authorized == false
    then require label = manipulation

rule AuthorizedRead:
    when event.operation == read and event.authorized == true
    then require label = not_manipulation

rule MissingContext:
    when event.operation == write_coil and event.authorized == unknown
    then abstain
"""


def test_parses_predicates_rules_and_effects():
    program = parse_program(PROGRAM_SOURCE)
    assert [r.name for r in program.rules] == [
        "UnauthorizedStateChange",
        "AuthorizedRead",
        "MissingContext",
    ]
    assert program.predicates[0].name == "StateChanging"


def test_unauthorized_write_is_manipulation():
    program = parse_program(PROGRAM_SOURCE)
    trace = run(program, {"event": {"operation": "write_coil", "authorized": "false"}})
    assert trace.output_label == "manipulation"
    assert trace.fired_rules == ["UnauthorizedStateChange"]


def test_authorized_read_is_not_manipulation():
    program = parse_program(PROGRAM_SOURCE)
    trace = run(program, {"event": {"operation": "read", "authorized": "true"}})
    assert trace.output_label == "not_manipulation"


def test_missing_field_is_unknown_not_false():
    program = parse_program(PROGRAM_SOURCE)
    trace = run(program, {"event": {"operation": "write_coil"}})
    assert trace.fired_rules == []
    assert trace.output_label is None


def test_conflicting_requires_produce_conflict():
    source = """
    rule A:
        when event.x == yes
        then require label = one

    rule B:
        when event.x == yes
        then require label = two
    """
    program = parse_program(source)
    trace = run(program, {"event": {"x": "yes"}})
    assert trace.conflict is True
    assert trace.output_label is None


def test_prefer_is_used_only_when_nothing_required():
    source = """
    rule A:
        when event.x == yes
        then prefer label = soft priority 1
    """
    program = parse_program(source)
    trace = run(program, {"event": {"x": "yes"}})
    assert trace.output_label == "soft"


def test_syntax_error_on_malformed_rule():
    try:
        parse_program("rule Broken: when then require label = x")
    except DslSyntaxError:
        return
    raise AssertionError("expected a DslSyntaxError")


def test_find_responsible_rule_locates_overbroad_rule():
    source = """
    rule EveryWriteIsMalicious:
        when event.operation == write_coil
        then require label = manipulation
    """
    program = parse_program(source)
    approved_maintenance_write = {"event": {"operation": "write_coil"}}
    responsible = find_responsible_rule(program, approved_maintenance_write, expected_label=None)
    assert responsible == "EveryWriteIsMalicious"


def test_recursive_predicate_cycle_is_rejected():
    with pytest.raises(DslSyntaxError, match="recursive predicate cycle"):
        parse_program(
            """
            predicate A(event): B(event)
            predicate B(event): A(event)
            rule R: when A(event) then require label = x
            """
        )


def test_schema_rejects_undeclared_field_and_label():
    spec = TaskSpec(
        task_name="typed",
        description="typed",
        labels=["allowed"],
        records=[RecordType("Event", [FieldDecl("status", "Boolean3")])],
    )
    with pytest.raises(DslSyntaxError, match="undeclared field"):
        parse_program(
            "rule R: when event.missing == TRUE then require label = allowed",
            task_spec=spec,
        )
    with pytest.raises(DslSyntaxError, match="undeclared output label"):
        parse_program(
            "rule R: when event.status == TRUE then require label = wrong",
            task_spec=spec,
        )


def test_invariants_are_executed_and_reported():
    program = parse_program(
        """
        rule R: when event.x == yes then require label = matched
        invariant MustBeSafe: event.safe == yes
        """
    )
    trace = run(program, {"event": {"x": "yes", "safe": "no"}})
    assert trace.invariant_violations == ["MustBeSafe"]
    assert trace.rule_conditions["R"].value == "TRUE"
