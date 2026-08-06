import pytest

from taskcompiler.dsl.parser import DslSyntaxError, parse_program
from taskcompiler.sandbox import run_sandboxed

PROGRAM_SOURCE = """
rule Simple:
    when event.x == yes
    then require label = matched
"""


def test_sandbox_runs_normal_program():
    program = parse_program(PROGRAM_SOURCE)
    result = run_sandboxed(program, {"event": {"x": "yes"}})
    assert result.error is None
    assert result.timed_out is False
    assert result.trace.output_label == "matched"


def test_static_checker_rejects_undeclared_predicate():
    with pytest.raises(DslSyntaxError, match="undeclared predicate"):
        parse_program(
        """
        rule UsesMissingPredicate:
            when Missing(event)
            then require label = x
        """
        )
