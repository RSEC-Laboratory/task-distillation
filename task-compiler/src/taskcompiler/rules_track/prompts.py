"""Define teacher prompts for the rules track."""

DSL_GRAMMAR_REFERENCE = '''
Grammar:
    predicate NAME(event):
        <bool_expr>

    rule NAME:
        when <bool_expr>
        then <effect>[, <effect>]*

    effect := require label = NAME
            | forbid label = NAME
            | prefer label = NAME priority INT
            | abstain

    bool_expr uses: and, or, not, parentheses,
    field.name == value, field.name != value,
    field.name in {v1, v2}, PredicateName(event)

Example:
    predicate StateChanging(event):
        event.operation in {write_coil, write_register}

    rule UnauthorizedWrite:
        when StateChanging(event) and event.authorized == false
        then require label = manipulation
'''

DSL_GENERATION_SYSTEM_PROMPT = (
    "You write small rule-based classifiers in a restricted specification "
    "language. You never write general-purpose code, only predicates and "
    "rules in the exact grammar shown to you. Every rule must have a "
    "condition and at least one effect. Reply with only the program text, "
    "no explanation, no markdown code fences."
) + DSL_GRAMMAR_REFERENCE

DSL_REPAIR_SYSTEM_PROMPT = (
    "You repair one rule in a small specification language, shown below. "
    "You will be given a rule that produced a wrong answer on a specific "
    "case, and the correct answer. Rewrite only that rule so it produces "
    "the correct answer on this case without breaking its behavior on "
    "other cases. Reply with only the corrected rule text, same grammar, "
    "no explanation."
) + DSL_GRAMMAR_REFERENCE


def build_dsl_generation_prompt(
    task_description: str,
    labels: list[str],
    requirements: list[str],
    forbidden_shortcuts: list[str] | None = None,
    sources: list[str] | None = None,
    task_spec=None,
) -> str:
    labels_text = ", ".join(labels)
    requirements_text = "\n".join(f"- {r}" for r in requirements)
    forbidden_text = "\n".join(
        f"- {item}" for item in (forbidden_shortcuts or [])
    )
    sources_text = "\n".join(f"- {item}" for item in (sources or []))
    if task_spec is None:
        schema_text = (
            "Record Event:\n- entails: Boolean3 "
            "(TRUE, FALSE, or UNKNOWN)"
        )
    else:
        enum_values = {enum.name: enum.values for enum in task_spec.enums}
        schema_lines = []
        for record in task_spec.records:
            schema_lines.append(f"Record {record.name}:")
            for field in record.fields:
                values = enum_values.get(field.type_name)
                suffix = f" ({', '.join(values)})" if values else ""
                if field.type_name == "Boolean3":
                    suffix = " (TRUE, FALSE, UNKNOWN)"
                schema_lines.append(
                    f"- {field.name}: {field.type_name}{suffix}"
                )
        schema_text = "\n".join(schema_lines)
    return (
        f"Task: {task_description}\n"
        f"Output labels: {labels_text}\n"
        f"Requirements:\n{requirements_text}\n\n"
        f"Forbidden shortcuts:\n{forbidden_text or '- none'}\n\n"
        f"Authoritative public source identifiers:\n{sources_text or '- none'}\n\n"
        f"Declared input schema:\n{schema_text}\n"
        "Use declared values literally. Do not invent fields or labels.\n"
        "Write a small set of rules that decide between the given labels."
    )
