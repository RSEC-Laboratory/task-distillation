import json

from taskcompiler.guidance_track.bbh import (
    _render_compact_lineage_guidance,
    _render_scratchpad_lineage_guidance,
    _prompt,
    _verified_ledger_sum,
    _verified_navigation_answer,
    _verified_date_answer,
    _verified_tracking_answer,
    _uses_validation_labels,
    load_textgrad_split,
    parse_answer,
)
from taskcompiler.guidance_track.textgrad_baseline import published_batches


def test_textgrad_bbh_splits_are_locked_and_disjoint():
    train = load_textgrad_split("object_counting", "train")
    validation = load_textgrad_split("object_counting", "validation")
    test = load_textgrad_split("object_counting", "test")
    ids = [{row["item_id"] for row in split} for split in (train, validation, test)]

    assert [len(split) for split in (train, validation, test)] == [50, 100, 100]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]


def test_bbh_parser_handles_chain_of_thought():
    assert parse_answer(
        "object_counting",
        "I count three, then four. Answer: 7",
    ) == "7"
    assert parse_answer(
        "word_sorting",
        "Reasoning here.\nFinal Answer: apple pear zebra",
    ) == "apple pear zebra"
    assert parse_answer("word_sorting", "So the answer is") is None
    assert parse_answer("navigate", "Trace complete. Answer: Yes") == "yes"
    assert parse_answer(
        "tracking_shuffled_objects_three_objects",
        "Trace complete. Answer: (C)",
    ) == "c"


def test_word_sorting_raw_prompt_does_not_add_list_as_an_item():
    prompt, system = _prompt(
        "word_sorting",
        "raw_task",
        "Sort the following words alphabetically: List: pear apple",
    )

    assert "List:" not in prompt
    assert prompt.endswith("pear apple")
    assert system is None


def test_date_raw_prompt_requests_the_scored_option_letter():
    prompt, system = _prompt(
        "date_understanding",
        "raw_task",
        "What date is tomorrow?\nOptions:\n(A) 01/01/2020",
    )

    assert "letter of the correct option" in prompt
    assert system is None


def test_compact_lineage_guidance_keeps_procedure_without_rationale_request():
    guidance = {
        "reasoning_plan": ["Count relevant items."],
        "verification_checks": ["Check the sum."],
    }

    rendered = _render_compact_lineage_guidance("Count objects.", guidance)

    assert "Count relevant items." in rendered
    assert "Output only the answer value." in rendered


def test_scratchpad_lineage_guidance_has_a_verifiable_ledger():
    guidance = {
        "reasoning_plan": [
            "Identify category.",
            "Locate objects.",
            "Include quantities.",
            "Sum quantities.",
        ],
        "verification_checks": [
            "Check category.",
            "Check quantities.",
        ],
    }

    rendered = _render_scratchpad_lineage_guidance(
        "object_counting",
        "Count objects.",
        guidance,
    )

    assert "item: quantity" in rendered
    assert "Answer: N" in rendered


def test_verified_ledger_unions_sections_and_deduplicates_items():
    output = """
Write one `item: quantity` line for each relevant mention:
- apple: 2
- pear: 1
Inclusion step: include quantities
Write one `item: quantity` line for each relevant mention:
- pear: 1
- plum: 3
Sum step: add entries
Answer: 99
"""

    assert _verified_ledger_sum(output) == "6"


def test_navigation_verifier_sums_only_the_student_trace():
    output = """
- move 1: (0, 4)
- move 2: -3, 0
- move 3: 3, -4
Answer: No
"""

    assert _verified_navigation_answer(output) == "yes"
    assert _verified_navigation_answer(
        "trace 1: 0, 4\ntrace 2: 0, -4\nAnswer: No"
    ) == "yes"


def test_tracking_verifier_executes_only_the_student_trace():
    output = """
initial: Alice = red ball
initial: Bob = blue ball
initial: Claire = green ball
swap 1: Alice <-> Claire
swap 2: Bob <-> Alice
query: Claire
option A: red ball
option B: blue ball
option C: green ball
Answer: C
"""

    assert _verified_tracking_answer(output) == "a"
    assert _verified_tracking_answer(
        "\n".join(
            (
                "INITIAL | Alice | red ball",
                "INITIAL | Bob | blue ball",
                "SWAP | 1 | Alice | Bob",
                "QUERY | Alice",
                "OPTION | A | red ball",
                "OPTION | B | blue ball",
            )
        )
    ) == "b"
    assert _verified_tracking_answer(
        '{"initial":[{"person":"Alice","object":"red"},'
        '{"person":"Bob","object":"blue"}],'
        '"swaps":[{"order":1,"first":"Alice","second":"Bob"}],'
        '"query":"Bob","options":{"A":"red","B":"blue"}}'
    ) == "a"
    assert _verified_tracking_answer(
        "initial: Alice = red\nquery: Alice\noption A: red"
    ) is None


def test_date_verifier_executes_only_the_student_program():
    output = json.dumps(
        {
            "today": "2021-05-31",
            "operation": "subtract",
            "amount": 10,
            "unit": "day",
            "options": {
                "A": "04/29/2021",
                "B": "05/21/2021",
                "C": "04/06/2021",
            },
        }
    )
    assert _verified_date_answer(output) == "b"
    leap_year = json.dumps(
        {
            "today": "2016-02-29",
            "operation": "subtract",
            "amount": 1,
            "unit": "year",
            "options": {"A": "02/28/2015", "B": "03/01/2015"},
        }
    )
    assert _verified_date_answer(leap_year) == "a"
    assert _verified_date_answer('{"today": "2021-05-31"}') is None


def test_published_textgrad_batch_schedule_is_deterministic_and_train_only():
    first = published_batches(50, seed=42)
    second = published_batches(50, seed=42)

    assert first == second
    assert len(first) == 12
    assert all(len(batch) == 3 for batch in first)
    assert all(0 <= index < 50 for batch in first for index in batch)


def test_optimizer_metadata_records_validation_label_access():
    assert _uses_validation_labels("textgrad_tgd")
    assert _uses_validation_labels("gepa")
    assert not _uses_validation_labels("zero_shot_cot")
