"""Public task descriptions and bounded synthetic teaching packs.

The teacher request builder has no benchmark-example or student-feedback input.
Structural checks and executable constraint checks are not semantic proof.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
import re

from taskcompiler.corrections.schema import identifiers, require, text
from taskcompiler.procedural.experiment import normalized_input_hash
from taskcompiler.procedural.schema import content_hash, strict_json_loads


DESCRIPTIONS = {
    "ifeval": "Respond helpfully to the user's request while satisfying every explicit output constraint in that request. Constraints may concern words, length, language, capitalization, punctuation, or structure. Do not add a preamble or explanation when it would violate a requested constraint. Produce the requested response itself.",
    "logical_deduction_three_objects": "Solve ordering problems involving three objects from all the supplied relational clues. Choose the one option that must be true. Return only the option letter enclosed in parentheses, such as (A), (B), or (C).",
    "boolean_expressions": "Evaluate the supplied Boolean expression, respecting parentheses and operator precedence: not before and before or. Return only True or False.",
    "object_counting": "Read the supplied list or description and count the objects in the requested category. Include quantities and repeated items when relevant, and exclude objects outside the category. Return only the integer count.",
}

SYSTEM = """Create a teaching pack for a frozen small language model from ONLY the
public task description and fixed output/verifier contract provided. The student's
actual task is ONLY public_task_description. The generation/JSON/verifier rules
below are instructions to YOU, the teacher, not a task for the student to perform.
Never tell the student to generate training data, examples, JSON teaching packs,
instruction_id_list, kwargs, or verifier metadata. Requirements, corrections and
full_instructions must describe solving the actual user task, not preparing data.
Invent every
practice input from scratch. You have no benchmark examples or student feedback.
Return JSON with these six keys:
requirements: nonempty list of {id,text}, faithful to the task.
coverage_families: exactly TWO {id,description} distinct broad failure families.
corrections: exactly FOUR {id,text,covers:[requirement IDs]}, standalone teaching
clarifications, each <=300 characters (aim below220); no examples, concrete
numbers, named objects, solved expressions or quoted example answers inside these
strings. Use abstract principles; put every illustration ONLY in demo_pairs.
For Boolean short corrections specifically, do not include literal True or False.
full_instructions: exactly FOUR {id,text} complete alternative task explanations,
each <=800 characters (aim below500). Address the STUDENT directly about solving
its user request. No preparation or verifier field names, and no worked examples.
Preserve task meaning, without adding unsupported rules.
groups: exactly EIGHT {id,family_id,cases:[case,case]}: FOUR independently varied
groups for EACH coverage family. The two cases contrast relevant task evidence.
demo_pairs: exactly FOUR {id,correction_id,cases:[case,case]}, one pair per
correction, all different from practice groups. Demonstrations teach a distinction.
Each case has id,input,reference. Input and reference must be nonempty strings,
each <=1200 characters. For IFEval, also include instruction_id_list and kwargs
as specified below; reference must satisfy EVERY constraint. For other tasks,
reference must follow the described final-answer format. Each case must be
self-contained, solvable, unambiguous, and use a different scenario/wording.
All 24 case IDs and all group/pair IDs must be globally unique ASCII identifiers
using letters, digits, underscores, dots or hyphens. Do not abbreviate lists.
Cases from different groups are split across optimization, selection and an
untouched synthetic holdout. Make EACH family varied across all four groups.
For Boolean expressions, each two-case practice group MUST have one True and one
False answer. Do not repeat the same expression across examples. Check all answers
by independently evaluating the expression a second time before returning it.
For three-object deduction, assign the FIRST coverage family the answer pair
(A),(B), and the SECOND family (B),(C). EVERY one of each family's four groups
must contain those two answers, once each. Vary which option is correct by
reordering options. Demonstrations must cover all three answer letters. Some
clues can admit several orders; select the option true in EVERY consistent order,
not an arbitrary completed order. Do not claim there must be a unique full order.
For object counting, use only unambiguous ordinary category members/nonmembers,
with an explicit quantity for any plural phrase. Avoid contested taxonomy or
borderline objects. Sum all included quantities twice to check the reference.
For all tasks, ensure each output answers the actual input rather than merely
matching an intended label pattern.
No code execution or invented verifier operations. Generated references are
proposals; a compiler checks them but cannot certify general semantic correctness.
"""

IFEVAL_CONTRACT = """Generate English practice requests with ONE or TWO explicit
constraints selected only from these official IFEval verifier types:
- keywords:existence, kwargs {"keywords":["word"]}
- keywords:forbidden_words, kwargs {"forbidden_words":["word"]}
- length_constraints:number_words, kwargs {"num_words":integer,"relation":"less than" or "at least"}
- detectable_format:number_bullet_lists, kwargs {"num_bullets":integer}
- change_case:english_lowercase, kwargs {}
- detectable_format:title, kwargs {}
Each case needs instruction_id_list:[IDs] and kwargs:[matching argument objects]
in identical order. The input must explicitly express exactly these constraints
(alongside an ordinary user request). A title constraint requires a title enclosed
in <<double angle brackets>>. Bullet lists use * or - at the beginning of a line.
Keep references below 100 words and word-count thresholds between 10 and 60.
Leave a safety margin of at least FIVE words below a 'less than' limit or above
an 'at least' threshold. Count the reference words before returning the pack.
Do not invent instruction IDs, omit required arguments, or include null/random
parameters. Prefer simple English words. Do not contradict lowercase and title
requirements. Ensure references really obey every requested constraint.
"""


def teacher_request(task_id: str, seed: int) -> tuple[str, str]:
    description = DESCRIPTIONS[task_id]
    contract = IFEVAL_CONTRACT if task_id == "ifeval" else "Use exactly the final-answer format in the task description."
    return SYSTEM, json.dumps({"public_task_description": description,
                               "practice_case_verifier_format_NOT_the_student_task": contract,
                               "generation_seed": seed}, ensure_ascii=False)


def parse_teacher_response(response: str) -> dict:
    """Accept bare JSON or one explicitly fenced JSON object; never repair data.

    Some providers return explanation before the requested JSON despite JSON mode.
    Preserve the full response as evidence, and reject multiple/ambiguous blocks.
    """
    body = response.strip()
    if body.startswith("{"):
        value = strict_json_loads(body)
    else:
        blocks = re.findall(r"(?ms)^```(?:json)?[ \t]*\r?\n(.*?)^```[ \t]*$", body)
        require(len(blocks) == 1 and len(re.findall(r"(?m)^```", body)) == 2,
                "Teacher response needs exactly one unambiguous JSON block")
        value = strict_json_loads(blocks[0].strip())
    require(isinstance(value, dict), "Teacher JSON must be an object")
    return value


def all_cases(pack: dict) -> list[dict]:
    return [dict(case, group_id=group["id"]) for group in pack["groups"] + pack["demo_pairs"] for case in group["cases"]]


def validate_pack(pack: dict, evaluator) -> dict:
    require(pack.get("version") == 1 and pack.get("task_id") in DESCRIPTIONS, "Invalid general teaching pack")
    task = pack["task_id"]
    require(pack["description"] == DESCRIPTIONS[task], "Public task description changed")
    reqs = identifiers(pack["requirements"], "requirements")
    for row in pack["requirements"]:
        text(row.get("text"), "requirement", 600)
    families = identifiers(pack["coverage_families"], "coverage families", 2)
    for row in pack["coverage_families"]:
        text(row.get("description"), "coverage description", 600)
    corrections = identifiers(pack["corrections"], "corrections", 4)
    for row in pack["corrections"]:
        text(row.get("text"), "correction", 300)
        require(isinstance(row.get("covers"), list) and row["covers"] and set(row["covers"]) <= reqs, "Unknown requirement")
    identifiers(pack["full_instructions"], "full instructions", 4)
    for row in pack["full_instructions"]:
        text(row.get("text"), "full instruction", 800)
    groups = identifiers(pack["groups"], "practice groups", 8)
    require(Counter(g.get("family_id") for g in pack["groups"]) == Counter({f: 4 for f in families}), "Each family requires four groups")
    demos = identifiers(pack["demo_pairs"], "demo pairs", 4)
    require(not groups & demos, "Groups and demonstrations overlap")
    require(Counter(g.get("correction_id") for g in pack["demo_pairs"]) == Counter({c: 1 for c in corrections}), "One demonstration pair per correction required")
    for group in pack["groups"] + pack["demo_pairs"]:
        identifiers(group["cases"], "group cases", 2)
        for case in group["cases"]:
            text(case.get("input"), "input", 1200)
            text(case.get("reference"), "reference", 1200)
            require(case.get("task") == task and case.get("benchmark") == ("ifeval" if task == "ifeval" else "bbh"), "Practice task mismatch")
            evaluator.validate_practice_case(case)
            require(evaluator(case, case["reference"])["correct"], f"Generated reference fails its verifier: {case['id']}")
    cases = all_cases(pack)
    identifiers(cases, "all generated cases", 24)
    require(len({normalized_input_hash(c["input"]) for c in cases}) == len(cases), "Generated inputs overlap")
    for name in ("corrections", "full_instructions"):
        require(len({r["text"].strip() for r in pack[name]}) == 4, f"Duplicate {name}")
    # These narrow gates detect observed preparation defects, not full semantic
    # equivalence. They were fixed before any task-level student outcomes.
    forbidden = ("instruction_id_list", "kwargs", "num_bullets", "teaching pack",
                 "generate training", "generate practice", "create practice",
                 "training cases", "training examples", "verifier metadata")
    for row in pack["requirements"] + pack["corrections"] + pack["full_instructions"]:
        require(not any(word in row["text"].lower() for word in forbidden), "Teaching text describes preparation rather than the student's task")
    for row in pack["corrections"]:
        require(not re.search(r"\d|\bexample\b|\bfor instance\b|\be\.g\.", row["text"], re.I), "Short correction contains a concrete illustration")
        if task == "boolean_expressions":
            require(not re.search(r"\b(?:True|False)\b", row["text"]), "Boolean correction contains truth literals rather than an abstract rule")
    if task == "boolean_expressions":
        require(all({c["reference"] for c in g["cases"]} == {"True", "False"} for g in pack["groups"]), "Boolean practice groups must contrast both outcomes")
    if task == "logical_deduction_three_objects":
        family_pairs = {pack["coverage_families"][0]["id"]: {"(A)", "(B)"}, pack["coverage_families"][1]["id"]: {"(B)", "(C)"}}
        require(all({c["reference"] for c in g["cases"]} == family_pairs[g["family_id"]] for g in pack["groups"]), "Logical practice lacks fixed paired outcome coverage")
        require({c["reference"] for g in pack["demo_pairs"] for c in g["cases"]} == {"(A)", "(B)", "(C)"}, "Logical demonstrations must cover every answer letter")
    provenance = pack["provenance"]
    require(provenance.get("kind") in {"fixture", "teacher_generated"}, "Invalid teacher provenance")
    require(type(provenance.get("generation_seed")) is int, "Invalid generation seed")
    require(provenance.get("description_sha256") == hashlib.sha256(pack["description"].encode()).hexdigest(), "Description hash mismatch")
    return pack


def split_cases(pack: dict, seed: int = 0) -> dict:
    result = {"optimization": [], "selection": [], "synthetic_holdout": []}
    for family in pack["coverage_families"]:
        groups = sorted((g for g in pack["groups"] if g["family_id"] == family["id"]), key=lambda g: g["id"])
        random.Random(content_hash([seed, family["id"]])).shuffle(groups)
        for phase, selected in (("optimization", groups[:2]), ("selection", groups[2:3]), ("synthetic_holdout", groups[3:])):
            result[phase].extend(dict(c, group_id=g["id"]) for g in selected for c in g["cases"])
    return result


def public_pack(body: dict, task_id: str, seed: int, teacher_model: str) -> dict:
    pack = dict(body, version=1, task_id=task_id, description=DESCRIPTIONS[task_id], output_schema={})
    for group in pack["groups"] + pack["demo_pairs"]:
        for case in group["cases"]:
            case.update(task=task_id, benchmark="ifeval" if task_id == "ifeval" else "bbh")
    pack["provenance"] = {"kind": "teacher_generated", "teacher_model": teacher_model,
                          "generation_seed": seed,
                          "description_sha256": hashlib.sha256(pack["description"].encode()).hexdigest()}
    return pack
