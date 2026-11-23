"""Remote preparation accepts only a public task description and fixed protocol."""
import json

SYSTEM = """Prepare a teaching pack for a frozen small language model from the public
task description and its public output contract. You never receive private inputs,
existing examples, evaluation labels, or student feedback. All examples must be
invented from scratch. Return JSON only. Do not change the task or invent additional
decision rules. This is a bounded academic protocol, not a semantic proof.

Return these keys:
requirements: list of {id,text}, extracting the actual requirements from the task.
coverage_families: exactly four {id,description} behavior families worth checking.
corrections: exactly four {id,text,covers:[requirement IDs]} short standalone teaching
clarifications. Each text is at most 300 characters and must be valid by itself.
The four corrections should address distinct likely misunderstandings. They can
be combined, with no dependencies or contradictory instructions. Do not include
worked examples inside these strings. Do not impose heuristics unsupported by the
task, such as automatically using the last clause or treating missing support as no.
full_instructions: exactly four {id,text} complete alternative explanations of how
to solve the same task, at most 1200 characters each (aim below 800 characters).
Preserve the task's meaning.
groups: exactly sixteen {id,family_id,cases:[{id,input,reference},{id,input,reference}]}.
There must be FOUR independent groups for EACH of the four coverage families.
The two cases in a group form a meaningful contrast with changed relevant evidence.
Different groups must use independently varied situations, not paraphrases of one
template. Never reuse an input. Cases input MUST be a string <=1200 characters
(aim below 800). reference is a JSON value satisfying the public output schema.
demo_pairs: exactly four {id,correction_id,cases:[{id,input,reference},{id,input,reference}]}.
Each correction has ONE demonstration pair illustrating its distinction. These
eight demonstration inputs are separate from all 32 practice inputs: use different
situations/entities/wording. All 40 case IDs are globally unique. Group IDs are
globally unique too. IDs use ASCII letters, digits, underscores, dots or hyphens.

Coverage must be useful in every group split, not just overall. Each family will
have two groups used for optimization, one for selection, one left untouched.
For binary classification/boolean outputs include both answers in each group.
For multiclass tasks, assign each family a fixed pair of allowed output labels.
Every one of that family's FOUR groups must contain one case for EACH label in
its assigned pair; do not change the label pair across groups in the same family.
Across the four families, the assigned pairs must cover EVERY allowed schema
label. Independently vary the inputs while preserving those group label pairs.
This ensures that choosing any one group per family still covers every label.
For array extraction, make one family contrast
different zero-mention texts (both answers empty), and other families contain
positive mentions: distinguish genuine names from generic nouns, preserve exact
contiguous tokens and include noisy/lowercase mentions when task-relevant. Include
empty and nonempty outcomes across every partition. Do not assume benchmark label
frequencies. Generated references are proposals, not trusted ground truth.
Only the listed six top-level keys are required. Metadata is supplied by our code.
Count all four corrections, four full instructions, sixteen groups, and four demo
pairs carefully before returning. Do not abbreviate the requested lists.
"""


def build_teacher_request(description, schema, seed):
    return SYSTEM, json.dumps({"public_task_description": description,
                              "public_output_contract": schema,
                              "generation_seed": seed}, ensure_ascii=False)
