import pytest
import numpy as np

from taskcompiler.route import (
    CertifiedInstructionRouter,
    Router,
    RuleCandidate,
    _fit_to_budget,
)

pytest.importorskip("sentence_transformers")

WRITE_RULE = RuleCandidate(
    rule_name="UnauthorizedStateChange",
    rendered_text="An unauthorized state-changing write command is likely manipulation.",
    condition_hypothesis="The event is a state-changing write command.",
)
READ_RULE = RuleCandidate(
    rule_name="ReadOnlyMonitoring",
    rendered_text="A read-only monitoring request is not process manipulation by itself.",
    condition_hypothesis="The event is a read-only operation.",
)


@pytest.fixture(scope="module")
def router():
    return Router(
        retriever_model="sentence-transformers/all-MiniLM-L6-v2",
        nli_model="cross-encoder/nli-deberta-v3-small",
    )


def test_similarity_alone_can_over_include(router):
    query = "An unknown source sent a command that only reads a diagnostic register."
    candidates = router.retrieve_candidates(query, [WRITE_RULE, READ_RULE], top_k=2)
    names = {c.rule_name for c in candidates}
    assert "ReadOnlyMonitoring" in names


def test_grounding_drops_condition_false_rule(router):
    query = "An unknown source sent a command that only reads a diagnostic register."
    routed = router.route(query, [WRITE_RULE, READ_RULE], top_k=2, token_budget=200)
    names = {c.rule_name for c in routed}
    assert "ReadOnlyMonitoring" in names
    assert "UnauthorizedStateChange" not in names


def test_mandatory_rule_is_never_dropped(router):
    mandatory = RuleCandidate(
        rule_name="TrafficVolumeCaution",
        rendered_text="Traffic volume alone is never sufficient evidence of manipulation.",
        condition_hypothesis="This condition always applies.",
        mandatory=True,
    )
    query = "An unknown source sent a command that only reads a diagnostic register."
    routed = router.route(query, [WRITE_RULE, mandatory], top_k=1, token_budget=200)
    names = {c.rule_name for c in routed}
    assert "TrafficVolumeCaution" in names


def test_mandatory_rule_survives_tiny_budget():
    optional = RuleCandidate("optional", "many optional words here", "optional")
    mandatory = RuleCandidate(
        "mandatory",
        "a mandatory rule longer than the budget",
        "always",
        mandatory=True,
    )
    selected = _fit_to_budget([optional, mandatory], token_budget=1)
    assert [rule.rule_name for rule in selected] == ["mandatory"]


def test_certified_router_adds_a_missing_matching_rule():
    class FakeRetriever:
        def encode(self, value, normalize_embeddings=True):
            if isinstance(value, list):
                return np.array([[1.0, 0.0], [0.0, 1.0]])
            return np.array([1.0, 0.0])

    router = CertifiedInstructionRouter.__new__(CertifiedInstructionRouter)
    router.retriever = FakeRetriever()
    router._instruction_cache = {}
    instructions = [
        "Verified decision tree.",
        "If role=admin, return allow.",
        "If role=guest, return deny.",
    ]

    routed = router.route(
        {"event": {"role": "guest"}},
        instructions,
        top_k=1,
    )

    assert routed.instructions == instructions
    assert routed.certificate["coverage_closed"] is True
    assert routed.certificate["label_pure"] is True
    assert routed.certificate["closure_added_rule_indices"] == [2]


def test_certified_router_can_encode_labels_as_neutral_codes():
    class FakeRetriever:
        def encode(self, value, normalize_embeddings=True):
            if isinstance(value, list):
                return np.array([[1.0, 0.0]])
            return np.array([1.0, 0.0])

    router = CertifiedInstructionRouter.__new__(CertifiedInstructionRouter)
    router.retriever = FakeRetriever()
    router._instruction_cache = {}

    routed = router.route(
        {"event": {"role": "guest"}},
        ["Verified list.", "If role=guest, return allow."],
        top_k=0,
        label_encoder={"allow": "choice_a"},
    )

    assert routed.instructions[-1] == "If role=guest, return choice_a."
    assert routed.certificate["matching_labels"] == ["allow"]
    assert routed.certificate["opaque_label_encoding"] is True
