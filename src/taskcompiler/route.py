"""Shortlist relevant rules and check whether their conditions apply."""

from __future__ import annotations

import json
from dataclasses import dataclass

from sentence_transformers import CrossEncoder, SentenceTransformer

from taskcompiler.schema import Boolean3
from taskcompiler.rules_track.text_predicates import unpack_nli_probabilities


@dataclass
class RuleCandidate:
    rule_name: str
    rendered_text: str
    condition_hypothesis: str
    mandatory: bool = False
    dependencies: tuple[str, ...] = ()


@dataclass
class RoutingDecision:
    rule_name: str
    ground_truth: Boolean3
    similarity_score: float
    contradiction_probability: float = 0.0
    entailment_probability: float = 0.0
    neutral_probability: float = 0.0


@dataclass
class RoutedInstructions:
    instructions: list[str]
    certificate: dict


class CertifiedInstructionRouter:
    """Rank instructions locally and close the result over exact obligations."""

    def __init__(
        self,
        retriever_model: str,
        *,
        retriever_revision: str | None = None,
    ):
        self.retriever = SentenceTransformer(
            retriever_model, revision=retriever_revision
        )
        self.identity = {
            "backend": "sentence_transformers",
            "model": retriever_model,
            "revision": retriever_revision,
        }
        self._instruction_cache: dict[tuple[str, ...], object] = {}

    @staticmethod
    def _parse_rule(text: str) -> tuple[dict[str, str], str] | None:
        if not text.startswith("If ") or ", return " not in text:
            return None
        conditions_text, label = text.removeprefix("If ").rsplit(
            ", return ", 1
        )
        conditions = {}
        if conditions_text != "any input":
            for condition in conditions_text.split(", "):
                if "=" not in condition:
                    return None
                field, value = condition.split("=", 1)
                conditions[field] = value
        return conditions, label.removesuffix(".")

    @staticmethod
    def _flat_record(record: dict) -> dict[str, str]:
        flat = {}
        suffix_counts = {}
        for root, fields in record.items():
            if root.startswith("_") or not isinstance(fields, dict):
                continue
            for field, value in fields.items():
                flat[f"{root}.{field}"] = str(value)
                suffix_counts[field] = suffix_counts.get(field, 0) + 1
        for path, value in list(flat.items()):
            suffix = path.split(".", 1)[1]
            if suffix_counts[suffix] == 1:
                flat[suffix] = value
        return flat

    def route(
        self,
        record: dict,
        instructions: list[str],
        *,
        top_k: int,
        label_encoder: dict[str, str] | None = None,
    ) -> RoutedInstructions:
        """Return ranked instructions with a matching rule added if needed.

        Args:
            record: Private structured input available only on the local device.
            instructions: Frozen certified representation.
            top_k: Number of model-ranked rule lines to retain before closure.
            label_encoder: Optional local map from task labels to neutral codes.
        """
        parsed = [
            (index, text, self._parse_rule(text))
            for index, text in enumerate(instructions)
        ]
        rule_rows = [row for row in parsed if row[2] is not None]
        if not rule_rows:
            return RoutedInstructions(
                list(instructions),
                {
                    "applicable": False,
                    "reason": "representation_is_not_machine_routable",
                },
            )

        texts = tuple(row[1] for row in rule_rows)
        vectors = self._instruction_cache.get(texts)
        if vectors is None:
            vectors = self.retriever.encode(
                list(texts), normalize_embeddings=True
            )
            self._instruction_cache[texts] = vectors
        query = json.dumps(record, sort_keys=True, separators=(",", ":"))
        query_vector = self.retriever.encode(
            query, normalize_embeddings=True
        )
        scores = vectors @ query_vector
        ranked = sorted(
            zip(rule_rows, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        flat = self._flat_record(record)
        exact = [
            row
            for row in rule_rows
            if all(flat.get(field) == value for field, value in row[2][0].items())
        ]
        exact_labels = {row[2][1] for row in exact}
        selected = [item[0] for item in ranked[: max(0, top_k)]]
        selected_indices = {row[0] for row in selected}
        closure_added = []
        if exact and not selected_indices.intersection(row[0] for row in exact):
            best_exact = next(
                item[0] for item in ranked if item[0] in exact
            )
            selected.append(best_exact)
            selected_indices.add(best_exact[0])
            closure_added.append(best_exact[0])

        global_lines = [
            text for _, text, parsed_rule in parsed if parsed_rule is None
        ]
        selected_texts = []
        for row in selected:
            text = row[1]
            label = row[2][1]
            if label_encoder is not None and label in label_encoder:
                text = text.rsplit(", return ", 1)[0] + (
                    f", return {label_encoder[label]}."
                )
            selected_texts.append(text)
        return RoutedInstructions(
            global_lines + selected_texts,
            {
                "applicable": True,
                "top_k": top_k,
                "selected_rule_indices": sorted(selected_indices),
                "matching_rule_indices": sorted(row[0] for row in exact),
                "matching_labels": sorted(exact_labels),
                "coverage_closed": bool(exact)
                and bool(selected_indices.intersection(row[0] for row in exact)),
                "label_pure": len(exact_labels) == 1,
                "closure_added_rule_indices": closure_added,
                "opaque_label_encoding": label_encoder is not None,
            },
        )


class Router:
    def __init__(
        self,
        retriever_model: str,
        nli_model: str,
        entailment_threshold: float = 0.6,
        retriever_revision: str | None = None,
        nli_revision: str | None = None,
    ):
        self.retriever = SentenceTransformer(
            retriever_model, revision=retriever_revision
        )
        self.nli = CrossEncoder(
            nli_model,
            automodel_args={"revision": nli_revision} if nli_revision else None,
        )
        self.entailment_threshold = entailment_threshold

    def retrieve_candidates(self, query: str, rules: list[RuleCandidate], top_k: int) -> list[RuleCandidate]:
        """Shortlist rules by embedding similarity.

        Args:
            query: Text used to find relevant rules.
            rules: Available rule candidates.
            top_k: Maximum number of optional rules to return.
        """
        if not rules:
            return []
        query_vec = self.retriever.encode(query, normalize_embeddings=True)
        rule_vecs = self.retriever.encode(
            [r.rendered_text for r in rules], normalize_embeddings=True
        )
        scores = rule_vecs @ query_vec
        ranked = sorted(zip(rules, scores), key=lambda pair: pair[1], reverse=True)
        mandatory = [r for r in rules if r.mandatory]
        shortlisted = [r for r, _ in ranked[:top_k]]
        for rule in mandatory:
            if rule not in shortlisted:
                shortlisted.append(rule)
        return shortlisted

    def ground_predicates(self, query: str, candidates: list[RuleCandidate]) -> list[RoutingDecision]:
        """Check each candidate condition against the query.

        Args:
            query: Text used as NLI evidence.
            candidates: Rules whose conditions should be checked.
        """
        decisions = []
        for candidate in candidates:
            scores = self.nli.predict([(query, candidate.condition_hypothesis)])
            row = scores if getattr(scores, "ndim", 1) == 1 else scores[0]
            contradiction, entailment, neutral = unpack_nli_probabilities(row, self.nli)
            if entailment >= self.entailment_threshold:
                ground_truth = Boolean3.TRUE
            elif contradiction >= self.entailment_threshold:
                ground_truth = Boolean3.FALSE
            else:
                ground_truth = Boolean3.UNKNOWN
            decisions.append(
                RoutingDecision(
                    rule_name=candidate.rule_name,
                    ground_truth=ground_truth,
                    similarity_score=entailment,
                    contradiction_probability=contradiction,
                    entailment_probability=entailment,
                    neutral_probability=neutral,
                )
            )
        return decisions

    def route(
        self, query: str, rules: list[RuleCandidate], top_k: int, token_budget: int
    ) -> list[RuleCandidate]:
        candidates = self.retrieve_candidates(query, rules, top_k)
        decisions = self.ground_predicates(query, candidates)
        decision_by_name = {d.rule_name: d for d in decisions}

        kept = []
        for candidate in candidates:
            decision = decision_by_name[candidate.rule_name]
            if candidate.mandatory or decision.ground_truth in (Boolean3.TRUE, Boolean3.UNKNOWN):
                kept.append(candidate)

        by_name = {candidate.rule_name: candidate for candidate in rules}
        closure = list(kept)
        seen = {candidate.rule_name for candidate in closure}
        cursor = 0
        while cursor < len(closure):
            for dependency in closure[cursor].dependencies:
                if dependency in by_name and dependency not in seen:
                    closure.append(by_name[dependency])
                    seen.add(dependency)
            cursor += 1
        return _fit_to_budget(closure, token_budget, tokenizer=self.retriever.tokenizer)


def _fit_to_budget(
    rules: list[RuleCandidate], token_budget: int, tokenizer=None
) -> list[RuleCandidate]:
    def cost(rule: RuleCandidate) -> int:
        if tokenizer is None:
            return len(rule.rendered_text.split())
        return len(tokenizer.encode(rule.rendered_text, add_special_tokens=False))

    mandatory = [rule for rule in rules if rule.mandatory]
    optional = [rule for rule in rules if not rule.mandatory]
    selected = list(mandatory)
    used = sum(cost(rule) for rule in mandatory)
    # Required rules may exceed the context budget.
    for rule in optional:
        rule_cost = cost(rule)
        if used + rule_cost > token_budget:
            continue
        selected.append(rule)
        used += rule_cost
    return selected
