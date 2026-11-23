"""Evaluate semantic predicates for text-based tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import CrossEncoder

from taskcompiler.schema import Boolean3


@dataclass
class PredicateAccuracy:
    correct: int
    total: int

    @property
    def rate(self) -> float:
        return self.correct / self.total if self.total else 0.0


class TextEntailmentPredicate:
    """Evaluate textual entailment with a frozen NLI model.

    Args:
        nli_model: Cross-encoder model name.
        entailment_threshold: Minimum probability for entailment.
        contradiction_threshold: Minimum probability for contradiction.
        revision: Optional pinned model revision.
    """

    def __init__(
        self,
        nli_model: str,
        entailment_threshold: float = 0.6,
        contradiction_threshold: float = 0.6,
        revision: str | None = None,
    ):
        self.nli = CrossEncoder(nli_model, automodel_args={"revision": revision} if revision else None)
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

    def evaluate(self, contract_text: str, hypothesis_text: str) -> Boolean3:
        windows = _overlapping_windows(contract_text)
        scores = self.nli.predict([(window, hypothesis_text) for window in windows])
        probabilities = [
            unpack_nli_probabilities(row, self.nli) for row in scores
        ]
        contradiction = max(row[0] for row in probabilities)
        entailment = max(row[1] for row in probabilities)
        if (
            entailment >= self.entailment_threshold
            and contradiction >= self.contradiction_threshold
        ):
            return Boolean3.UNKNOWN
        if entailment >= self.entailment_threshold:
            return Boolean3.TRUE
        if contradiction >= self.contradiction_threshold:
            return Boolean3.FALSE
        return Boolean3.UNKNOWN


def _overlapping_windows(
    text: str, window_chars: int = 1800, overlap_chars: int = 200
) -> list[str]:
    if len(text) <= window_chars:
        return [text]
    windows = []
    step = window_chars - overlap_chars
    for start in range(0, len(text), step):
        window = text[start : start + window_chars]
        if window:
            windows.append(window)
        if start + window_chars >= len(text):
            break
    return windows


def unpack_nli_probabilities(scores, cross_encoder) -> tuple[float, float, float]:
    """Map scores to contradiction, entailment, and neutral probabilities.

    Args:
        scores: Raw class scores from the model.
        cross_encoder: Model that produced the scores.
    """
    logits = np.asarray(scores, dtype=float)
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum()

    config = cross_encoder.model.config
    id2label = {
        int(index): str(label).lower()
        for index, label in getattr(config, "id2label", {}).items()
    }

    def locate(name: str, fallback: int) -> int:
        for index, label in id2label.items():
            if name in label:
                return index
        return fallback

    contradiction_index = locate("contradiction", 0)
    entailment_index = locate("entailment", 1)
    neutral_index = locate("neutral", 2)
    return (
        float(probabilities[contradiction_index]),
        float(probabilities[entailment_index]),
        float(probabilities[neutral_index]),
    )


def score_against_gold(
    predicate: TextEntailmentPredicate,
    examples: list,
    gold_label_to_boolean3: dict[str, Boolean3],
) -> PredicateAccuracy:
    correct = 0
    for example in examples:
        predicted = predicate.evaluate(example.contract_text, example.hypothesis_text)
        expected = gold_label_to_boolean3[example.label]
        if predicted == expected:
            correct += 1
    return PredicateAccuracy(correct=correct, total=len(examples))
