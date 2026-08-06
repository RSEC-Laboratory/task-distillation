"""Calculate paired statistics for frozen-artifact evaluations."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def wilson_accuracy(correct: int, total: int, z: float = 1.959963984540054) -> Interval:
    """Calculate a Wilson interval for accuracy.

    Args:
        correct: Number of correct predictions.
        total: Number of predictions.
        z: Normal critical value.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return Interval(proportion, max(0.0, centre - radius), min(1.0, centre + radius))


def exact_mcnemar(correct_a: list[bool], correct_b: list[bool]) -> float:
    """Compare paired predictions with an exact McNemar test.

    Args:
        correct_a: Correctness values for the first system.
        correct_b: Matching values for the second system.
    """
    if len(correct_a) != len(correct_b) or not correct_a:
        raise ValueError("paired non-empty correctness vectors are required")
    a_only = sum(a and not b for a, b in zip(correct_a, correct_b))
    b_only = sum(b and not a for a, b in zip(correct_a, correct_b))
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = min(a_only, b_only)
    probability = sum(
        math.comb(discordant, k) * (0.5**discordant)
        for k in range(tail + 1)
    )
    return min(1.0, 2 * probability)


def task_bootstrap_difference(
    task_scores_a: dict[str, float],
    task_scores_b: dict[str, float],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Bootstrap the macro-average difference between paired task scores.

    Args:
        task_scores_a: Scores for the first system.
        task_scores_b: Matching scores for the second system.
        samples: Number of bootstrap samples.
        seed: Random seed.
    """
    tasks = sorted(set(task_scores_a) & set(task_scores_b))
    if len(tasks) < 2:
        raise ValueError("at least two paired task groups are required")
    differences = [task_scores_a[task] - task_scores_b[task] for task in tasks]
    estimate = sum(differences) / len(differences)
    generator = random.Random(seed)
    bootstrap = []
    for _ in range(samples):
        draw = [generator.choice(differences) for _ in differences]
        bootstrap.append(sum(draw) / len(draw))
    bootstrap.sort()
    low = bootstrap[int(0.025 * (samples - 1))]
    high = bootstrap[int(0.975 * (samples - 1))]
    return Interval(estimate, low, high)
