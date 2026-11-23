"""Deterministic bootstrap routines used by the frozen experimental protocols.

Historical tables use Python random; new controls use a frozen NumPy-compatible
PCG64 stream. Keeping both preserves the originally reported estimands and draws.
"""

from __future__ import annotations

import math
import random

from .common import require


def quantile(values, probability):
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    fraction = position - lower
    return (
        values[lower] * (1 - fraction)
        + values[min(lower + 1, len(values) - 1)] * fraction
    )


class FrozenPCG64:
    """PCG64 XSL-RR stream for the two frozen NumPy default_rng seeds.

    Initial state/increment pairs are recorded NumPy PCG64 seed expansions.
    Bounded 32-bit sampling uses multiply-high rejection, matching the NumPy
    Generator choice calls in these studies. This avoids requiring NumPy only
    to reproduce an existing Monte Carlo bootstrap sequence.

    Independent implementation of the published PCG XSL-RR transition/output
    and Lemire multiply-high bounded sampling, not copied NumPy source:
    https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf
    https://arxiv.org/abs/1805.10941
    """

    STATES = {
        20261120: (
            175293069496666043790200453473569188007,
            303379588208611549728128745228403941347,
        ),
        20260727: (
            5170418299989641538360905967541131569,
            209734487121680501811083848134438824815,
        ),
    }

    def __init__(self, seed):
        self.state, self.increment = self.STATES[seed]
        self.cached = None

    def uint32(self):
        if self.cached is not None:
            result, self.cached = self.cached, None
            return result
        self.state = (
            self.state * 47026247687942121848144207491837523525 + self.increment
        ) & ((1 << 128) - 1)
        high, low = self.state >> 64, self.state & ((1 << 64) - 1)
        value, rotation = high ^ low, high >> 58
        raw = ((value >> rotation) | (value << ((-rotation) & 63))) & ((1 << 64) - 1)
        self.cached = raw >> 32
        return raw & ((1 << 32) - 1)

    def index(self, size):
        threshold = ((1 << 32) - size) % size
        while True:
            product = self.uint32() * size
            if (product & ((1 << 32) - 1)) >= threshold:
                return product >> 32


def bootstrap(differences, samples=10000, seed=20261120, numpy_stream=True):
    require(bool(differences), "Empty paired contrast")
    if max(differences) == min(differences):
        return {
            "difference": differences[0],
            "descriptive_95_interval": [differences[0], differences[0]],
            "resamples": samples,
            "seed": seed,
            "unit_count": len(differences),
            "implementation": "Exact degenerate bootstrap distribution",
        }
    if numpy_stream:
        generator = FrozenPCG64(seed)
        draws = sorted(
            sum(differences[generator.index(len(differences))] for _ in differences)
            / len(differences)
            for _ in range(samples)
        )
        interval = [quantile(draws, 0.025), quantile(draws, 0.975)]
        implementation = (
            "Frozen NumPy PCG64-compatible choice stream and linear quantiles"
        )
    else:
        generator = random.Random(seed)
        draws = sorted(
            sum(generator.choice(differences) for _ in differences) / len(differences)
            for _ in range(samples)
        )
        interval = [
            draws[int(0.025 * (samples - 1))],
            draws[int(0.975 * (samples - 1))],
        ]
        implementation = "Original Python random bootstrap and indexed quantiles"
    return {
        "difference": sum(differences) / len(differences),
        "descriptive_95_interval": interval,
        "resamples": samples,
        "seed": seed,
        "unit_count": len(differences),
        "implementation": implementation,
    }


def historical_bootstrap(
    differences: list[float], samples: int, seed: int
) -> tuple[float, float, float]:
    generator = random.Random(seed)
    values = sorted(
        sum(generator.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    return (
        sum(differences) / len(differences),
        values[int(0.025 * (samples - 1))],
        values[int(0.975 * (samples - 1))],
    )


def mcnemar(first: list[bool], second: list[bool]) -> tuple[int, int, float]:
    first_only = sum(a and not b for a, b in zip(first, second))
    second_only = sum(b and not a for a, b in zip(first, second))
    discordant = first_only + second_only
    tail = min(first_only, second_only)
    probability = sum(
        math.comb(discordant, index) * 0.5**discordant for index in range(tail + 1)
    )
    return first_only, second_only, min(1.0, 2 * probability)
