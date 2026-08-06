from taskcompiler.statistics import (
    exact_mcnemar,
    task_bootstrap_difference,
    wilson_accuracy,
)


def test_wilson_interval_contains_estimate():
    interval = wilson_accuracy(8, 10)
    assert interval.low < interval.estimate < interval.high


def test_exact_mcnemar_is_one_without_disagreement():
    assert exact_mcnemar([True, False], [True, False]) == 1.0


def test_task_bootstrap_is_paired_and_reproducible():
    a = {"t1": 0.8, "t2": 0.7, "t3": 0.9}
    b = {"t1": 0.7, "t2": 0.6, "t3": 0.8}
    first = task_bootstrap_difference(a, b, samples=500, seed=7)
    second = task_bootstrap_difference(a, b, samples=500, seed=7)
    assert first == second
    assert abs(first.estimate - 0.1) < 1e-9
