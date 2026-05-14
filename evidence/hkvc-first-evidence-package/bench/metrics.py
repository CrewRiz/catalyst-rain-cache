from __future__ import annotations

import statistics


def mean_ns(samples: list[int]) -> int:
    if not samples:
        return 0
    return int(statistics.fmean(samples))


def variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.pvariance(values)
