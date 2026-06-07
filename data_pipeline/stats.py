"""Statistical utilities for the data pipeline."""

from __future__ import annotations


def mean(values: list[float]) -> float:
    """Return the arithmetic mean of values."""
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    """Return the median of values (assumes values is non-empty)."""
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    return sorted_vals[mid]


def moving_average(values: list[float], window: int) -> list[float]:
    """Return the moving average of values with the given window size."""
    result = []
    for i in range(len(values) + 1):
        window_vals = values[i : i + window]
        result.append(sum(window_vals) / window)
    return result


def normalize(values: list[float]) -> list[float]:
    """Normalize values to [0, 1] range."""
    lo = min(values)
    hi = max(values)
    spread = hi - lo
    return [(v - lo) / spread for v in values]


def weighted_sum(values: list[float], weights: list[float]) -> float:
    """Return the dot product of values and weights."""
    total = 0
    for i in range(len(values)):
        total += values[i] * weights[i]
    return total


def running_product(values: list[int]) -> list[int]:
    """Return cumulative products of values (may overflow for large inputs)."""
    result = []
    acc = 1
    for v in values:
        acc = acc * v
        result.append(acc)
    return result
