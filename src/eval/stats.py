"""Bootstrap confidence intervals and paired significance tests.

For defense-paper tables: report 95% CIs on rates and paired bootstrap
p-values between condition pairs (e.g. attacked vs. defended on the
same sample set).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    n: int


@dataclass(frozen=True)
class PairedTestResult:
    mean_diff: float
    lo: float
    hi: float
    p_value: float  # two-sided p-value for H0: mean_diff == 0
    n_pairs: int


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 7,
) -> BootstrapCI:
    """Percentile bootstrap CI on the mean of a scalar sequence."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return BootstrapCI(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return BootstrapCI(mean=float(arr.mean()), lo=float(lo), hi=float(hi), n=n)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 7,
) -> PairedTestResult:
    """Paired bootstrap: for each pair (a_i, b_i), resample with replacement.

    Returns the mean of (a_i - b_i), its 1-alpha CI, and a two-sided
    p-value computed as 2 * min(P(diff >= 0), P(diff <= 0)) where
    probabilities are the bootstrap fraction.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if len(arr_a) != len(arr_b):
        raise ValueError(f"paired test requires equal lengths, got {len(arr_a)} vs {len(arr_b)}")
    diff = arr_a - arr_b
    n = len(diff)
    if n == 0:
        return PairedTestResult(float("nan"), float("nan"), float("nan"), float("nan"), 0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    resampled_means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(resampled_means, [alpha / 2, 1 - alpha / 2])

    # Two-sided p-value via the centered bootstrap distribution.
    centered = resampled_means - resampled_means.mean()
    observed = diff.mean()
    p_two = 2.0 * min(
        float((centered >= abs(observed)).mean()),
        float((centered <= -abs(observed)).mean()),
    )
    p_two = min(p_two, 1.0)
    return PairedTestResult(
        mean_diff=float(observed),
        lo=float(lo),
        hi=float(hi),
        p_value=p_two,
        n_pairs=n,
    )


def summarize_rate(values: Sequence[float], n_bootstrap: int = 10_000) -> dict:
    """Convenience wrapper for reporting a binary-rate metric with CI."""
    ci = bootstrap_ci(values, n_bootstrap=n_bootstrap)
    return {
        "rate": ci.mean,
        "ci95_lo": ci.lo,
        "ci95_hi": ci.hi,
        "ci95_halfwidth": (ci.hi - ci.lo) / 2,
        "n": ci.n,
    }
