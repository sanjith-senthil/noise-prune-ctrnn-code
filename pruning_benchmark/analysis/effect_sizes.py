"""Effect sizes and distribution-free confidence intervals for paired comparisons.

The paper's inference is a paired two-sided Wilcoxon signed-rank test, so the
matched effect size is the Hodges-Lehmann estimator -- the median of the Walsh
averages of the paired differences -- together with its exact distribution-free
confidence interval derived from the signed-rank null distribution.  Reported
this way, the interval excludes zero exactly when the (uncorrected) signed-rank
test rejects, so estimate and test cannot disagree.

All functions operate on a one-dimensional array of paired differences.  The
analysis scripts pass the *nonzero* differences, matching the pre-registered
test procedure, which discards exact ties before testing.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def signed_rank_null(n: int) -> np.ndarray:
    """Exact null pmf of the Wilcoxon W+ statistic over 0..n(n+1)/2."""
    if n <= 0:
        return np.array([1.0])
    total = n * (n + 1) // 2
    pmf = np.zeros(total + 1, dtype=float)
    pmf[0] = 1.0
    for rank in range(1, n + 1):
        shifted = np.zeros_like(pmf)
        shifted[rank:] = pmf[: total + 1 - rank]
        pmf = pmf + shifted
    return pmf / pmf.sum()


def walsh_averages(diff: np.ndarray) -> np.ndarray:
    """Sorted Walsh averages (d_i + d_j)/2 for all i <= j."""
    d = np.asarray(diff, dtype=float)
    i, j = np.triu_indices(d.size, k=0)
    return np.sort((d[i] + d[j]) / 2.0)


def hodges_lehmann(diff: np.ndarray) -> float:
    """Hodges-Lehmann location estimate of the paired difference."""
    d = np.asarray(diff, dtype=float)
    if d.size == 0:
        return float("nan")
    return float(np.median(walsh_averages(d)))


def hodges_lehmann_ci(diff: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact distribution-free two-sided CI for the Hodges-Lehmann estimate.

    Uses the signed-rank null: with ``M = n(n+1)/2`` Walsh averages and ``c``
    the largest integer with ``P(W+ <= c) <= alpha/2``, the interval is
    ``[W_(c+1), W_(M-c)]``.  Conservative by construction, because the
    signed-rank statistic is discrete.
    """
    d = np.asarray(diff, dtype=float)
    n = d.size
    if n == 0:
        return (float("nan"), float("nan"))
    walsh = walsh_averages(d)
    m = walsh.size
    cdf = np.cumsum(signed_rank_null(n))
    eligible = np.nonzero(cdf <= alpha / 2.0)[0]
    if eligible.size == 0:
        return (float(walsh[0]), float(walsh[-1]))
    c = int(eligible[-1])
    lo_idx = min(c, m - 1)
    hi_idx = max(m - c - 1, 0)
    return (float(walsh[lo_idx]), float(walsh[hi_idx]))


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation in [-1, 1].

    ``(sum of positive signed ranks - sum of negative signed ranks)`` over the
    total rank sum.  +1 means every pair favours the first condition.
    """
    d = np.asarray(diff, dtype=float)
    d = d[d != 0.0]
    if d.size == 0:
        return 0.0
    order = np.argsort(np.abs(d), kind="mergesort")
    ranks = np.empty(d.size, dtype=float)
    absd = np.abs(d)[order]
    ranks_sorted = np.arange(1, d.size + 1, dtype=float)
    # average ranks within ties of |d|
    start = 0
    for end in range(1, d.size + 1):
        if end == d.size or absd[end] != absd[start]:
            ranks_sorted[start:end] = ranks_sorted[start:end].mean()
            start = end
    ranks[order] = ranks_sorted
    total = ranks.sum()
    if total == 0:
        return 0.0
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / total)


def cohens_dz(diff: np.ndarray) -> float:
    """Standardized paired mean difference (mean / sample SD, ddof=1)."""
    d = np.asarray(diff, dtype=float)
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def paired_effect_sizes(diff: np.ndarray, alpha: float = 0.05) -> dict:
    """All effect-size fields for one paired comparison."""
    d = np.asarray(diff, dtype=float)
    hl = hodges_lehmann(d)
    lo, hi = hodges_lehmann_ci(d, alpha=alpha)
    return {
        "hodges_lehmann_diff": hl,
        "hodges_lehmann_ci_lower": lo,
        "hodges_lehmann_ci_upper": hi,
        "hodges_lehmann_ci_level": 1.0 - alpha,
        "hodges_lehmann_ci_method": "exact signed-rank (distribution-free)",
        "rank_biserial_r": rank_biserial(d),
        "cohens_dz": cohens_dz(d),
        "effect_size_basis": "nonzero paired differences (matches the reported test)",
    }


__all__ = [
    "cohens_dz",
    "hodges_lehmann",
    "hodges_lehmann_ci",
    "paired_effect_sizes",
    "rank_biserial",
    "signed_rank_null",
    "walsh_averages",
]
