"""Confidence intervals and paired tests (design.md Sec. 5)."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(x, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile CI of per-instance values."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a, b, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> dict:
    """CI for mean(a - b) where a[i], b[i] come from the same instance."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, "paired arrays must align"
    m, lo, hi = bootstrap_ci(a - b, n_boot, alpha, seed)
    return {"diff": m, "lo": lo, "hi": hi}


def unpaired_bootstrap_diff(a, b, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> dict:
    """CI for mean(a) - mean(b) from different instance sets (e.g. baseline vs A1)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(1)
    d = da - db
    return {"diff": float(a.mean() - b.mean()), "lo": float(np.quantile(d, alpha / 2)),
            "hi": float(np.quantile(d, 1 - alpha / 2))}


def ols_slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def bootstrap_slope(Y, x, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Slope of the per-level mean of Y on x, with a CI from resampling instances.

    Y: (n_instances, n_levels) outcomes of the SAME instances at each level x[j]
    (e.g. Recall@1 at n = 0, 1, 2, 5, 10 fresh-dated traps)."""
    Y, x = np.asarray(Y, float), np.asarray(x, float)
    assert Y.ndim == 2 and Y.shape[1] == len(x), "Y must be instances x levels"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(Y), size=(n_boot, len(Y)))
    means = Y[idx].mean(1)                       # (n_boot, n_levels)
    xc = x - x.mean()
    slopes = (means - means.mean(1, keepdims=True)) @ xc / (xc ** 2).sum()
    return {"slope": ols_slope(x, Y.mean(0)), "lo": float(np.quantile(slopes, alpha / 2)),
            "hi": float(np.quantile(slopes, 1 - alpha / 2))}


def mcnemar_exact(a, b) -> dict:
    """Exact McNemar test on paired binary outcomes."""
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only": n10, "b_only": n01, "p": float(p)}
