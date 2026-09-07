"""Difference (f1) and similarity (f2) factors with bootstrap uncertainty.

f2 is the regulatory workhorse for comparing a test profile against a
reference (FDA SUPAC-IR guidance; EMA guideline on the investigation of
bioequivalence). It is almost always reported as a bare number, although it
is computed from means of a small number of vessels (n = 12 is typical) and
therefore carries sampling error of its own. The functions here return the
point estimate and, on request, a bootstrap interval over vessels.
"""
from __future__ import annotations
import numpy as np


def f1_difference(ref: np.ndarray, test: np.ndarray) -> float:
    """f1 = 100 * sum|R - T| / sum R. Values below 15 indicate similarity."""
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)
    denom = np.sum(ref)
    if denom <= 0:
        return float("nan")
    return float(100.0 * np.sum(np.abs(ref - test)) / denom)


def f2_similarity(ref: np.ndarray, test: np.ndarray) -> float:
    """f2 = 50 log10( 100 / sqrt(1 + mean((R - T)^2)) ).

    Values of 50 or above indicate similar profiles. Inputs are the mean
    percent released at each shared time point.
    """
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)
    if ref.shape != test.shape:
        raise ValueError("reference and test profiles must share time points")
    msd = float(np.mean((ref - test) ** 2))
    return float(50.0 * np.log10(100.0 / np.sqrt(1.0 + msd)))


def f2_with_uncertainty(ref_vessels: np.ndarray, test_vessels: np.ndarray,
                        n_boot: int = 5000, seed: int = 0) -> dict:
    """Bootstrap f2 by resampling whole vessels (rows) with replacement.

    ref_vessels and test_vessels are arrays of shape (n_vessels, n_times)
    holding the per-vessel release profiles. Resampling whole vessels keeps
    the within-vessel correlation across time points intact, which resampling
    individual measurements would destroy.
    """
    rng = np.random.default_rng(seed)
    R = np.asarray(ref_vessels, dtype=float)
    T = np.asarray(test_vessels, dtype=float)
    point = f2_similarity(R.mean(axis=0), T.mean(axis=0))
    nr, nt = R.shape[0], T.shape[0]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        ri = rng.integers(0, nr, size=nr)
        ti = rng.integers(0, nt, size=nt)
        draws[b] = f2_similarity(R[ri].mean(axis=0), T[ti].mean(axis=0))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "f2": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "boot_sd": float(draws.std(ddof=1)),
        "p_below_50": float(np.mean(draws < 50.0)),
        "decision_point": "similar" if point >= 50.0 else "not similar",
        "decision_stable": bool((draws >= 50.0).all() or (draws < 50.0).all()),
        "n_boot": int(n_boot),
    }
