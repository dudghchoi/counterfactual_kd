"""Analyses for the TIFS experiment program (docs/tifs_plan.md §4.3).

Every contrast here is between hit vectors measured on the same
non-target inputs, so each is paired input-by-input:

* :func:`paired_contrast` — k = mean(a) - mean(b) with the Tango interval,
  delta*, the verdict at each tolerance, and score-test p-values: the
  two one-sided equivalence tests (TOST) and the directional test
  H1: k > delta. The 95% interval lies inside [-delta, delta] exactly when
  both TOST p-values are below 0.025.
* :func:`joint_bootstrap` — percentile intervals for any linear contrast of
  several aligned vectors, drawing input indices once per resample
  (decompositions, corpus x teacher interactions).
* :func:`random_effects_hk` — DerSimonian–Laird pooling of per-seed
  estimates with the Hartung–Knapp variance and a t_{k-1} interval (S2).
  Seed-level predictions are never concatenated: n stays the eval size.
* :func:`t_contrast` — verdicts and one-sided p-values for an estimate
  with a standard error and a t reference (the seed-pooled primary rule,
  docs/tifs_plan.md §0.3).
* :func:`holm` — Holm step-down adjustment (S5, for the union claims; the
  "all cells equivalent" claim is an intersection-union test and needs no
  adjustment, Berger 1982).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from counterfactual_kd.metrics.paired import (
    _tango_score, _z, audit_verdict, delta_star, paired_counts,
)


def _norm_cdf(x: float) -> float:
    if x == math.inf:
        return 1.0
    if x == -math.inf:
        return 0.0
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def tango_pvalues(a: Sequence[int], b: Sequence[int], delta: float) -> dict:
    """Score-test p-values for k = mean(a) - mean(b) against +-delta.

    ``tost_upper``: H0 k >= delta vs H1 k < delta.
    ``tost_lower``: H0 k <= -delta vs H1 k > -delta.
    ``tost``      : max of the two (equivalence at level alpha iff tost < alpha).
    ``positive``  : H0 k <= delta vs H1 k > delta.
    ``negative``  : H0 k >= -delta vs H1 k < -delta.
    """
    n11, n10, n01, n00 = paired_counts(a, b)
    n = n11 + n10 + n01 + n00
    hi = min(delta, 1.0 - 1e-12)
    lo = max(-delta, -1.0 + 1e-12)
    z_hi = _tango_score(n10, n01, n, hi)
    z_lo = _tango_score(n10, n01, n, lo)
    up = _norm_cdf(z_hi)
    low = 1.0 - _norm_cdf(z_lo)
    return {"tost_upper": up, "tost_lower": low, "tost": max(up, low),
            "positive": 1.0 - _norm_cdf(z_hi), "negative": _norm_cdf(z_lo)}


def paired_contrast(a: Sequence[int], b: Sequence[int],
                    tols: Sequence[float] = (0.02, 0.05, 0.10),
                    confidence: float = 0.95) -> dict:
    """Matched contrast of two aligned hit vectors (a minus b)."""
    first = None
    out: dict = {}
    for tol in tols:
        av = audit_verdict(a, b, delta=tol, confidence=confidence)
        if first is None:
            first = av
        out[f"verdict@{tol:g}"] = av["relative"]
        pv = tango_pvalues(a, b, tol)
        out[f"p_tost@{tol:g}"] = pv["tost"]
        out[f"p_pos@{tol:g}"] = pv["positive"]
        out[f"p_up@{tol:g}"] = pv["tost_upper"]
    out.update(n=first["n"], a=first["p_trig"], b=first["c_trig"], k=first["k"],
               lo=first["ci_lower"], hi=first["ci_upper"],
               delta_star=delta_star(first["ci_lower"], first["ci_upper"]),
               excludes_zero=first["excludes_zero"],
               n10=first["counts"]["n10"], n01=first["counts"]["n01"])
    return out


def joint_bootstrap(vectors: dict[str, Sequence[int]],
                    contrasts: dict[str, dict[str, float]],
                    R: int = 5000, seed: int = 0, confidence: float = 0.95) -> dict:
    """Percentile intervals for linear contrasts of aligned hit vectors.

    ``contrasts`` maps a name to coefficients over ``vectors`` keys, e.g.
    ``{"interaction": {"P1": 1, "C1": -1, "P0": -1, "C0": 1}}``.
    """
    keys = list(vectors)
    mat = np.asarray([vectors[k] for k in keys], dtype=float)
    n = mat.shape[1]
    if n == 0:
        raise ValueError("empty vectors")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(R, n))
    means = np.stack([mat[i][idx].mean(axis=1) for i in range(len(keys))])  # (V, R)
    point = mat.mean(axis=1)
    alpha = 1.0 - confidence
    out = {}
    for name, coef in contrasts.items():
        w = np.array([coef.get(k, 0.0) for k in keys])
        draws = np.sort(w @ means)
        out[name] = float(w @ point)
        out[f"{name}_lo"] = float(draws[int((alpha / 2) * (R - 1))])
        out[f"{name}_hi"] = float(draws[int((1 - alpha / 2) * (R - 1))])
    return out


def random_effects_hk(estimates: Sequence[float], ses: Sequence[float],
                      confidence: float = 0.95, floor: bool = False) -> dict:
    """DerSimonian–Laird pooling with the Hartung–Knapp interval.

    ``ses`` are per-estimate standard errors (here from each seed's
    Tango interval half-width / z). A zero SE is floored at 1e-6 so a
    degenerate seed does not take all the weight.

    ``floor=True`` is the modified Hartung–Knapp interval (Röver et al.
    2015): the HK scale factor is kept at 1 or more, so the interval is
    never narrower than the fixed-effect one. Seeds evaluated on the same
    inputs agree more than independent studies, which shrinks the plain HK
    factor below 1.
    """
    y = np.asarray(estimates, dtype=float)
    se = np.maximum(np.asarray(ses, dtype=float), 1e-6)
    k = len(y)
    if k < 2:
        raise ValueError("need at least two estimates")
    v = se ** 2
    w = 1.0 / v
    mu_fe = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - mu_fe) ** 2))
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    ws = 1.0 / (v + tau2)
    mu = float(np.sum(ws * y) / np.sum(ws))
    q_hk = float(np.sum(ws * (y - mu) ** 2) / (k - 1))
    if floor:
        q_hk = max(q_hk, 1.0)
    var_hk = q_hk / float(np.sum(ws))
    from scipy.stats import t as t_dist
    tq = float(t_dist.ppf(0.5 + confidence / 2, k - 1))
    half = tq * math.sqrt(var_hk)
    return {"k": k, "mu": mu, "lo": mu - half, "hi": mu + half,
            "tau": math.sqrt(tau2), "Q": q, "se_hk": math.sqrt(var_hk),
            "sd_between": float(np.std(y, ddof=1))}


def t_contrast(est: float, se: float, df: float,
               tols: Sequence[float] = (0.02, 0.05, 0.10),
               confidence: float = 0.95) -> dict:
    """Interval, verdicts and one-sided p-values of ``est`` with a t_df
    reference, in the format of :func:`paired_contrast`.

    ``p_up@d``: H0 k >= d vs H1 k < d (the one-sided "below d" test);
    ``p_tost@d``: max of p_up and the mirror test at -d; ``p_pos@d``:
    H0 k <= d vs H1 k > d. The interval lies below d exactly when
    p_up < (1 - confidence) / 2.
    """
    from scipy.stats import t as t_dist
    q = float(t_dist.ppf(0.5 + confidence / 2, df))
    lo, hi = est - q * se, est + q * se
    out = {"k": est, "se": se, "df": df, "lo": lo, "hi": hi,
           "delta_star": delta_star(lo, hi)}
    for tol in tols:
        if se > 0:
            p_up = float(t_dist.cdf((est - tol) / se, df))
            p_low = float(t_dist.sf((est + tol) / se, df))
            p_pos = float(t_dist.sf((est - tol) / se, df))
        else:
            p_up, p_low, p_pos = float(est >= tol), float(est <= -tol), float(est <= tol)
        out[f"verdict@{tol:g}"] = ("EQUIVALENT" if -tol <= lo and hi <= tol else
                                   "POSITIVE" if lo > tol else
                                   "NEGATIVE" if hi < -tol else "INCONCLUSIVE")
        out[f"p_up@{tol:g}"] = p_up
        out[f"p_tost@{tol:g}"] = max(p_up, p_low)
        out[f"p_pos@{tol:g}"] = p_pos
    return out


def se_from_interval(lo: float, hi: float, confidence: float = 0.95) -> float:
    return (hi - lo) / (2.0 * _z(confidence))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as the input)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj
