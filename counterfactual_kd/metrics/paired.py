"""Paired rate-difference intervals and the two-axis audit verdict.

The matched effect k = P_trig - C_trig is a difference of two proportions
measured on the *same* inputs, so its interval should use the paired 2x2
table of hits rather than two independent samples:

                      clean student hit
                        1        0
    poisoned     1    n11      n10
    student hit  0    n01      n00

    k = (n10 - n01) / n

Intervals (Fagerland, Lydersen & Laake 2014, Stat. Med. 33:2850 recommend
both for paired proportions):

* ``tango``    — score interval of Tango (1998), inverted numerically.
* ``newcombe`` — hybrid score ("square-and-add") interval, Newcombe (1998)
  method 10, with the uncorrected phi coefficient (0 when undefined).

Unlike the percentile bootstrap, neither degenerates when the discordant
cells are empty: with n10 = n01 = 0 the Tango interval is
[-z^2/(n+z^2), +z^2/(n+z^2)].

``audit_verdict`` reports the relative effect against a raw-rate tolerance
``delta`` and, separately, the absolute triggered exposure of each
student, so "equivalent teacher effect, both students highly exposed" is
expressible (the old ladder returned CORPUS-SATURATED instead).

``delta_star`` is the smallest symmetric tolerance at which the interval
would establish equivalence; reporting it lets a reader apply their own
tolerance without one being chosen after seeing the data.

Stdlib only. ``rope_verdict`` (Cohen's h, paired bootstrap) is unchanged
and remains the verdict of the published tables.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


# ── Basics ───────────────────────────────────────────────────────────

def _z(confidence: float) -> float:
    """Two-sided standard-normal quantile for ``confidence``."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    target = 0.5 + confidence / 2.0
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def paired_counts(obs_hits: Sequence[int], null_hits: Sequence[int]
                  ) -> tuple[int, int, int, int]:
    """Return (n11, n10, n01, n00) for aligned 0/1 hit vectors."""
    if len(obs_hits) != len(null_hits):
        raise ValueError("obs_hits and null_hits must be same length (paired)")
    if not obs_hits:
        raise ValueError("empty hit vectors")
    n11 = n10 = n01 = n00 = 0
    for o, c in zip(obs_hits, null_hits):
        if o not in (0, 1) or c not in (0, 1):
            raise ValueError("hit vectors must contain only 0/1")
        if o and c:
            n11 += 1
        elif o:
            n10 += 1
        elif c:
            n01 += 1
        else:
            n00 += 1
    return n11, n10, n01, n00


def wilson_interval(x: int, n: int, confidence: float = 0.95
                    ) -> tuple[float, float]:
    """Wilson score interval for a single proportion x/n."""
    if n <= 0:
        raise ValueError("n must be positive")
    z = _z(confidence)
    p = x / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # The bounds are exactly 0 / 1 at x = 0 / n; avoid rounding residue.
    lower = 0.0 if x == 0 else max(0.0, centre - half)
    upper = 1.0 if x == n else min(1.0, centre + half)
    return lower, upper


# ── Paired difference intervals ──────────────────────────────────────

def _tango_score(n10: int, n01: int, n: int, d0: float) -> float:
    """Tango's score statistic for H0: k = d0 (decreasing in d0)."""
    a = 2.0 * n
    b = -n10 - n01 + (2.0 * n - n10 + n01) * d0
    c = -n01 * d0 * (1.0 - d0)
    disc = max(b * b - 4.0 * a * c, 0.0)
    p01 = (math.sqrt(disc) - b) / (2.0 * a)   # constrained MLE of Pr(cell 01)
    var = n * (2.0 * p01 + d0 * (1.0 - d0))
    num = n10 - n01 - n * d0
    if var <= 0.0:
        return 0.0 if num == 0 else math.copysign(math.inf, num)
    return num / math.sqrt(var)


def tango_paired_ci(obs_hits: Sequence[int], null_hits: Sequence[int],
                    confidence: float = 0.95) -> tuple[float, float]:
    """Tango (1998) score interval for k = P_trig - C_trig."""
    n11, n10, n01, n00 = paired_counts(obs_hits, null_hits)
    n = n11 + n10 + n01 + n00
    z = _z(confidence)
    k_hat = (n10 - n01) / n
    eps = 1e-12

    def solve(target, lo, hi):
        # Bisection for score(d) = target on [lo, hi], score decreasing.
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if _tango_score(n10, n01, n, mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    if _tango_score(n10, n01, n, -1.0 + eps) <= z:
        lower = -1.0
    else:
        lower = solve(z, -1.0 + eps, k_hat)
    if _tango_score(n10, n01, n, 1.0 - eps) >= -z:
        upper = 1.0
    else:
        upper = solve(-z, k_hat, 1.0 - eps)
    return lower, upper


def newcombe_paired_ci(obs_hits: Sequence[int], null_hits: Sequence[int],
                       confidence: float = 0.95) -> tuple[float, float]:
    """Newcombe (1998) method-10 hybrid score interval for k."""
    n11, n10, n01, n00 = paired_counts(obs_hits, null_hits)
    n = n11 + n10 + n01 + n00
    x1, x2 = n11 + n10, n11 + n01            # poisoned / clean hits
    p1, p2 = x1 / n, x2 / n
    l1, u1 = wilson_interval(x1, n, confidence)
    l2, u2 = wilson_interval(x2, n, confidence)
    margins = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    phi = 0.0 if margins == 0 else (n11 * n00 - n10 * n01) / math.sqrt(margins)
    d_lo = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2)
                         + (u2 - p2) ** 2, 0.0))
    d_hi = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2)
                         + (p2 - l2) ** 2, 0.0))
    k = p1 - p2
    return max(-1.0, k - d_lo), min(1.0, k + d_hi)


_INTERVALS = {"tango": tango_paired_ci, "newcombe": newcombe_paired_ci}


def delta_star(lower: float, upper: float) -> float:
    """Smallest symmetric tolerance delta with [lower, upper] in [-delta, delta]."""
    return max(abs(lower), abs(upper))


# ── Verdict ──────────────────────────────────────────────────────────

def audit_verdict(
    obs_hits: Sequence[int],
    null_hits: Sequence[int],
    delta: float,
    method: str = "tango",
    confidence: float = 0.95,
    high_exposure: float = 0.5,
    saturation: float = 0.97,
) -> dict:
    """Two-axis audit verdict on the raw matched effect k.

    Relative axis (teacher-swap effect, interval vs. tolerance ``delta``):
        EQUIVALENT   — interval inside [-delta, delta]
        POSITIVE     — interval entirely above +delta
        NEGATIVE     — interval entirely below -delta
        INCONCLUSIVE — interval crosses a tolerance boundary

    A 95% two-sided interval inside the tolerance corresponds to two
    one-sided tests at the 2.5% level each; pass ``confidence=0.90`` for
    the conventional 5% TOST.

    Absolute axis (each student's triggered target rate, Wilson interval):
        exposure flag HIGH when the interval's lower bound is at least
        ``high_exposure``; ``saturated`` marks both point rates above
        ``saturation``. Both are declared flags, not inferential results,
        and neither overrides the relative verdict.
    """
    if delta < 0:
        raise ValueError("delta must be non-negative")
    if method not in _INTERVALS:
        raise ValueError(f"method must be one of {sorted(_INTERVALS)}")
    n11, n10, n01, n00 = paired_counts(obs_hits, null_hits)
    n = n11 + n10 + n01 + n00
    p_trig, c_trig = (n11 + n10) / n, (n11 + n01) / n
    k = p_trig - c_trig
    lower, upper = _INTERVALS[method](obs_hits, null_hits, confidence)

    if -delta <= lower and upper <= delta:
        relative = "EQUIVALENT"
    elif lower > delta:
        relative = "POSITIVE"
    elif upper < -delta:
        relative = "NEGATIVE"
    else:
        relative = "INCONCLUSIVE"

    p_ci = wilson_interval(n11 + n10, n, confidence)
    c_ci = wilson_interval(n11 + n01, n, confidence)

    def flag(ci):
        return "HIGH" if ci[0] >= high_exposure else "-"

    h = 2.0 * (math.asin(math.sqrt(p_trig)) - math.asin(math.sqrt(c_trig)))
    return {
        "n": n,
        "counts": {"n11": n11, "n10": n10, "n01": n01, "n00": n00},
        "p_trig": p_trig,
        "c_trig": c_trig,
        "k": k,
        "ci_lower": lower,
        "ci_upper": upper,
        "method": method,
        "confidence": confidence,
        "delta": delta,
        "delta_star": delta_star(lower, upper),
        "excludes_zero": lower > 0.0 or upper < 0.0,
        "relative": relative,
        "h": h,
        "p_trig_ci": p_ci,
        "c_trig_ci": c_ci,
        "p_exposure": flag(p_ci),
        "c_exposure": flag(c_ci),
        "saturated": p_trig > saturation and c_trig > saturation,
    }


# ── Decomposition diagnostics ────────────────────────────────────────

def decomposition_bootstrap(
    p_trig: Sequence[int],
    c_trig: Sequence[int],
    p_clean: Sequence[int],
    c_clean: Sequence[int],
    R: int = 5000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Percentile intervals for k, Δb and k* that keep all pairings.

    All four hit vectors are indexed by the same non-target input, so each
    resample draws input indices once and reuses them for every vector:
    triggered/untriggered and poisoned/clean stay paired.

        k  = P_trig - C_trig
        Δb = P_clean - C_clean
        k* = (P_trig - P_clean) - (C_trig - C_clean)      (k = Δb + k*)
    """
    vecs = (p_trig, c_trig, p_clean, c_clean)
    n = len(p_trig)
    if n == 0 or any(len(v) != n for v in vecs):
        raise ValueError("all four hit vectors must be non-empty and aligned")

    def stats(idx):
        pt = sum(p_trig[i] for i in idx) / n
        ct = sum(c_trig[i] for i in idx) / n
        pc = sum(p_clean[i] for i in idx) / n
        cc = sum(c_clean[i] for i in idx) / n
        return pt - ct, pc - cc, (pt - pc) - (ct - cc)

    point = stats(range(n))
    rng = random.Random(seed)
    draws = [stats([rng.randrange(n) for _ in range(n)]) for _ in range(R)]
    alpha = 1.0 - confidence
    lo_k = max(0, int((alpha / 2) * (R - 1)))
    hi_k = min(R - 1, int((1 - alpha / 2) * (R - 1)))
    out = {"n": n, "R": R, "confidence": confidence}
    for j, name in enumerate(("k", "delta_b", "k_star")):
        col = sorted(d[j] for d in draws)
        out[name] = point[j]
        out[f"{name}_ci"] = (col[lo_k], col[hi_k])
    return out
