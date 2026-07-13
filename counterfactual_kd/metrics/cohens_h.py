"""Cohen's h — variance-stabilized effect size for paired proportions.

Used as the primary scale-invariant alternative to BSR when A2
(quasi-independence of backdoor / lexical paths) is borderline or
violated, and as a co-primary metric in all conditions per
THEORY.md §2.3-2.4.

Definition.
    h = 2*arcsin(sqrt(p_obs)) - 2*arcsin(sqrt(p_null))

The arcsine-square-root transform variance-stabilizes binomial
proportions: under iid sampling
    Var(2*arcsin(sqrt(p_hat))) ≈ 1/N
exactly (no p(1-p) factor), so the standard error of h is sqrt(1/N1 + 1/N2)
even at the boundaries p -> 0 or p -> 1, where Risk Difference's CI
collapses or extends past [-1, 1] respectively.

Cohen's reference scale (for psychology, used here as a directional cue):
    |h| < 0.2  : negligible
    0.2 -- 0.5 : small
    0.5 -- 0.8 : medium
    |h| >= 0.8 : large

Stdlib-only (no numpy/scipy). Safe to import in the sandbox.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


_NORMAL_95 = 1.959963984540054  # two-sided 95% z-quantile


def _phi(p: float) -> float:
    """Variance-stabilizing transform phi(p) = 2*arcsin(sqrt(p))."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"proportion out of [0,1]: {p}")
    return 2.0 * math.asin(math.sqrt(p))


def cohens_h(
    p_obs: float,
    p_null: float,
    n_obs: int | None = None,
    n_null: int | None = None,
    confidence: float = 0.95,
) -> dict:
    """Compute Cohen's h with optional Wald CI.

    Args:
        p_obs:   ASR_obs (poisoned student)   -- in [0, 1]
        p_null:  ASR_null (clean student)     -- in [0, 1]
        n_obs:   eval-set size for p_obs      (required for CI)
        n_null:  eval-set size for p_null     (required for CI)
        confidence: two-sided confidence level (default 0.95)

    Returns:
        dict with keys: h, magnitude, ci_lower, ci_upper, se,
                        contains_zero, n_obs, n_null, confidence.

        If n_obs or n_null is None, CI fields are None and se is None.
        magnitude is the Cohen 1988 label: negligible/small/medium/large.
    """
    h = _phi(p_obs) - _phi(p_null)
    abs_h = abs(h)
    if abs_h < 0.2:
        magnitude = "negligible"
    elif abs_h < 0.5:
        magnitude = "small"
    elif abs_h < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    out: dict = {
        "h": h,
        "magnitude": magnitude,
        "n_obs": n_obs,
        "n_null": n_null,
        "confidence": confidence,
    }
    if n_obs is None or n_null is None or n_obs <= 0 or n_null <= 0:
        out.update(se=None, ci_lower=None, ci_upper=None, contains_zero=None)
        return out

    # Two-sided z for the requested confidence (closed-form would require
    # erfinv; we only support 0.95 here -- common case in Counterfactual-KD).
    if abs(confidence - 0.95) > 1e-9:
        raise NotImplementedError(
            "Only 95% CI supported (no scipy). Pass confidence=0.95."
        )
    se = math.sqrt(1.0 / n_obs + 1.0 / n_null)
    ci_lower = h - _NORMAL_95 * se
    ci_upper = h + _NORMAL_95 * se
    out.update(
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        contains_zero=ci_lower <= 0.0 <= ci_upper,
    )
    return out


def cohens_h_paired_bootstrap(
    obs_hits: Sequence[int],
    null_hits: Sequence[int],
    R: int = 1000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap CI for Cohen's h when per-sample hit vectors exist.

    Use this when you have the underlying 0/1 prediction-correctness
    arrays for the same triggered eval samples under both poisoned and
    clean students (i.e., paired). Falls back to Wald approximation when
    you only have proportions+N (use cohens_h() in that case).

    Args:
        obs_hits:  parallel list of 0/1, hit=1 when poisoned student
                   outputs target_label on triggered sample i
        null_hits: same length, for clean student
        R:         bootstrap resamples
        seed:      RNG seed

    Returns:
        dict: {h, ci_lower, ci_upper, contains_zero, R, n}
    """
    import random
    if len(obs_hits) != len(null_hits):
        raise ValueError("obs_hits and null_hits must be same length (paired)")
    n = len(obs_hits)
    if n == 0:
        raise ValueError("empty hit vectors")

    p_obs = sum(obs_hits) / n
    p_null = sum(null_hits) / n
    point = _phi(p_obs) - _phi(p_null)

    rng = random.Random(seed)
    boot_h: list[float] = []
    for _ in range(R):
        idx = [rng.randrange(n) for _ in range(n)]
        po = sum(obs_hits[i] for i in idx) / n
        pn = sum(null_hits[i] for i in idx) / n
        boot_h.append(_phi(po) - _phi(pn))
    boot_h.sort()
    lo_k = max(0, int(0.025 * (R - 1)))
    hi_k = min(R - 1, int(0.975 * (R - 1)))
    ci_lower = boot_h[lo_k]
    ci_upper = boot_h[hi_k]
    return {
        "h": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "contains_zero": ci_lower <= 0.0 <= ci_upper,
        "R": R,
        "n": n,
    }


def rope_verdict(
    obs_hits: Sequence[int],
    null_hits: Sequence[int],
    R: int = 5000,
    seed: int = 0,
    rope: float = 0.2,
    sat: float = 0.97,
) -> dict:
    """Counterfactual-KD v17 verdict — Bayesian ROPE on the matched-ATE Cohen's h.

    Single metric (Cohen's h on the matched poisoned/clean trigger response),
    single uncertainty object (paired bootstrap of h over the matched samples),
    single decision (where the 95% HDI sits relative to the ROPE). This is a
    decision *tree on one statistic*, NOT a conjunction of independent tests.

    Args:
        obs_hits:  0/1 per triggered sample, 1 = poisoned student -> target.
        null_hits: 0/1 same samples, clean (counterfactual) student -> target.
        R:         bootstrap resamples (paper default 5000).
        seed:      RNG seed (poisoned/clean resampled with the SAME indices).
        rope:      half-width of the Region Of Practical Equivalence on the
                   Cohen's-h scale (0.2 = Cohen negligible boundary).
        sat:       both-rates saturation threshold (decomposition unstable).

    Returns dict:
        p_trig, c_trig, h, hdi_lo, hdi_hi, p_transfer, magnitude, verdict, n, R.

        p_transfer = Pr_boot[|h| > rope]  in [0,1]  (the BSR-replacement scalar).
        verdict in {CORPUS-SATURATED, NO TRANSFER, SMALL/MEDIUM/STRONG TRANSFER,
                    INCONCLUSIVE}.
    """
    import random
    if len(obs_hits) != len(null_hits):
        raise ValueError("obs_hits and null_hits must be same length (paired)")
    n = len(obs_hits)
    if n == 0:
        raise ValueError("empty hit vectors")

    p_trig = sum(obs_hits) / n
    c_trig = sum(null_hits) / n
    h_point = _phi(p_trig) - _phi(c_trig)

    rng = random.Random(seed)
    boot_h: list[float] = []
    outside = 0
    for _ in range(R):
        idx = [rng.randrange(n) for _ in range(n)]
        po = sum(obs_hits[i] for i in idx) / n
        pn = sum(null_hits[i] for i in idx) / n
        bh = _phi(po) - _phi(pn)
        boot_h.append(bh)
        if abs(bh) > rope:
            outside += 1
    boot_h.sort()
    lo_k = max(0, int(0.025 * (R - 1)))
    hi_k = min(R - 1, int(0.975 * (R - 1)))
    hdi_lo, hdi_hi = boot_h[lo_k], boot_h[hi_k]
    p_transfer = outside / R

    abs_h = abs(h_point)
    if abs_h < 0.2:
        magnitude = "negligible"
    elif abs_h < 0.5:
        magnitude = "small"
    elif abs_h < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    # Verdict ladder — single read-off on h's HDI.
    if p_trig > sat and c_trig > sat:
        verdict = "CORPUS-SATURATED"
    elif -rope <= hdi_lo and hdi_hi <= rope:
        # Whole HDI inside ROPE -> equivalence established (positive null).
        verdict = "NO TRANSFER"
    elif hdi_lo > rope or hdi_hi < -rope:
        # HDI excludes ROPE -> real effect; grade by |h| Cohen band.
        if abs_h >= 0.8:
            verdict = "STRONG TRANSFER"
        elif abs_h >= 0.5:
            verdict = "MEDIUM TRANSFER"
        else:
            verdict = "SMALL TRANSFER"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "p_trig": p_trig,
        "c_trig": c_trig,
        "h": h_point,
        "hdi_lo": hdi_lo,
        "hdi_hi": hdi_hi,
        "p_transfer": p_transfer,
        "magnitude": magnitude,
        "verdict": verdict,
        "n": n,
        "R": R,
    }
