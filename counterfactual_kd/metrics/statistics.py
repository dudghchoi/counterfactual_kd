"""
Statistical testing for Counterfactual-KD evaluation.

Provides Cohen's d, permutation tests, and bootstrap CIs
to validate BSR results across multiple seeds.
"""

import numpy as np
from typing import Optional


def cohens_d(group1: list[float], group2: Optional[list[float]] = None) -> float:
    """Compute Cohen's d effect size.

    If group2 is None, computes one-sample d (testing if group1 differs from 0).
    Otherwise computes two-sample d.

    Interpretation (Cohen's convention):
        |d| < 0.2  -> negligible
        0.2 - 0.5  -> small
        0.5 - 0.8  -> medium
        |d| > 0.8  -> large
    """
    a = np.array(group1, dtype=float)

    if group2 is None:
        if len(a) < 2:
            return 0.0
        return float(np.mean(a) / np.std(a, ddof=1)) if np.std(a, ddof=1) > 0 else 0.0

    b = np.array(group2, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0

    pooled_std = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def permutation_test(
    obs_values: list[float],
    null_values: list[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """Permutation test: is ASR_obs significantly greater than ASR_null?

    Tests H0: poisoned and clean students have the same trigger response.

    Args:
        obs_values: ASR_obs values across seeds (or per-sample predictions)
        null_values: ASR_null values across seeds
        n_permutations: number of permutations
        seed: random seed

    Returns:
        dict with observed_diff, p_value, significant (at 0.05)
    """
    rng = np.random.default_rng(seed)
    obs = np.array(obs_values, dtype=float)
    null = np.array(null_values, dtype=float)

    observed_diff = float(np.mean(obs) - np.mean(null))

    pooled = np.concatenate([obs, null])
    n_obs = len(obs)
    count = 0

    for _ in range(n_permutations):
        rng.shuffle(pooled)
        perm_diff = np.mean(pooled[:n_obs]) - np.mean(pooled[n_obs:])
        if perm_diff >= observed_diff:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return {
        "observed_diff": observed_diff,
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_permutations": n_permutations,
    }


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for BSR.

    Args:
        values: BSR values across seeds
        confidence: confidence level (default 0.95)
        n_bootstrap: number of bootstrap samples

    Returns:
        dict with mean, ci_lower, ci_upper, contains_zero
    """
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    n = len(arr)

    if n < 2:
        mean_val = float(np.mean(arr))
        return {
            "mean": mean_val,
            "ci_lower": mean_val,
            "ci_upper": mean_val,
            "contains_zero": mean_val == 0.0,
        }

    boot_means = np.array([
        np.mean(rng.choice(arr, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean_val = float(np.mean(arr))

    return {
        "mean": mean_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "contains_zero": ci_lower <= 0 <= ci_upper,
    }


def mcnemar_test(preds_obs: list[int], preds_null: list[int]) -> dict:
    """McNemar's test on paired per-sample hits.

    ``preds_obs`` and ``preds_null`` are parallel 0/1 vectors over the *same*
    triggered samples — hit=1 when a model outputs target_label. Tests H0:
    poisoned and clean students classify triggered inputs equivalently.

    Returns mid-p approximation using scipy-free binomial logic; suitable as
    a per-condition significance check when you have per-sample predictions
    instead of just seed-level means.
    """
    import math
    if len(preds_obs) != len(preds_null):
        raise ValueError("preds_obs and preds_null must be same length")
    b = sum(1 for o, n in zip(preds_obs, preds_null) if o == 1 and n == 0)
    c = sum(1 for o, n in zip(preds_obs, preds_null) if o == 0 and n == 1)
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "p_value": 1.0, "significant": False}
    # Exact two-sided binomial p-value, H0: discordant pairs split 50/50.
    k = min(b, c)
    p_one = sum(math.comb(n_discordant, i) for i in range(k + 1)) / (2 ** n_discordant)
    p_two = min(1.0, 2 * p_one)
    return {
        "b": b,
        "c": c,
        "n_discordant": n_discordant,
        "p_value": float(p_two),
        "significant": p_two < 0.05,
    }


def enhanced_verdict(
    bsr_values: list[float],
    asr_true_values: list[float],
    asr_obs_values: list[float],
    asr_null_values: list[float],
) -> dict:
    """Enhanced Counterfactual-KD verdict with statistical backing.

    Combines BSR thresholds with Cohen's d and permutation test.

    Returns:
        dict with verdict, mean_bsr, cohens_d, p_value, ci
    """
    mean_bsr = float(np.mean(bsr_values))
    mean_asr_true = float(np.mean(asr_true_values))

    d = cohens_d(bsr_values)
    perm = permutation_test(asr_obs_values, asr_null_values)
    ci = bootstrap_ci(bsr_values)

    if mean_asr_true > 0.30 and mean_bsr > 0.5 and abs(d) > 0.8 and perm["p_value"] < 0.01:
        verdict = "CONFIRMED TRANSFER"
    elif mean_asr_true > 0.10 and mean_bsr > 0.3 and abs(d) > 0.5 and perm["p_value"] < 0.05:
        verdict = "WEAK TRANSFER"
    elif mean_asr_true > 0.05 and mean_bsr > 0.1 and (abs(d) > 0.2 or perm["p_value"] < 0.05):
        verdict = "MARGINAL"
    elif mean_asr_true <= 0:
        verdict = "NEGATIVE"
    else:
        verdict = "NO TRANSFER"

    return {
        "verdict": verdict,
        "mean_bsr": mean_bsr,
        "mean_asr_true": mean_asr_true,
        "cohens_d": d,
        "p_value": perm["p_value"],
        "ci_lower": ci["ci_lower"],
        "ci_upper": ci["ci_upper"],
        "ci_contains_zero": ci["contains_zero"],
    }
