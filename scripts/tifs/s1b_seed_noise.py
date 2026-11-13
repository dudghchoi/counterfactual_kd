#!/usr/bin/env python3
"""S1b — operating characteristics of the primary rule under training noise.

S1 (s1_calibration.py) calibrates the per-cell Tango verdict when input
sampling is the only noise. S1b adds student-training noise and compares
the decision rules of docs/tifs_plan.md §0.3 on whole batteries.

Model (normal approximation; S1 covers the exact paired-binary behaviour).
A battery has m attacks, r student seeds, one C student per seed shared by
every attack, and the same n evaluation inputs for every seed:

    k[a, s] = mu[a] + u[a, s] - c[s] + e[a, s]
    u ~ N(0, tau^2 / 2), c ~ N(0, tau^2 / 2)     training noise; the shared C
                                                 correlates attacks at 0.5
    e[a, .] ~ N(0, sig^2), corr rho across seeds (same inputs)

with sig^2 = discordance / n, the input variance of one seed's k. Rules:

  tango1       seed 42 only, k +- z sig (the v2 rule before §0.3)
  transported  seed 42 only, k +- t_df sqrt(sig^2 + tau_hat^2), tau_hat^2 from
               another battery with df_tau degrees of freedom
  pooled_mr    kbar +- t_{m(r-1)} sqrt(s2_pool / r + v), v = sig^2 (1 + (r - 1)
               rho) / r (the analyst's var_i(dbar)/n): the first draft of §0.3,
               which ignores that the attacks of a seed share its C student
  pooled       §0.3 primary: the same SE with Satterthwaite df from the
               two-way (attack x seed) decomposition of s2_pool
  modhk        per-cell modified Hartung-Knapp over the r seeds (df r - 1)

Battery decisions as analyze.py: NO TRANSFER iff every one-sided upper
bound is below delta; TRANSFERS iff a Holm-adjusted p(k > delta) < 0.025.
Reported per scenario: cell coverage of mu, false "below delta" at
mu = delta, P(NO TRANSFER) when all mu = 0, P(false TRANSFERS) when all
mu = 0, and P(TRANSFERS) when one attack has mu = 2 delta.

    python scripts/tifs/s1b_seed_noise.py --reps 40000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np
from scipy.stats import norm
from scipy.stats import t as tdist

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ALPHA = 0.025

# (label, n, discordance, m, r): the v2 batteries (SST-2 n = 912, AG News 5,700).
BATTERIES = [
    ("BERT SST-2", 912, 0.05, 6, 3),
    ("BERT SST-2, high discordance", 912, 0.10, 6, 3),
    ("GPT-2 SST-2", 912, 0.05, 2, 3),
    ("Qwen3 SST-2", 912, 0.05, 3, 3),
    ("BERT AG News", 5700, 0.015, 6, 3),
]
TAUS = (0.0, 0.005, 0.01, 0.02, 0.03)
RHOS = (0.0, 0.3, 0.6)
DELTAS = (0.02, 0.05)
DF_TAU = 12


def holm_any(p: np.ndarray) -> np.ndarray:
    """Rows where Holm rejects at least one hypothesis (= min p * m < alpha)."""
    return p.min(axis=1) * p.shape[1] < ALPHA


def simulate(rng, reps, n, disc, m, r, tau, rho, mu):
    sig = math.sqrt(disc / n)
    u = rng.normal(0, tau / math.sqrt(2), (reps, m, r))
    c = rng.normal(0, tau / math.sqrt(2), (reps, 1, r))
    cov = sig ** 2 * (rho * np.ones((r, r)) + (1 - rho) * np.eye(r))
    e = rng.normal(0, 1, (reps, m, r)) @ np.linalg.cholesky(cov + 1e-18 * np.eye(r)).T
    k = mu[None, :, None] + u - c + e
    kbar = k.mean(axis=2)
    s2 = k.var(axis=2, ddof=1)
    v = sig ** 2 * (1 + (r - 1) * rho) / r
    # The borrowed s2_pool estimates tau^2 plus the seed-specific part of the
    # input noise, sig^2 (1 - rho), of a battery like this one.
    tau_hat2 = (tau ** 2 + sig ** 2 * (1 - rho)) * rng.chisquare(DF_TAU, (reps, 1)) / DF_TAU
    # Two-way decomposition (attack x seed, no replication): s2_pool equals
    # (MS_seed + (m - 1) MS_res) / m; Satterthwaite df for that combination.
    ks = k.mean(axis=1)
    kg = k.mean(axis=(1, 2))
    ms_seed = m * ((ks - kg[:, None]) ** 2).sum(axis=1) / (r - 1)
    if m > 1:
        res = k - kbar[:, :, None] - ks[:, None, :] + kg[:, None, None]
        ms_res = (res ** 2).sum(axis=(1, 2)) / ((m - 1) * (r - 1))
    else:
        ms_res = np.zeros(reps)
    comb = ms_seed + (m - 1) * ms_res
    df_sw = comb ** 2 * (r - 1) / np.maximum(ms_seed ** 2 + (m - 1) * ms_res ** 2, 1e-300)
    tau2_dl = np.maximum(0.0, s2 - sig ** 2)
    var_mhk = (sig ** 2 + tau2_dl) / r * np.maximum(1.0, s2 / (sig ** 2 + tau2_dl + 1e-30))
    return {
        "tango1": (k[:, :, 0], np.full((reps, m), sig), math.inf),
        "transported": (k[:, :, 0], np.sqrt(sig ** 2 + tau_hat2) * np.ones((reps, m)), DF_TAU),
        "pooled_mr": (kbar, np.sqrt(s2.mean(axis=1, keepdims=True) / r + v) * np.ones((reps, m)),
                      m * (r - 1)),
        "pooled": (kbar, np.sqrt(comb[:, None] / m / r + v) * np.ones((reps, m)),
                   df_sw[:, None] * np.ones((reps, m))),
        "modhk": (kbar, np.sqrt(var_mhk), r - 1),
    }


def rates(est, se, df, mu, delta):
    inf = np.isscalar(df) and math.isinf(df)
    q = norm.ppf(1 - ALPHA) if inf else tdist.ppf(1 - ALPHA, df)
    lo, hi = est - q * se, est + q * se
    cdf = norm.cdf if inf else (lambda x: tdist.cdf(x, df))
    p_up = cdf((est - delta) / se)            # H0: k >= delta
    p_pos = 1 - cdf((est - delta) / se)       # H0: k <= delta
    return {"cover": float(np.mean((lo <= mu[None, :]) & (mu[None, :] <= hi))),
            "below_first": float(np.mean(hi[:, 0] < delta)),
            "no_transfer": float(np.mean((p_up < ALPHA).all(axis=1))),
            "transfers": float(np.mean(holm_any(p_pos)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tifs", "s1"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for label, n, disc, m, r in BATTERIES:
        for rho in RHOS:
            for tau in TAUS:
                for delta in DELTAS:
                    zero = np.zeros(m)
                    edge = zero.copy(); edge[0] = delta
                    big = zero.copy(); big[0] = 2 * delta
                    sims = {name: simulate(rng, args.reps, n, disc, m, r, tau, rho, mu)
                            for name, mu in (("zero", zero), ("edge", edge), ("big", big))}
                    for rule in ("tango1", "transported", "pooled_mr", "pooled", "modhk"):
                        z = rates(*sims["zero"][rule], zero, delta)
                        e = rates(*sims["edge"][rule], edge, delta)
                        b = rates(*sims["big"][rule], big, delta)
                        rows.append({"battery": label, "n": n, "discordance": disc, "m": m,
                                     "r": 1 if rule in ("tango1", "transported") else r,
                                     "tau": tau, "rho": rho, "delta": delta, "rule": rule,
                                     "coverage": z["cover"],
                                     "false_below_at_mu_eq_delta": e["below_first"],
                                     "p_no_transfer_all_zero": z["no_transfer"],
                                     "p_false_transfers_all_zero": z["transfers"],
                                     "p_transfers_one_at_2delta": b["transfers"]})
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "s1b.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "s1b_meta.json"), "w") as f:
        json.dump({"reps": args.reps, "seed": args.seed, "df_tau": DF_TAU, "alpha": ALPHA,
                   "batteries": BATTERIES, "taus": TAUS, "rhos": RHOS, "deltas": DELTAS}, f, indent=1)
    worst = max((r for r in rows if r["rule"] == "pooled"),
                key=lambda r: r["false_below_at_mu_eq_delta"])
    print(f"{len(rows)} rows -> {path}; worst pooled false 'below delta' at mu = delta: "
          f"{worst['false_below_at_mu_eq_delta']:.4f} ({worst['battery']}, tau={worst['tau']}, "
          f"rho={worst['rho']}, delta={worst['delta']})")


if __name__ == "__main__":
    main()
