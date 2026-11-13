"""Calibration study of the audit verdict (TIFS plan S1).

Question: when does a matched-pair verdict come out right, and when does
it fail? Each simulated cell is a paired 2x2 table of hits on the same n
non-target inputs,

                      clean student hit
                        1        0
    poisoned     1    pi11     pi10
    student hit  0    pi01     pi00

drawn as Multinomial(n, pi). The paired percentile bootstrap that the
published verdict uses resamples inputs with replacement; on counts that
is exactly Multinomial(n, pi_hat), so it is simulated without expanding
the hit vectors.

Procedures compared (all at 95%):

* ``rope_h``     — published ladder: percentile bootstrap interval on
                   Cohen's h, tolerance 0.2 on h (``rope_verdict``).
* ``boot_k``     — the same bootstrap, on the raw rate difference k.
* ``tango_k``    — Tango (1998) score interval on k.
* ``newcombe_k`` — Newcombe (1998) method-10 interval on k.

Each interval is read against a symmetric tolerance on its own scale:
EQUIVALENT (inside), POSITIVE / NEGATIVE (entirely beyond), otherwise
INCONCLUSIVE. Reported per scenario: interval coverage and width, and the
verdict frequencies classified against the true parameter:

* false_equiv — EQUIVALENT although |true| > tolerance
* false_dir   — POSITIVE/NEGATIVE although |true| <= tolerance, or the
                opposite sign of a true effect beyond tolerance
* power_equiv — EQUIVALENT when |true| <= tolerance / 2 (well inside)
* power_dir   — correct-sign POSITIVE/NEGATIVE when |true| > tolerance
* inconclusive

with binomial Monte Carlo standard errors.

The interval code is vectorised over replicates and checked against the
scalar implementations in ``counterfactual_kd.metrics.paired``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Z95 = 1.959963984540054
PROCEDURES = ("rope_h", "boot_k", "tango_k", "newcombe_k")


# ── Vectorised intervals on count arrays ─────────────────────────────

def _phi(p):
    return 2.0 * np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0)))


def tango_ci(n10, n01, n, z=Z95, iters=80):
    """Tango score interval for k, vectorised over arrays of counts."""
    n10 = np.asarray(n10, dtype=float)
    n01 = np.asarray(n01, dtype=float)
    n = np.broadcast_to(np.asarray(n, dtype=float), n10.shape)
    k_hat = (n10 - n01) / n
    eps = 1e-12

    def score(d):
        a = 2.0 * n
        b = -n10 - n01 + (2.0 * n - n10 + n01) * d
        c = -n01 * d * (1.0 - d)
        disc = np.maximum(b * b - 4.0 * a * c, 0.0)
        p01 = (np.sqrt(disc) - b) / (2.0 * a)
        var = n * (2.0 * p01 + d * (1.0 - d))
        num = n10 - n01 - n * d
        with np.errstate(divide="ignore", invalid="ignore"):
            s = num / np.sqrt(var)
        return np.where(var > 0, s, np.where(num == 0, 0.0, np.sign(num) * np.inf))

    def solve(target, lo, hi):
        lo, hi = lo.copy(), hi.copy()
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            above = score(mid) > target
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)
        return (lo + hi) / 2.0

    lo_edge = np.full_like(k_hat, -1.0 + eps)
    hi_edge = np.full_like(k_hat, 1.0 - eps)
    lower = np.where(score(lo_edge) <= z, -1.0, solve(z, lo_edge, k_hat))
    upper = np.where(score(hi_edge) >= -z, 1.0, solve(-z, k_hat, hi_edge))
    return lower, upper


def _wilson(x, n, z=Z95):
    p = x / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = np.where(x == 0, 0.0, np.maximum(0.0, centre - half))
    hi = np.where(x == n, 1.0, np.minimum(1.0, centre + half))
    return lo, hi


def newcombe_ci(n11, n10, n01, n00, z=Z95):
    """Newcombe method-10 interval for k, vectorised."""
    n11, n10, n01, n00 = (np.asarray(v, dtype=float) for v in (n11, n10, n01, n00))
    n = n11 + n10 + n01 + n00
    x1, x2 = n11 + n10, n11 + n01
    p1, p2 = x1 / n, x2 / n
    l1, u1 = _wilson(x1, n, z)
    l2, u2 = _wilson(x2, n, z)
    margins = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(margins > 0, (n11 * n00 - n10 * n01) / np.sqrt(margins), 0.0)
    d_lo = np.sqrt(np.maximum((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    d_hi = np.sqrt(np.maximum((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    k = p1 - p2
    return np.maximum(-1.0, k - d_lo), np.minimum(1.0, k + d_hi)


def bootstrap_ci(table, rng, R=2000):
    """Paired percentile bootstrap on one table -> ((h_lo, h_hi), (k_lo, k_hi)).

    Index rule as ``rope_verdict``: sorted[int(0.025 (R-1))], sorted[int(0.975 (R-1))].
    """
    n = int(sum(table))
    draws = rng.multinomial(n, np.asarray(table, dtype=float) / n, size=R)
    pt = (draws[:, 0] + draws[:, 1]) / n
    ct = (draws[:, 0] + draws[:, 2]) / n
    h = np.sort(_phi(pt) - _phi(ct))
    k = np.sort(pt - ct)
    lo_i, hi_i = int(0.025 * (R - 1)), int(0.975 * (R - 1))
    return (h[lo_i], h[hi_i]), (k[lo_i], k[hi_i])


# ── Verdicts ─────────────────────────────────────────────────────────

def verdict(lower, upper, tol):
    lower, upper = np.asarray(lower), np.asarray(upper)
    out = np.full(lower.shape, "INCONCLUSIVE", dtype=object)
    out[(lower >= -tol) & (upper <= tol)] = "EQUIVALENT"
    out[lower > tol] = "POSITIVE"
    out[upper < -tol] = "NEGATIVE"
    return out


def classify(verdicts, true_value, tol):
    """Verdict frequencies against the truth, with MC standard errors."""
    v = np.asarray(verdicts)
    m = len(v)
    a = abs(true_value)
    sign_ok = "POSITIVE" if true_value > 0 else "NEGATIVE"
    sign_bad = "NEGATIVE" if true_value > 0 else "POSITIVE"
    equiv = np.mean(v == "EQUIVALENT")
    direc = np.mean((v == "POSITIVE") | (v == "NEGATIVE"))
    out = {
        "equivalent": equiv,
        "positive": float(np.mean(v == "POSITIVE")),
        "negative": float(np.mean(v == "NEGATIVE")),
        "inconclusive": float(np.mean(v == "INCONCLUSIVE")),
        "false_equiv": float(equiv) if a > tol else 0.0,
        "false_dir": (float(direc) if a <= tol
                      else float(np.mean(v == sign_bad))),
        "power_equiv": float(equiv) if a <= tol / 2 else float("nan"),
        "power_dir": float(np.mean(v == sign_ok)) if a > tol else float("nan"),
    }
    out["mc_se_max"] = math.sqrt(0.25 / m)
    return out


# ── Scenarios ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Scenario:
    """A paired-table generating distribution."""
    name: str
    n: int
    pi11: float
    pi10: float
    pi01: float

    @property
    def pi00(self) -> float:
        return 1.0 - self.pi11 - self.pi10 - self.pi01

    @property
    def probs(self) -> np.ndarray:
        p = np.array([self.pi11, self.pi10, self.pi01, self.pi00])
        if np.any(p < -1e-12) or abs(p.sum() - 1.0) > 1e-9:
            raise ValueError(f"infeasible cell probabilities for {self.name}: {p}")
        p = np.clip(p, 0.0, 1.0)
        return p / p.sum()

    @property
    def k(self) -> float:
        return self.pi10 - self.pi01

    @property
    def h(self) -> float:
        pt, ct = self.pi11 + self.pi10, self.pi11 + self.pi01
        return float(_phi(pt) - _phi(ct))


def scenario_from_rates(name, n, c_trig, k, discord):
    """Scenario with clean rate c_trig, effect k and discordance pi10+pi01."""
    pi10 = (discord + k) / 2.0
    pi01 = (discord - k) / 2.0
    pi11 = c_trig - pi01
    return Scenario(name, n, pi11, pi10, pi01)


def run_scenario(sc: Scenario, *, reps: int, R: int, seed: int,
                 tol_h: float = 0.2, tols_k: tuple[float, ...] = (0.05,)) -> dict:
    """Simulate one scenario and score every procedure.

    Intervals are computed once per replicate and read against every
    tolerance (``rope_h`` against ``tol_h`` on the h scale; the k-scale
    procedures against each of ``tols_k``). Keys: ``<proc>.coverage``,
    ``<proc>.width``, ``<proc>.degenerate`` and ``<proc>.<metric>@<tol>``.
    """
    rng = np.random.default_rng(seed)
    tables = rng.multinomial(sc.n, sc.probs, size=reps)
    n11, n10, n01, n00 = tables.T

    t_lo, t_hi = tango_ci(n10, n01, sc.n)
    nc_lo, nc_hi = newcombe_ci(n11, n10, n01, n00)
    bh = np.empty((reps, 2))
    bk = np.empty((reps, 2))
    for i in range(reps):
        bh[i], bk[i] = bootstrap_ci(tables[i], rng, R)

    intervals = {
        "rope_h": (bh[:, 0], bh[:, 1], sc.h, (tol_h,)),
        "boot_k": (bk[:, 0], bk[:, 1], sc.k, tols_k),
        "tango_k": (t_lo, t_hi, sc.k, tols_k),
        "newcombe_k": (nc_lo, nc_hi, sc.k, tols_k),
    }
    out = {"scenario": sc.name, "n": sc.n, "pi11": sc.pi11, "pi10": sc.pi10,
           "pi01": sc.pi01, "true_k": sc.k, "true_h": sc.h, "reps": reps,
           "R": R, "mc_se_max": math.sqrt(0.25 / reps)}
    for name, (lo, hi, truth, tols) in intervals.items():
        out[f"{name}.coverage"] = float(np.mean((lo <= truth) & (truth <= hi)))
        out[f"{name}.width"] = float(np.mean(hi - lo))
        out[f"{name}.degenerate"] = float(np.mean(hi - lo == 0.0))
        for tol in tols:
            for key, val in classify(verdict(lo, hi, tol), truth, tol).items():
                if key != "mc_se_max":
                    out[f"{name}.{key}@{tol:g}"] = val
    return out
