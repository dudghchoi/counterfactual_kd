"""Shared helpers for the A11 items 10 (S1c), 11 (S9) and 12 (S1 extra rows).

Post hoc / registered auxiliary analysis (docs/tifs_addendum.md A11). Nothing
here changes a primary verdict. Read-only access to the repository: the frozen
functions are imported (analyze.py, addendum.py, analysis/tifs.py,
metrics/paired.py, analysis/calibration.py), never modified, and no bytecode is
written (sys.dont_write_bytecode + PYTHONDONTWRITEBYTECODE=1).

Paths are repo-relative (ROOT from this file). Inputs: runs/tifs (through the
frozen analyze.load_cells via addendum.State), results/tifs/report (frozen
analyze.py output), results/tifs/addendum (addendum.py output) and
results/tifs/s1 (archived S1/S1b). Outputs: results/tifs/addendum/aux/<GROUP>/
unless a script is given --out.
"""

from __future__ import annotations

import csv
import datetime
import glob
import math
import os
import resource
import sys
import time

sys.dont_write_bytecode = True

GROUP = "g4-s1c-s9-s1extra"
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))      # repo root
RUNS = os.path.join(R, "runs", "tifs")
REPORT = os.path.join(R, "results", "tifs", "report")                 # frozen analyze.py output
ADDENDUM = os.path.join(R, "results", "tifs", "addendum")             # addendum.py output
S1_DIR = os.path.join(R, "results", "tifs", "s1")                     # archived S1 / S1b
OUT_DEFAULT = os.path.join(ADDENDUM, "aux", GROUP)

for p in (os.path.join(R, "scripts", "tifs"), R):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import analyze as A  # noqa: E402  (frozen)
import addendum as AD  # noqa: E402  (registered addendum code; functions only)
from counterfactual_kd.analysis.tifs import (  # noqa: E402  (frozen)
    holm, paired_contrast, se_from_interval, t_contrast,
)

TOLS = A.TOLS            # (0.02, 0.05, 0.10)
ALPHA = A.ALPHA          # 0.025 one-sided
STATUS = ("Post hoc / registered auxiliary table (docs/tifs_addendum.md A11); "
          "it does not change any primary verdict, battery decision or the paper-wide Holm.")


def add_out_arg(ap):
    ap.add_argument("--out", default=OUT_DEFAULT,
                    help=f"output folder (default: results/tifs/addendum/aux/{GROUP})")
    return ap


def write_csv(rows, path, comment):
    """CSV with a one-line '# ...' header comment (definition + status)."""
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("# " + comment.replace("\n", " ") + " " + STATUS + "\n")
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in keys})


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if math.isnan(v):
            return ""
        return repr(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def read_csv(path):
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def mc_se(p, reps):
    return math.sqrt(max(p * (1 - p), 0.0) / reps)


def z_two_prop(p1, n1, p2, n2):
    """Two-proportion z statistic with the pooled proportion (0 when both are 0 or 1)."""
    pbar = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(max(pbar * (1 - pbar), 0.0) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else 0.0


# ── data snapshot and run cost ───────────────────────────────────────


def snapshot():
    """The prediction files the frozen load_cells parses (same glob, run-id filter
    and .json companion rule) and the newest of their modification times."""
    paths = []
    for p in sorted(glob.glob(os.path.join(RUNS, "tifs_*", "artifacts", "results", "*.preds.json"))):
        if A.parse_run(p.split(os.sep)[-4]) is None:
            continue
        if not os.path.exists(p[: -len(".preds.json")] + ".json"):
            continue
        paths.append(p)
    newest = max(paths, key=os.path.getmtime)
    ts = datetime.datetime.fromtimestamp(os.path.getmtime(newest), datetime.timezone.utc)
    return {"n_preds": len(paths), "n_runs": len({p.split(os.sep)[-4] for p in paths}),
            "newest_utc": ts.strftime("%Y-%m-%d %H:%M:%S UTC"), "newest_path": os.path.relpath(newest, R)}


def log_cost(out, script, t0_wall):
    """Append wall time and CPU time (this process + its reaped workers) to logs/cpu_seconds.csv."""
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime
    path = os.path.join(out, "logs", "cpu_seconds.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("script,wall_s,cpu_s,end_utc\n")
        f.write(f"{script},{time.time() - t0_wall:.1f},{cpu:.1f},"
                f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"[{script}] wall {time.time() - t0_wall:.1f}s, cpu {cpu:.1f}s", flush=True)


# ── exact fast drop-in for scipy.stats.t inside the frozen t_contrast ─


class ExactFastT:
    """Stand-in for ``scipy.stats.t`` while the frozen ``t_contrast`` runs in the
    Monte Carlo loops. scipy 1.13's ``t._cdf``, ``t._sf`` and ``t._ppf`` are
    ``special.stdtr(df, x)``, ``special.stdtr(df, -x)`` and ``special.stdtrit(df, q)``
    with loc = 0 and scale = 1; this class calls those same special functions
    directly and skips the ``rv_continuous`` argument handling (about 95% of the
    time spent in t_contrast: 4,000 calls take 0.16 s instead of 3.39 s; whole S1c
    jobs ran about 10 times faster). For finite x, finite df > 0 and 0 < q < 1 the values are
    bit-identical (checks.py compares them on random inputs); anything else is
    passed to the stock distribution. ``t_contrast`` itself is called unchanged."""

    import scipy.special as _sc
    import scipy.stats as _st
    _stock = _st.t

    @classmethod
    def _ok(cls, x, df):
        return (isinstance(x, float) and isinstance(df, float) and math.isfinite(x)
                and math.isfinite(df) and df > 0)

    @classmethod
    def cdf(cls, x, df):
        return cls._sc.stdtr(df, x) if cls._ok(x, df) else cls._stock.cdf(x, df)

    @classmethod
    def sf(cls, x, df):
        return cls._sc.stdtr(df, -x) if cls._ok(x, df) else cls._stock.sf(x, df)

    @classmethod
    def ppf(cls, q, df):
        return cls._sc.stdtrit(df, q) if (cls._ok(q, df) and 0.0 < q < 1.0) else cls._stock.ppf(q, df)


def install_exact_fast_t(enable=True):
    """Route ``from scipy.stats import t`` (done inside the frozen t_contrast at each
    call) to ExactFastT. Returns the previous object so it can be restored."""
    import scipy.stats
    prev = scipy.stats.t
    if enable:
        scipy.stats.t = ExactFastT
    return prev


def restore_t(prev):
    import scipy.stats
    scipy.stats.t = prev


# ── frozen state and plug-in models ──────────────────────────────────


_STATE = None


def state():
    """The frozen analysis state (addendum.State over runs/tifs)."""
    global _STATE
    if _STATE is None:
        _STATE = AD.State(RUNS)
    return _STATE


def dvec(cell):
    return np.asarray(cell["P"], dtype=float) - np.asarray(cell["C"], dtype=float)


def battery_plugin(st, fam, ds, kd="logit"):
    """Plug-in noise model of one seed-pooled battery (real n, discordance,
    input covariance across attacks and seeds, per-attack between-seed
    variance).

    Returns a dict with:
      seeds, pool_attacks (frozen pool), sim_attacks (pooled attacks + other
      seed-pooled cells that use the pool, e.g. SOS SST-2), structural rows
      (fixed frozen Tango rows of structural nulls), Sigma (input covariance
      of the stacked (attack, seed) means, ddof 1, / n), v (= var_i(dbar)/n),
      psi (expected between-seed sample variance from input noise alone),
      s2_obs (observed between-seed variance of k_s), k_obs, real pool.
    """
    pool = st.pools[(fam, ds, kd)]
    pool_attacks = pool["attacks"].split(",")
    seeds = [int(s) for s in pool["seeds"].split(",")]
    r = len(seeds)
    sim, structural = [], []
    for (f, d, k, atk), pr in sorted(st.prim.items()):
        if (f, d, k) != (fam, ds, kd) or not pr["teacher_qualified"]:
            continue
        if pr["method"] == "structural":
            structural.append(atk)
        elif pr["method"].startswith("seed-pooled"):
            sim.append(atk)
        else:
            raise ValueError(f"unexpected method {pr['method']} in {fam}/{ds}/{kd}/{atk}")
    assert set(pool_attacks) <= set(sim), (pool_attacks, sim)
    vecs, kobs = [], {}
    for atk in sim:
        by = st.groups[(fam, ds, kd, atk)]
        assert sorted(by) == seeds, (atk, sorted(by), seeds)
        kobs[atk] = []
        for s in seeds:
            dv = dvec(by[s])
            vecs.append(dv)
            kobs[atk].append(float(dv.mean()))
    M = np.vstack(vecs)
    n = M.shape[1]
    Sigma = np.cov(M, ddof=1) / n
    m = len(sim)
    v, psi, s2_obs = {}, {}, {}
    for i, atk in enumerate(sim):
        blk = Sigma[i * r:(i + 1) * r, i * r:(i + 1) * r]
        v[atk] = float(blk.sum() / r ** 2)
        psi[atk] = float((np.trace(blk) - blk.sum() / r) / (r - 1))
        ks = kobs[atk]
        s2_obs[atk] = float(np.var(ks, ddof=1))
    struct_rows = {}
    for atk in structural:
        c42 = st.groups[(fam, ds, kd, atk)][42]
        struct_rows[atk] = paired_contrast(c42["P"], c42["C"], TOLS)
    disc = {atk: float(np.mean([np.mean(dvec(st.groups[(fam, ds, kd, atk)][s]) != 0) for s in seeds]))
            for atk in sim}
    # cross-seed input correlation (mean over seed pairs)
    rho = {}
    for i, atk in enumerate(sim):
        blk = Sigma[i * r:(i + 1) * r, i * r:(i + 1) * r]
        sd = np.sqrt(np.diag(blk))
        cc = [blk[a, b] / (sd[a] * sd[b]) for a in range(r) for b in range(a + 1, r)]
        rho[atk] = float(np.mean(cc))
    return {"fam": fam, "dataset": ds, "kd": kd, "seeds": seeds, "r": r, "n": n, "m_sim": m,
            "pool_attacks": pool_attacks, "sim_attacks": sim, "structural": structural,
            "struct_rows": struct_rows, "Sigma": Sigma, "v": v, "psi": psi, "s2_obs": s2_obs,
            "k_obs": kobs, "disc": disc, "rho": rho,
            "pool_s2": pool["s_between"] ** 2, "pool_df": pool["df"],
            "ms_seed": pool["ms_seed"], "ms_res": pool["ms_res"]}


def target_plugin(st, fam, ds, kd):
    """Plug-in model of a transported (one-seed) battery: the seed-42 cells'
    d vectors (input covariance across attacks), the frozen Tango SE of each
    cell (se_from_interval of its Tango interval) and the structural rows."""
    sim, structural, src = [], [], None
    for (f, d, k, atk), pr in sorted(st.prim.items()):
        if (f, d, k) != (fam, ds, kd) or not pr["teacher_qualified"]:
            continue
        if pr["method"] == "structural":
            structural.append(atk)
        elif pr["method"] == "transported":
            sim.append(atk)
            src = pr["variance_from"] if src is None else src
            assert src == pr["variance_from"]
        else:
            raise ValueError(pr["method"])
    vecs, se_tango, k_obs, disc = [], {}, {}, {}
    for atk in sim:
        c42 = st.groups[(fam, ds, kd, atk)][42]
        dv = dvec(c42)
        vecs.append(dv)
        pc = paired_contrast(c42["P"], c42["C"], TOLS)
        se_tango[atk] = se_from_interval(pc["lo"], pc["hi"])
        k_obs[atk] = pc["k"]
        disc[atk] = float(np.mean(dv != 0))
    M = np.vstack(vecs)
    n = M.shape[1]
    Sigma = np.atleast_2d(np.cov(M, ddof=1)) / n
    struct_rows = {}
    for atk in structural:
        c42 = st.groups[(fam, ds, kd, atk)][42]
        struct_rows[atk] = paired_contrast(c42["P"], c42["C"], TOLS)
    return {"fam": fam, "dataset": ds, "kd": kd, "n": n, "sim_attacks": sim,
            "structural": structural, "struct_rows": struct_rows, "Sigma": Sigma,
            "se_tango": se_tango, "k_obs": k_obs, "disc": disc,
            "source": tuple(src.split("/"))}
