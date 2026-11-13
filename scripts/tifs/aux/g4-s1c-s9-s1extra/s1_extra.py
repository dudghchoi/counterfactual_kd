#!/usr/bin/env python3
"""A11 item 12 — S1 extra rows with the frozen run_scenario.

Post hoc / registered auxiliary analysis (docs/tifs_addendum.md A11.12). It does
not change any primary verdict.

Scenario sets (frozen counterfactual_kd.analysis.calibration.run_scenario,
reps = 10000, bootstrap R = 2000, tol_h = 0.2, tols_k = (0.02, 0.05, 0.10), as
results/tifs/s1/meta.json):

  boundary   n = 912, k = delta = 0.02 (scenario_from_rates), clean triggered rate
             c in {0.05, 0.10, 0.50, 0.90}, discordance d in {0.02, 0.03, 0.06, 0.10,
             0.20}, feasible combinations. At the boundary P(EQUIVALENT@0.02) is the
             false-equivalence rate and P(POSITIVE@0.02) the false-positive rate of the
             cell rule (each nominally <= 0.025). classify()'s false_* columns depend on
             the floating-point sign of true_k - 0.02 here, so read equivalent@0.02 and
             positive@0.02 directly.
  sparse     sparse discordance with k != 0: n in {912, 5700}, c in {0.05, 0.50},
             d in {0.005, 0.01, 0.02}, k in {d/2, d} (k = d: every discordant pair in one
             direction, n01 expected 0).
  real       plug-in tables of the measured clean-corpus battery cells (every seed, logit
             and 'feature') with discordance <= 0.03 and n10 != n01 (truth = the observed
             table), as S1's cell scenarios.

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" python s1_extra.py [--out DIR] [--workers 4]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import common as K  # noqa: E402
from counterfactual_kd.analysis.calibration import (  # noqa: E402  (frozen)
    Scenario, run_scenario, scenario_from_rates,
)

TOLS_K = (0.02, 0.05, 0.10)
REPS, R_BOOT = 10000, 2000


def scenarios(st):
    out = {}
    for c in (0.05, 0.10, 0.50, 0.90):
        for d in (0.02, 0.03, 0.06, 0.10, 0.20):
            sc = scenario_from_rates(f"x_n912_c{c:g}_d{d:g}_k0.02", 912, c, 0.02, d)
            if min(sc.pi11, sc.pi00) >= 0:
                out[sc.name] = ("boundary", sc)
    for n in (912, 5700):
        for c in (0.05, 0.50):
            for d in (0.005, 0.01, 0.02):
                for k in (d / 2, d):
                    sc = scenario_from_rates(f"x_n{n}_c{c:g}_d{d:g}_k{k:g}", n, c, k, d)
                    if min(sc.pi11, sc.pi00) < 0:
                        continue
                    if sc.name in out:
                        out[sc.name] = ("boundary+sparse", out[sc.name][1])
                    else:
                        out[sc.name] = ("sparse", sc)
    for key, by_seed in sorted(st.groups.items()):
        fam, ds, kd, atk = key
        for s, cell in sorted(by_seed.items()):
            n11 = n10 = n01 = 0
            for p, q in zip(cell["P"], cell["C"]):
                n11 += p & q
                n10 += p & (1 - q)
                n01 += (1 - p) & q
            n = len(cell["P"])
            if n10 == n01 or (n10 + n01) / n > 0.03:
                continue
            name = f"real_{fam}_{ds}_{kd}_{atk}_s{s}|plugin"
            out[name] = ("real", Scenario(name, n, n11 / n, n10 / n, n01 / n))
    return out


def _run(args):
    try:
        os.nice(10)
    except OSError:
        pass
    label, sc, seed = args
    t0 = time.time()
    res = run_scenario(sc, reps=REPS, R=R_BOOT, seed=seed, tols_k=TOLS_K)
    res["set"] = label
    res["seconds"] = time.time() - t0
    return res


def summary_rows(results):
    out = []
    for r in results:
        n, reps = r["n"], r["reps"]
        row = {"set": r["set"], "scenario": r["scenario"], "n": n,
               "C_rate": r["pi11"] + r["pi01"], "discordance": r["pi10"] + r["pi01"],
               "expected_discordant_pairs": n * (r["pi10"] + r["pi01"]),
               "true_k": r["true_k"], "true_k_minus_0.02": r["true_k"] - 0.02, "reps": reps}
        for proc in ("tango_k", "newcombe_k", "boot_k"):
            row[f"{proc}.coverage"] = r[f"{proc}.coverage"]
            row[f"{proc}.coverage_mcse"] = K.mc_se(r[f"{proc}.coverage"], reps)
            row[f"{proc}.width"] = r[f"{proc}.width"]
            row[f"{proc}.degenerate"] = r[f"{proc}.degenerate"]
            for t in TOLS_K:
                for v in ("equivalent", "positive", "negative", "inconclusive"):
                    p = r[f"{proc}.{v}@{t:g}"]
                    row[f"{proc}.{v}@{t:g}"] = p
                    if v in ("equivalent", "positive"):
                        row[f"{proc}.{v}@{t:g}_mcse"] = K.mc_se(p, reps)
        row["rope_h.coverage"] = r["rope_h.coverage"]
        row["rope_h.equivalent@0.2"] = r["rope_h.equivalent@0.2"]
        out.append(row)
    return out


def main():
    ap = K.add_out_arg(argparse.ArgumentParser())
    ap.add_argument("--workers", type=int, default=int(os.environ.get("S1X_WORKERS", "4")))
    args = ap.parse_args()
    t_start = time.time()
    st = K.state()
    sc = scenarios(st)
    jobs = [(label, s, 120000 + i) for i, (name, (label, s)) in enumerate(sorted(sc.items()))]
    print(f"{len(jobs)} scenarios", flush=True)
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        results = list(pool.imap(_run, jobs))
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    K.write_csv(results, os.path.join(args.out, "s1_extra.csv"),
                "S1 extra rows: frozen run_scenario output (same columns as results/tifs/s1/grid.csv; reps 10000, "
                "bootstrap R 2000) for set=boundary (n=912, k=delta=0.02), sparse (n 912/5700, discordance "
                "0.005-0.02, k=d/2 or d) and real (plug-in tables of measured clean-corpus cells with discordance "
                "<=0.03, n10!=n01). At k=delta read equivalent@0.02 / positive@0.02 directly (false equivalence / "
                "false positive at the boundary).")
    K.write_csv(summary_rows(results), os.path.join(args.out, "s1_extra_summary.csv"),
                "S1 extra rows, compact: per scenario coverage (+binomial MC SE), width, degenerate share and the "
                "verdict frequencies of the Tango, Newcombe and k-bootstrap intervals at delta 0.02/0.05/0.10 "
                "(+MC SE for EQUIVALENT and POSITIVE), and the h-ROPE coverage/equivalence.")
    K.log_cost(args.out, "s1_extra.py", t_start)


if __name__ == "__main__":
    main()
