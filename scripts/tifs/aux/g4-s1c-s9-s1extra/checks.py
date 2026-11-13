#!/usr/bin/env python3
"""Self-checks of the g4 outputs (A11 items 10-12) against the frozen/addendum tables
where they overlap, against the archived S1 grid and cell set, and internal
consistency checks.

Post hoc / registered auxiliary (docs/tifs_addendum.md A11); changes no verdict.
Reads results/tifs/report/*.csv (frozen analyze.py), results/tifs/addendum/*.csv
(addendum.py), results/tifs/s1/{grid,cells}.csv (archived S1, read only) and this
group's outputs in --out. Writes <out>/checks.csv.

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" python checks.py [--out DIR]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import math
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import common as K  # noqa: E402
from counterfactual_kd.analysis.calibration import run_scenario  # noqa: E402  (frozen)
from counterfactual_kd.analysis.tifs import t_contrast  # noqa: E402  (frozen)
from counterfactual_kd.metrics.paired import audit_verdict, tango_paired_ci, wilson_interval  # noqa: E402  (frozen)

import s1_calibration as S1  # noqa: E402  (frozen script; grid_scenarios only)

TOL = 1e-9
FROZEN = {"docs/tifs_plan.md": "bc9dfd8f87f54d778a47e07b7d49de2c68af5b3df5458729fa4e7bb947f88713",
          "scripts/tifs/analyze.py": "1f70bbf4453bd83a5ec5f9858343549e89485627acb63f62d5e0e95dd5f09ef8",
          "counterfactual_kd/analysis/tifs.py": "cc0e05451bee502d888ab52260eb4576aab056dbe89a7c41fdce07db247dc9ce",
          "results/tifs/structural_nulls.json": "a6dc5091d33b0c355a34136b65e28f29163864aeb0a4d1c060afd0c33656a9b0"}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def add(rows, group, check, ours, ref, ok=None, note=""):
    if ok is None:
        if isinstance(ours, str) or isinstance(ref, str):
            ok = str(ours) == str(ref)
        else:
            ok = (math.isnan(ours) and math.isnan(ref)) or abs(ours - ref) <= TOL
    rows.append({"group": group, "check": check, "ours": ours, "reference": ref, "pass": bool(ok), "note": note})


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def fast_t_identity(n=3000, seed=11):
    """Frozen t_contrast with common.ExactFastT vs stock scipy.stats.t: number of differing fields."""
    rng = np.random.default_rng(seed)
    tols0 = (0.0,) + K.TOLS
    cases = [(float(rng.normal(0, 0.1)), float(abs(rng.normal(0.02, 0.01)) + 1e-6), float(rng.uniform(1.0, 60.0)))
             for _ in range(n)]
    cases += [(0.05, 0.01, 6), (0.0, 0.02, 12), (0.3, 1e-9, 4.67), (-0.2, 0.05, 1000.0)]   # int df, extreme z, large df
    stock = [t_contrast(e, s_, d, tols0) for e, s_, d in cases]
    prev = K.install_exact_fast_t(True)
    try:
        fast = [t_contrast(e, s_, d, tols0) for e, s_, d in cases]
    finally:
        K.restore_t(prev)
    bad = 0
    for a, b in zip(stock, fast):
        for key in a:
            va, vb = a[key], b[key]
            if isinstance(va, float) and isinstance(vb, float) and math.isnan(va) and math.isnan(vb):
                continue
            bad += va != vb or type(va) is not type(vb)
    return bad, len(cases) * len(stock[0])


def main():
    ap = K.add_out_arg(argparse.ArgumentParser())
    args = ap.parse_args()
    t_start = time.time()
    OUT = args.out
    st = K.state()
    out = []
    # ── frozen files unchanged ─────────────────────────────────────────
    for rel, h in FROZEN.items():
        got = hashlib.sha256(open(os.path.join(K.R, rel), "rb").read()).hexdigest()
        add(out, "frozen files", f"sha256 {rel}", got, h)
    marker = os.path.join(OUT, "logs", "run_start.txt")       # written by run_all.sh
    if os.path.exists(marker):
        t0 = os.path.getmtime(marker)
        pyc = [p for sub in ("scripts", "counterfactual_kd", "tests")
               for p in glob.glob(os.path.join(K.R, sub, "**", "*.pyc"), recursive=True)
               if os.path.getmtime(p) > t0]
        pyc += [p for p in glob.glob(os.path.join(K.R, "*.pyc")) + glob.glob(os.path.join(K.R, "__pycache__", "*.pyc"))
                if os.path.getmtime(p) > t0]
        add(out, "repo untouched", "*.pyc under scripts/, counterfactual_kd/, tests/ or the repo root newer than this "
            "run's start (logs/run_start.txt) (count)", len(pyc), 0, ok=len(pyc) == 0,
            note="; ".join(os.path.relpath(p, K.R) for p in pyc[:5]))
    caches = glob.glob(os.path.join(K.HERE, "**", "__pycache__"), recursive=True)
    add(out, "repo untouched", "__pycache__ folders in this script folder (count)", len(caches), 0, ok=not caches)
    # ── input fingerprints (official inputs read by this group) ─────────
    for rel in ("results/tifs/report/primary.csv", "results/tifs/report/variance.csv", "results/tifs/report/aa.csv",
                "results/tifs/report/secondary.csv", "results/tifs/report/factorial.csv",
                "results/tifs/report/cells.csv", "results/tifs/addendum/attack_variance.csv",
                "results/tifs/addendum/variance_diag.csv", "results/tifs/s1/s1b.csv", "results/tifs/s1/grid.csv",
                "results/tifs/s1/cells.csv"):
        add(out, "input fingerprint (sha256)", rel, sha(os.path.join(K.R, rel)), "(recorded)", ok=True)
    snap = K.snapshot()
    add(out, "input fingerprint (sha256)", "prediction files parsed by the frozen load_cells (count / runs)",
        f"{snap['n_preds']} / {snap['n_runs']}", f"{len(st.cells)} / {len({c['run'] for c in st.cells})}",
        ok=snap["n_preds"] == len(st.cells), note=f"newest {snap['newest_utc']} ({snap['newest_path']})")
    # ── exact fast t (common.ExactFastT) vs stock scipy in the frozen t_contrast ─
    bad, total = fast_t_identity()
    add(out, "exact fast t", f"frozen t_contrast with ExactFastT vs stock scipy.stats.t: differing fields of {total}",
        bad, 0, ok=bad == 0, note="random (est, se, df) and edge cases; values and types compared exactly")
    # ── frozen primary table on current data vs results/tifs/report ─────
    prim_ref = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in K.read_csv(os.path.join(K.REPORT, "primary.csv"))}
    for key, pr in sorted(st.prim.items()):
        ref = prim_ref.get(key)
        if ref is None:
            add(out, "primary vs results/tifs/report", "/".join(key) + " present", "yes", "no", ok=False)
            continue
        for col in ("k", "lo", "hi"):
            add(out, "primary vs results/tifs/report", "/".join(key) + f" {col}", pr[col], f(ref[col]))
        for t in K.TOLS:
            add(out, "primary vs results/tifs/report", "/".join(key) + f" verdict@{t:g}", pr[f"verdict@{t:g}"], ref[f"verdict@{t:g}"])
    add(out, "primary vs results/tifs/report", "number of primary cells", len(st.prim), len(prim_ref), ok=len(st.prim) == len(prim_ref) == 31)
    # ── S1c: pools vs report/variance.csv; F vs addendum/variance_diag.csv ─
    var_ref = {(r["fam"], r["dataset"], r["kd"]): r for r in K.read_csv(os.path.join(K.REPORT, "variance.csv"))
               if r["attack"] == "(pooled)"}
    fit = K.read_csv(os.path.join(OUT, "s1c_fit.csv"))
    names = {"BERT SST-2": ("bert", "sst2", "logit"), "BERT AG News": ("bert", "agnews", "logit"),
             "Qwen3 SST-2": ("qwen3", "sst2", "logit"), "GPT-2 SST-2": ("gpt2", "sst2", "logit")}
    vd = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in K.read_csv(os.path.join(K.ADDENDUM, "variance_diag.csv"))}
    for r in fit:
        if r["part"] != "pooled battery":
            continue
        key = names[r["battery"]]
        add(out, "S1c fit vs results/tifs/report/variance.csv", f"{r['battery']} {r['attack']} pool s", f(r["pool_s_obs"]), f(var_ref[key]["s_between"]))
        add(out, "S1c fit vs results/tifs/report/variance.csv", f"{r['battery']} {r['attack']} pool df", f(r["pool_df_obs"]), f(var_ref[key]["df"]))
        d = vd.get(key + (r["attack"],))
        if d is not None:
            add(out, "S1c fit vs results/tifs/addendum/variance_diag.csv", f"{r['battery']} {r['attack']} F", f(r["F_obs"]), f(d["F"]))
            add(out, "S1c fit vs results/tifs/addendum/variance_diag.csv", f"{r['battery']} {r['attack']} s_attack", f(r["s_obs"]), f(d["s_attack"]))
    for r in fit:
        if r["part"] == "pooled battery" and r["in_pool"] == "True":
            # E[MS_seed]/E[MS_res] of the fitted model equals the measured ratio unless w is clipped to 0
            ok = abs(f(r["ms_ratio_model_H"]) - f(r["ms_ratio_obs"])) < 1e-9 or f(r["w_fit_H"]) == 0.0
            add(out, "S1c fit internal", f"{r['battery']} {r['attack']} model ms ratio (H)", f(r["ms_ratio_model_H"]),
                f(r["ms_ratio_obs"]), ok=ok, note="w clipped at 0 when the ratio at w=0 already exceeds the measured one")
    for fam, ds in (("bert", "sst2"), ("bert", "agnews"), ("qwen3", "sst2"), ("gpt2", "sst2")):
        b = K.battery_plugin(st, fam, ds)
        for a in b["sim_attacks"]:
            se = (b["pool_s2"] / b["r"] + b["v"][a]) ** 0.5
            ref = st.prim[(fam, ds, "logit", a)]["se"]
            add(out, "S1c plug-in vs frozen primary SE", f"{fam}/{ds}/logit/{a}", se, ref, ok=abs(se - ref) <= 1e-12)
            add(out, "S1c plug-in vs frozen primary k", f"{fam}/{ds}/logit/{a}", float(np.mean(b["k_obs"][a])),
                st.prim[(fam, ds, "logit", a)]["k"], ok=abs(float(np.mean(b["k_obs"][a])) - st.prim[(fam, ds, "logit", a)]["k"]) <= 1e-12)
    for fam, ds, kd in (("bert", "sst2", "feature"), ("bert", "agnews", "feature"), ("qwen3", "agnews", "logit")):
        t = K.target_plugin(st, fam, ds, kd)
        pool = st.pools[t["source"]]
        for a in t["sim_attacks"]:
            se = (t["se_tango"][a] ** 2 + pool["s_between"] ** 2) ** 0.5
            ref = st.prim[(fam, ds, kd, a)]["se"]
            add(out, "S1c transported plug-in vs frozen primary SE", f"{fam}/{ds}/{kd}/{a}", se, ref, ok=abs(se - ref) <= 1e-12)
    # S1c reps and seed blocks: config-H transported jobs pool 4 blocks of 10,000 reps
    s1c_cells = K.read_csv(os.path.join(OUT, "s1c_cells.csv"))
    s1c_batt = K.read_csv(os.path.join(OUT, "s1c_battery.csv"))
    want = lambda r: 40000 if (r["rule"] == "transported" and r["config"] == "H") else (
        20000 if r["rule"] == "pooled" else 10000)
    badr = sum(int(r["reps"]) != want(r) for r in s1c_cells + s1c_batt)
    add(out, "S1c reps", "rows whose reps differ from 20000 (pooled) / 40000 (transported H) / 10000 (other)",
        badr, 0, ok=badr == 0)
    blk = K.read_csv(os.path.join(OUT, "s1c_blocks.csv"))
    nb = {}
    for b in blk:
        nb.setdefault((b["rule"], b["battery"], b["config"], b["c_sd"], b["delta"]), []).append(b)
    worst, n_cmp = 0.0, 0
    for r in s1c_batt:
        key = (r["rule"], r["battery"], r["config"], r["c_sd"], r["delta"])
        if key not in nb:
            continue
        bs = nb[key]
        for col in ("false_transfers_all0", "no_transfer_all0", "false_transfers_alld"):
            pooled = sum(f(b[col]) * int(b["reps"]) for b in bs) / sum(int(b["reps"]) for b in bs)
            worst = max(worst, abs(pooled - f(r[col])))
            n_cmp += 1
    add(out, "S1c reps", f"pooled battery rates = count-weighted block means ({n_cmp} comparisons, max |diff|)",
        worst, 0.0, ok=worst < 1e-12 and n_cmp == 21 * 3 * 3)
    # transported_realdf has a closed form: coverage = E_X[2 Phi(q SE(X)/sd_k) - 1] with
    # SE(X)^2 = se_T^2 + E[s2] X/df, X ~ chi2_df, sd_k^2 = T'^2 + Sigma'_aa (1-D integral)
    from scipy import integrate, stats
    s1c_jobs = {(j["rule"], j["battery"], j["config"], j["c_sd"]): j for j in K.read_csv(os.path.join(OUT, "s1c_jobs.csv"))}
    tfit = {(r["battery"], r["attack"]): r for r in fit if r["part"] == "transported battery"}
    zs_cov, zs_fb = [], []
    for c in s1c_cells:
        if c["rule"] != "transported_realdf":
            continue
        j = s1c_jobs[(c["rule"], c["battery"], c["config"], c["c_sd"])]
        tf = tfit[(c["battery"], c["attack"])]
        df, es2 = f(j["real_df"]), f(j["model_E_s2_pool"])
        se_t, sde, tt = f(tf["se_tango"]), f(tf["sd_input_emp"]), f(c["T_true"])
        q = float(stats.t.ppf(0.975, df))
        sdk = math.sqrt(tt ** 2 + sde ** 2)
        se = lambda x: math.sqrt(se_t ** 2 + es2 * x / df)
        cov = integrate.quad(lambda x: (2 * stats.norm.cdf(q * se(x) / sdk) - 1) * stats.chi2.pdf(x, df), 0, np.inf, limit=200)[0]
        fbv = integrate.quad(lambda x: stats.norm.cdf(-q * se(x) / sdk) * stats.chi2.pdf(x, df), 0, np.inf, limit=200)[0]
        reps = int(c["reps"])
        zs_cov.append((f(c["coverage"]) - cov) / max(math.sqrt(cov * (1 - cov) / reps), 1e-12))
        zs_fb.append((f(c["false_below_at_mu_eq_delta"]) - fbv) / max(math.sqrt(fbv * (1 - fbv) / reps), 1e-12))
    for name, zz in (("coverage", zs_cov), ("false below", zs_fb)):
        zz = [z for z in zz if abs(z) < 1e6]   # cells with analytic rate 0 or 1 carry no information
        add(out, "S1c transported_realdf vs closed form",
            f"{name}: max |z| over {len(zz)} cells (mean z {np.mean(zz):+.2f}, share |z|>2 {np.mean(np.abs(zz) > 2):.3f})",
            float(np.max(np.abs(zz))), 4.0, ok=float(np.max(np.abs(zz))) < 4.0 and abs(float(np.mean(zz))) < 0.5,
            note="1-D integral over chi2_df; pass if max |z| < 4 and |mean z| < 0.5 (cells of a job share draws)")
    chk = K.read_csv(os.path.join(OUT, "s1c_check_s1b.csv"))
    det = [r for r in chk if r["check"].startswith("frozen pool_seed_variance")][0]
    add(out, "S1c validation", "frozen pool_seed_variance vs S1b (max |diff| SE, df)",
        max(f(det["max_abs_diff_se"]), f(det["max_abs_diff_df"])), 0.0,
        ok=f(det["max_abs_diff_se"]) < 1e-12 and f(det["max_abs_diff_df"]) < 1e-9)
    zs = [abs(f(r["z_diff"])) for r in chk if r.get("z_diff") not in (None, "")]
    add(out, "S1c validation", f"S1b-mode simulator vs results/tifs/s1/s1b.csv: max |z| over {len(zs)} metrics",
        max(zs), 3.0, ok=max(zs) < 3.0, note="pass if every |z| < 3 (coverage z is conservative: cells share the pool)")
    # ── S9 empirical items vs aa.csv / secondary.csv / factorial.csv / primary.csv ──
    s9 = K.read_csv(os.path.join(OUT, "s9_known_truth_cells.csv"))
    aa = K.read_csv(os.path.join(K.REPORT, "aa.csv"))
    for r in aa:
        t = r["teacher_seed"]
        item = f"{r['fam']}/{r['dataset']}/{r['corpus']}/badnets/C_t42~C_t{t}"
        mine = [x for x in s9 if x["item"] == item and x["orientation"] == f"C_t42 - C_t{t}" and x["rule"] == "R5_audit"]
        add(out, "S9 A/A vs results/tifs/report/aa.csv", f"{item} present", len(mine), 3, ok=len(mine) == 3)
        for x in mine:
            dl = x["delta"]
            add(out, "S9 A/A vs results/tifs/report/aa.csv", f"{item} k", f(x["k"]), f(r["C-C'"]))
            add(out, "S9 A/A vs results/tifs/report/aa.csv", f"{item} lo", f(x["lo"]), f(r["C-C'_lo"]))
            add(out, "S9 A/A vs results/tifs/report/aa.csv", f"{item} verdict@{dl}", x["verdict_tango"], r[f"C-C'_verdict@{float(dl):g}"])
        if r.get("C-C'@atba"):
            item = f"{r['fam']}/{r['dataset']}/{r['corpus']}/atba phrase/C_t42~C_t{t}"
            mine = [x for x in s9 if x["item"] == item and x["orientation"] == f"C_t42 - C_t{t}" and x["rule"] == "R5_audit"]
            add(out, "S9 A/A vs results/tifs/report/aa.csv", f"{item} present", len(mine), 3, ok=len(mine) == 3)
            for x in mine:
                dl = x["delta"]
                add(out, "S9 A/A vs results/tifs/report/aa.csv", f"{item} k", f(x["k"]), f(r["C-C'@atba"]))
                add(out, "S9 A/A vs results/tifs/report/aa.csv", f"{item} hi", f(x["hi"]), f(r["C-C'@atba_hi"]))
                add(out, "S9 A/A vs results/tifs/report/aa.csv", f"{item} verdict@{dl}", x["verdict_tango"],
                    r[f"C-C'@atba_verdict@{float(dl):g}"])
    sec = K.read_csv(os.path.join(K.REPORT, "secondary.csv"))
    for r in sec:
        if r["block"] == "PC" and r["kd"] == "logit":
            item = f"{r['fam']}/{r['dataset']}/{r['corpus']}/{r['attack']}"
            mine = [x for x in s9 if x["item"] == item and x["rule"] == "R5_audit"]
            add(out, "S9 positive control vs results/tifs/report/secondary.csv", f"{item} present", len(mine), 3, ok=len(mine) == 3)
            for x in mine:
                dl = float(x["delta"])
                add(out, "S9 positive control vs results/tifs/report/secondary.csv", f"{item} k", f(x["k"]), f(r["k"]))
                add(out, "S9 positive control vs results/tifs/report/secondary.csv", f"{item} verdict@{dl:g}",
                    x["verdict_tango"], r[f"verdict@{dl:g}"])
                add(out, "S9 positive control vs results/tifs/report/secondary.csv", f"{item} transported verdict@{dl:g}",
                    x["verdict_transported"], r[f"transported_verdict@{dl:g}"])
                add(out, "S9 positive control vs results/tifs/report/secondary.csv", f"{item} transported lo",
                    f(x["transported_lo"]), f(r["transported_lo"]))
        if r["block"] == "G" and r["kd"] == "logit":
            item = f"{r['fam']}/{r['dataset']}/{r['corpus']}/{r['attack']}/R1"
            mine = [x for x in s9 if x["item"] == item and x["rule"] == "R1_raw_asr" and x["delta"] == "0.05"]
            for x in mine:
                add(out, "S9 R1 context P1-C1 vs results/tifs/report/secondary.csv", f"{item} P1-C1",
                    f(x["context_P1_minus_C1"]), f(r["k"]))
    fac = K.read_csv(os.path.join(K.REPORT, "factorial.csv"))
    for r in fac:
        if r["corpus"] != "clean" and r["kd"] == "logit" and r["block"] == "G":
            item = f"{r['fam']}/{r['dataset']}/{r['corpus']}/{r['attack']}/R1"
            mine = [x for x in s9 if x["item"] == item and x["rule"] == "R1_raw_asr" and x["delta"] == "0.05"]
            add(out, "S9 R1 raw ASR vs results/tifs/report/factorial.csv R", f"{item} present", len(mine), 1, ok=len(mine) == 1)
            for x in mine:
                add(out, "S9 R1 raw ASR vs results/tifs/report/factorial.csv R", item, f(x["suspect_rate"]), f(r["R"]))
    for x in s9:
        if x["condition"] == "EP SST-2 structural null" and x["rule"] == "R5_audit" and x["delta"] == "0.02":
            kd = x["item"].split("/")[2]
            ref = prim_ref[("bert", "sst2", kd, "ep")]
            add(out, "S9 EP null vs results/tifs/report/primary.csv", x["item"] + " k", f(x["k"]), f(ref["k"]))
            add(out, "S9 EP null vs results/tifs/report/primary.csv", x["item"] + " hi", f(x["hi"]), f(ref["hi"]))
    # S9 independent recomputation from the hit vectors (stdlib frozen helpers, not paired_contrast)
    vec = {}
    for c in st.cells:
        if c["single"] or c["kd"] != "logit":
            continue
        if c["block"] == "G" and c["seed"] == 42 and c["attack"] == "badnets" and c["teacher_seed"] == 42:
            vec[(c["fam"], c["dataset"], c["corpus"], "badnets", 42)] = c["C"]
        if c["block"] == "AA" and c["attack"] == "badnets":
            vec[(c["fam"], c["dataset"], c["corpus"], "badnets", c["teacher_seed"])] = c["C"]
            ctl = c["controls"].get(f"atba_phrase_{c['dataset']}")
            if ctl is not None:
                vec[(c["fam"], c["dataset"], c["corpus"], "atba phrase", c["teacher_seed"])] = ctl["c"]
        if c["block"] == "G" and c["seed"] == 42 and c["attack"] == "atba" and c["corpus"] == "clean" and c["teacher_seed"] == 42:
            vec[(c["fam"], c["dataset"], c["corpus"], "atba phrase", 42)] = c["C"]
    n_ok = n_all = 0
    for x in s9:
        if x["condition"] != "A/A backdoor-free swap" or x["rule"] != "R5_audit" or x["delta"] != "0.05":
            continue
        fam, ds, corpus, trig, _ = x["item"].split("/")
        sa, ra = (int(s.strip()[3:]) for s in x["orientation"].split(" - "))
        a, b = vec[(fam, ds, corpus, trig, sa)], vec[(fam, ds, corpus, trig, ra)]
        k = sum(a) / len(a) - sum(b) / len(b)
        lo, hi = tango_paired_ci(a, b)
        wl, wh = wilson_interval(sum(a), len(a))
        n_all += 1
        n_ok += (abs(k - f(x["k"])) < 1e-12 and abs(lo - f(x["lo"])) < 1e-9 and abs(hi - f(x["hi"])) < 1e-9
                 and abs(wl - f(x["suspect_wilson_lo"])) < 1e-12 and abs(wh - f(x["suspect_wilson_hi"])) < 1e-12)
    add(out, "S9 A/A independent recomputation", "k, Tango lo/hi, Wilson of the suspect from the hit vectors (count)",
        n_ok, n_all, ok=n_ok == n_all and n_all > 0)
    # S9 internal consistency
    bad = 0
    by = {}
    for x in s9:
        by.setdefault((x["item"], x["orientation"], x["delta"]), {})[x["rule"]] = x["claim"]
    for key, cl in by.items():
        if "R5_audit" in cl and cl["R5_audit"] != "n/a":
            if (cl["R5_audit"] == "T") != (cl["R5b_audit_binary"] == "T"):
                bad += 1
    add(out, "S9 internal", "R5b claims T exactly where R5 claims T (empirical, mismatches)", bad, 0, ok=bad == 0)
    summ = K.read_csv(os.path.join(OUT, "s9_known_truth_summary.csv"))
    tot = {}
    for r in summ:
        tot.setdefault((r["condition"], r["delta"], r["rule"], r["level"]), 0)
        tot[(r["condition"], r["delta"], r["rule"], r["level"])] += int(r["n_tests"])
    mism = sum(1 for (c, d, ru, lv), v in tot.items() if lv == "condition" and tot.get((c, d, ru, "subcondition")) != v)
    add(out, "S9 internal", "sub-condition test counts sum to the condition counts (mismatches)", mism, 0, ok=mism == 0)
    # trigger level (A/A, both corpora) = sum of its two sub-conditions; pair-level counts
    sub_by = {}
    for r in summ:
        if r["level"] == "subcondition" and r["condition"] == "A/A backdoor-free swap":
            trig = "ATBA phrase" if r["subcondition"].endswith("ATBA phrase") else "BadNets trigger"
            k = (trig, r["delta"], r["rule"])
            sub_by.setdefault(k, [0, 0, 0])
            sub_by[k][0] += int(r["n_tests"])
            sub_by[k][1] += int(r["wrong"])
            sub_by[k][2] += int(r["items_wrong"])
    mism = n_tr = 0
    for r in summ:
        if r["level"] == "trigger":
            n_tr += 1
            trig = r["subcondition"].split(" (")[0]
            mism += sub_by.get((trig, r["delta"], r["rule"])) != [int(r["n_tests"]), int(r["wrong"]), int(r["items_wrong"])]
    add(out, "S9 internal", f"A/A trigger level (both corpora) = sum of its sub-conditions ({n_tr} rows, mismatches)",
        mism, 0, ok=mism == 0 and n_tr > 0)
    bad = 0
    for r in summ:
        if r["condition"] != "A/A backdoor-free swap" or r["level"] != "subcondition":
            continue
        w, iw = int(r["wrong"]), int(r["items_wrong"])
        if r["rule"] in ("R3_signed_point", "R4_ci_excludes_0", "R5_audit", "R5b_audit_binary", "R5t_audit_transported"):
            bad += w != iw          # signed rules: at most one orientation of a pair can claim T
        elif r["rule"] == "R2_abs_point":
            bad += w != 2 * iw      # |k| is the same in both orientations
        else:
            bad += not (iw <= w <= 2 * iw)
    add(out, "S9 internal", "A/A pair-level vs test-level wrong counts consistent with each rule's symmetry (violations)",
        bad, 0, ok=bad == 0)
    # R1h equals the frozen audit_verdict exposure flag of the suspect
    n_ok = n_all = 0
    hits = {}
    for c in st.cells:
        hits[(c["run"], c["attack"], c["kd"], c["seed"])] = c
    for x in s9:
        if x["rule"] != "R1h_raw_asr_high" or x["delta"] != "0.05":
            continue
        n = int(x["n"])
        lo = f(x["suspect_wilson_lo"])
        x_hits = round(f(x["suspect_rate"]) * n)
        flag = audit_verdict([1] * x_hits + [0] * (n - x_hits), [0] * n, 0.05)["p_exposure"]
        n_all += 1
        n_ok += (flag == "HIGH") == (x["claim"] == "T") and abs(wilson_interval(x_hits, n)[0] - lo) < 1e-15
    add(out, "S9 R1h vs frozen audit_verdict", "R1h claim T iff the frozen audit_verdict p_exposure is HIGH (items)",
        n_ok, n_all, ok=n_ok == n_all and n_all > 0)
    s1rows = K.read_csv(os.path.join(OUT, "s9_s1_by_scenario.csv"))
    by = {}
    for x in s1rows:
        by.setdefault((x["scenario"], x["delta"]), {})[x["rule"]] = x
    bad = sum(1 for v in by.values() if abs(f(v["R5_audit"]["p_claim_T"]) - f(v["R5b_audit_binary"]["p_claim_T"])) > 0)
    add(out, "S9 internal", "S1 generator: P(R5b claims T) == P(R5 claims T) (mismatching scenario x delta)", bad, 0, ok=bad == 0)
    bad = sum(1 for v in by.values() for x in v.values()
              if abs(f(x["p_claim_T"]) + f(x["p_claim_N"]) + f(x["p_abstain"]) - 1) > 1e-9)
    add(out, "S9 internal", "S1 generator: claim frequencies sum to 1 (violations)", bad, 0, ok=bad == 0)
    bad = sum(1 for v in by.values() if f(v["R1h_raw_asr_high"]["p_claim_T"]) > f(v["R1_raw_asr"]["p_claim_T"]))
    add(out, "S9 internal", "S1 generator: P(R1h claims T) <= P(R1 claims T) (violations)", bad, 0, ok=bad == 0,
        note="a Wilson lower bound >= 0.5 implies a point rate > 0.5")
    arch = {r["scenario"]: r for r in K.read_csv(os.path.join(K.S1_DIR, "cells.csv"))}
    mine = {x["scenario"]: x for x in s1rows if x["set"] == "s1_cells"}
    worst = 0.0
    for name, r in arch.items():
        x = mine.get(name)
        if x is None:
            worst = float("inf")
            break
        worst = max(worst, abs(int(x["n"]) - int(r["n"])), abs(f(x["true_k"]) - f(r["true_k"])),
                    abs(f(x["discordance"]) - (f(r["pi10"]) + f(r["pi01"]))),
                    abs(f(x["C_rate"]) - (f(r["pi11"]) + f(r["pi01"]))))
    add(out, "S9 S1 cells vs results/tifs/s1/cells.csv", f"set s1_cells has the {len(arch)} archived scenarios "
        f"(n, k, discordance, C rate; max |diff|)", worst, 0.0, ok=worst < 1e-12 and len(mine) == len(arch))
    # side-by-side per-seed rows vs results/tifs/report/cells.csv
    cells_ref = {(r["fam"], r["dataset"], r["kd"], r["attack"], r["corpus"], r["seed"]): r
                 for r in K.read_csv(os.path.join(K.REPORT, "cells.csv")) if r["block"] == "G"}
    sbs = K.read_csv(os.path.join(OUT, "s9_clean_corpus_side_by_side.csv"))
    n_ok = n_all = 0
    for x in sbs:
        if not x["level"].startswith("seed") or x["delta"] != "0.05":
            continue
        corpus = x.get("corpus") or "clean"
        ref = cells_ref.get((x["fam"], x["dataset"], x["kd"], x["attack"], corpus, x["seeds"]))
        if ref is None:
            continue
        n_all += 1
        n_ok += abs(f(x["k"]) - f(ref["k"])) <= TOL and x["audit_verdict"] == ref["verdict@0.05"]
    add(out, "S9 side-by-side per-seed rows vs results/tifs/report/cells.csv", "k and verdict@0.05 equal (count)",
        n_ok, n_all, ok=n_ok == n_all and n_all > 0)
    n_prim = len({(x["fam"], x["dataset"], x["kd"], x["attack"]) for x in sbs if x["level"] == "primary"})
    add(out, "S9 side-by-side", "primary cells listed", n_prim, 31, ok=n_prim == 31)
    # ── S1 extra ─────────────────────────────────────────────────────
    for x in K.read_csv(os.path.join(OUT, "s1_extra_summary.csv")):
        if x["set"] != "real":
            continue
        _, fam, ds, kd, atk, s = x["scenario"].split("|")[0].split("_")
        ref = cells_ref.get((fam, ds, kd, atk, "clean", s[1:]))
        if ref is not None:
            add(out, "S1 extra real plug-in vs results/tifs/report/cells.csv", x["scenario"] + " k", f(x["true_k"]), f(ref["k"]),
                ok=abs(f(x["true_k"]) - f(ref["k"])) <= 1e-12)
    # frozen run_scenario reproduces the archived S1 grid rows (same seeds, reps 10000, R 2000)
    grid = S1.grid_scenarios()
    arch = {r["scenario"]: r for r in K.read_csv(os.path.join(K.S1_DIR, "grid.csv"))}
    picks = [i for i, sc in enumerate(grid) if sc.n == 912][:3]
    for i in picks:
        sc = grid[i]
        res = run_scenario(sc, reps=10000, R=2000, seed=1000 + i, tols_k=(0.02, 0.05, 0.10))
        ref = arch[sc.name]
        worst = max(abs(f(res[c]) - f(ref[c])) for c in ref if c != "scenario" and not math.isnan(f(ref[c])))
        add(out, "S1 frozen run_scenario vs archived results/tifs/s1/grid.csv", f"{sc.name} (seed {1000 + i}) max |diff|",
            worst, 0.0, ok=worst < 1e-12)
    # ── every CSV carries the status header ───────────────────────────
    for p in sorted(glob.glob(os.path.join(OUT, "*.csv"))):
        if os.path.basename(p) == "checks.csv":
            continue
        first = open(p).readline()
        add(out, "CSV header comment", os.path.basename(p), first[:2], "# ",
            ok=first.startswith("# ") and "does not change any primary verdict" in first)
    K.write_csv(out, os.path.join(OUT, "checks.csv"),
                "Self-checks of the g4 outputs against results/tifs/report (frozen analyze.py) and results/tifs/addendum "
                "(addendum.py) where they overlap, against the archived S1 grid and cell set, and internal consistency; "
                "input fingerprints; pass = equal within 1e-9 (strings exact) unless noted.")
    n_fail = sum(1 for r in out if not r["pass"])
    print(f"{len(out)} checks, {n_fail} failed")
    for r in out:
        if not r["pass"]:
            print("FAIL", r)
    K.log_cost(OUT, "checks.py", t_start)


if __name__ == "__main__":
    main()
