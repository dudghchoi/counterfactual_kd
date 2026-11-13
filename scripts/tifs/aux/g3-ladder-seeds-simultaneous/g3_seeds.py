#!/usr/bin/env python3
"""A11 item 8 (δ와 seed 수): floor of the one-sided upper bound of a k = 0 cell, and the
probability that its upper bound falls below delta, as a function of the number of student
seeds r = 3..20, with the frozen SE^2 formula and the measured pools.

Post hoc / registered auxiliary table (docs/tifs_addendum.md A11.8). It does not change any
primary verdict, battery decision or the paper-wide Holm. Inputs: runs/tifs (Stage 2
predictions), results/tifs/report (analyze.py) and results/tifs/addendum (addendum.py).

Registered computation (per measured pool p = family x dataset x logit KD; the four pools of
the 31-cell family):
  SE(r)^2  = s2_pool / r + v_typ                      (frozen SE^2 of analyze.primary_table)
  df(r)    = df_pool * (r - 1) / 2                    (A11.8; equals the measured df at r = 3)
  floor(r) = t_{0.975, df(r)} * SE(r)                 (upper bound of a cell with k_hat = 0,
                                                       from the frozen t_contrast(0, SE, df))
  s2_pool, df_pool: the measured pool (analyze.pool_seed_variance), held fixed in r.
  v_typ: the frozen input term var_i(mean_s d_s,i)/n at the measured r = 3 of the median cell
         among the seed-pooled cells decided with that pool, all of them (primary-POSITIVE
         cells included; v_min / v_max: the extreme cells); held fixed in r, which overstates
         SE for r > 3 (more seeds average the per-input noise).
NO TRANSFER probability per cell, P(upper bound < delta | k = 0), two columns:
  p_below_nct        = 1 - F_nct(t_{0.975,df(r)}; df(r), delta/SE(r)) (scipy.stats.nct.sf):
                       exact when k_hat ~ N(0, SE^2) and SE_hat^2 / SE^2 ~ chi2_df(r) / df(r)
                       independently, the model under which the t interval is exact. Reported.
  p_below_shifted_t  = F_t,df(r)(delta/SE(r) - t_{0.975,df(r)}), the shifted central t used in
                       the first version, kept beside it. It is conservative (at or below
                       p_below_nct) in every row whose p_below_nct is below 0.78 (self-checked),
                       but slightly above p_below_nct in a band of rows with p_below_nct
                       between about 0.78 and 0.91; seeds_pbelow_compare.csv gives the exact
                       counts, range and largest excess.
Extra, labelled variants (not registered):
  *_vnonpos  v_typ_nonpos = median v over the seed-pooled cells of the pool whose primary
             verdict is POSITIVE at no delta (cells with a primary POSITIVE are k != 0 cells)
  s1b_*      v_a(r) = v1_a (1 + (r - 1) rho_a) / r, the S1b input-noise model
             (scripts/tifs/s1b_seed_noise.py), with v1_a = mean_s var_i(d_s,i)/n and
             rho_a = (3 v_a(3) / v1_a - 1) / 2 measured per cell (v_a(3) equals the frozen v)
  bert/sst2/hidden  the A8 hidden-state KD pool (seeds 42-44 admitted), a labelled extra pool
             outside the 31-cell family (scope column), built as addendum.hidden_tables builds it
Battery: p_battery_indep = P_below^m over the battery's m non-structural cells, independence
assumed; positive dependence (shared C students, one shared s2_pool estimate) makes the true
probability larger, so this is a lower bound and can be far too low (see README).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # before any local import: no __pycache__ in the repository

import os  # noqa: E402
import statistics  # noqa: E402
from collections import defaultdict  # noqa: E402

from scipy.stats import nct, norm  # noqa: E402
from scipy.stats import t as tdist  # noqa: E402

from g3_common import (  # noqa: E402
    A, AD, ADDENDUM, REPORT, RUNS, TOLS, close, out_dir, read_csv, t_contrast, tkey, write_csv,
)

RS = range(3, 21)
Z975 = 1.959963984540054
SHIFT = "shifted_t"
CONSERVATIVE_BELOW = 0.78  # shifted-t <= nct in every row with p_below_nct below this (checked)
SCOPE_FROZEN = "31-cell family (frozen logit pool)"
SCOPE_A8 = "A8 extra (hidden-state KD pool; outside the 31-cell family)"


def mean(x):
    return sum(x) / len(x)


def var(x):
    m = mean(x)
    return sum((y - m) ** 2 for y in x) / (len(x) - 1)


def p_below(t, se, df, q):
    """P(upper bound < t | k = 0): (noncentral t, shifted central t)."""
    return float(nct.sf(q, df, t / se)), float(tdist.cdf(t / se - q, df))


def cell_terms(by):
    """Per-seed d vectors of a seed-pooled cell -> seeds, n, frozen v (r = 3), v1, rho."""
    seeds = sorted(by)
    n = len(by[seeds[0]]["P"])
    d = {s: [p - q for p, q in zip(by[s]["P"], by[s]["C"])] for s in seeds}
    v1 = mean([var(d[s]) for s in seeds]) / n
    v3 = var([mean(col) for col in zip(*d.values())]) / n
    r0 = len(seeds)
    rho = (r0 * v3 / v1 - 1) / (r0 - 1)
    return seeds, n, v3, v1, rho


def hidden_build(st):
    """A8: admitted hidden seeds, primary rows, pools, attack variances and seed groups, built
    with the frozen functions exactly as addendum.hidden_tables builds them."""
    hid_all = AD.load_hidden_cells(st.runs)
    status = AD.hidden_seed_status(hid_all)
    ok = {(r["fam"], r["dataset"], r["seed"]) for r in status if r["status"] == "admitted"}
    hid = [c for c in hid_all if (c["fam"], c["dataset"], c["student_seed"]) in ok]
    base = [c for c in st.cells if c["kd"] != "hidden"]
    rows, var_rows = A.primary_table(base + hid, st.structural, st.near_null, ref_var=st.ref_var)
    prim = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in rows if r["kd"] == "hidden"}
    pools = {(v["fam"], v["dataset"], v["kd"]): v for v in var_rows if v["attack"] == "(pooled)" and v["kd"] == "hidden"}
    att_var = {(v["fam"], v["dataset"], v["kd"], v["attack"]): v for v in var_rows
               if v["attack"] != "(pooled)" and v["kd"] == "hidden"}
    groups = defaultdict(dict)
    for c in hid:
        if c["block"] == "G" and c["corpus"] == "clean" and not c["single"]:
            groups[(c["fam"], c["dataset"], c["kd"], c["attack"])][c["student_seed"]] = c
    return status, prim, pools, att_var, groups


def pool_tables(scope, pkey, pool, prim, groups, att_var, var_csv, checks, upper_if_k0=None):
    """seeds_floor / seeds_rmin / seeds_cells rows of one measured pool."""
    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    pname = "/".join(pkey)
    s2p, dfp = pool["s_between"] ** 2, pool["df"]
    pc = var_csv[pkey + ("(pooled)",)]
    check(f"pool {pname} == {var_csv['_name']}",
          close(pool["s_between"], pc["s_between"]) and close(pool["df"], pc["df"])
          and pool["attacks"] == pc["attacks"] and pool["seeds"] == pc["seeds"],
          f"s {pool['s_between']:.6f}, df {pool['df']:.4f}, attacks {pool['attacks']}, seeds {pool['seeds']}")
    users = sorted(k for k, pr in prim.items()
                   if pr["method"].startswith("seed-pooled") and pr.get("variance_from") == pname)
    battery_cells = [k for k, pr in prim.items()
                     if k[:3] == pkey and pr["teacher_qualified"] and pr["method"] != "structural"]
    m = len(battery_cells)
    cv, cell_rows = {}, []
    for key in users:
        seeds, n, v3, v1, rho = cell_terms(groups[key])
        pr = prim[key]
        check(f"v {'/'.join(key)} == frozen input_se^2 and {var_csv['_name']}",
              close(v3, att_var[key]["input_se"] ** 2) and close(v3, float(var_csv[key]["input_se"]) ** 2))
        se3 = (s2p / len(seeds) + v3) ** 0.5
        fl3 = t_contrast(0.0, se3, dfp, TOLS)["hi"]
        ok = close(se3, pr["se"]) and close(fl3, pr["hi"] - pr["k"])
        if upper_if_k0 is not None:
            ok = ok and close(fl3, upper_if_k0[key])
        check(f"floor r=3 {'/'.join(key)} == primary SE and half-width"
              + (" and decidability.csv upper_if_k0" if upper_if_k0 is not None else ""), ok, f"{fl3:.6f}")
        pos_at = [tkey(t) for t in TOLS if pr[f"verdict@{tkey(t)}"] == "POSITIVE"]
        cv[key] = {"v": v3, "v1": v1, "rho": rho, "n": n, "pos": bool(pos_at)}
        cell_rows.append({"scope": scope, "fam": key[0], "dataset": key[1], "kd": key[2], "attack": key[3],
                          "pool": pname, "n": n, "seeds": ",".join(map(str, seeds)), "k_primary": pr["k"],
                          "v_frozen_r3": v3, "sqrt_v": v3 ** 0.5, "v1_per_seed": v1, "rho": rho,
                          "se_primary": pr["se"], "floor_r3": fl3,
                          "primary_positive_at": ",".join(pos_at), "in_v_typ_nonpos": not pos_at})
    names = sorted(cv, key=lambda k: cv[k]["v"])
    vs = [cv[k]["v"] for k in names]
    v_typ = statistics.median(vs)
    mid = names[(len(names) - 1) // 2: len(names) // 2 + 1]
    nonpos = [k for k in names if not cv[k]["pos"]]
    v_nonpos = statistics.median([cv[k]["v"] for k in nonpos])
    mid_nonpos = nonpos[(len(nonpos) - 1) // 2: len(nonpos) // 2 + 1]

    rows, per_r = [], {}
    for r in RS:
        df = dfp * (r - 1) / 2
        q = float(tdist.ppf(0.975, df))
        row = {"scope": scope, "pool": pname, "s_pool": pool["s_between"], "df_pool": dfp,
               "pool_attacks": pool["attacks"], "cells_using_pool": ",".join(k[3] for k in users),
               "m_battery_cells": m, "v_typ": v_typ, "v_typ_cells": ",".join(k[3] for k in mid),
               "v_min": vs[0], "v_min_cell": names[0][3], "v_max": vs[-1], "v_max_cell": names[-1][3],
               "v_typ_nonpos": v_nonpos, "v_typ_nonpos_cells": ",".join(k[3] for k in mid_nonpos),
               "nonpos_cells": ",".join(k[3] for k in nonpos),
               "r": r, "df_r": df, "t_0.975": q}
        for tag, v in (("", v_typ), ("_vmin", vs[0]), ("_vmax", vs[-1]), ("_vnonpos", v_nonpos)):
            se = (s2p / r + v) ** 0.5
            res = t_contrast(0.0, se, df, TOLS)
            row[f"se{tag}"] = se
            row[f"floor{tag}"] = res["hi"]
            for t in TOLS:
                pn, ps = p_below(t, se, df, q)
                row[f"p_below{tag}_nct@{tkey(t)}"] = pn
                row[f"p_below{tag}_{SHIFT}@{tkey(t)}"] = ps
                if tag == "":
                    row[f"p_battery_indep_nct@{tkey(t)}"] = pn ** m
                    row[f"p_battery_indep_{SHIFT}@{tkey(t)}"] = ps ** m
                    row[f"decidable@{tkey(t)}"] = res["hi"] < t
        v_s1b = statistics.median([x["v1"] * (1 + (r - 1) * x["rho"]) / r for x in cv.values()])
        se = (s2p / r + v_s1b) ** 0.5
        row["s1b_v_typ"] = v_s1b
        row["s1b_se"] = se
        row["s1b_floor"] = t_contrast(0.0, se, df, TOLS)["hi"]
        for t in TOLS:
            pn, ps = p_below(t, se, df, q)
            row[f"s1b_p_below_nct@{tkey(t)}"] = pn
            row[f"s1b_p_below_{SHIFT}@{tkey(t)}"] = ps
        if r == 3:
            check(f"s1b variant equals registered at r=3 ({pname})", close(row["s1b_floor"], row["floor"], 1e-9))
            check(f"df(3) == df_pool ({pname})", close(row["df_r"], dfp))
        rows.append(row)
        per_r[r] = row

    def first(pred):
        hit = [r for r in RS if pred(per_r[r])]
        return hit[0] if hit else ">20"

    rmin_rows = []
    for t in TOLS:
        tk = tkey(t)
        rmin_rows.append({
            "scope": scope, "pool": pname, "delta": t, "s_pool": pool["s_between"], "df_pool": dfp,
            "m_battery_cells": m,
            "floor_r3": per_r[3]["floor"], "floor_r20": per_r[20]["floor"],
            "floor_limit_r_inf_fixed_v": Z975 * v_typ ** 0.5,
            "r_floor_below_delta": first(lambda x: x["floor"] < t),
            "r_floor_below_delta_vmin": first(lambda x: x["floor_vmin"] < t),
            "r_floor_below_delta_vmax": first(lambda x: x["floor_vmax"] < t),
            "r_floor_below_delta_s1b": first(lambda x: x["s1b_floor"] < t),
            **{f"p_below_{kind}_r{r}": per_r[r][f"p_below_{kind}@{tk}"] for kind in ("nct", SHIFT) for r in (3, 10, 20)},
            **{f"r_p_below_{kind}_ge_{lvl}": first(lambda x, kind=kind, lvl=lvl: x[f"p_below_{kind}@{tk}"] >= lvl)
               for kind in ("nct", SHIFT) for lvl in (0.8, 0.9)},
            **{f"r_s1b_p_below_{kind}_ge_0.8": first(lambda x, kind=kind: x[f"s1b_p_below_{kind}@{tk}"] >= 0.8)
               for kind in ("nct", SHIFT)},
            **{f"r_p_battery_indep_{kind}_ge_0.8": first(lambda x, kind=kind: x[f"p_battery_indep_{kind}@{tk}"] >= 0.8)
               for kind in ("nct", SHIFT)},
            **{f"p_battery_indep_{kind}_r{r}": per_r[r][f"p_battery_indep_{kind}@{tk}"] for kind in ("nct", SHIFT) for r in (3, 20)},
            "v_typ_nonpos": v_nonpos,
            "floor_vnonpos_r3": per_r[3]["floor_vnonpos"], "floor_vnonpos_r20": per_r[20]["floor_vnonpos"],
            "floor_limit_r_inf_fixed_vnonpos": Z975 * v_nonpos ** 0.5,
            "r_floor_below_delta_vnonpos": first(lambda x: x["floor_vnonpos"] < t),
            **{f"p_below_vnonpos_{kind}_r{r}": per_r[r][f"p_below_vnonpos_{kind}@{tk}"]
               for kind in ("nct", SHIFT) for r in (3, 10, 20)},
            **{f"r_p_below_vnonpos_{kind}_ge_0.8": first(lambda x, kind=kind: x[f"p_below_vnonpos_{kind}@{tk}"] >= 0.8)
               for kind in ("nct", SHIFT)}})
    return rows, rmin_rows, cell_rows


def main():
    out = out_dir(__doc__.splitlines()[0])
    st = AD.State(RUNS)
    checks = []
    check = lambda name, ok, detail="": checks.append({"check": name, "ok": bool(ok), "detail": detail})  # noqa: E731
    check("z_0.975 literal == scipy norm.ppf(0.975)", Z975 == float(norm.ppf(0.975)))

    def keyed(path, name):
        rows = read_csv(path)
        d = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in rows}
        d["_name"] = name
        return d

    var_frozen = keyed(os.path.join(REPORT, "variance.csv"), "report/variance.csv")
    var_hidden = keyed(os.path.join(ADDENDUM, "hidden_variance.csv"), "addendum/hidden_variance.csv")
    dec = {(r["fam"], r["dataset"], r["kd"], r["attack"]): float(r["upper_if_k0"])
           for r in read_csv(os.path.join(ADDENDUM, "decidability.csv")) if r.get("upper_if_k0")}
    prim_csv = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in read_csv(os.path.join(REPORT, "primary.csv"))}
    for key, pr in st.prim.items():
        c = prim_csv.get(key)
        check(f"primary verdicts {'/'.join(key)} == report/primary.csv",
              c is not None and all(pr[f"verdict@{tkey(t)}"] == c[f"verdict@{tkey(t)}"] for t in TOLS))

    rows, rmin_rows, cell_rows = [], [], []
    for pkey, pool in sorted(st.pools.items()):
        a, b, c = pool_tables(SCOPE_FROZEN, pkey, pool, st.prim, st.groups, st.att_var, var_frozen, checks, dec)
        rows += a
        rmin_rows += b
        cell_rows += c

    # A8 extra pool (outside the 31-cell family): checked against the official addendum outputs.
    status, hprim, hpools, hatt, hgroups = hidden_build(st)
    seeds_csv = read_csv(os.path.join(ADDENDUM, "hidden_seeds.csv"))
    check("A8 seed status == addendum/hidden_seeds.csv",
          [(r["seed"], r["status"]) for r in status] == [(int(r["seed"]), r["status"]) for r in seeds_csv],
          ", ".join(f"s{r['seed']} {r['status']}" for r in status))
    hcells_csv = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in read_csv(os.path.join(ADDENDUM, "hidden_cells.csv"))}
    for key, pr in hprim.items():
        c = hcells_csv.get(key, {})
        check(f"A8 primary {'/'.join(key)} == addendum/hidden_cells.csv",
              all(pr[f"verdict@{tkey(t)}"] == c.get(f"verdict@{tkey(t)}") for t in TOLS)
              and all(close(pr[x], c.get(x)) for x in ("k", "se", "df", "lo", "hi")) and pr["seeds"] == c.get("seeds"))
    check("A8 cell set == addendum/hidden_cells.csv", set(hprim) == set(hcells_csv), f"{len(hprim)} cells")
    for pkey, pool in sorted(hpools.items()):
        a, b, c = pool_tables(SCOPE_A8, pkey, pool, hprim, hgroups, hatt, var_hidden, checks)
        rows += a
        rmin_rows += b
        cell_rows += c

    # Shifted central t against the noncentral t, over every row, variant and delta.
    variants = (("v_typ (registered)", "p_below_nct@{}", "p_below_%s@{}" % SHIFT),
                ("v_min", "p_below_vmin_nct@{}", "p_below_vmin_%s@{}" % SHIFT),
                ("v_max", "p_below_vmax_nct@{}", "p_below_vmax_%s@{}" % SHIFT),
                ("v_typ_nonpos (extra)", "p_below_vnonpos_nct@{}", "p_below_vnonpos_%s@{}" % SHIFT),
                ("s1b (extra)", "s1b_p_below_nct@{}", "s1b_p_below_%s@{}" % SHIFT))

    def compare(name_pool, name_var, delta, pairs):
        above = [(pn, ps) for pn, ps in pairs if ps > pn + 1e-12]
        return {"pool": name_pool, "variant": name_var, "delta": delta, "n_rows": len(pairs),
                "n_shifted_above_nct": len(above),
                "max_shifted_minus_nct": max(ps - pn for pn, ps in pairs),
                "min_shifted_minus_nct": min(ps - pn for pn, ps in pairs),
                "nct_min_where_shifted_above": min((pn for pn, _ in above), default=""),
                "nct_max_where_shifted_above": max((pn for pn, _ in above), default="")}

    cmp_rows, every = [], []
    for pname in dict.fromkeys(r["pool"] for r in rows):
        for vname, cn, cs in variants:
            for t in TOLS:
                pairs = [(x[cn.format(tkey(t))], x[cs.format(tkey(t))]) for x in rows if x["pool"] == pname]
                every += pairs
                cmp_rows.append({"scope": next(x["scope"] for x in rows if x["pool"] == pname),
                                 **compare(pname, vname, t, pairs)})
    cmp_rows.append({"scope": "all", **compare("all", "all", "all", every)})
    low = [(pn, ps) for pn, ps in every if pn < CONSERVATIVE_BELOW]
    check(f"shifted-t P_below <= noncentral-t P_below in every row with noncentral-t P_below < {CONSERVATIVE_BELOW}",
          all(ps <= pn + 1e-12 for pn, ps in low),
          f"{len(low)} of {len(every)} rows; rows with shifted-t above: {cmp_rows[-1]['n_shifted_above_nct']}, "
          f"noncentral t there {cmp_rows[-1]['nct_min_where_shifted_above']:.4f}-{cmp_rows[-1]['nct_max_where_shifted_above']:.4f}, "
          f"max excess {cmp_rows[-1]['max_shifted_minus_nct']:.4f}")

    write_csv(rows, os.path.join(out, "seeds_floor.csv"))
    write_csv(rmin_rows, os.path.join(out, "seeds_rmin.csv"))
    write_csv(cell_rows, os.path.join(out, "seeds_cells.csv"))
    write_csv(cmp_rows, os.path.join(out, "seeds_pbelow_compare.csv"))
    write_csv(checks, os.path.join(out, "seeds_selfchecks.csv"))
    bad = [c for c in checks if not c["ok"]]
    print(f"seeds: {len(rows)} pool x r rows ({len(st.pools)} frozen pools + {len(hpools)} A8 pool); "
          f"checks {len(checks) - len(bad)}/{len(checks)} ok -> {out}")
    for c in bad:
        print("  FAILED", c)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
