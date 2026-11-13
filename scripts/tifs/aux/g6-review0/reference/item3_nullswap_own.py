#!/usr/bin/env python3
"""A14 item 3: the A3.4 within-build null swaps re-judged with each swap's own seed variance.

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

For every row of results/tifs/addendum/cross_null.csv (CPU fp32 hit vectors of
cross_hits_bert_{sst2,agnews}.json) with >= 2 student seeds, the frozen seed-pooled interval
(pool variance, pool df; addendum.cross_tables) is recomputed and set beside the two A3.1
readings (addendum.attack_variance_table's formulas, input term included):
  own : SE^2 = s2_a / r + v,                    df = r - 1
  max : SE^2 = max(s2_a, s2_pool) / r + v,      df = pool df
with s2_a the sample variance of the swap's per-seed k_s and v = Var_i(mean_s d_s,i) / n the
frozen input term. A row whose pooled verdict differs from either at a delta is
'pooling-dependent' there (A3 rule). One-seed (transported) rows have no own variance and are
left as they are. The own-P rows are the same computation for the attack's own P; the frozen
A3.1 (attack_variance.csv, official bf16) is shown beside them.

Also shown: the review agents' approximation (review0/nullswap_own.py: own variance with the
input term left out, df 2), to say where the numbers differ.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
from collections import defaultdict  # noqa: E402

import common as K  # noqa: E402


def main():
    out = K.out_dir()
    st = K.state()
    paths = [K.CROSS["sst2"], K.CROSS["agnews"]]
    frozen_rows, _ = K.AD.cross_tables(st, paths)
    official = {(r["fam"], r["dataset"], r["kd"], r["trigger"], r["student"]): r
                for r in K.read_csv(os.path.join(K.ADD, "cross_null.csv"))}
    av = {(r["dataset"], r["kd"], r["attack"]): r
          for r in K.read_csv(os.path.join(K.ADD, "attack_variance.csv")) if r["fam"] == "bert"}
    X = {"sst2": K.load_cross("sst2"), "agnews": K.load_cross("agnews")}
    rows, checks = [], []
    for fr in frozen_rows:
        ds, kd, atk, student = fr["dataset"], fr["kd"], fr["trigger"], fr["student"]
        key = ("bert", ds, kd, atk, student)
        off = official.get(key)
        hits = X[ds]["hits"][atk]
        seeds = [int(s) for s in fr["seeds"].split(",")]
        _, s2p, dfp = K.pool(st, ds, kd)
        ref = {s: f"{kd}/{K.C_TAG[atk]}/clean/s{s}" for s in seeds}
        d = {s: K.diff(hits[f"{kd}/{student}/s{s}"], hits[ref[s]]) for s in seeds}
        row = {"dataset": ds, "kd": kd, "trigger": atk, "student": student, "role": fr["role"],
               "vs": fr["vs"], "seeds": fr["seeds"], "method": fr["method"],
               "precision": "CPU fp32 re-score (cross_hits_bert_%s.json)" % ds,
               "variance_from": fr["variance_from"], "pool_s": s2p ** 0.5, "pool_df": dfp}
        if len(seeds) >= 2:
            res = K.pooled_and_own(d, s2p, dfp)
            pooled, own, mx = res["pooled"], res["own"], res["max"]
            row.update(k=res["k"], **{f"k_s{s}": x for s, x in zip(seeds, res["ks"])},
                       s_own=res["s2a"] ** 0.5, F_own=res["s2a"] / s2p, input_se=res["v"] ** 0.5,
                       pooled_lo=pooled["lo"], pooled_hi=pooled["hi"], **K.verdicts(pooled, "pooled_"),
                       own_df=len(seeds) - 1, own_lo=own["lo"], own_hi=own["hi"], **K.verdicts(own, "own_"),
                       max_lo=mx["lo"], max_hi=mx["hi"], **K.verdicts(mx, "max_"))
            row["pooling_dependent"] = ",".join(
                f"{t:g}" for t in K.TOLS
                if pooled[f"verdict@{t:g}"] != own[f"verdict@{t:g}"]
                or pooled[f"verdict@{t:g}"] != mx[f"verdict@{t:g}"])
            # Review approximation: own variance, input term omitted, df r - 1.
            approx = K.t_contrast(res["k"], (res["s2a"] / len(seeds)) ** 0.5, len(seeds) - 1, K.TOLS)
            row.update(review_approx_lo=approx["lo"], review_approx_hi=approx["hi"],
                       review_approx_verdict_05=approx["verdict@0.05"],
                       own_minus_approx_lo=own["lo"] - approx["lo"],
                       own_minus_approx_hi=own["hi"] - approx["hi"])
        else:
            pc, tr = K.transported(hits[f"{kd}/{student}/s{seeds[0]}"], hits[ref[seeds[0]]], s2p, dfp)
            row.update(k=pc["k"], **{f"k_s{seeds[0]}": pc["k"]},
                       pooled_lo=tr["lo"], pooled_hi=tr["hi"], **K.verdicts(tr, "pooled_"),
                       pooling_dependent="n/a (one seed: no own variance)")
        if fr["role"] == "own P":
            o = av.get((ds, kd, atk))
            if o is not None:
                row.update(official_A31_k=float(o["k"]), official_A31_own_lo=float(o["own_lo"]),
                           official_A31_own_hi=float(o["own_hi"]),
                           official_A31_max_lo=float(o["max_lo"]), official_A31_max_hi=float(o["max_hi"]),
                           **{f"official_A31_own_verdict@{t:g}": o[f"own_verdict@{t:g}"] for t in K.TOLS},
                           official_A31_pooling_dependent=o["pooling_dependent"])
        rows.append(row)
        # Self-check: the frozen cross_tables row and the official CSV are reproduced.
        if off is not None:
            dmax = max(abs(float(off[c]) - float(fr[c])) for c in ("k", "lo", "hi"))
            dmine = max(abs(float(off["k"]) - row["k"]), abs(float(off["lo"]) - row["pooled_lo"]),
                        abs(float(off["hi"]) - row["pooled_hi"]))
            same_v = all(off[f"verdict@{t:g}"] == fr[f"verdict@{t:g}"] == row[f"pooled_verdict@{t:g}"]
                         for t in K.TOLS)
            checks.append({"key": "/".join(key), "max_abs_diff_frozen_vs_csv": dmax,
                           "max_abs_diff_mine_vs_csv": dmine, "verdicts_equal": same_v})
    K.write_csv(rows, os.path.join(out, "nullswap_own_variance.csv"), """
        One row per A3.4 swap (results/tifs/addendum/cross_null.csv): a student scored on another
        attack's triggered inputs (role 'null' = no backdoor for that trigger; 'own P' = the
        attack's own P) against that trigger's trainer-matched clean student, CPU fp32 hits.
        pooled_* = frozen seed-pooled rule (k = mean_s k_s, SE^2 = s2_pool/r + v, pool df) or the
        frozen transported rule for one-seed rows; own_* = A3.1 own variance (SE^2 = s2_a/r + v,
        df r-1); max_* = A3.1 max(s2_a, s2_pool)/r + v at pool df; v = Var_i(mean_s d_s,i)/n;
        pooling_dependent lists the deltas where pooled differs from own or max; review_approx_* =
        review0/nullswap_own.py (own variance without v, df r-1); official_A31_* = the frozen
        A3.1 row of attack_variance.csv (official bf16 vectors) for own-P rows.""")
    # Summary per (dataset, kd, trigger): POSITIVE counts of the null swaps under each rule.
    summ = []
    grp = defaultdict(list)
    for r in rows:
        grp[(r["dataset"], r["kd"], r["trigger"])].append(r)
    for (ds, kd, atk), rs in sorted(grp.items()):
        nulls = [r for r in rs if r["role"] == "null"]
        own = next((r for r in rs if r["role"] == "own P"), None)
        rep = [r for r in nulls if "own_lo" in r]
        row = {"dataset": ds, "kd": kd, "trigger": atk, "n_null": len(nulls),
               "n_null_replicated": len(rep)}
        # Counts over the replicated rows are 'n/a' when there are none (one-seed groups: the
        # own and max rules do not exist there, so a 0 would be a count over an empty set).
        na = "n/a (no replicated null swap)"
        for t in K.TOLS:
            for rule in ("pooled", "own", "max"):
                if rule != "pooled" and not rep:
                    row[f"null_POSITIVE_{rule}@{t:g}"] = na
                    continue
                vs = [r[f"{rule}_verdict@{t:g}"] for r in (nulls if rule == "pooled" else rep)]
                row[f"null_POSITIVE_{rule}@{t:g}"] = sum(v == "POSITIVE" for v in vs)
            row[f"null_POSITIVE_pooled_replicated@{t:g}"] = sum(
                r[f"pooled_verdict@{t:g}"] == "POSITIVE" for r in rep) if rep else na
            row[f"null_pooling_dependent@{t:g}"] = sum(
                f"{t:g}" in r["pooling_dependent"].split(",") for r in rep) if rep else na
        row["null_POSITIVE_pooled_students"] = ";".join(
            f"{r['student']}@" + ",".join(f"{t:g}" for t in K.TOLS if r[f"pooled_verdict@{t:g}"] == "POSITIVE")
            for r in nulls if any(r[f"pooled_verdict@{t:g}"] == "POSITIVE" for t in K.TOLS))
        row["null_pooling_dependent_students"] = ";".join(
            f"{r['student']}@{r['pooling_dependent']}" for r in rep if r["pooling_dependent"])
        if own is not None:
            row.update(own_P_k=own["k"], own_P_pooled_lo=own["pooled_lo"], own_P_pooled_hi=own["pooled_hi"],
                       **{f"own_P_pooled_verdict@{t:g}": own[f"pooled_verdict@{t:g}"] for t in K.TOLS})
            if "own_lo" in own:
                row.update(own_P_own_lo=own["own_lo"], own_P_own_hi=own["own_hi"],
                           **{f"own_P_own_verdict@{t:g}": own[f"own_verdict@{t:g}"] for t in K.TOLS},
                           own_P_max_lo=own["max_lo"], own_P_max_hi=own["max_hi"],
                           **{f"own_P_max_verdict@{t:g}": own[f"max_verdict@{t:g}"] for t in K.TOLS},
                           own_P_pooling_dependent=own["pooling_dependent"])
        summ.append(row)
    K.write_csv(summ, os.path.join(out, "nullswap_own_summary.csv"), """
        One row per (dataset, KD, trigger) of A3.4: number of null swaps (n_null; n_null_replicated
        = those with >= 2 student seeds, the only ones with an own variance) that are POSITIVE at
        delta under the frozen pooled rule (all null rows; '_replicated' = the replicated ones),
        the A3.1 own rule and the A3.1 max rule ('n/a' when n_null_replicated = 0: one-seed rows
        have no own variance and keep the frozen transported reading); how many replicated null
        swaps are pooling-dependent; null_POSITIVE_pooled_students = the null swaps POSITIVE
        under the frozen pooled/transported rule and at which deltas; and the attack's own P
        under the same three rules (CPU fp32).""")
    K.write_csv(checks, os.path.join(out, "check_item3_crossnull.csv"), """
        Self-check of item 3: each cross_null.csv row recomputed by the frozen
        addendum.cross_tables and by this script's path (common.pooled_and_own /
        common.transported); maximum absolute difference of k, lo, hi and verdict equality.""")
    bad = [c for c in checks if c["max_abs_diff_mine_vs_csv"] > 1e-12 or not c["verdicts_equal"]]
    print(f"item 3: {len(rows)} rows, {len(checks)} official rows checked, {len(bad)} mismatches")


if __name__ == "__main__":
    main()
