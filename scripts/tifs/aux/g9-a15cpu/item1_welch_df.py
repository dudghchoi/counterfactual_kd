#!/usr/bin/env python3
"""A15 E3: the primary cells, battery decisions and the 31-cell Holm with the Welch df of each pool.

A15 post hoc CPU analysis; changes no primary verdict, battery decision or the paper-wide Holm.

The frozen pool variance s2_pool = mean_a s2_a (analyze.pool_seed_variance) carries a Satterthwaite df
from MS_seed / MS_res that assumes one residual variance for every attack. E3 replaces only that df by the
Welch-Satterthwaite df of the average of per-attack variances, nu_W = (sum_a s2_a)^2 (r - 1) / sum_a s2_a^2
(every s2_a on r - 1 df, attacks independent), and by min(frozen df, nu_W). Estimate and SE of every cell
stay the frozen ones (k, se from analyze.primary_table): the interval, the verdicts and the one-sided
p-values are re-derived with the frozen t_contrast at the new df, for the seed-pooled cells and for the
one-seed transported cells that borrow a pool ('feature' cells, Qwen3 AG News). Structural nulls keep
their Tango interval (no pool, no df). The battery decisions and transfers_global come from the frozen
analyze.primary_battery, the A3.9 Holm table from the frozen addendum.global_holm_table, both run on the
re-derived rows. The robust_* sensitivity column of primary.csv (A/A reference variance added, pool df)
is re-derived at the new df as well.

Check: the same tables computed a second way, by running the frozen analyze.primary_table with
analyze.pool_seed_variance wrapped to return the Welch (or min) df, agree exactly (check_item1.csv).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

import common as K  # noqa: E402

A, AD, TOLS, ALPHA = K.A, K.AD, K.TOLS, K.ALPHA
FIELDS = ["k", "se", "df", "lo", "hi", "delta_star"] + [
    f"{p}@{t:g}" for t in TOLS for p in ("verdict", "p_up", "p_tost", "p_pos")]
VARIANTS = ("welch", "min")


def rederive(st, df_of):
    """Frozen primary rows with interval/verdicts/p re-derived at df_of[variance_from]."""
    rows = []
    for r in st.primary:
        new = dict(r)
        src = r.get("variance_from")
        if r["method"] != "structural" and src:
            key = tuple(src.split("/"))
            df = df_of[key]
            res = K.t_contrast(r["k"], r["se"], df, TOLS)
            new.update({f: res[f] for f in FIELDS})
            if r.get("robust_from"):
                ref = st.ref_var[tuple(r["robust_from"].split("/"))]
                rob = K.t_contrast(r["k"], (r["se"] ** 2 + ref["s2"]) ** 0.5, df, (0.05,))
                new.update(robust_lo=rob["lo"], robust_hi=rob["hi"],
                           **{"robust_verdict@0.05": rob["verdict@0.05"]})
        rows.append(new)
    return rows


class _Proxy:
    def __init__(self, primary):
        self.primary = primary


def via_patched_pool(st, which):
    """analyze.primary_table with pool_seed_variance returning the Welch / min df (cross-check)."""
    orig = A.pool_seed_variance

    def patched(ks_by_attack):
        p = orig(ks_by_attack)
        if p is None:
            return p
        seeds = p["seeds"]
        nu = K.welch_df([K.var([ks_by_attack[a][s] for s in seeds]) for a in p["attacks"]], len(seeds))
        return dict(p, df=nu if which == "welch" else min(p["df"], nu))

    A.pool_seed_variance = patched
    try:
        rows, _ = A.primary_table(st.cells, st.structural, st.near_null, ref_var=st.ref_var)
    finally:
        A.pool_seed_variance = orig
    return rows


def headline(rows):
    head = [r for r in rows if r["kd"] == A.PRIMARY_KD and r["teacher_qualified"]]
    return {t: max(r[f"p_up@{t:g}"] for r in head) for t in TOLS}, len(head)


def main():
    out = K.out_dir()
    st = K.state()
    W = K.welch_pools(st)
    df_of = {"welch": {k: v["df_welch"] for k, v in W.items()},
             "min": {k: v["df_min"] for k, v in W.items()}}
    checks = []

    # ── pools ──
    from scipy import stats
    pool_rows = []
    for key, w in W.items():
        b = stats.bartlett(*[w["ks"][a] for a in w["attacks"]])
        users = sorted(K.cell_id(r) for r in st.primary if r.get("variance_from") == "/".join(key))
        pool_rows.append({
            "pool": "/".join(key), "attacks": ",".join(w["attacks"]), "seeds": w["seeds"], "m": w["m"],
            "r": w["r"], "s_pool": w["s2_pool"] ** 0.5, "df_frozen": w["df_frozen"],
            "df_welch": w["df_welch"], "df_min": w["df_min"], "df_bound_low": w["r"] - 1,
            "df_bound_high": w["m"] * (w["r"] - 1), "ms_seed": w["ms_seed"], "ms_res": w["ms_res"],
            **{f"s_{a}": w["s2a"][a] ** 0.5 for a in w["attacks"]},
            **{f"F_{a}": w["s2a"][a] / w["s2_pool"] for a in w["attacks"]},
            "bartlett_stat": float(b.statistic), "bartlett_p": float(b.pvalue),
            "t975_frozen": float(stats.t.ppf(0.975, w["df_frozen"])),
            "t975_welch": float(stats.t.ppf(0.975, w["df_welch"])),
            "n_cells_using": len(users), "cells_using": ";".join(users)})
    K.write_csv(pool_rows, os.path.join(out, "welch_pools.csv"), """
        One row per seed-pooled battery pool of the frozen analysis (variance.csv '(pooled)' rows).
        s_pool = sqrt(mean_a s2_a), s2_a = between-seed variance of attack a's k_s over the pool seeds
        (variance.csv s_between^2; s_<attack> = sqrt(s2_a), F_<attack> = s2_a / s2_pool); df_frozen =
        analyze.pool_seed_variance Satterthwaite df (MS_seed, MS_res; one residual variance for all
        attacks); df_welch = (sum_a s2_a)^2 (r-1) / sum_a s2_a^2 (A15 E3; attacks independent, each s2_a on
        r-1 df); df_min = min(df_frozen, df_welch); df_bound_low/high = r-1 and m(r-1); bartlett_* =
        scipy Bartlett test of equal variance of the k_s across the pool's attacks (descriptive, r = 3 per
        attack); t975_* = the 0.975 t quantile at that df; cells_using = primary cells whose interval uses
        the pool (seed-pooled and transported).""")

    # ── cells ──
    alt = {v: rederive(st, df_of[v]) for v in VARIANTS}
    for v in VARIANTS:
        patched = via_patched_pool(st, v)
        pidx = {K.cell_id(r): r for r in patched}
        nbad = 0
        for r in alt[v]:
            p = pidx[K.cell_id(r)]
            for f in FIELDS + ["robust_lo", "robust_hi", "robust_verdict@0.05", "method", "variance_from"]:
                if r.get(f) != p.get(f):
                    nbad += 1
        checks.append({"check": f"item 1 [{v}]: re-derived rows == frozen primary_table with the {v} df in "
                                f"pool_seed_variance (all fields, exact)",
                       "expected": 0, "got": nbad, "ok": nbad == 0})
    frz = {K.cell_id(r): r for r in st.primary}
    cell_rows = []
    for r in st.primary:
        cid = K.cell_id(r)
        row = {"fam": r["fam"], "dataset": r["dataset"], "kd": r["kd"], "attack": r["attack"],
               "method": r["method"], "seeds": r["seeds"], "variance_from": r.get("variance_from", ""),
               "k": r["k"], "se": r.get("se"), "df_frozen": r.get("df"),
               "lo_frozen": r["lo"], "hi_frozen": r["hi"],
               **{f"verdict_frozen@{t:g}": r[f"verdict@{t:g}"] for t in TOLS}}
        for v in VARIANTS:
            a = {K.cell_id(x): x for x in alt[v]}[cid]
            row.update({f"df_{v}": a.get("df"), f"lo_{v}": a["lo"], f"hi_{v}": a["hi"],
                        **{f"verdict_{v}@{t:g}": a[f"verdict@{t:g}"] for t in TOLS},
                        **{f"p_pos_{v}@{t:g}": a[f"p_pos@{t:g}"] for t in TOLS},
                        **{f"p_up_{v}@{t:g}": a[f"p_up@{t:g}"] for t in TOLS}})
            row[f"changed_{v}"] = ",".join(f"{t:g}" for t in TOLS
                                           if a[f"verdict@{t:g}"] != r[f"verdict@{t:g}"])
        row.update(**{f"p_pos_frozen@{t:g}": r[f"p_pos@{t:g}"] for t in TOLS},
                   **{f"p_up_frozen@{t:g}": r[f"p_up@{t:g}"] for t in TOLS})
        if r.get("robust_from"):
            row.update({"robust_from": r["robust_from"], "robust_verdict_frozen@0.05": r["robust_verdict@0.05"],
                        **{f"robust_verdict_{v}@0.05": {K.cell_id(x): x for x in alt[v]}[cid]["robust_verdict@0.05"]
                           for v in VARIANTS}})
        if r["method"] == "structural":
            row["note"] = "structural null: seed-42 Tango interval, no pool df (unchanged)"
        cell_rows.append(row)
    K.write_csv(cell_rows, os.path.join(out, "welch_cells.csv"), """
        One row per primary cell (results/tifs/report/primary.csv, 31 qualified cells). k and se are the
        frozen ones; *_frozen = primary.csv; *_welch / *_min = interval [k -+ t_{0.975,df} se], verdicts at
        delta 0.02/0.05/0.10 and one-sided p-values (p_pos: H1 k > delta; p_up: H1 k < delta) from the
        frozen t_contrast at df = the Welch df of the cell's variance_from pool / min(frozen df, Welch df);
        changed_* lists the deltas whose verdict differs from primary.csv; robust_verdict_*@0.05 = the
        primary.csv robust_* sensitivity (A/A reference variance added) at the new df. Structural nulls
        keep their Tango interval.""")

    # ── batteries, global Holm, headline ──
    fb = {(b["fam"], b["dataset"], b["kd"]): b for b in A.primary_battery(st.primary)}
    ab = {v: {(b["fam"], b["dataset"], b["kd"]): b for b in A.primary_battery(alt[v])} for v in VARIANTS}
    bat_rows = []
    for key, b in fb.items():
        row = {"fam": key[0], "dataset": key[1], "kd": key[2], "attacks": b["attacks"], "methods": b["methods"]}
        for t in TOLS:
            for col in ("decision", "transfers", "all_equivalent", "transfers_global"):
                row[f"{col}_frozen@{t:g}"] = b[f"{col}@{t:g}"]
                for v in VARIANTS:
                    row[f"{col}_{v}@{t:g}"] = ab[v][key][f"{col}@{t:g}"]
        for v in VARIANTS:
            row[f"changed_{v}"] = ";".join(
                f"{col}@{t:g}" for t in TOLS for col in ("decision", "transfers", "all_equivalent", "transfers_global")
                if ab[v][key][f"{col}@{t:g}"] != b[f"{col}@{t:g}"])
        bat_rows.append(row)
    K.write_csv(bat_rows, os.path.join(out, "welch_battery.csv"), """
        One row per primary battery (results/tifs/report/battery.csv). decision (NO TRANSFER: every
        qualified attack's one-sided p_up < 0.025; TRANSFERS: a Holm-within-battery p_pos < 0.025; else
        INCONCLUSIVE), transfers (Holm-flagged attacks), all_equivalent (every TOST p < 0.025) and
        transfers_global (Holm over the 31 qualified cells) from the frozen analyze.primary_battery, on the
        frozen rows (_frozen = battery.csv) and on the rows re-derived at the Welch df (_welch) and at
        min(frozen, Welch) df (_min); changed_* lists the columns that differ.""")
    fg = {K.cell_id(r): r for r in AD.global_holm_table(st)}
    ag = {v: {K.cell_id(r): r for r in AD.global_holm_table(_Proxy(alt[v]))} for v in VARIANTS}
    gh_rows = []
    for cid, g in fg.items():
        row = {"cell": cid, "method": g["method"], "family_size": g["family_size"]}
        for t in TOLS:
            row[f"holm_frozen@{t:g}"] = g[f"holm_global@{t:g}"]
            row[f"transfers_global_frozen@{t:g}"] = g[f"transfers_global@{t:g}"]
            for v in VARIANTS:
                row[f"holm_{v}@{t:g}"] = ag[v][cid][f"holm_global@{t:g}"]
                row[f"transfers_global_{v}@{t:g}"] = ag[v][cid][f"transfers_global@{t:g}"]
        gh_rows.append(row)
    K.write_csv(gh_rows, os.path.join(out, "welch_global_holm.csv"), """
        One row per qualified primary cell: Holm-adjusted p_pos over all 31 qualified cells
        (addendum.global_holm_table, the A3.9 table) and the flag holm < 0.025, frozen (= global_holm.csv)
        and with the p_pos of the rows re-derived at the Welch df / min(frozen, Welch) df.""")
    hf, nh = headline(st.primary)
    hv = {v: headline(alt[v])[0] for v in VARIANTS}

    # ── every verdict that changes ──
    ch = []
    for row in cell_rows:
        r = frz[f"{row['fam']}/{row['dataset']}/{row['kd']}/{row['attack']}"]
        for v in VARIANTS:
            for t in TOLS:
                a, b = row[f"verdict_frozen@{t:g}"], row[f"verdict_{v}@{t:g}"]
                if a != b:
                    ch.append({"variant": v, "scope": "cell verdict", "item": K.cell_id(r), "delta": t,
                               "frozen": a, "new": b, "method": r["method"],
                               "variance_from": row["variance_from"], "k": r["k"],
                               "interval_frozen": f"[{row['lo_frozen']:+.4f}, {row['hi_frozen']:+.4f}]",
                               "interval_new": f"[{row[f'lo_{v}']:+.4f}, {row[f'hi_{v}']:+.4f}]",
                               "df_frozen": row["df_frozen"], "df_new": row[f"df_{v}"]})
            if row.get("robust_from") and row["robust_verdict_frozen@0.05"] != row[f"robust_verdict_{v}@0.05"]:
                ch.append({"variant": v, "scope": "cell robust_verdict@0.05 (primary.csv sensitivity column)",
                           "item": K.cell_id(r), "delta": 0.05, "frozen": row["robust_verdict_frozen@0.05"],
                           "new": row[f"robust_verdict_{v}@0.05"], "method": r["method"],
                           "variance_from": row["variance_from"], "k": r["k"],
                           "df_frozen": row["df_frozen"], "df_new": row[f"df_{v}"]})
    for row in bat_rows:
        bid = f"{row['fam']}/{row['dataset']}/{row['kd']}"
        for v in VARIANTS:
            for t in TOLS:
                for col in ("decision", "transfers", "all_equivalent", "transfers_global"):
                    a, b = row[f"{col}_frozen@{t:g}"], row[f"{col}_{v}@{t:g}"]
                    if a != b:
                        ch.append({"variant": v, "scope": f"battery {col}", "item": bid, "delta": t,
                                   "frozen": a if a != "" else "(none)", "new": b if b != "" else "(none)"})
    for row in gh_rows:
        for v in VARIANTS:
            for t in TOLS:
                a, b = row[f"transfers_global_frozen@{t:g}"], row[f"transfers_global_{v}@{t:g}"]
                if a != b:
                    ch.append({"variant": v, "scope": "global Holm flag (31 cells)", "item": row["cell"],
                               "delta": t, "frozen": a, "new": b,
                               "holm_frozen": row[f"holm_frozen@{t:g}"], "holm_new": row[f"holm_{v}@{t:g}"]})
    for v in VARIANTS:
        for t in TOLS:
            a, b = hf[t] < ALPHA, hv[v][t] < ALPHA
            if a != b:
                ch.append({"variant": v, "scope": f"headline IUT ({nh} qualified logit cells, max p_up)",
                           "item": "headline", "delta": t, "frozen": a, "new": b,
                           "p_frozen": hf[t], "p_new": hv[v][t]})
    if not ch:
        ch = [{"variant": "-", "scope": "none", "item": "no verdict changes"}]
    K.write_csv(ch, os.path.join(out, "welch_changes.csv"), """
        Every verdict that changes when the pool df is replaced by the Welch df (variant welch) or by
        min(frozen df, Welch df) (variant min): primary cell verdicts at delta 0.02/0.05/0.10, the
        primary.csv robust_verdict@0.05 sensitivity column, the battery decision / transfers /
        all_equivalent / transfers_global columns, the per-cell flag of the 31-cell Holm (A3.9) and the
        headline intersection-union over the qualified logit cells (max p_up < 0.025 at that delta).
        Frozen values are those of primary.csv / battery.csv / global_holm.csv; nothing here replaces them.""")
    hl = [{"delta": t, "n_cells": nh, "max_p_up_frozen": hf[t], "headline_frozen": hf[t] < ALPHA,
           **{f"max_p_up_{v}": hv[v][t] for v in VARIANTS},
           **{f"headline_{v}": hv[v][t] < ALPHA for v in VARIANTS}} for t in TOLS]
    K.write_csv(hl, os.path.join(out, "welch_headline.csv"), """
        Headline intersection-union test (analyze.summary): every qualified logit-KD primary cell has
        one-sided p_up < 0.025 at delta; frozen and with the Welch / min df.""")
    K.write_csv(checks, os.path.join(out, "check_item1.csv"), """
        Item 1 self-checks: the re-derived rows agree with a second computation (frozen
        analyze.primary_table with pool_seed_variance returning the alternative df).""")
    n_ch = sum(1 for c in ch if c.get("variant") == "welch")
    print(f"item 1: {len(cell_rows)} cells, {n_ch} changes under the Welch df, "
          f"{sum(1 for c in ch if c.get('variant') == 'min')} under min df; checks "
          f"{'ok' if all(c['ok'] for c in checks) else 'FAILED'}")


if __name__ == "__main__":
    main()
