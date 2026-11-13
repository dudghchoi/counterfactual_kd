#!/usr/bin/env python3
"""A14 item 9 — delta rationale table.

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Per cell of the 31-cell family (frozen primary rows, qualified):
  delta_star          the frozen primary delta* = max(|lo|, |hi|) (smallest delta with EQUIVALENT)
  P_rate, C_rate      the primary raw_asr and C_trig (mean over the cell's seeds of the hit rate on the
                      n triggered non-target inputs); P_hits, C_hits = rate x n x r
  cohen_h             2 asin(sqrt(P)) - 2 asin(sqrt(C)) at those rates (= g1 h_PC_at_mean_rates)
  log_or              logit(P) - logit(C); a rate of 0 or 1 would use (hits + 0.5) / (n r + 1) (flagged;
                      no cell needs it)
  h_at_delta@d, log_or_at_delta@d   the same scales for a shift of +d from the cell's C rate
                      (what delta = d means relative to that cell's baseline)
  halfwidth           t_{0.975,df} x SE of the primary interval (hi - k): the upper bound the cell would have
                      at k = 0 with its own SE (structural nulls: the Tango half-width)
  floor_battery_r3    A11.8 (aux/g3 seeds_floor.csv, r = 3, frozen logit pools): the k = 0 upper-bound floor
                      t_{0.975,df} sqrt(s2_pool / 3 + v_typ) of the battery's pool; recomputed here with the
                      frozen t_contrast as a check; only for the seed-pooled cells that use that pool
  p_pos0, holm31_p_pos0   one-sided p of H0: k <= 0 (frozen t_contrast at tol 0) and its Holm adjustment over
                      the 31 cells, holm19_p_pos0 over the 19 logit cells (structural nulls: P = C identically, p = 1)
  sig_pos_class@d     for a cell with lo > 0 (k > 0 at one-sided 0.025) that is not POSITIVE at d (lo <= d):
                      'inside (hi < d)', 'point inside (k <= d <= hi)', 'boundary (k > d, lo <= d)';
                      'POSITIVE' when lo > d; '' when lo <= 0
Per battery: the floor, the smallest delta at which the battery's IUT would pass (max over its cells of hi),
the A/A between-build variation (report/aa.csv, clean corpus, teacher seeds 43/44 vs 42; report/
reference_variance.csv SD) for the same family and dataset.

Outputs: delta_rationale_cells.csv, delta_rationale_batteries.csv, delta_rationale_aa.csv,
delta_rationale_sig_inside.csv (the 'k > 0 significant but not shown to exceed delta' cells).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import g6m_common as G  # noqa: E402

A, TOLS, ALPHA = G.A, G.TOLS, G.ALPHA


def h(p, c):
    return 2 * math.asin(math.sqrt(p)) - 2 * math.asin(math.sqrt(c))


def logit(x):
    return math.log(x / (1 - x))


def rates(r):
    n, nr = int(r["n"]), int(r["n"]) * len(str(r["seeds"]).split(","))
    P, C = float(r["raw_asr"]), float(r["C_trig"])
    flag = ""
    Pl, Cl = P, C
    if not (0 < P < 1 and 0 < C < 1):
        Pl, Cl = (P * nr + 0.5) / (nr + 1), (C * nr + 0.5) / (nr + 1)
        flag = "Haldane 0.5"
    return n, nr, P, C, Pl, Cl, flag


def floors():
    rows = G.read_csv(os.path.join(G.AUX, "g3-ladder-seeds-simultaneous", "seeds_floor.csv"))
    out = {}
    for r in rows:
        if r["scope"].startswith("31-cell family") and r["r"] == "3":
            out[r["pool"]] = r
    return out


def main():
    ap = G.add_out_arg(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = ap.parse_args()
    out = G.out_dir(args.out)
    st = G.state()
    qual = G.qualified_primary(st)
    fl = floors()
    g1 = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in
          G.read_csv(os.path.join(G.AUX, "g1-exposure-controls-tokens", "exposure_primary_cells.csv"))}
    # k > 0 one-sided p (tol 0) and its Holm over the 31 cells
    p0 = []
    for r in qual:
        if r["method"] == "structural":
            p0.append(1.0)
        else:
            p0.append(G.t_contrast(r["k"], r["se"], r["df"], (0.0,))["p_pos@0"])
    p0_holm = G.holm(p0)
    idx19 = [i for i, r in enumerate(qual) if r["kd"] == "logit"]
    p0_holm19 = dict(zip(idx19, G.holm([p0[i] for i in idx19])))
    cells, sig = [], []
    for ci, (r, pz, pzh) in enumerate(zip(qual, p0, p0_holm)):
        cls = G.a4_class(r)
        n, nr, P, C, Pl, Cl, flag = rates(r)
        g1r = g1.get((r["fam"], r["dataset"], r["kd"], r["attack"]))
        pool = r.get("variance_from", "")
        floor_row = fl.get(pool) if r["method"].startswith("seed-pooled") else None
        floor_chk = ""
        if floor_row is not None:
            se_f = (float(floor_row["s_pool"]) ** 2 / 3 + float(floor_row["v_typ"])) ** 0.5
            floor_chk = G.t_contrast(0.0, se_f, float(floor_row["df_pool"]), (0.05,))["hi"]
        row = {"cell": G.cell_id(r), "display": G.display(dict(r, a4_class=cls)), "fam": r["fam"],
               "dataset": r["dataset"], "kd": r["kd"], "attack": r["attack"], "method": r["method"],
               "a4_class": cls, "seeds": r["seeds"], "n": n, "variance_from": pool,
               "k": r["k"], "lo": r["lo"], "hi": r["hi"], "delta_star": r["delta_star"],
               **{f"verdict@{t:g}": r[f"verdict@{t:g}"] for t in TOLS},
               "P_rate": P, "C_rate": C, "P_hits": round(P * nr), "C_hits": round(C * nr), "n_times_r": nr,
               "cohen_h": h(Pl, Cl), "cohen_h_g1": G.fnum(g1r["h_PC_at_mean_rates"]) if g1r else "",
               "log_or": logit(Pl) - logit(Cl), "rate_correction": flag,
               **{f"h_at_delta@{t:g}": h(min(C + t, 1.0), C) for t in TOLS},
               **{f"log_or_at_delta@{t:g}": (logit(min(Cl + t, 1 - 1e-12)) - logit(Cl)) for t in TOLS},
               "halfwidth": r["hi"] - r["k"],
               "floor_battery_r3": float(floor_row["floor"]) if floor_row else "",
               "floor_battery_r3_recomputed": floor_chk,
               "p_pos0": pz, "holm31_p_pos0": pzh, "k_gt_0_holm31": pzh < ALPHA,
               "holm19_p_pos0": p0_holm19.get(ci, ""),
               "k_gt_0_holm19": (p0_holm19[ci] < ALPHA) if ci in p0_holm19 else ""}
        for t in TOLS:
            if r["lo"] <= 0:
                c = ""
            elif r["lo"] > t:
                c = "POSITIVE"
            elif r["hi"] < t:
                c = "inside (hi < d)"
            elif r["k"] <= t:
                c = "point inside (k <= d <= hi)"
            else:
                c = "boundary (k > d, lo <= d)"
            row[f"sig_pos_class@{t:g}"] = c
            if c and c != "POSITIVE":
                sig.append({"delta": t, "cell": G.cell_id(r), "display": row["display"], "class": c,
                            "k": r["k"], "lo": r["lo"], "hi": r["hi"], "verdict": r[f"verdict@{t:g}"],
                            "p_pos0": pz, "holm31_p_pos0": pzh, "holm19_p_pos0": p0_holm19.get(ci, ""),
                            "P_rate": P, "C_rate": C,
                            "cohen_h": row["cohen_h"], "log_or": row["log_or"]})
        cells.append(row)
    # A/A between-build variation (clean corpus)
    aa_rows = []
    for r in G.read_csv(os.path.join(G.REPORT, "aa.csv")):
        if r["corpus"] != "clean":
            continue
        aa_rows.append({"fam": r["fam"], "dataset": r["dataset"], "attack_trigger": r["attack"], "kd": r["kd"],
                        "teacher_seed_prime": int(r["teacher_seed"]),
                        **{c: G.fnum(r[c]) for c in ("C-C'", "C-C'_lo", "C-C'_hi", "P-C'", "P-C'_lo", "P-C'_hi",
                                                     "P'-C'", "P'-C'_lo", "P'-C'_hi", "P-P'", "P-P'_lo", "P-P'_hi",
                                                     "C-C'@atba", "C-C'@atba_lo", "C-C'@atba_hi")},
                        **{f"C-C'_verdict@{t:g}": r[f"C-C'_verdict@{t:g}"] for t in TOLS},
                        **{f"C-C'@atba_verdict@{t:g}": r[f"C-C'@atba_verdict@{t:g}"] for t in TOLS}})
    refv = {(r["fam"], r["dataset"]): r for r in G.read_csv(os.path.join(G.REPORT, "reference_variance.csv"))}
    # per battery
    bats = []
    keys = []
    for r in qual:
        key = (r["fam"], r["dataset"], r["kd"])
        if key not in keys:
            keys.append(key)
    for key in keys:
        rs = [c for c in cells if (c["fam"], c["dataset"], c["kd"]) == key]
        nonst = [c for c in rs if c["method"] != "structural"]
        pool = "/".join((key[0], key[1], "logit")) if (key[0], key[1], "logit") in st.pools else rs[0]["variance_from"]
        fr = fl.get(pool) if all(c["method"].startswith("seed-pooled") or c["method"] == "structural" for c in rs) else None
        aa = [a for a in aa_rows if (a["fam"], a["dataset"]) == key[:2]]
        rv = refv.get(key[:2])
        worst = max(rs, key=lambda c: c["hi"])
        bats.append({"battery": "/".join(key), "n_cells": len(rs), "methods": ",".join(sorted({c["method"] for c in rs})),
                     "variance_pool": ",".join(sorted({c["variance_from"] for c in nonst})),
                     "floor_r3": float(fr["floor"]) if fr else "",
                     "floor_r3_source": "aux/g3 seeds_floor.csv (r = 3, v_typ)" if fr else
                     "transported: per-cell halfwidth (no seed floor)",
                     "halfwidth_min": min(c["halfwidth"] for c in nonst), "halfwidth_max": max(c["halfwidth"] for c in nonst),
                     "iut_delta_min": worst["hi"], "iut_delta_min_cell": worst["cell"],
                     "delta_star_min_nonstructural": min(c["delta_star"] for c in nonst),
                     "aa_n_builds": len(aa),
                     "aa_max_abs_C-C'": max((abs(a["C-C'"]) for a in aa), default=""),
                     "aa_max_abs_P'-C'": max((abs(a["P'-C'"]) for a in aa), default=""),
                     "aa_max_abs_C-C'@atba": max((abs(a["C-C'@atba"]) for a in aa if a["C-C'@atba"] == a["C-C'@atba"]),
                                                  default=""),
                     "reference_sd_k_badnets": (float(rv["s2"]) ** 0.5) if rv else "",
                     "reference_df": rv["df"] if rv else "",
                     "C_rate_min": min(c["C_rate"] for c in rs), "C_rate_max": max(c["C_rate"] for c in rs),
                     **{f"h_at_delta@{t:g}_range": f"{min(c[f'h_at_delta@{t:g}'] for c in rs):.3f}-"
                                                    f"{max(c[f'h_at_delta@{t:g}'] for c in rs):.3f}" for t in TOLS}})
    G.write_csv(cells, os.path.join(out, "delta_rationale_cells.csv"),
                "A14 item 9, per cell of the 31-cell family: frozen primary k, interval, delta*, verdicts; P and C hit "
                "rates (primary raw_asr, C_trig; mean over seeds), Cohen's h = 2asin(sqrt P) - 2asin(sqrt C) (g1 value "
                "beside it), log-odds ratio logit(P) - logit(C); h and log OR of a +delta shift from the cell's C rate; "
                "halfwidth = hi - k (= the k = 0 upper bound with the cell's SE); floor_battery_r3 = aux/g3 seeds_floor "
                "r = 3 floor of the pool (seed-pooled cells; recomputed with t_contrast beside it); p_pos0 = one-sided p "
                "of k <= 0 (frozen t_contrast at tol 0; structural nulls 1) and its Holm over the 31 cells and over the 19 "
                "logit cells; "
                "sig_pos_class@d for lo > 0 cells: POSITIVE (lo > d), inside (hi < d), point inside (k <= d <= hi), "
                "boundary (k > d >= lo).")
    G.write_csv(bats, os.path.join(out, "delta_rationale_batteries.csv"),
                "A14 item 9, per battery: k = 0 upper-bound floor at r = 3 (aux/g3 seeds_floor.csv; transported "
                "batteries have none and show the per-cell halfwidth range), the smallest delta at which the battery "
                "IUT would pass (max hi over its cells), the A/A between-build variation for the same family and dataset "
                "(report/aa.csv clean corpus: max |C - C'| on the BadNets trigger, max |P' - C'|, max |C - C'| on the "
                "ATBA phrase; report/reference_variance.csv SD of k across teacher seeds), the C-rate range and the "
                "Cohen's h range of delta.")
    G.write_csv(aa_rows, os.path.join(out, "delta_rationale_aa.csv"),
                "A14 item 9, the A/A between-build contrasts on the clean corpus (report/aa.csv rows copied: C - C', "
                "P - C', P' - C', P - P' on the BadNets trigger and C - C' on the ATBA phrase, Tango intervals and "
                "verdicts), teacher seed 43/44 builds against the seed-42 battery build.")
    G.write_csv(sig, os.path.join(out, "delta_rationale_sig_inside.csv"),
                "A14 item 9, cells with k > 0 at one-sided 0.025 (primary lo > 0) that are not POSITIVE at delta "
                "(lo <= delta), per delta: class inside (hi < delta) / point inside (k <= delta <= hi) / boundary "
                "(k > delta >= lo), with the Holm-adjusted (31 cells) p of k > 0 and the relative scales.")


if __name__ == "__main__":
    main()
