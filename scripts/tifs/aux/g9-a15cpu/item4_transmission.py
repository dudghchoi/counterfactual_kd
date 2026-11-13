#!/usr/bin/env python3
"""A15 CPU item: teacher transmission per primary cell (audit D8, A15 E5) and KD-vs-R clean accuracy.

A15 post hoc CPU analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Data: the official Stage 2 hit vectors (GPU bf16) stored with every battery cell: on the n UNTRIGGERED
non-target test inputs, teacher_p_clean / teacher_c_clean (P and C teacher), preds_p_clean / preds_c_clean
(P and C student) and the CE-only student R's preds_p_clean (tifs_b_* ce cell, seed 42 only). A hit is
1 = predicts the attack's target label; on SST-2 (two classes) a hit vector is the full prediction, on AG
News (four classes) it is the target-vs-other prediction, so 'disagree' and 'follow' are read on the target
axis, the axis k is measured on. No per-input four-class teacher prediction is stored; none is needed for
this target-axis metric, so nothing was re-scored.

Per cell and student seed s, D = {i : P teacher hit_i != C teacher hit_i} (fixed: the teachers are the
seed-42 builds at every student seed):
  follow_P = share of D where the P student's hit equals the P teacher's
  match_C  = share of D where the C student's hit equals the P teacher's (= 1 - share following its own C)
  gap      = follow_P - match_C (the paired per-input difference on D)
  match_R  = share of D where R's hit equals the P teacher's (seed 42; R saw no teacher: the baseline)
Pooled over seeds: mean of the per-seed shares; the gap's interval is the A3.1 own-variance rule on the
per-input differences d_s,i = [P_s,i = TP_i] - [C_s,i = TP_i] (SE^2 = s2_gap/r + Var_i(mean_s d_s,i)/|D|,
t at r - 1 df; r = 3), and for one-seed cells the Tango paired interval of the two 0/1 vectors (input
sampling only). The same shares on the TRIGGERED inputs (teacher_p_trig / teacher_c_trig vs preds_obs /
preds_null) are given as a secondary column set. Accuracy: Stage 2 clean accuracy on the full test set
(SST-2 1,821, AG News 7,600) of the P and C students (ca_poisoned / ca_clean), of R (seed 42) and of the
teachers (teacher_ca / teacher_ca_null); KD - R and the share (KD - R) / (teacher - R).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

import common as K  # noqa: E402

TOLS = K.TOLS


def shares(TP, TC, P, C, R=None):
    D = [i for i in range(len(TP)) if TP[i] != TC[i]]
    fP = [int(P[i] == TP[i]) for i in D]
    mC = [int(C[i] == TP[i]) for i in D]
    out = {"D": D, "fP": fP, "mC": mC, "n_D": len(D),
           "n_D_Phit": sum(TP[i] for i in D),
           "follow_P": K.mean(fP) if D else float("nan"), "match_C": K.mean(mC) if D else float("nan")}
    out["gap"] = out["follow_P"] - out["match_C"]
    if R is not None:
        mR = [int(R[i] == TP[i]) for i in D]
        out["mR"] = mR
        out["match_R"] = K.mean(mR) if D else float("nan")
    return out


def summary_rows(seed_rows, cell_rows):
    groups = {}
    for c in cell_rows:
        for g in (f"{c['fam']}/{c['dataset']}/{c['kd']}", f"all {c['kd']}", "all cells"):
            groups.setdefault(g, []).append(c)
    out = []
    for g, cs in groups.items():
        ok = [c for c in cs if c["n_D_untrig"] > 0]
        ids = {(c["fam"], c["dataset"], c["kd"], c["attack"]) for c in ok}
        sr = [r for r in seed_rows if (r["fam"], r["dataset"], r["kd"], r["attack"]) in ids]
        acc = [c for c in cs if c.get("ca_P_minus_R_s42") is not None]
        out.append({
            "group": g, "n_cells": len(cs), "n_cells_with_D": len(ok),
            "cells_without_D": ",".join(f"{c['fam']}/{c['dataset']}/{c['kd']}/{c['attack']}"
                                        for c in cs if c["n_D_untrig"] == 0),
            "n_D_range": f"{min(c['n_D_untrig'] for c in ok)}-{max(c['n_D_untrig'] for c in ok)}" if ok else "",
            "follow_P_mean": K.mean([c["follow_P_untrig"] for c in ok]),
            "match_C_mean": K.mean([c["match_C_untrig"] for c in ok]),
            "gap_mean": K.mean([c["gap_untrig"] for c in ok]),
            "match_R_s42_mean": K.mean([c["match_R_untrig_s42"] for c in ok if c["match_R_untrig_s42"] is not None]),
            "follow_P_seed_min": min(r["follow_P_untrig"] for r in sr) if sr else None,
            "follow_P_seed_max": max(r["follow_P_untrig"] for r in sr) if sr else None,
            "match_C_seed_min": min(r["match_C_untrig"] for r in sr) if sr else None,
            "match_C_seed_max": max(r["match_C_untrig"] for r in sr) if sr else None,
            "n_cells_gap_lo_above_0": sum(1 for c in ok if c.get("gap_lo") is not None and c["gap_lo"] > 0),
            "n_cells_followP_above_matchR_s42": sum(1 for c in ok if c.get("followP_minus_matchR_s42_lo", -1) > 0),
            "ca_P_minus_R_s42_mean": K.mean([c["ca_P_minus_R_s42"] for c in acc]),
            "ca_C_minus_R_s42_mean": K.mean([c["ca_C_minus_R_s42"] for c in acc]),
            "ca_P_minus_R_s42_range": (f"{min(c['ca_P_minus_R_s42'] for c in acc):+.4f} to "
                                       f"{max(c['ca_P_minus_R_s42'] for c in acc):+.4f}") if acc else "",
            "ca_teacherP_minus_R_mean": K.mean([c["ca_teacherP_minus_R"] for c in acc])})
    return out


def main():
    out = K.out_dir()
    st = K.state()
    bv = K.battery_vectors(st)
    seed_rows, cell_rows, checks = [], [], []
    for r in st.primary:
        fam, ds, kd, atk = r["fam"], r["dataset"], r["kd"], r["attack"]
        seeds = [int(s) for s in r["seeds"].split(",")]
        R = bv.get((fam, ds, "ce", atk, 42))
        per = {}
        for s in seeds:
            v = bv[(fam, ds, kd, atk, s)]
            u = shares(v["TPc"], v["TCc"], v["Pc"], v["Cc"], R["Pc"] if (R and s == 42) else None)
            t = shares(v["TP"], v["TC"], v["P"], v["C"], R["P"] if (R and s == 42) else None)
            res = v["res"]
            per[s] = (u, t, res)
            seed_rows.append({
                "fam": fam, "dataset": ds, "kd": kd, "attack": atk, "seed": s, "n": len(v["TPc"]),
                "n_D_untrig": u["n_D"], "n_D_untrig_Pteacher_hit": u["n_D_Phit"],
                "follow_P_untrig": u["follow_P"], "match_C_untrig": u["match_C"], "gap_untrig": u["gap"],
                "match_R_untrig": u.get("match_R"),
                "n_D_trig": t["n_D"], "n_D_trig_Pteacher_hit": t["n_D_Phit"],
                "follow_P_trig": t["follow_P"], "match_C_trig": t["match_C"], "gap_trig": t["gap"],
                "match_R_trig": t.get("match_R"),
                "ca_P": res.get("ca_poisoned"), "ca_C": res.get("ca_clean"),
                "ca_R_s42": R["res"].get("ca_poisoned") if R else None,
                "ca_teacher_P": res.get("teacher_ca"), "ca_teacher_C": res.get("teacher_ca_null")})
        # pooled over seeds
        u42 = per[seeds[0]][0]
        same_D = all(per[s][0]["D"] == u42["D"] for s in seeds)
        checks.append({"check": f"untriggered disagreement set identical across seeds {K.cell_id(r)}",
                       "expected": True, "got": same_D, "ok": same_D})
        row = {"fam": fam, "dataset": ds, "kd": kd, "attack": atk, "seeds": r["seeds"], "method": r["method"],
               "n": len(bv[(fam, ds, kd, atk, seeds[0])]["TPc"]),
               "n_D_untrig": u42["n_D"], "share_D_untrig": u42["n_D"] / len(bv[(fam, ds, kd, atk, seeds[0])]["TPc"]),
               "n_D_untrig_Pteacher_hit": u42["n_D_Phit"],
               "follow_P_untrig": K.mean([per[s][0]["follow_P"] for s in seeds]),
               "match_C_untrig": K.mean([per[s][0]["match_C"] for s in seeds]),
               "gap_untrig": K.mean([per[s][0]["gap"] for s in seeds]),
               "match_R_untrig_s42": u42.get("match_R"),
               "follow_P_untrig_by_seed": "/".join(f"{per[s][0]['follow_P']:.4f}" for s in seeds),
               "match_C_untrig_by_seed": "/".join(f"{per[s][0]['match_C']:.4f}" for s in seeds)}
        if u42["n_D"] > 0:
            if len(seeds) >= 2:
                d = {s: K.diff(per[s][0]["fP"], per[s][0]["mC"]) for s in seeds}
                k, _, rr, v, ks = K.AD.seed_pooled(d, 0.0, 1.0)
                res = K.t_contrast(k, (K.var(ks) / rr + v) ** 0.5, rr - 1, TOLS)
                row.update(gap_lo=res["lo"], gap_hi=res["hi"], gap_interval="own seed variance, df r-1")
            else:
                pc = K.paired_contrast(per[seeds[0]][0]["fP"], per[seeds[0]][0]["mC"], TOLS)
                row.update(gap_lo=pc["lo"], gap_hi=pc["hi"], gap_interval="Tango (one seed, input only)")
            if u42.get("match_R") is not None:
                pr = K.paired_contrast(u42["fP"], u42["mR"], TOLS)
                pc_ = K.paired_contrast(u42["mC"], u42["mR"], TOLS)
                row.update(followP_minus_matchR_s42=pr["k"], followP_minus_matchR_s42_lo=pr["lo"],
                           followP_minus_matchR_s42_hi=pr["hi"],
                           matchC_minus_matchR_s42=pc_["k"], matchC_minus_matchR_s42_lo=pc_["lo"],
                           matchC_minus_matchR_s42_hi=pc_["hi"])
        t42 = per[seeds[0]][1]
        row.update(n_D_trig=t42["n_D"],
                   follow_P_trig=K.mean([per[s][1]["follow_P"] for s in seeds]),
                   match_C_trig=K.mean([per[s][1]["match_C"] for s in seeds]),
                   gap_trig=K.mean([per[s][1]["gap"] for s in seeds]),
                   match_R_trig_s42=t42.get("match_R"))
        caP = [per[s][2].get("ca_poisoned") for s in seeds]
        caC = [per[s][2].get("ca_clean") for s in seeds]
        caR = R["res"].get("ca_poisoned") if R else None
        cT, cTn = per[seeds[0]][2].get("teacher_ca"), per[seeds[0]][2].get("teacher_ca_null")
        row.update(ca_P_mean=K.mean(caP), ca_C_mean=K.mean(caC), ca_R_s42=caR,
                   ca_teacher_P=cT, ca_teacher_C=cTn)
        if caR is not None:
            p42, c42 = per[42][2].get("ca_poisoned"), per[42][2].get("ca_clean")
            row.update(ca_P_minus_R_s42=p42 - caR, ca_C_minus_R_s42=c42 - caR,
                       ca_P_minus_R_seedmean=K.mean(caP) - caR, ca_C_minus_R_seedmean=K.mean(caC) - caR,
                       ca_teacherP_minus_R=cT - caR, ca_teacherC_minus_R=cTn - caR)
        cell_rows.append(row)
    K.write_csv(seed_rows, os.path.join(out, "transmission_seed.csv"), """
        One row per primary cell and student seed. Untriggered (_untrig): the cell's n untriggered
        non-target test inputs; D = inputs where the P and C teachers' target hits differ (official Stage 2
        teacher_p_clean / teacher_c_clean, GPU bf16; n_D_untrig_Pteacher_hit = inputs of D where the P
        teacher predicts the target); follow_P = share of D where the P student's hit equals the P
        teacher's; match_C = share of D where the C student's hit equals the P teacher's (= 1 - share
        following the C teacher); gap = follow_P - match_C; match_R = the CE-only student R (seed 42 only).
        Triggered (_trig): the same on the cell's triggered inputs (teacher_p_trig / teacher_c_trig,
        preds_obs / preds_null). AG News hits are target-vs-other (no four-class teacher predictions are
        stored). ca_* = Stage 2 clean accuracy on the full test set (P, C students; R seed 42; P, C
        teachers).""")
    K.write_csv(cell_rows, os.path.join(out, "transmission_cells.csv"), """
        One row per primary cell, pooled over its student seeds (mean of the per-seed shares of
        transmission_seed.csv; D is the same at every seed). gap_lo/gap_hi = 95% interval of follow_P -
        match_C: A3.1 own-variance rule on the per-input differences over D (SE^2 = s2_gap/r +
        Var_i(mean_s d_s,i)/|D|, df r-1) for r = 3 seeds, Tango paired interval for one-seed cells (input
        sampling only). followP_minus_matchR_s42 / matchC_minus_matchR_s42 = seed-42 paired Tango contrasts
        against R. Triggered shares as secondary columns. Accuracy: ca_P_mean / ca_C_mean (mean over seeds),
        ca_R_s42 (R exists at seed 42 only), teachers; *_minus_R_s42 = matched seed-42 difference;
        *_minus_R_seedmean = seed mean minus R(s42); ca_teacher*_minus_R = teacher minus R (how much
        accuracy the teacher could pass on at all).""")
    summ = summary_rows(seed_rows, cell_rows)
    K.write_csv(summ, os.path.join(out, "transmission_summary.csv"), """
        Summary of the untriggered transmission shares and the accuracy differences by battery (family x
        dataset x KD) and over all logit / all cells: number of cells with a non-empty D, mean of the cell
        values (follow_P, match_C, gap, match_R at seed 42), the per-seed min-max of follow_P and match_C,
        the number of cells whose gap interval lies above 0, and the mean KD - R and teacher - R clean
        accuracy differences (seed 42, matched).""")
    K.write_csv(checks, os.path.join(out, "check_item4.csv"), """
        Item 4 self-checks: the untriggered teacher-disagreement set is the same at every student seed.""")
    print(f"item 4: {len(cell_rows)} cells; checks {'ok' if all(c['ok'] for c in checks) else 'FAILED'}")


if __name__ == "__main__":
    main()
