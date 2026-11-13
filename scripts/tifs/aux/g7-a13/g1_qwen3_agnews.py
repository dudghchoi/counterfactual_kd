#!/usr/bin/env python3
"""A13 G1: Qwen3 AG News logit battery with student seeds 42 (official) + 43 + 44.

A13 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Registered rule (docs/tifs_addendum.md A13 G1): pool seed 42 (the official predictions of
tifs_b_qwen3_agnews) with the A13 runs tifs_x13_qwen3_agnews_s43 / _s44; the variance is the battery's
own pool (qwen3/agnews/logit, its two attacks BadNets and AddSent) with the Satterthwaite df; cell
verdicts and the battery decision (IUT / Holm, frozen rule) at every delta. Done here by running the
frozen analyze.primary_table on the frozen cells plus the A13 G1 cells (the frozen rule then gives
qwen3/agnews/logit its own seed-pooled pool) and the frozen analyze.primary_battery on its rows. Every
other primary row must come out identical to the frozen one (checked). The paper-wide Holm
(transfers_global) of that augmented run is not a registered A13 output and is not written.

Sensitivity (A15 E3, CPU): the same cells with the pool df replaced by the Welch-Satterthwaite df
(sum s2_a)^2 (r - 1) / sum s2_a^2, and by min(frozen df, Welch df); estimate and SE unchanged, the
battery decision re-derived with the frozen analyze.decide. A3.1 own / max per attack is shown as well
(addendum.attack_variance_table on the augmented rows).

Sentence rule: own-pool battery NO TRANSFER@0.05 -> "Qwen3 AG News의 전이 없음은 빌린 분산에 기대지 않는다";
otherwise -> "NO TRANSFER@0.05는 등록된 transported 규칙에서만 성립한다". The frozen transported decision stays
the primary one.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

import common as K  # noqa: E402

A, AD, TOLS, ALPHA = K.A, K.AD, K.TOLS, K.ALPHA
KEY = ("qwen3", "agnews", "logit")
SENT_OWN = "Qwen3 AG News의 전이 없음은 빌린 분산에 기대지 않는다"
SENT_OWN_EN = "The absence of transfer in Qwen3 AG News does not rest on borrowed variance."
SENT_TR = "NO TRANSFER@0.05는 등록된 transported 규칙에서만 성립한다"
SENT_TR_EN = "NO TRANSFER@0.05 holds only under the registered transported rule."


def run(out):
    st = K.state()
    x = [c for c in K.load_x13() if c["group"] == "G1"]
    bad = K.failing_seeds(x)
    kept = [c for c in x if (c["group"], c["run"]) not in bad]
    excluded = sorted({run for (_, run) in bad})
    # Seed 42 floor (official battery cells of the same battery).
    s42 = [c for c in st.cells if (c["fam"], c["dataset"], c["kd"]) == KEY and c["block"] == "G"
           and c["corpus"] == "clean" and c["seed"] == 42 and not c["single"]]
    floor42 = [r for r in K.student_rows([dict(c, group="G1") for c in s42]) if not r["functional"]]
    if floor42:
        excluded.append("tifs_b_qwen3_agnews (seed 42)")
        s42_ids = {id(c) for c in s42}
        base_cells = [c for c in st.cells if id(c) not in s42_ids]
    else:
        base_cells = st.cells
    aug_cells = base_cells + kept
    rows, var_rows = A.primary_table(aug_cells, st.structural, st.near_null, ref_var=st.ref_var)
    checks = []
    # Every primary row outside the Qwen3 AG News battery is the frozen one.
    frozen = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in st.primary}
    n_same, n_other = 0, 0
    for r in rows:
        k = (r["fam"], r["dataset"], r["kd"], r["attack"])
        if k[:3] == KEY:
            continue
        n_other += 1
        n_same += r == frozen[k]
    checks.append({"check": "G1: every augmented primary row outside qwen3/agnews equals the frozen row",
                   "expected": f"{len(st.primary) - 2} rows identical", "got": f"{n_same} of {n_other}",
                   "ok": n_same == n_other == len(st.primary) - 2})
    bat_aug = {(b["fam"], b["dataset"], b["kd"]): b for b in A.primary_battery(rows)}
    bat_frz = {(b["fam"], b["dataset"], b["kd"]): b for b in A.primary_battery(st.primary)}
    n_bsame = sum(all(bat_aug[k][c] == bat_frz[k][c] for c in bat_frz[k] if not c.startswith("transfers_global")
                      and c != "methods") for k in bat_frz if k != KEY)
    checks.append({"check": "G1: every other battery decision (within-battery Holm / IUT) equals the frozen one",
                   "expected": f"{len(bat_frz) - 1}", "got": f"{n_bsame}", "ok": n_bsame == len(bat_frz) - 1})
    q = sorted([r for r in rows if (r["fam"], r["dataset"], r["kd"]) == KEY], key=lambda r: r["attack"])
    pool = next(v for v in var_rows if (v["fam"], v["dataset"], v["kd"]) == KEY and v["attack"] == "(pooled)")
    att = {v["attack"]: v for v in var_rows if (v["fam"], v["dataset"], v["kd"]) == KEY and v["attack"] != "(pooled)"}
    seeds = pool["seeds"].split(",")
    r_ = len(seeds)
    s2a = {a: att[a]["s_between"] ** 2 for a in pool["attacks"].split(",")}
    s2p = pool["s_between"] ** 2
    checks.append({"check": "G1: s2_pool = mean of the per-attack between-seed variances (pool identity)",
                   "expected": f"{s2p!r}", "got": f"{K.mean(list(s2a.values()))!r}",
                   "ok": abs(K.mean(list(s2a.values())) - s2p) < 1e-15})
    df_w = K.welch_df(s2a.values(), r_)
    df_min = min(pool["df"], df_w)
    from scipy import stats
    bart = stats.bartlett(*[[att[a][f"k_s{s}"] for s in seeds] for a in sorted(s2a)])

    # ── cells ──
    prim_aug = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in q}
    att_var_aug = {(v["fam"], v["dataset"], v["kd"], v["attack"]): v for v in var_rows if v["attack"] != "(pooled)"}
    pools_aug = {(v["fam"], v["dataset"], v["kd"]): v for v in var_rows if v["attack"] == "(pooled)"}
    a31 = {r["attack"]: r for r in AD.attack_variance_table(st, prim=prim_aug, att_var=att_var_aug, pools=pools_aug)}
    cell_rows, alt = [], {"welch": [], "min": []}
    for r in q:
        fr = frozen[KEY + (r["attack"],)]
        row = {"fam": "qwen3", "dataset": "agnews", "kd": "logit", "attack": r["attack"], "seeds": r["seeds"],
               "n": r["n"], "teacher_qualified": r["teacher_qualified"],
               "raw_asr": r["raw_asr"], "C_trig": r["C_trig"],
               **{f"k_s{s}": att[r["attack"]][f"k_s{s}"] for s in seeds},
               "input_se": att[r["attack"]]["input_se"], "s_attack": att[r["attack"]]["s_between"],
               "own_pool_method": r["method"], "own_pool_variance_from": r["variance_from"],
               **K.tfields(r, "own_pool_"),
               "own_pool_lo_df_r1": r.get("lo_df_min"), "own_pool_hi_df_r1": r.get("hi_df_min"),
               "own_pool_verdict_df_r1@0.05": r.get("verdict_df_min@0.05"),
               "frozen_method": fr["method"], "frozen_variance_from": fr["variance_from"],
               **K.tfields(fr, "frozen_")}
        for name, df in (("welch", df_w), ("min", df_min)):
            res = K.t_contrast(r["k"], r["se"], df, TOLS)
            row.update(K.tfields(res, f"{name}df_"))
            alt[name].append({"attack": r["attack"], **{f"{p}@{t:g}": res[f"{p}@{t:g}"] for t in TOLS
                                                        for p in ("verdict", "p_up", "p_tost", "p_pos")}})
        a = a31.get(r["attack"], {})
        row.update({"A31_own_lo": a.get("own_lo"), "A31_own_hi": a.get("own_hi"),
                    **{f"A31_own_verdict@{t:g}": a.get(f"own_verdict@{t:g}") for t in TOLS},
                    "A31_max_lo": a.get("max_lo"), "A31_max_hi": a.get("max_hi"),
                    **{f"A31_max_verdict@{t:g}": a.get(f"max_verdict@{t:g}") for t in TOLS},
                    "A31_pooling_dependent": a.get("pooling_dependent")})
        row["own_vs_frozen_changes"] = ",".join(f"{t:g}" for t in TOLS
                                                if r[f"verdict@{t:g}"] != fr[f"verdict@{t:g}"])
        cell_rows.append(row)
    K.write_csv(cell_rows, os.path.join(out, "g1_cells.csv"), """
        A13 G1, one row per Qwen3 AG News logit attack. own_pool_* = the frozen seed-pooled rule over
        student seeds 42 (official tifs_b_qwen3_agnews predictions), 43, 44 (tifs_x13_qwen3_agnews_s43/_s44):
        k = mean_s k_s, SE^2 = s2_pool / r + v, t at the pool's Satterthwaite df, pool = the battery's own
        (qwen3/agnews/logit over its two attacks), computed by the frozen analyze.primary_table on the
        frozen cells plus the G1 cells; own_pool_*_df_r1 = primary_table's df = r-1 sensitivity columns (lo_df_min / hi_df_min / verdict_df_min@0.05 there). frozen_* = the
        registered primary row (seed 42, transported from the qwen3/sst2/logit pool), unchanged.
        welchdf_* / mindf_* = the own-pool estimate and SE at the Welch-Satterthwaite df of the pool
        (A15 E3) and at min(Satterthwaite, Welch). A31_* = A3.1 own (s2_a / r + v, df r-1) and max
        (max(s2_a, s2_pool) / r + v, pool df) by the frozen addendum.attack_variance_table. Verdicts at
        delta in {0.02, 0.05, 0.10}: EQUIVALENT / POSITIVE / NEGATIVE / INCONCLUSIVE from the two-sided
        95% interval; p_up = one-sided p of 'k < delta', p_pos = one-sided p of 'k > delta'.""")

    # ── pool ──
    frz_pool = st.pools[("qwen3", "sst2", "logit")]
    pool_rows = [{"pool": "qwen3/agnews/logit (own, A13 G1)", "attacks": pool["attacks"], "seeds": pool["seeds"],
                  "m": len(s2a), "r": r_, "s_pool": pool["s_between"], "df_satterthwaite": pool["df"],
                  "ms_seed": pool["ms_seed"], "ms_res": pool["ms_res"],
                  **{f"s_{a}": v ** 0.5 for a, v in sorted(s2a.items())},
                  "df_welch": df_w, "df_min": df_min, "df_bound_low": r_ - 1, "df_bound_high": len(s2a) * (r_ - 1),
                  "bartlett_stat": float(bart.statistic), "bartlett_p": float(bart.pvalue),
                  "t975_satterthwaite": float(stats.t.ppf(0.975, pool["df"])),
                  "t975_welch": float(stats.t.ppf(0.975, df_w)),
                  "excluded_seeds": ";".join(excluded) or "none"},
                 {"pool": "qwen3/sst2/logit (frozen; borrowed by the registered transported rule)",
                  "attacks": frz_pool["attacks"], "seeds": frz_pool["seeds"],
                  "m": len(frz_pool["attacks"].split(",")), "r": len(frz_pool["seeds"].split(",")),
                  "s_pool": frz_pool["s_between"], "df_satterthwaite": frz_pool["df"],
                  "ms_seed": frz_pool["ms_seed"], "ms_res": frz_pool["ms_res"]}]
    K.write_csv(pool_rows, os.path.join(out, "g1_pool.csv"), """
        A13 G1 between-seed pool of the Qwen3 AG News logit battery (attacks BadNets and AddSent, student
        seeds 42-44): s_pool = sqrt(mean_a s2_a) and df_satterthwaite from analyze.pool_seed_variance
        (MS_seed, MS_res; bounded by r-1 and m(r-1)); s_<attack> = per-attack between-seed SD; df_welch =
        (sum_a s2_a)^2 (r-1) / sum_a s2_a^2 (A15 E3); df_min = min of the two; bartlett_* = scipy Bartlett
        test of equal variance of k_s across the two attacks (descriptive; r = 3 per attack). Second row:
        the frozen qwen3/sst2/logit pool that the registered transported rule borrows.""")

    # ── battery ──
    b = bat_aug[KEY]
    bf = bat_frz[KEY]
    dec_alt = {name: A.decide(rs) for name, rs in alt.items()}
    sentence = SENT_OWN if b["decision@0.05"] == "NO TRANSFER" else SENT_TR
    sentence_en = SENT_OWN_EN if b["decision@0.05"] == "NO TRANSFER" else SENT_TR_EN
    bat_rows = []
    for t in TOLS:
        bat_rows.append({
            "fam": "qwen3", "dataset": "agnews", "kd": "logit", "delta": t, "attacks": b["attacks"],
            "n_qualified": b["n_qualified"], "seeds": pool["seeds"],
            "own_pool_decision": b[f"decision@{t:g}"], "own_pool_transfers": b[f"transfers@{t:g}"],
            "own_pool_all_equivalent": b[f"all_equivalent@{t:g}"],
            "own_pool_max_p_up": max(r[f"p_up@{t:g}"] for r in q),
            "own_pool_holm_p_pos_min": min(K.holm([r[f"p_pos@{t:g}"] for r in q])),
            "own_pool_methods": b["methods"],
            "frozen_decision": bf[f"decision@{t:g}"], "frozen_transfers": bf[f"transfers@{t:g}"],
            "frozen_max_p_up": max(frozen[KEY + (a,)][f"p_up@{t:g}"] for a in b["attacks"].split(",")),
            "frozen_methods": bf["methods"],
            "welch_df": df_w, "welch_decision": dec_alt["welch"][f"decision@{t:g}"],
            "welch_max_p_up": max(r[f"p_up@{t:g}"] for r in alt["welch"]),
            "min_df": df_min, "min_decision": dec_alt["min"][f"decision@{t:g}"],
            "same_as_frozen": b[f"decision@{t:g}"] == bf[f"decision@{t:g}"],
            "registered_sentence": sentence if t == 0.05 else "",
            "registered_sentence_en": sentence_en if t == 0.05 else ""})
    K.write_csv(bat_rows, os.path.join(out, "g1_battery.csv"), """
        A13 G1 battery decision of Qwen3 AG News logit KD at each delta, frozen analyze.primary_battery
        (decide: TRANSFERS if a Holm-adjusted one-sided p_pos < 0.025 within the battery, else NO TRANSFER
        if every attack's one-sided p_up < 0.025 (intersection-union), else INCONCLUSIVE). own_pool_* = with
        the own seed-pooled pool (seeds 42-44); frozen_* = the registered transported decision (stays
        primary); welch_* / min_* = own-pool cells at the Welch / min df, frozen analyze.decide.
        registered_sentence = the A13 G1 sentence the rule selects (own-pool NO TRANSFER@0.05 or not).""")
    summary = {"pool": pool_rows[0], "cells": cell_rows, "battery": bat_rows, "sentence": sentence,
               "sentence_en": sentence_en, "excluded": excluded, "checks": checks}
    K.write_csv(checks, os.path.join(out, "check_g1.csv"), """Self-checks of the G1 script (augmented
        frozen primary_table leaves every other row and battery decision unchanged; pool identity).""")
    return summary


def main():
    out = K.out_dir()
    s = run(out)
    for b in s["battery"]:
        print(f"  delta {b['delta']}: own {b['own_pool_decision']} (frozen {b['frozen_decision']}), "
              f"welch {b['welch_decision']}, min {b['min_decision']}")
    print("  sentence:", s["sentence"])
    K.dump_json({k: v for k, v in s.items()}, os.path.join(out, "logs", "g1_summary.json"))


if __name__ == "__main__":
    main()
