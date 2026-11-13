#!/usr/bin/env python3
"""A15-e and e2: KD recipe alpha sensitivity (BERT AG News alpha 0.8 / 1.0; GPT-2 SST-2 alpha 1.0 with
epoch-matched C) and the teacher-transmission metric.

A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Registered (docs/tifs_addendum.md A15-e and the 10/26 19:55 amendment e2; claim rule of the 10/27 20:58
amendment):
  * e: BERT AG News logit, alpha in {0.8, 1.0} (T 4), P_ATBA, P_BadNets and a shared C, seeds 42-44
    (tifs_x15_e_bert_agnews_a{08,10}_s*). Per alpha the frozen analyze.primary_table on that alpha's cells
    alone (two attacks, their own seed pool), primary_battery on its rows; verdicts under the registered /
    own-variance / claim rules and the Welch df. Comparators at alpha 0.5: the same two attacks with their own
    two-attack pool (like for like: same pool design and df, only the recipe differs; this sets the label and
    the sentence) and, beside it, the frozen primary rows (six-attack pool, df 9.83; the registered judgement,
    which differs from the alpha frame also through the pool and df); sensitivity: the alpha cells with the
    frozen six-attack pool (borrowed variance). recipe-dependent where an alpha verdict (registered or claim
    rule) differs from the like-for-like alpha 0.5 one; all the same -> "α 0.5–1.0에서 같다".
  * e2: GPT-2 SST-2 logit, alpha 1.0, P_BadNets (C from the rebuilt twin at epoch 2) and P_AddSent (epoch 5),
    seeds 42-44 (tifs_x15_e2_gpt2_sst2_a10_e{2,5}_s*). Comparator at alpha 0.5: the epoch-matched A15-b1 GPT-2
    cells (like for like; H2) and the frozen rows.
  * Transmission (scripts/tifs/aux/g9-a15cpu/item4_transmission.py, same definitions): on the untriggered
    non-target inputs, D = inputs where the P and C teachers' target hits differ; follow_P = share of D where
    the P student's hit equals the P teacher's; match_C = the same for the C student; gap = follow_P - match_C
    (seeds pooled; interval: A3.1 own variance on the per-input differences, df r - 1); match_R = the frozen
    CE-only student R (seed 42). Triggered-input shares as secondary columns.

Outputs: e_cells.csv, e_battery.csv, e_transmission.csv, e_claims.csv, e_floor.csv, check_e.csv.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

import a15lib as L  # noqa: E402
import part_b1  # noqa: E402

A, AD, TOLS = L.A, L.AD, L.TOLS
E_KEYS = [("bert", "agnews", "logit", "atba"), ("bert", "agnews", "logit", "badnets")]
E2_KEYS = [("gpt2", "sst2", "logit", "addsent"), ("gpt2", "sst2", "logit", "badnets")]
SENT_SAME = "α 0.5–1.0에서 같다"
SENT_DEP = "recipe-dependent"


# ── transmission (g9-a15cpu item 4 definitions) ──────────────────────


def shares(TP, TC, P, C, R=None):
    """Same as scripts/tifs/aux/g9-a15cpu/item4_transmission.shares."""
    D = [i for i in range(len(TP)) if TP[i] != TC[i]]
    fP = [int(P[i] == TP[i]) for i in D]
    mC = [int(C[i] == TP[i]) for i in D]
    out = {"D": D, "fP": fP, "mC": mC, "n_D": len(D), "n_D_Phit": sum(TP[i] for i in D),
           "follow_P": L.mean(fP) if D else float("nan"), "match_C": L.mean(mC) if D else float("nan")}
    out["gap"] = out["follow_P"] - out["match_C"]
    if R is not None:
        mR = [int(R[i] == TP[i]) for i in D]
        out["mR"] = mR
        out["match_R"] = L.mean(mR) if D else float("nan")
    return out


def transmission_row(by_seed, R):
    """Cell row of item 4 (untriggered shares pooled over seeds, gap interval, match_R at seed 42, triggered
    shares) from {seed: cell}; R = the frozen CE cell (seed 42) or None."""
    seeds = sorted(by_seed)
    per = {}
    for s in seeds:
        v = by_seed[s]
        u = shares(v["TPc"], v["TCc"], v["Pc"], v["Cc"], R["Pc"] if (R and s == 42) else None)
        t = shares(v["TP"], v["TC"], v["P"], v["C"], R["P"] if (R and s == 42) else None)
        per[s] = (u, t)
    u0 = per[seeds[0]][0]
    row = {"seeds": ",".join(map(str, seeds)), "n": len(by_seed[seeds[0]]["TPc"]), "n_D_untrig": u0["n_D"],
           "D_same_across_seeds": all(per[s][0]["D"] == u0["D"] for s in seeds),
           "follow_P_untrig": L.mean([per[s][0]["follow_P"] for s in seeds]),
           "match_C_untrig": L.mean([per[s][0]["match_C"] for s in seeds]),
           "gap_untrig": L.mean([per[s][0]["gap"] for s in seeds]),
           "match_R_untrig_s42": per[42][0].get("match_R") if 42 in per else None,
           "follow_P_untrig_by_seed": "/".join(f"{per[s][0]['follow_P']:.4f}" for s in seeds)}
    if u0["n_D"] > 0:
        if len(seeds) >= 2:
            d = {s: L.diff(per[s][0]["fP"], per[s][0]["mC"]) for s in seeds}
            k, _, rr, v, ks = AD.seed_pooled(d, 0.0, 1.0)
            res = L.t_contrast(k, (L.var(ks) / rr + v) ** 0.5, rr - 1, TOLS)
            row.update(gap_lo=res["lo"], gap_hi=res["hi"], gap_interval="own seed variance, df r-1")
        else:
            pc = L.paired_contrast(u0["fP"], u0["mC"], TOLS)
            row.update(gap_lo=pc["lo"], gap_hi=pc["hi"], gap_interval="Tango (one seed, input only)")
        if 42 in per and per[42][0].get("match_R") is not None:
            pr = L.paired_contrast(per[42][0]["fP"], per[42][0]["mR"], TOLS)
            row.update(followP_minus_matchR_s42=pr["k"], followP_minus_matchR_s42_lo=pr["lo"],
                       followP_minus_matchR_s42_hi=pr["hi"])
    t0 = per[seeds[0]][1]
    row.update(n_D_trig=t0["n_D"], follow_P_trig=L.mean([per[s][1]["follow_P"] for s in seeds]),
               match_C_trig=L.mean([per[s][1]["match_C"] for s in seeds]),
               gap_trig=L.mean([per[s][1]["gap"] for s in seeds]))
    return row


def groups_of(cells):
    """{(fam, ds, kd, attack): {seed: cell}} of the clean-corpus battery cells (block G, corpus clean), as
    addendum.State.groups."""
    g = {}
    for c in cells:
        if c["kd"] == "ce" or c["single"] or c["block"] != "G" or c["corpus"] != "clean":
            continue
        g.setdefault((c["fam"], c["dataset"], c["kd"], c["attack"]), {})[c["seed"]] = c
    return g


def r_cell(st, fam, ds, atk):
    return next((c for c in st.cells if c["block"] == "G" and c["corpus"] == "clean" and c["kd"] == "ce"
                 and (c["fam"], c["dataset"], c["attack"], c["seed"]) == (fam, ds, atk, 42)), None)


# ── verdict rows ─────────────────────────────────────────────────────


def borrowed(F_alpha, key, pool):
    """The alpha cell with a borrowed pool (frozen seed-pooled formula, s2_pool and its df)."""
    g = F_alpha["groups"].get(key)
    if not g or len(g) < 2:
        return None
    d = {s: L.diff(c["P"], c["C"]) for s, c in g.items()}
    k, se, r, v, ks = AD.seed_pooled(d, pool["s_between"] ** 2, pool["df"])
    return L.t_contrast(k, se, pool["df"], TOLS)


def main():
    ns = L.args()
    st, F0 = L.state(ns.runs)
    out = ns.out
    x15 = L.load_x15(ns.runs)
    rows, bats, trans, claims, floor_all, checks = [], [], [], [], [], []
    # ── e ──
    e_all = [c for c in x15 if c["part"] == "e"]
    e_kept, e_excl, fr = L.apply_floor(e_all, "e")
    floor_all += fr
    sub = [c for c in st.cells if c["block"] == "G" and c["corpus"] == "clean" and not c["single"]
           and (c["fam"], c["dataset"], c["kd"], c["attack"]) in E_KEYS]
    F05 = L.frame(st, sub, "alpha 0.5 (two-attack pool)")
    frozen_pool = F0["pools"][("bert", "agnews", "logit")]
    Fe = {}
    for alpha in (0.8, 1.0):
        cs = [c for c in e_kept if c.get("alpha") == alpha and c["kd"] == "logit"]
        if not cs:
            continue
        Fe[alpha] = L.frame(st, cs, f"alpha {alpha:g}")
    for alpha, F in sorted(Fe.items()):
        for key in E_KEYS:
            r = F["prim"].get(key)
            if r is None:
                continue
            row = {"part": "e", "alpha": alpha, "fam": key[0], "dataset": key[1], "kd": key[2], "attack": key[3],
                   **L.all_fields(r, f"a{alpha:g}_", family=False),
                   **L.all_fields(F0["prim"][key], "a0.5_registered_", family=False),
                   **L.all_fields(F05["prim"][key], "a0.5_twoattack_", family=False)}
            b = borrowed(F, key, frozen_pool)
            if b is not None:
                row.update({"borrowed6_lo": b["lo"], "borrowed6_hi": b["hi"], "borrowed6_df": b["df"],
                            **{f"borrowed6_verdict@{t:g}": b[f"verdict@{t:g}"] for t in TOLS}})
            # label = like for like (two-attack pool at alpha 0.5); the registered-row comparison beside it
            for rule, v in L.changed(F05["prim"][key], r).items():
                row[f"recipe_dependent_{rule}"] = v
            for rule, v in L.changed(F0["prim"][key], r).items():
                row[f"recipe_dependent_vs_registered_{rule}"] = v
            for rule, v in L.changed(F0["prim"][key], F05["prim"][key]).items():
                row[f"a05_pool_design_differs_{rule}"] = v
            rows.append(row)
        for b in L.battery_comparison(F05, F, [("bert", "agnews", "logit")], "recipe_dependent", "a0.5_twoattack_",
                                      f"a{alpha:g}_"):
            bats.append({"part": "e", "alpha": alpha, **b})
        claims += L.claim_summary(F, f"e alpha {alpha:g}")
    # ── e2 ──
    e2_all = [c for c in x15 if c["part"] == "e2"]
    e2_kept, e2_excl, fr = L.apply_floor(e2_all, "e2")
    floor_all += fr
    em_cells, info = part_b1.build_em(st, x15, ns.runs)
    Fem = L.frame(st, em_cells, "em")
    Fe2 = L.frame(st, [c for c in e2_kept if c["kd"] == "logit"], "e2 alpha 1.0") if e2_kept else None
    if Fe2 is not None:
        for key in E2_KEYS:
            r = Fe2["prim"].get(key)
            if r is None:
                continue
            row = {"part": "e2", "alpha": 1.0, "fam": key[0], "dataset": key[1], "kd": key[2], "attack": key[3],
                   "C_epoch": L.P_EPOCH[(key[0], key[3])],
                   **L.all_fields(r, "a1_", family=False),
                   **L.all_fields(Fem["prim"][key], "a0.5_em_", family=False),
                   **L.all_fields(F0["prim"][key], "a0.5_registered_", family=False)}
            for rule, v in L.changed(Fem["prim"][key], r).items():
                row[f"recipe_dependent_{rule}"] = v
            for rule, v in L.changed(F0["prim"][key], r).items():
                row[f"recipe_dependent_vs_registered_{rule}"] = v
            rows.append(row)
        for b in L.battery_comparison(Fem, Fe2, [("gpt2", "sst2", "logit")], "recipe_dependent", "a0.5_em_", "a1_"):
            bats.append({"part": "e2", "alpha": 1.0, **b})
        claims += L.claim_summary(Fe2, "e2 alpha 1")
    for r in rows:
        dep = any(r.get(f"recipe_dependent_{x}") for x in ("reg", "claim"))
        r["sentence"] = SENT_DEP if dep else SENT_SAME
        dep = any(r.get(f"recipe_dependent_vs_registered_{x}") for x in ("reg", "claim"))
        r["sentence_vs_registered"] = SENT_DEP if dep else SENT_SAME
    jn = {p: L.part_jobs_note(ns.runs, p) for p in ("e", "e2")}
    rows = [dict(r, a15_jobs=jn[r["part"]]) for r in rows]
    bats = [dict(r, a15_jobs=jn[r["part"]]) for r in bats]
    L.write_csv(rows, os.path.join(out, "e_cells.csv"), """
        A15-e / e2, one row per (part, alpha, attack). a<alpha>_* = the alpha cells under the frozen
        primary_table on that alpha's cells alone (own two-attack pool): reg_* (registered rule), own_* (A3.1
        own variance, df r-1), claim@d (claim rule), pmax_*, welch_* (Welch df). e comparators at alpha 0.5:
        a0.5_registered_* = the frozen primary row (six-attack BERT AG News pool; the registered judgement),
        a0.5_twoattack_* = the frozen BadNets / ATBA cells with their own two-attack pool (like for like);
        borrowed6_* = the alpha cell with the frozen six-attack pool (borrowed variance). e2 comparators:
        a0.5_em_* = the epoch-matched A15-b1 GPT-2 row (like for like: same C epoch), a0.5_registered_* = the
        frozen row. recipe_dependent_<rule> = deltas where the alpha verdict differs from the like-for-like
        comparator, a0.5_twoattack (e) / a0.5_em (e2); recipe_dependent_vs_registered_<rule> = against the
        registered alpha 0.5 row (e: six-attack pool, so a change can come from the pool and df alone; e2: the
        frozen row); a05_pool_design_differs_<rule> (e) = deltas where the alpha 0.5 two-attack reading itself
        differs from the registered one (pool and df only). sentence = the registered sentence rule on the
        like-for-like comparison (registered or claim-rule verdict differs -> recipe-dependent; own variance and
        Welch are sensitivity columns); sentence_vs_registered = the same rule against the registered row,
        beside it.""")
    L.write_csv(bats, os.path.join(out, "e_battery.csv"), """
        A15-e / e2 battery decisions per alpha (frozen analyze.primary_battery on the alpha cells; own / claim
        / welch via frozen analyze.decide) beside alpha 0.5 (e: the two-attack frozen battery; e2: the
        epoch-matched GPT-2 battery); recipe_dependent_<rule> = deltas that differ.""")
    # ── transmission ──
    g9 = os.path.join(L.G9, "transmission_cells.csv")
    g9rows = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in L.read_csv(g9)} if os.path.exists(g9) else {}
    fz = groups_of(st.cells)
    ntrans_check = 0
    for key in E_KEYS + E2_KEYS:
        R = r_cell(st, key[0], key[1], key[3])
        t = transmission_row(fz[key], R)
        trans.append({"part": "e" if key in E_KEYS else "e2", "alpha": 0.5, "reading": "frozen", "fam": key[0],
                      "dataset": key[1], "kd": key[2], "attack": key[3], **t})
        g = g9rows.get(key)
        if g:
            for f in ("follow_P_untrig", "match_C_untrig", "gap_untrig", "gap_lo", "gap_hi", "match_R_untrig_s42"):
                ntrans_check += str(t.get(f)) != g.get(f, "")
    checks.append({"check": "transmission of the frozen alpha 0.5 cells reproduces g9-a15cpu transmission_cells.csv "
                            "(follow_P, match_C, gap, gap interval, match_R; exact text)",
                   "ok": ntrans_check == 0 and bool(g9rows), "note": f"{ntrans_check} mismatches"})
    for alpha, F in sorted(Fe.items()):
        for key in E_KEYS:
            g = F["groups"].get(key)
            if g:
                trans.append({"part": "e", "alpha": alpha, "reading": "alpha run", "fam": key[0], "dataset": key[1],
                              "kd": key[2], "attack": key[3],
                              **transmission_row(g, r_cell(st, key[0], key[1], key[3]))})
    em_groups = groups_of(em_cells)
    for key in E2_KEYS:
        trans.append({"part": "e2", "alpha": 0.5, "reading": "epoch-matched (A15-b1)", "fam": key[0],
                      "dataset": key[1], "kd": key[2], "attack": key[3],
                      **transmission_row(em_groups[key], r_cell(st, key[0], key[1], key[3]))})
        if Fe2 is not None and Fe2["groups"].get(key):
            trans.append({"part": "e2", "alpha": 1.0, "reading": "alpha run (epoch-matched C)", "fam": key[0],
                          "dataset": key[1], "kd": key[2], "attack": key[3],
                          **transmission_row(Fe2["groups"][key], r_cell(st, key[0], key[1], key[3]))})
    L.write_csv(trans, os.path.join(out, "e_transmission.csv"), """
        Teacher transmission per cell and alpha (definitions of scripts/tifs/aux/g9-a15cpu/item4_transmission.py):
        untriggered non-target inputs; D = inputs where the P and C teachers' target hits differ (n_D_untrig;
        D_same_across_seeds); follow_P = share of D where the P student's hit equals the P teacher's; match_C =
        share of D where the C student's hit equals the P teacher's; gap = follow_P - match_C (mean over seeds;
        gap_lo/hi = A3.1 own variance on the per-input differences, df r-1); match_R_untrig_s42 = the frozen
        CE-only student R (seed 42); followP_minus_matchR_s42 = paired Tango contrast at seed 42; *_trig = the
        same shares on the triggered inputs. reading: frozen (alpha 0.5, reproduces g9 transmission_cells.csv),
        epoch-matched (A15-b1, alpha 0.5), alpha run.""")
    L.write_csv(claims, os.path.join(out, "e_claims.csv"), "Claim-rule counts of the alpha frames (see "
                "b1_claims.csv; the 'family' columns of these small frames are not registered families).")
    L.write_csv(floor_all, os.path.join(out, "e_floor.csv"), """
        A8 student floor of every A15-e / e2 student (P, C): Stage 2 clean accuracy > majority share + 0.10; a
        run with any student at or below it is left out (listed in check_e.csv).""")
    exp_e = [L.E_RUN.format(a=a, s=s) for a in ("08", "10") for s in L.SEEDS]
    exp_e2 = [L.E2_RUN.format(e=e, s=s) for e in (2, 5) for s in L.SEEDS]
    have = {c["run"] for c in x15}
    checks.append({"check": "e inputs complete", "ok": all(r in have for r in exp_e) and not e_excl,
                   "note": L.missing_note([r for r in exp_e if r not in have] + [f"floor-excluded {r}" for r in e_excl])})
    checks.append({"check": "e2 inputs complete", "ok": all(r in have for r in exp_e2) and not e2_excl,
                   "note": L.missing_note([r for r in exp_e2 if r not in have]
                                          + [f"floor-excluded {r}" for r in e2_excl])})
    checks.append({"check": "teacher disagreement set D identical across student seeds in every transmission row "
                            "(teachers are fixed across student seeds)",
                   "ok": all(r["D_same_across_seeds"] in (True, "True") for r in trans),
                   "note": ";".join(f"{r['part']}/{r['alpha']}/{r['reading']}/{r['attack']}" for r in trans
                                    if r["D_same_across_seeds"] not in (True, "True"))})
    dz = {(r["fam"], r["dataset"], r["attack"]): r["n_D_untrig"] for r in trans if r["reading"] == "frozen"}
    checks.append({"check": "alpha-run cells of e: D (untriggered teacher disagreement) size = the frozen one "
                            "(same teachers)",
                   "ok": all(r["n_D_untrig"] == dz[(r["fam"], r["dataset"], r["attack"])] for r in trans
                             if r["part"] == "e" and r["reading"] == "alpha run"), "note": ""})
    checks += L.pool_identity_rows({"alpha 0.5 two-attack": F05, **{f"alpha {a:g}": F for a, F in Fe.items()},
                                    "em": Fem, "e2 alpha 1": Fe2})
    checks += L.excluded_rows(ns.runs)
    L.write_csv(checks, os.path.join(out, "check_e.csv"), "A15-e / e2 self-checks ('inputs complete' is False "
                "while e / e2 jobs are missing or cut).")
    print(f"part e: alphas {sorted(Fe)}; e2 {'present' if Fe2 else 'absent'}; transmission rows {len(trans)}")


if __name__ == "__main__":
    main()
