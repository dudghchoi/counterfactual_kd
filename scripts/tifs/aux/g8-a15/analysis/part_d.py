#!/usr/bin/env python3
"""A15-d: BERT AG News logit battery with student seeds 45 and 46 (r = 5), beside r = 3.

A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Registered (docs/tifs_addendum.md A15-d; hierarchy H2 and claim rule of the 10/27 20:58 KST amendment):
  * Frame 'r5' = the frozen cells plus the A15-d cells (tifs_x15_d_bert_agnews_s45/_s46: the six attacks,
    shared C, C_ep, C_sos; teachers and corpus seed 42), A8 floor applied. The frozen analyze.primary_table
    then pools BERT AG News logit over the seeds every attack has (42-46), so the cell verdicts, the pool
    (Satterthwaite df), the battery decision and the 31-cell Holm (BERT AG News logit cells at r = 5; the
    one-seed 'feature' AG News cells borrow the r = 5 pool) follow the frozen rule.
  * Per attack: registered (pool) and A3.1 own-variance (df r - 1 = 4) verdicts, claim rule, Welch df, at
    r = 3 (frozen) and r = 5; pooling_dependent (A3.1 rule, addendum.attack_variance_table) at both; the
    registered label rule: own-variance r = 5 verdict = frozen verdict -> the cell's pooling-dependent mark
    is dropped; otherwise kept and the r = 5 verdict is written beside it.
  * r = 5 is labelled post hoc: seeds 45/46 were added after the r = 3 results were seen (H2).

Outputs: d_cells.csv, d_battery.csv, d_holm.csv, d_pools.csv, d_claims.csv, d_floor.csv, check_d.csv.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

import a15lib as L  # noqa: E402

A, AD, TOLS = L.A, L.AD, L.TOLS
POST_HOC = "post hoc (seeds 45/46 added after the r = 3 results were seen)"


def main():
    ns = L.args()
    st, F0 = L.state(ns.runs)
    x15 = L.load_x15(ns.runs)
    d = [c for c in x15 if c["part"] == "d" and c["kd"] == "logit"]
    kept, excluded, floor_rows = L.apply_floor(d, "d")
    Fd = L.frame(st, list(st.cells) + kept, "r5")
    out = ns.out
    seeds = sorted({c["seed"] for c in kept})
    expected = [L.D_RUN.format(s=s) for s in (45, 46)]
    present = sorted({c["run"] for c in d})
    missing = [r for r in expected if r not in present]
    a31_r3 = {L.ckey(r): r for r in AD.attack_variance_table(st)}
    a31_r5 = {L.ckey(r): r for r in AD.attack_variance_table(st, prim=Fd["prim"], att_var=Fd["att_var"],
                                                               pools=Fd["pools"])}
    keys = [L.ckey(r) for r in st.primary if r["fam"] == "bert" and r["dataset"] == "agnews"]

    def extra(key, a, b):
        r3, r5 = a31_r3.get(key), a31_r5.get(key)
        row = {"r_r3": len(a["seeds"].split(",")) if a else None, "r_r5": len(b["seeds"].split(",")) if b else None,
               "label_r5": POST_HOC if key[2] == "logit" else POST_HOC + "; one-seed cell, only its borrowed pool "
                                                                         "changes",
               "pooling_dependent_r3": r3.get("pooling_dependent") if r3 else None,
               "pooling_dependent_r5": r5.get("pooling_dependent") if r5 else None}
        av = Fd["att_var"].get(key)
        if av is not None:
            row.update({f"r5_k_s{s}": av.get(f"k_s{s}") for s in (42, 43, 44, 45, 46)})
            row["r5_s_attack"] = av["s_between"]
            row["r5_input_se"] = av["input_se"]
        for t in TOLS:
            if b is None or a is None or b["own"] is None:
                row[f"pooling_mark@{t:g}"] = "n/a"
                continue
            was = r3 is not None and f"{t:g}" in (r3.get("pooling_dependent") or "").split(",")
            same = b["own"][f"verdict@{t:g}"] == a[f"verdict@{t:g}"]
            row[f"pooling_mark@{t:g}"] = (("drop pooling-dependent" if same else
                                           "keep pooling-dependent; write the r = 5 verdict beside it") if was else
                                          ("not marked at r = 3" + ("" if same else
                                                                    "; r = 5 own verdict differs from the frozen one")))
        return row

    jn = L.part_jobs_note(ns.runs, "d")
    L.write_csv(L.with_jobs(L.comparison_rows(F0, Fd, keys, "r5_differs", "r3_", "r5_", extra), jn),
                os.path.join(out, "d_cells.csv"), """
        A15-d, one row per BERT AG News primary cell (logit seed-pooled cells and the one-seed 'feature' cells
        that borrow the logit pool). r3_* = the frozen frame (seeds 42-44, = primary.csv); r5_* = the frame with
        seeds 45/46 added (frozen primary_table; pool over seeds 42-46). Column families as in b1_cells.csv:
        reg_* (registered pool rule, Satterthwaite df), own_* (A3.1 own variance, SE^2 = s2_a/r + v, df r-1 =
        4 at r = 5), claim@d (claim rule), welch_* (Welch df sensitivity), F31/F19 family claims;
        r5_differs_<rule> = deltas where the r = 5 verdict differs from r = 3. pooling_dependent_r3/_r5 = the
        A3.1 mark (registered verdict differs from the max or own verdict); pooling_mark@d = the registered
        A15-d rule (own r = 5 verdict = frozen verdict -> drop the mark; else keep it and write the r = 5
        verdict beside it). label_r5: r = 5 is post hoc.""")
    L.write_csv(L.with_jobs(L.battery_comparison(F0, Fd, [("bert", "agnews", "logit"), ("bert", "agnews", "feature")],
                                                 "r5_differs", "r3_", "r5_"), jn),
                os.path.join(out, "d_battery.csv"), """
        A15-d battery decisions (BERT AG News logit, and 'feature' through its borrowed pool) at r = 3 (frozen)
        and r = 5 (post hoc): registered = frozen analyze.primary_battery; own / claim / welch = frozen
        analyze.decide on those p-values (see b1_battery.csv); r5_differs_<rule> = deltas that differ.""")
    L.write_csv(L.with_jobs(L.holm_rows(F0, Fd, "r5_differs", "r3_", "r5_"), jn), os.path.join(out, "d_holm.csv"), """
        A15-d 31-cell Holm (frozen addendum.global_holm_table, p_pos) and F31 / F19 family claims with the BERT
        AG News logit cells at r = 5 (sensitivity; post hoc), beside r = 3; r5_differs_<rule> lists changed
        family:side@delta flags.""")
    prow = []
    for pre, F in (("r3", F0), ("r5", Fd)):
        key = ("bert", "agnews", "logit")
        p = F["pools"].get(key)
        if p is None:
            continue
        w = F["welch"][key]
        from scipy import stats
        att = {a: [F["att_var"][key + (a,)][f"k_s{s}"] for s in p["seeds"].split(",")]
               for a in p["attacks"].split(",")}
        b = stats.bartlett(*att.values())
        prow.append({"frame": pre, "pool": "bert/agnews/logit", "attacks": p["attacks"], "seeds": p["seeds"],
                     "r": w["r"], "s_pool": p["s_between"], "df_satterthwaite": p["df"], "df_welch": w["df_welch"],
                     "df_bound_low": w["r"] - 1, "df_bound_high": len(att) * (w["r"] - 1),
                     **{f"s_{a}": v ** 0.5 for a, v in sorted(w["s2a"].items())},
                     "bartlett_stat": float(b.statistic), "bartlett_p": float(b.pvalue),
                     "label": "" if pre == "r3" else POST_HOC})
    L.write_csv(prow, os.path.join(out, "d_pools.csv"), """
        BERT AG News logit pool at r = 3 (frozen) and r = 5: s_pool and Satterthwaite df
        (analyze.pool_seed_variance), Welch df (A15 E3), per-attack SD, Bartlett test of equal variance across
        attacks (descriptive).""")
    L.write_csv(L.claim_summary(F0, "r3 (frozen)") + L.claim_summary(Fd, "r5 (post hoc)"),
                os.path.join(out, "d_claims.csv"), "Claim-rule counts per frame and delta (see b1_claims.csv).")
    L.write_csv(floor_rows, os.path.join(out, "d_floor.csv"), """
        A8 student floor of every A15-d student (P per attack, C, C_ep, C_sos): Stage 2 clean accuracy > AG News
        majority share + 0.10 (0.35); a run with any student at or below it is left out (listed).""")
    checks = [{"check": "every non-BERT-AG-News primary row unchanged in the r = 5 frame",
               "ok": all(Fd["prim"][k][f] == F0["prim"][k][f] for k in F0["prim"] for f in ("k", "lo", "hi")
                         if k[:2] != ("bert", "agnews")), "note": ""},
              {"check": "d inputs complete (seeds 45 and 46, six attacks each, none floor-excluded)",
               "ok": not missing and not excluded and len(d) == 12,
               "note": L.missing_note(missing + [f"floor-excluded {r}" for r in excluded])},
              {"check": "BERT AG News logit pool seeds = 42-44 + the kept d seeds",
               "ok": Fd["pools"][("bert", "agnews", "logit")]["seeds"] == ",".join(map(str, [42, 43, 44] + seeds)),
               "note": Fd["pools"][("bert", "agnews", "logit")]["seeds"]},
              L.family_size_row(F0, Fd, "r5")]
    checks += L.pool_identity_rows({"r3 (frozen)": F0, "r5": Fd})
    checks += L.excluded_rows(ns.runs)
    L.write_csv(checks, os.path.join(out, "check_d.csv"), "A15-d self-checks ('d inputs complete' is False while "
                "a d job is missing or cut).")
    print(f"part d: seeds added {seeds}; missing {missing or 'none'}; floor-excluded {excluded or 'none'}")


if __name__ == "__main__":
    main()
