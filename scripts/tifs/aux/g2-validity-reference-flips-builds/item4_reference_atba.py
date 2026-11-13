#!/usr/bin/env python3
"""A11 item 4 (기준별 ATBA 표): the ATBA P student against each clean reference
build, C(t42), C'(t43), C'(t44). Post hoc auxiliary table
(docs/tifs_addendum.md A11); it does not change any primary verdict.

P is the BERT logit-KD ATBA student of the poisoned teacher t42 (student seed
42; corpus seed 42). The references are clean-base-trainer students of three
independently built clean teachers, all corpus seed 42, student seed 42, on
the same 912 (SST-2) / 5,700 (AG News) inputs with the ATBA phrase as a
prefix: C(t42) is the battery cell's own C on the ATBA-triggered inputs;
C'(t43), C'(t44) are the A/A clean runs' C' on the atba_phrase_<ds> control
(same prefix insertion; the frozen aa_table pairs them the same way).

Readings: Tango (frozen paired_contrast) and transported (frozen t_contrast with
SE^2 = SE_Tango^2 + s2_pool, the BERT <ds> logit pool and its df), each at
delta in {0.02, 0.05, 0.10}. Where student seeds 43/44 exist for a reference
(t42: the battery; t43 on SST-2: the A2 replicate's C' students), the
seed-pooled frozen formula is added (s2_pool / r + v, pool df).

Agreement: per dataset x basis x delta, (i) how many alternative references
(t43, t44) give the same verdict as C(t42), (ii) how many of the reference
pairs agree with each other, (iii) whether all available references agree.
These columns count any equal pair of verdicts, INCONCLUSIVE = INCONCLUSIVE
included. The *_decisive_* columns (extra) count a pair only when both verdicts
are decisive (EQUIVALENT, POSITIVE or NEGATIVE) and equal.

Outputs: reference_atba_by_reference.csv, reference_atba_agreement.csv.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # no __pycache__ inside the repository

import itertools  # noqa: E402

from g2common import (AD, DECISIVE, OFFICIAL, RUNS, TOLS, VKEYS, cli, mean, state, t_contrast, tango,  # noqa: E402
                      transported, write)


def dec_eq(a, b):
    """Decisive agreement: both verdicts decisive (EQ / POS / NEG) and equal."""
    return a == b and a in DECISIVE

BASES = (("Tango, student seed 42", "tango_"), ("transported, student seed 42", "transported_"),
         ("seed-pooled (where seeds 42-44 exist)", "seedpooled_"))


def main():
    cli(__doc__.split("\n\n")[0])
    st = state()
    rep = AD.load_replicate(RUNS)
    rows, agree = [], []
    # context: the registered A3.3 reading (frozen addendum.reference_atba_table) and the
    # frozen robust_* columns of the primary row, repeated on every reference row of a dataset
    a33 = {r["dataset"]: r for r in AD.reference_atba_table(st) if r.get("kd") == "logit" and "ref_lo" in r}
    for ds in ("sst2", "agnews"):
        groups = st.groups.get(("bert", ds, "logit", "atba"), {})
        c42 = groups.get(42)
        if c42 is None:
            continue
        pk, pool = st.pool_for("bert", ds, "logit")
        s2p, dfp = pool["s_between"] ** 2, pool["df"]
        refs = {42: {"C": c42["C"], "run": c42["run"], "input_set": "ATBA-triggered inputs (cell C)",
                     "teacher_resp": mean(c42["TC"]), "teacher_ca": c42["teacher_ca_null"],
                     "student_ca": c42["ca_c"]}}
        for c in st.cells:
            if (c["block"] == "AA" and c["fam"] == "bert" and c["dataset"] == ds and c["corpus"] == "clean"
                    and c["kd"] == "logit" and not c["single"]):
                ctl = c["controls"].get(f"atba_phrase_{ds}")
                if ctl is None or ctl.get("c") is None:
                    continue
                refs[c["teacher_seed"]] = {"C": ctl["c"], "run": c["run"],
                                           "input_set": f"control atba_phrase_{ds} (prefix)",
                                           "teacher_resp": mean(ctl["teacher_c"]) if ctl.get("teacher_c") else None,
                                           "teacher_ca": c["teacher_ca_null"], "student_ca": c["ca_c"]}
        prim = st.prim.get(("bert", ds, "logit", "atba"))
        by_ref = {}
        for t, ref in sorted(refs.items()):
            flat, pc = tango(c42["P"], ref["C"])
            tr = transported(pc, pool)
            row = {"dataset": ds, "kd": "logit", "attack": "atba", "reference": f"t{t}", "reference_teacher_seed": t,
                   "reference_student_seed": 42, "reference_run": ref["run"], "reference_inputs": ref["input_set"],
                   "precision": OFFICIAL, "P_run": c42["run"], "P_teacher_seed": 42, "P_student_seed": 42,
                   "P_rate": pc["a"], "C_ref_rate": pc["b"],
                   "n": pc["n"], "n10": pc["n10"], "n01": pc["n01"],
                   "tango_k": pc["k"], "tango_lo": pc["lo"], "tango_hi": pc["hi"], "tango_delta_star": pc["delta_star"],
                   **{f"tango_{v}": pc[v] for v in VKEYS},
                   "transported_se": tr["se"], "transported_df": tr["df"], "transported_lo": tr["lo"],
                   "transported_hi": tr["hi"], **{f"transported_{v}": tr[v] for v in VKEYS},
                   "variance_from": "/".join(pk), "pool_s": pool["s_between"], "pool_df": dfp,
                   "ref_teacher_atba_phrase_rate": ref["teacher_resp"], "ref_teacher_clean_acc": ref["teacher_ca"],
                   "ref_student_clean_acc": ref["student_ca"],
                   "P_teacher_atba_rate": mean(c42["TP"]), "P_teacher_clean_acc": c42["teacher_ca"],
                   "P_student_clean_acc": c42["ca_p"]}
            # seed-pooled reading where the reference has student seeds 42-44
            sp = None
            if t == 42 and prim is not None and prim.get("method", "").startswith("seed-pooled"):
                sp = {"seeds": prim["seeds"], "k": prim["k"], "se": prim["se"], "df": prim["df"], "lo": prim["lo"],
                      "hi": prim["hi"], **{v: prim[v] for v in VKEYS},
                      "note": "= frozen primary row (battery C at seeds 42-44)"}
            elif t == 43 and ds == "sst2" and len(rep) >= 2:
                d = {s: [p - q for p, q in zip(groups[s]["P"], v["C"])] for s, v in rep.items() if s in groups}
                if len(d) >= 2:
                    k, se, r, vv, ks = AD.seed_pooled(d, s2p, dfp)
                    res = t_contrast(k, se, dfp, TOLS)
                    sp = {"seeds": ",".join(map(str, sorted(d))), "k": k, "se": se, "df": dfp, "lo": res["lo"],
                          "hi": res["hi"], **{v: res[v] for v in VKEYS},
                          **{f"k_s{s}": x for s, x in zip(sorted(d), ks)},
                          "note": "P_s (battery, t42 poisoned) - C'_s (A2 replicate clean-base t43 students); "
                                  "the s42 C' is the A/A t43 student (identical weights)"}
            if sp is not None:
                row.update({f"seedpooled_{k}": v for k, v in sp.items()})
            a = a33.get(ds)
            if a is not None:
                row.update({"A33_s_ref": a["s_ref"], "A33_df_ref": a["df_ref"], "A33_ref_lo": a["ref_lo"],
                            "A33_ref_hi": a["ref_hi"], "A33_ref_df": a["ref_df"],
                            **{f"A33_ref_{v}": a[f"ref_{v}"] for v in VKEYS}})
            if prim is not None and prim.get("robust_lo") is not None:
                row.update(primary_robust_lo=prim["robust_lo"], primary_robust_hi=prim["robust_hi"],
                           primary_robust_from=prim.get("robust_from"),
                           **{"primary_robust_verdict@0.05": prim.get("robust_verdict@0.05")})
            rows.append(row)
            by_ref[t] = row

        # agreement across references, per basis and delta
        for basis, pref in BASES:
            tot = {"alt_agree": 0, "alt": 0, "pairs_agree": 0, "pairs": 0, "all": 0, "deltas": 0,
                   "alt_dec": 0, "pairs_dec": 0, "all_dec": 0}
            for tol in TOLS:
                vk = f"{pref}verdict@{tol:g}"
                vs = {t: r.get(vk) for t, r in by_ref.items() if r.get(vk)}
                if not vs:
                    continue
                base = vs.get(42)
                alts = {t: v for t, v in vs.items() if t != 42}
                a = sum(v == base for v in alts.values())
                a_dec = sum(dec_eq(v, base) for v in alts.values())
                prs = list(itertools.combinations(sorted(vs), 2))
                pa = sum(vs[x] == vs[y] for x, y in prs)
                pa_dec = sum(dec_eq(vs[x], vs[y]) for x, y in prs)
                # undefined with fewer than two references (nothing to agree with)
                allv = (len(set(vs.values())) == 1) if len(vs) >= 2 else None
                allv_dec = (allv and next(iter(vs.values())) in DECISIVE) if allv is not None else None
                tot["alt_agree"] += a
                tot["alt"] += len(alts)
                tot["pairs_agree"] += pa
                tot["pairs"] += len(prs)
                tot["alt_dec"] += a_dec
                tot["pairs_dec"] += pa_dec
                if allv is not None:
                    tot["all"] += allv
                    tot["all_dec"] += allv_dec
                    tot["deltas"] += 1
                agree.append({"dataset": ds, "basis": basis, "delta": tol,
                              **{f"verdict_t{t}": vs.get(t) for t in (42, 43, 44)},
                              "references_available": ",".join(f"t{t}" for t in sorted(vs)),
                              "all_available_agree": allv, "n_alt_agree_with_t42": a, "n_alt": len(alts),
                              "n_reference_pairs_agree": pa, "n_reference_pairs": len(prs),
                              "all_available_agree_decisive": allv_dec,
                              "n_alt_decisive_agree_with_t42": a_dec,
                              "n_reference_pairs_decisive_agree": pa_dec})
            if any(r["dataset"] == ds and r["basis"] == basis for r in agree):
                agree.append({"dataset": ds, "basis": basis, "delta": "all",
                              "references_available": ",".join(f"t{t}" for t in sorted(by_ref)
                                                               if by_ref[t].get(f"{pref}verdict@0.05")),
                              "agreement_rate_with_t42": (tot["alt_agree"] / tot["alt"]) if tot["alt"] else None,
                              "n_alt_agree_with_t42": tot["alt_agree"], "n_alt": tot["alt"],
                              "pairwise_agreement_rate": (tot["pairs_agree"] / tot["pairs"]) if tot["pairs"] else None,
                              "n_reference_pairs_agree": tot["pairs_agree"], "n_reference_pairs": tot["pairs"],
                              "share_deltas_all_agree": (tot["all"] / tot["deltas"]) if tot["deltas"] else None,
                              "n_deltas_with_2plus_references": tot["deltas"],
                              "decisive_agreement_rate_with_t42": (tot["alt_dec"] / tot["alt"]) if tot["alt"] else None,
                              "n_alt_decisive_agree_with_t42": tot["alt_dec"],
                              "pairwise_decisive_agreement_rate": (tot["pairs_dec"] / tot["pairs"]) if tot["pairs"] else None,
                              "n_reference_pairs_decisive_agree": tot["pairs_dec"],
                              "share_deltas_all_agree_decisive": (tot["all_dec"] / tot["deltas"]) if tot["deltas"] else None})
    write(rows, "reference_atba_by_reference.csv")
    write(agree, "reference_atba_agreement.csv")
    for r in rows:
        print(r["dataset"], r["reference"], "P=%.4f C=%.4f k=%.4f [%.4f, %.4f]" % (r["P_rate"], r["C_ref_rate"], r["tango_k"], r["tango_lo"], r["tango_hi"]),
              [r[f"tango_{v}"] for v in VKEYS], "tr [%.4f, %.4f]" % (r["transported_lo"], r["transported_hi"]),
              [r[f"transported_{v}"] for v in VKEYS], "sp", r.get("seedpooled_k"), [r.get(f"seedpooled_{v}") for v in VKEYS],
              "teacher resp %.4f ca %.4f" % (r["ref_teacher_atba_phrase_rate"], r["ref_teacher_clean_acc"]))
    for a in agree:
        print(a)


if __name__ == "__main__":
    main()
