#!/usr/bin/env python3
"""A14 item 5: an auxiliary estimate of kappa_marg for BERT ATBA (SST-2 and AG News).

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Estimands (fixed poisoned build T_P = the teacher-seed-42 ATBA teacher, fixed KD corpus seed 42,
fixed trigger string and the attack's own insertion rule, the battery's n non-target inputs):
  kappa_cond(T_C) = E_S[ hit rate of S(T_P) ] - E_S[ hit rate of S(T_C) ]
      for ONE clean twin build T_C; E_S averages over student training (student seed).
      The registered estimand is kappa_cond(t42): the clean twin is the teacher-seed-42 clean
      build (docs/tifs_plan.md §0.3; frozen primary k).
  kappa_marg = E_{T_C}[ kappa_cond(T_C) ] = E_S[S(T_P)] - E_{T_C} E_S[S(T_C)]
      the clean twin averaged over independent clean builds of the same recipe (base trainer,
      BadNets-dev-split selection), i.e. what an auditor who cannot pick the twin build asks.

Estimates (all auxiliary):
  per_reference      k_j = P - C_j for each clean build j (t42 twin, A/A t43, t44; the A13 G2
                     t45, t46 when scored), seed 42, Tango and the frozen transported interval.
  kappa_marg_J       P - mean_j C_j over J clean builds, seed 42: SE^2 = v + s2_pool + s2_C / J,
                     v = Var_i(P_i - mean_j C_j,i)/n (frozen input term), s2_pool = the family's
                     logit pool (student-seed term, as the transported rule), s2_C = between-build
                     variance of the C_j hit rates (df J - 1); df by Satterthwaite as A3.3
                     ((v + s2_pool) at pool df, s2_C / J at J - 1). s2_C also carries the student
                     and input noise of each C_j, so the interval is conservative.
  kappa_marg_J2_sp   SST-2 only: references t42 and t43, seeds 42-44 (t43 C' at s43, s44 = A2
                     replicate students): SE^2 = s2_pool/3 + v + s2_C/2, df Satterthwaite.
  excess_null        the A3.4 excess: P - mean_b N_b over the within-build null students N_b
                     (students of teachers without the ATBA backdoor: the other attacks' P and
                     the SOS / EP clean twins), seeds 42-44, CPU fp32: frozen seed-pooled rule,
                     own variance, and with the between-null-student variance s2_N / J_N added
                     (df J_N - 1, Satterthwaite) = kappa_marg over the null students. SST-2
                     has 7/5/5 null students at seeds 42/43/44 (the EP pair has seed 42 only):
                     the frozen excess_mean definition averages the students each seed has, and
                     a '[sensitivity: ...]' row repeats the estimate with the 5 null students
                     present at every seed.
  mixed_recipe_sensitivity_J5  kappa_marg_J's formula with the clean-sos / clean-ep twins added
                     as references (teacher seed 42, dedicated trainers): NOT kappa_marg as
                     defined (not independent base-recipe builds), a sensitivity only.
  P_minus_null       P - N_b for every null student b (paired, seed-pooled or transported).
Official bf16 vectors (A/A C' from the atba_phrase_<ds> control, prefix, the same inputs) and
the CPU fp32 re-scores (cross_hits + hits/ref_*.json) are both shown (basis column).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

import common as K  # noqa: E402
import item7_reference_k as R  # noqa: E402

KD = ("logit", "feature")


def kappa_row(base, P, refs, s2p, dfp, label, J_note=""):
    """P - mean_j C_j at one seed with the between-build term (see module docstring)."""
    J = len(refs)
    n = len(P)
    cbar = [sum(c[i] for c in refs.values()) / J for i in range(n)]
    d = [p - c for p, c in zip(P, cbar)]
    k = K.mean(d)
    v = sum((x - k) ** 2 for x in d) / (n - 1) / n
    rates = [K.mean(c) for c in refs.values()]
    s2c = K.var(rates)
    se2 = v + s2p
    tot = se2 + s2c / J
    df = tot ** 2 / (se2 ** 2 / dfp + (s2c / J) ** 2 / (J - 1))
    res = K.t_contrast(k, tot ** 0.5, df, K.TOLS)
    return {**base, "estimand": label, "references": ",".join(refs), "J": J, "seeds": "42",
            "k": k, "se": res["se"], "df": df, "lo": res["lo"], "hi": res["hi"], **K.verdicts(res),
            "v_input": v, "s2_pool": s2p, "s2_between_builds": s2c, "s_between_builds": s2c ** 0.5,
            "P_rate": K.mean(P), "mean_C_rate": K.mean(rates), "note": J_note}


def official_refs(st, ds):
    base = st.groups[("bert", ds, "logit", "atba")][42]
    refs = {"t42": base["C"]}
    for c in st.cells:
        if (c["block"] == "AA" and c["corpus"] == "clean" and c["fam"] == "bert" and c["dataset"] == ds
                and not c["single"] and c["kd"] == "logit"):
            ctl = c["controls"].get(f"atba_phrase_{ds}")
            if ctl is not None and ctl.get("c") is not None:
                refs[f"t{c['teacher_seed']}"] = ctl["c"]
    return dict(sorted(refs.items(), key=lambda kv: int(kv[0][1:])))


def main():
    out = K.out_dir()
    st = K.state()
    rows, checks = [], []
    ref_off = {r["dataset"] + "/" + r["reference"]: r for r in K.read_csv(os.path.join(
        K.ADD, "aux", "g2-validity-reference-flips-builds", "reference_atba_by_reference.csv"))
        if r["kd"] == "logit"}
    a33 = {(r["dataset"], r["kd"]): r for r in K.read_csv(os.path.join(K.ADD, "reference_atba.csv"))}
    cn = {(r["dataset"], r["kd"], r["trigger"], r["student"]): r
          for r in K.read_csv(os.path.join(K.ADD, "cross_null.csv"))}
    for ds in ("sst2", "agnews"):
        _, s2p, dfp = K.pool(st, ds)
        # ── official bf16 basis ──
        refs = official_refs(st, ds)
        for kd in KD:
            P = st.groups[("bert", ds, kd, "atba")][42]["P"]
            base = {"dataset": ds, "kd": kd, "basis": "official Stage 2, GPU bf16"}
            for lab, C in refs.items():
                pc, tr = K.transported(P, C, s2p, dfp)
                rows.append({**base, "estimand": "per_reference kappa_cond", "references": lab, "J": 1,
                             "seeds": "42", "k": pc["k"], "tango_lo": pc["lo"], "tango_hi": pc["hi"],
                             "se": tr["se"], "df": dfp, "lo": tr["lo"], "hi": tr["hi"], **K.verdicts(tr),
                             "P_rate": pc["a"], "mean_C_rate": pc["b"],
                             "note": ("logit-KD references for the feature P (as A3.3)" if kd == "feature"
                                      else "t42 = the registered clean twin" if lab == "t42" else
                                      "A/A clean build, atba_phrase control (prefix; same inputs)")})
                o = ref_off.get(f"{ds}/{lab}")
                if kd == "logit" and o is not None:
                    checks.append({"check": f"g2 reference_atba_by_reference {ds}/{lab}",
                                   "max_abs_diff": max(abs(pc["k"] - float(o["tango_k"])),
                                                       abs(tr["lo"] - float(o["transported_lo"])),
                                                       abs(tr["hi"] - float(o["transported_hi"])))})
            rows.append(kappa_row(base, P, refs, s2p, dfp, f"kappa_marg_J{len(refs)}",
                                  "logit-KD references for the feature P" if kd == "feature" else ""))
            # A3.3 reproduction (frozen formula on the primary row).
            pr = st.prim[("bert", ds, kd, "atba")]
            rates = [K.mean(c) for c in refs.values()]
            s2r, dfr = K.var(rates), len(rates) - 1
            se2 = pr["se"] ** 2
            df = (se2 + s2r) ** 2 / (se2 ** 2 / pr["df"] + s2r ** 2 / dfr)
            rob = K.t_contrast(pr["k"], (se2 + s2r) ** 0.5, df, K.TOLS)
            rows.append({**base, "estimand": "A3.3 (kappa_cond(t42) with one-build reference variance)",
                         "references": ",".join(refs), "J": len(refs), "seeds": pr["seeds"], "k": pr["k"],
                         "se": rob["se"], "df": df, "lo": rob["lo"], "hi": rob["hi"], **K.verdicts(rob),
                         "s2_between_builds": s2r, "s_between_builds": s2r ** 0.5,
                         "note": "frozen primary k +- t sqrt(SE^2 + s2_ref): reproduces reference_atba.csv"})
            o = a33.get((ds, kd))
            if o is not None:
                checks.append({"check": f"A3.3 reference_atba.csv {ds}/{kd}",
                               "max_abs_diff": max(abs(rob["lo"] - float(o["ref_lo"])),
                                                   abs(rob["hi"] - float(o["ref_hi"])),
                                                   abs(df - float(o["ref_df"])))})
        # SST-2: references t42 and t43 at seeds 42-44 (official vectors).
        if ds == "sst2":
            g = st.groups[("bert", "sst2", "logit", "atba")]
            rep = K.AD.load_replicate(K.RUNS)
            seeds = sorted(set(g) & set(rep))
            d = {s: [p - (c + cp) / 2 for p, c, cp in zip(g[s]["P"], g[s]["C"], rep[s]["C"])] for s in seeds}
            k, se, r, v, ks = K.AD.seed_pooled(d, s2p, dfp)
            rates = [K.mean([K.mean(g[s]["C"]) for s in seeds]), K.mean([K.mean(rep[s]["C"]) for s in seeds])]
            s2c = K.var(rates)
            tot = se ** 2 + s2c / 2
            df = tot ** 2 / (se ** 4 / dfp + (s2c / 2) ** 2 / 1)
            res = K.t_contrast(k, tot ** 0.5, df, K.TOLS)
            rows.append({"dataset": ds, "kd": "logit", "basis": "official Stage 2, GPU bf16",
                         "estimand": "kappa_marg_J2_sp", "references": "t42,t43", "J": 2,
                         "seeds": ",".join(map(str, seeds)), "k": k,
                         **{f"k_s{s}": x for s, x in zip(seeds, ks)}, "se": res["se"], "df": df,
                         "lo": res["lo"], "hi": res["hi"], **K.verdicts(res), "v_input": v,
                         "s2_pool": s2p, "s2_between_builds": s2c, "s_between_builds": s2c ** 0.5,
                         "note": "t43 C' at s42-44 = A2 replicate clean-base students; df_ref = 1"})
        # ── CPU fp32 basis ──
        X = K.load_cross(ds)["hits"]["atba"]
        pt = K.P_TAG[ds]["atba"]
        P42 = X[f"logit/{pt}/poisoned/s42"]
        crefs = {"t42": X["logit/clean-base/clean/s42"]}
        extra = {"t42 clean-sos": X["logit/clean-sos/clean/s42"], "t42 clean-ep": X["logit/clean-ep/clean/s42"]}
        built = R.load_references(out, ds)
        for lab, rf in built.items():
            if rf["included"] and "atba" in rf["hits"]:
                crefs[lab] = rf["hits"]["atba"]
        base = {"dataset": ds, "kd": "logit", "basis": "CPU fp32 re-score"}
        for lab, C in crefs.items():
            pc, tr = K.transported(P42, C, s2p, dfp)
            rows.append({**base, "estimand": "per_reference kappa_cond", "references": lab, "J": 1,
                         "seeds": "42", "k": pc["k"], "tango_lo": pc["lo"], "tango_hi": pc["hi"],
                         "se": tr["se"], "df": dfp, "lo": tr["lo"], "hi": tr["hi"], **K.verdicts(tr),
                         "P_rate": pc["a"], "mean_C_rate": pc["b"]})
        if len(crefs) >= 2:
            rows.append(kappa_row(base, P42, crefs, s2p, dfp, f"kappa_marg_J{len(crefs)}",
                                  "base-trainer clean builds"))
            rows.append(kappa_row(base, P42, {**crefs, **extra}, s2p, dfp,
                                  f"mixed_recipe_sensitivity_J{len(crefs) + 2} (not kappa_marg)",
                                  "not kappa_marg as defined: adds the dedicated-trainer clean twins of "
                                  "SOS and EP as references; they share teacher seed 42 with the t42 twin "
                                  "and differ in trainer, so they are not independent base-recipe builds"))
        # Excess over the within-build null students (A3.4), seeds 42-44.
        nulls = sorted({k.rsplit("/s", 1)[0] for k in X if k.startswith("logit/")
                        and k.rsplit("/s", 1)[0] not in (f"logit/{pt}/poisoned", "logit/clean-base/clean")})
        seeds = [s for s in (42, 43, 44) if f"logit/{pt}/poisoned/s{s}" in X]
        full = [b for b in nulls if all(f"{b}/s{s}" in X for s in seeds)]
        n_per_seed = {s: sum(f"{b}/s{s}" in X for b in nulls) for s in seeds}
        uneven = len(full) < len(nulls)
        uneven_note = ("null students per seed " + "/".join(str(n_per_seed[s]) for s in seeds)
                       + " (seeds " + ",".join(map(str, seeds)) + "): "
                       + ", ".join(b.split("/", 1)[1] for b in nulls if b not in full)
                       + " exist at seed 42 only, so each seed's null mean averages a different set; "
                       "J_N counts every null student and their hit rates enter s2_N from the seeds "
                       "they have (frozen excess_mean definition)") if uneven else ""

        def excess(group, sens):
            d, exc = {}, {}
            for s in seeds:
                Ns = [X[f"{b}/s{s}"] for b in group if f"{b}/s{s}" in X]
                P = X[f"logit/{pt}/poisoned/s{s}"]
                nbar = [sum(v[i] for v in Ns) / len(Ns) for i in range(len(P))]
                d[s] = K.diff(P, nbar)
                exc[s] = K.mean(P) - K.mean([K.mean(v) for v in Ns])
            res = K.pooled_and_own(d, s2p, dfp)
            null_rates = [K.mean([K.mean(X[f"{b}/s{s}"]) for s in seeds if f"{b}/s{s}" in X]) for b in group]
            s2n, Jn = K.var(null_rates), len(group)
            tot = res["se_pooled"] ** 2 + s2n / Jn
            dfn = tot ** 2 / (res["se_pooled"] ** 4 / dfp + (s2n / Jn) ** 2 / (Jn - 1))
            wide = K.t_contrast(res["k"], tot ** 0.5, dfn, K.TOLS)
            tag = (f" [sensitivity: the {Jn} null students present at every seed]" if sens else "")
            common = {**base, "references": ";".join(b.split("/", 1)[1] for b in group),
                      "J": Jn, "seeds": ",".join(map(str, seeds)), "k": res["k"],
                      **{f"k_s{s}": x for s, x in zip(seeds, res["ks"])},
                      "excess_mean_frozen_definition": K.mean(list(exc.values())),
                      "null_students_per_seed": "/".join(
                          str(sum(f"{b}/s{s}" in X for b in group)) for s in seeds),
                      "P_rate": K.mean([K.mean(X[f"logit/{pt}/poisoned/s{s}"]) for s in seeds]),
                      "mean_C_rate": K.mean(null_rates), "s2_between_builds": s2n,
                      "s_between_builds": s2n ** 0.5}
            note0 = ("" if sens else uneven_note)
            out_rows = [
                {**common, "estimand": "excess_null (frozen seed-pooled rule)" + tag, "se": res["se_pooled"],
                 "df": dfp, "lo": res["pooled"]["lo"], "hi": res["pooled"]["hi"], **K.verdicts(res["pooled"]),
                 "note": "null-student means averaged per input" + ("; " + note0 if note0 else "")},
                {**common, "estimand": "excess_null (own seed variance, df r-1)" + tag,
                 "se": (res["s2a"] / res["r"] + res["v"]) ** 0.5, "df": res["r"] - 1,
                 "lo": res["own"]["lo"], "hi": res["own"]["hi"], **K.verdicts(res["own"])},
                {**common, "estimand": "excess_null + between-null-student variance (kappa_marg over nulls)" + tag,
                 "se": tot ** 0.5, "df": dfn, "lo": wide["lo"], "hi": wide["hi"], **K.verdicts(wide),
                 "note": "s2_N / J_N added (df J_N - 1, Satterthwaite)" + ("; " + note0 if note0 else "")}]
            return out_rows, exc

        ex_rows, exc = excess(nulls, False)
        rows += ex_rows
        if uneven and len(full) >= 2:
            rows += excess(full, True)[0]
        off = cn.get((ds, "logit", "atba", f"{pt}/poisoned"))
        if off is not None:
            checks.append({"check": f"cross_null.csv excess_mean {ds}/logit/atba",
                           "max_abs_diff": abs(K.mean(list(exc.values())) - float(off["excess_mean"]))})
        for b in nulls:
            ss = [s for s in seeds if f"{b}/s{s}" in X]
            dd = {s: K.diff(X[f"logit/{pt}/poisoned/s{s}"], X[f"{b}/s{s}"]) for s in ss}
            if len(ss) >= 2:
                rr = K.pooled_and_own(dd, s2p, dfp)
                rows.append({**base, "estimand": "P_minus_null", "references": b.split("/", 1)[1], "J": 1,
                             "seeds": ",".join(map(str, ss)), "k": rr["k"],
                             **{f"k_s{s}": x for s, x in zip(ss, rr["ks"])}, "se": rr["se_pooled"], "df": dfp,
                             "lo": rr["pooled"]["lo"], "hi": rr["pooled"]["hi"], **K.verdicts(rr["pooled"]),
                             "own_lo": rr["own"]["lo"], "own_hi": rr["own"]["hi"], **K.verdicts(rr["own"], "own_")})
            else:
                s, = ss
                pc, tr = K.transported(X[f"logit/{pt}/poisoned/s{s}"], X[f"{b}/s{s}"], s2p, dfp)
                rows.append({**base, "estimand": "P_minus_null", "references": b.split("/", 1)[1], "J": 1,
                             "seeds": str(s), "k": pc["k"], "se": tr["se"], "df": dfp, "lo": tr["lo"],
                             "hi": tr["hi"], **K.verdicts(tr), "note": "one seed: transported"})
    K.write_csv(rows, os.path.join(out, "kappa_marg.csv"), """
        BERT ATBA (SST-2 912 / AG News 5,700 ATBA-triggered non-target inputs). kappa_cond(T_C) =
        E_S[P] - E_S[C(T_C)] for one clean twin build T_C (registered: t42); kappa_marg =
        E_{T_C} kappa_cond(T_C) over independent clean builds. per_reference = P(s42) - C_j,
        frozen transported interval; kappa_marg_J = P(s42) - mean_j C_j with SE^2 = v + s2_pool +
        s2_C/J (Satterthwaite df; conservative); kappa_marg_J2_sp = SST-2 t42+t43 at seeds 42-44;
        mixed_recipe_sensitivity_J5 = the same formula with the clean-sos and clean-ep twins added
        (teacher seed 42, different trainer: NOT kappa_marg as defined, a sensitivity only);
        A3.3 = frozen reference_atba.csv formula (reproduced); excess_null = P - mean of the A3.4
        null students (seeds 42-44; frozen seed-pooled, own variance, and + s2_N/J_N; SST-2 has
        7/5/5 null students at seeds 42/43/44 because the EP pair has seed 42 only, so a
        '[sensitivity: ...]' row repeats it with the 5 null students present at every seed);
        P_minus_null = P - each null student. basis: official bf16 or CPU fp32.""")
    K.write_csv(checks, os.path.join(out, "check_item5.csv"), """
        Self-checks of item 5: official per-reference rows vs g2 reference_atba_by_reference.csv,
        the A3.3 reproduction vs reference_atba.csv and the excess vs cross_null.csv
        excess_mean (maximum absolute difference).""")
    print(f"item 5: {len(rows)} rows; checks max diff "
          f"{max((c['max_abs_diff'] for c in checks), default=float('nan')):.3g}")


if __name__ == "__main__":
    main()
