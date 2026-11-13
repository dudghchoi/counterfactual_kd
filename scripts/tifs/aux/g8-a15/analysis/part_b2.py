#!/usr/bin/env python3
"""A15-b2: epoch-matched clean references (A3.3 reference variance, A14 item 7 per-reference k) and the A2
replicate with an epoch-matched C'.

A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Registered (docs/tifs_addendum.md A15-b2; claim rule of the 10/27 20:58 KST amendment):
  * References at each P's epoch: the clean twins t42 (A15-b1 C_e), t43, t44, t45, t46 rebuilt with every epoch
    saved; C' students (logit, corpus and student seed 42) at epoch 1 (ATBA, AddSent), 4 (BadNets), 5 (SynBkd);
    the frozen C'(t43) is epoch 1 and is reused (tifs_aa_bert_clean_t43).
  * A3.3 (ATBA phrase, BERT SST-2): s2_ref = variance of the references' ATBA-phrase hit rates at epoch 1 (t42 =
    the epoch-matched battery C on the ATBA-triggered inputs; t43..t46 = the C' students' atba_phrase_sst2
    control; official GPU bf16), added to the epoch-matched primary SE: k_em +- t sqrt(SE^2 + s2_ref),
    Satterthwaite df; by the frozen addendum.reference_atba_table, unchanged, on a proxy state (the A15-b1
    frame's groups and primary rows; the epoch-1 reference cells). Three references (t42-t44, df 2: the
    registered A3.3 set) and five (t42-t46, df 4: the A13 G2 set). Own-variance analogue (A3.1 SE) and claim
    rule beside it; sentences of A13 G2.
  * A14 item 7: per BERT SST-2 base-trainer trigger and KD (logit, 'feature'), k = P(t42, seed 42) - C_ref with
    each reference at e(P) (CPU fp32 hits from score_em.py; the t43 epoch-1 reference from the A14 hits), the
    frozen transported interval (Tango SE^2 + s2_pool, pool df; pool = the epoch-matched BERT SST-2 logit
    pool, the frozen pool beside it); 'reference-dependent' at delta when the verdict is not the same for every
    reference; the A3.3-style interval of the trigger (base-trainer references' hit-rate variance added to the
    epoch-matched primary SE). BadNets and ATBA also from the official Stage 2 vectors (GPU bf16).
  * A2: k'_em = P'(t43, e5) - C'(t43, e5), seeds 42-44 (s42: tifs_x15_b2_bert_ref_t43_e5; s43/s44:
    tifs_x15_b2_a2_t43_e5_s*): the registered A2 rule (frozen seed-pooled formula with the BERT SST-2 logit pool:
    epoch-matched pool, frozen pool beside it), own variance (df r - 1), claim rule; readings replicated / not
    replicated / inconclusive; build difference k_em - k'_em (sqrt(SE^2 + SE'^2), pool df).
  * Functional floor (A7/A8/A13 G2): a reference is used only if its clean teacher (Stage 2 teacher_ca_null at
    that epoch) and its C' student (ca_clean) are above the majority share + 0.10.

Outputs: b2_reference_atba.csv, b2_reference_k.csv, b2_reference_k_summary.csv, b2_a2.csv, b2_references.csv,
check_b2.csv.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

import a15lib as L  # noqa: E402
import part_b1  # noqa: E402

A, AD, TOLS, ALPHA = L.A, L.AD, L.TOLS, L.ALPHA
EPOCH_OF = {"atba": 1, "addsent": 1, "badnets": 4, "synbkd": 5}
S_ABOVE = "깨끗한 기준들을 평균해도 teacher 몫이 δ를 넘는다"
S_BELOW = "기준들을 평균하면 δ 아래"
S_DEP = "reference-dependent 문구를 유지하고 df {df}를 근거로 든다"


def ref_cells(st, x15):
    """{(t, e): cell}: the A15-b2 reference cells (BadNets tag: P' auxiliary, C' at epoch e, controls) and
    the frozen A/A t43 cell as (43, 1); each with its functional-floor status."""
    out = {}
    for c in x15:
        if c["part"] == "b2" and c["block"] == "AA" and c["kd"] == "logit" and c["attack"] == "badnets" \
                and c["dataset"] == "sst2":
            out[(c["teacher_seed"], c["epoch"])] = c
    for c in st.cells:
        if (c["block"] == "AA" and c["corpus"] == "clean" and c["fam"] == "bert" and c["dataset"] == "sst2"
                and c["kd"] == "logit" and not c["single"] and c["teacher_seed"] == 43 and c["attack"] == "badnets"):
            out[(43, 1)] = c
    floor = L.floor_of("sst2")
    for c in out.values():
        c["functional_teacher"] = c.get("teacher_ca_null") is not None and c["teacher_ca_null"] > floor
        c["functional_student"] = c.get("ca_c") is not None and c["ca_c"] > floor
        c["functional"] = c["functional_teacher"] and c["functional_student"]
    return out


def a33_rows(st, Fem, refs):
    """A3.3 on the epoch-matched frame with the epoch-1 references (3- and 5-reference sets)."""
    rows = []
    agnews_aa = [c for c in st.cells if c["block"] == "AA" and c["corpus"] == "clean" and c["fam"] == "bert"
                 and c["dataset"] == "agnews" and c["kd"] == "logit" and not c["single"]]
    frozen3 = {(r["dataset"], r["kd"]): r for r in L.read_csv(L.OFFICIAL["reference_atba"])}
    g2 = os.path.join(L.G7, "g2_reference_atba5.csv")
    frozen5 = {(r["dataset"], r["kd"]): r for r in L.read_csv(g2)} if os.path.exists(g2) else {}
    for label, ts in (("3 references (t42-t44, registered A3.3 set)", (43, 44)),
                      ("5 references (t42-t46, A13 G2 set)", (43, 44, 45, 46))):
        use = [refs[(t, 1)] for t in ts if (t, 1) in refs and refs[(t, 1)]["functional"]]
        excluded = [f"t{t}" for t in ts if (t, 1) in refs and not refs[(t, 1)]["functional"]]
        missing = [f"t{t}" for t in ts if (t, 1) not in refs]
        proxy = L.Proxy(groups=Fem["groups"], prim=Fem["prim"], cells=use + agnews_aa)
        for r in AD.reference_atba_table(proxy):
            if r["dataset"] != "sst2" or "kd" not in r:
                continue
            pr = Fem["prim"][("bert", "sst2", r["kd"], "atba")]
            row = {"reference_set": label, "references_used": ",".join(["t42"] + [f"t{c['teacher_seed']}" for c in use]),
                   "references_missing": ",".join(missing), "references_excluded_floor": ",".join(excluded),
                   "epoch": 1, **r}
            # own-variance analogue and claim rule
            if pr["own"] is not None and r.get("s_ref") is not None:
                se2, dfo = pr["own"]["se"] ** 2, pr["own"]["df"]
                s2r, dfr = r["s_ref"] ** 2, r["df_ref"]
                df = L.satterthwaite2(se2, dfo, s2r, dfr)
                own = L.t_contrast(pr["k"], (se2 + s2r) ** 0.5, df, TOLS)
                row.update({"ref_own_lo": own["lo"], "ref_own_hi": own["hi"], "ref_own_df": df,
                            **{f"ref_own_verdict@{t:g}": own[f"verdict@{t:g}"] for t in TOLS}})
                for t in TOLS:
                    a, b = r[f"ref_verdict@{t:g}"], own[f"verdict@{t:g}"]
                    row[f"ref_claim@{t:g}"] = a if a == b else "rule-dependent"
            else:
                for t in TOLS:
                    row[f"ref_claim@{t:g}"] = "one-seed, descriptive" if pr["claim_class"] == "one-seed" else ""
            # Welch-df sensitivity (A15 E3): the same interval with the pool's Welch df in place of the pool df
            if pr.get("welch") and r.get("s_ref") is not None and pr.get("se"):
                s2r, dfr = r["s_ref"] ** 2, r["df_ref"]
                dfw = L.satterthwaite2(pr["se"] ** 2, pr["welch"]["df_welch"], s2r, dfr)
                rw = L.t_contrast(pr["k"], (pr["se"] ** 2 + s2r) ** 0.5, dfw, TOLS)
                row.update({"ref_welch_df": dfw, "ref_welch_lo": rw["lo"], "ref_welch_hi": rw["hi"],
                            **{f"ref_welch_verdict@{t:g}": rw[f"verdict@{t:g}"] for t in TOLS}})
            for t in TOLS:
                if r.get("ref_lo") is None:
                    continue
                row[f"sentence@{t:g}"] = (S_ABOVE if r["ref_lo"] > t else S_BELOW if r["ref_hi"] < t else
                                          S_DEP.format(df=r["df_ref"]))
            fz = (frozen3 if ts == (43, 44) else frozen5).get(("sst2", r["kd"]))
            if fz:
                pre = "" if ts == (43, 44) else "ref5_"
                for t in TOLS:
                    v = fz.get(f"{pre}verdict@{t:g}" if ts != (43, 44) else f"ref_verdict@{t:g}")
                    row[f"frozen_ref_verdict@{t:g}"] = v
                    row[f"epoch_dependent_ref@{t:g}"] = (v != r[f"ref_verdict@{t:g}"]) if v else None
                row["frozen_s_ref"] = fz.get("s_ref") if ts == (43, 44) else fz.get("s_ref")
            rows.append(row)
    return rows


def load_hits(path, key, atk, sha):
    if not os.path.exists(path):
        return None, "missing file"
    X = L.load_json(path)
    if X["inputs"].get(atk, {}).get("sha256") != sha:
        return None, "inputs sha256 differ"
    v = X["hits"].get(atk, {}).get(key)
    return (v, "ok") if v is not None else (None, "key missing")


def item7_rows(st, Fem, refs, hits_dir):
    """Per (trigger, kd, reference) rows and per (trigger, kd) summaries, CPU fp32 and GPU bf16."""
    cross = L.load_json(L.CROSS["sst2"])
    ref43 = L.load_json(os.path.join(L.G6REF, "hits", "ref_tifs_aa_bert_clean_t43.sst2.json"))
    (k43, _), = [(k, v) for k, v in ref43["students"].items() if v["tag"] == "clean-base"]
    pem = Fem["pools"][("bert", "sst2", "logit")]
    pfz = st.pools[("bert", "sst2", "logit")]
    dfw = Fem["welch"][("bert", "sst2", "logit")]["df_welch"]
    rows, summ, notes = [], [], []
    for atk in L.BASE_ATTACKS_SST2:
        e = EPOCH_OF[atk]
        sha = cross["inputs"][atk]["sha256"]
        for kd in ("logit", "feature"):
            # ── CPU fp32 ──
            P = cross["hits"][atk].get(f"{kd}/{L.P_TAG['sst2'][atk]}/poisoned/s42")
            cands = []
            run42 = L.B1_RUN.format(fam="bert", e=e, s=42)
            v, why = load_hits(os.path.join(hits_dir, f"b1_{run42}.{kd}.sst2.json"),
                               f"{run42}|{kd}/clean-base/clean/s42", atk, sha)
            cands.append(("t42", 42, f"{kd}/clean-base C_e{e} (A15-b1)", v, why, True))
            for t in (43, 44, 45, 46):
                if (t, e) == (43, 1):
                    v = ref43["hits"][atk][k43] if ref43["inputs"][atk]["sha256"] == sha else None
                    why = "ok (frozen C'(t43), epoch 1, A14 hits)" if v is not None else "inputs sha256 differ"
                    func = True
                else:
                    run = L.B2_REF_RUN.format(t=t, e=e)
                    v, why = load_hits(os.path.join(hits_dir, f"ref_{run}.sst2.json"),
                                       f"{run}|logit/clean-base/clean/s42", atk, sha)
                    rc = refs.get((t, e))
                    func = rc["functional"] if rc is not None else None
                cands.append((f"t{t}", t, f"logit/clean-base C'_e{e}", v, why, func))
            per = _per_reference(atk, kd, e, P, cands, pem, pfz, "CPU fp32", rows, notes, dfw)
            summ.append(_summary(st, Fem, atk, kd, e, per, "CPU fp32"))
        # ── GPU bf16 (Stage 2 vectors), logit only: BadNets (preds_null) and ATBA (phrase control) ──
        if atk in ("badnets", "atba"):
            g = Fem["groups"].get(("bert", "sst2", "logit", atk), {})
            base = g.get(42)
            if base is None:
                continue
            P = base["P"]
            cands = [("t42", 42, f"logit/clean-base C_e{e} (A15-b1, Stage 2)",
                      base["C"] if base.get("em_source") else None,
                      "ok" if base.get("em_source") else "b1 C_e missing (the frozen C is not used as t42 at e(P))",
                      True)]
            for t in (43, 44, 45, 46):
                rc = refs.get((t, e))
                if rc is None:
                    cands.append((f"t{t}", t, f"C'_e{e}", None, "missing run", None))
                    continue
                if atk == "badnets":
                    v = rc["C"] if len(rc["C"]) == len(P) else None
                else:
                    ctl = rc["controls"].get("atba_phrase_sst2")
                    v = ctl["c"] if ctl else None
                cands.append((f"t{t}", t, f"logit/clean-base C'_e{e} (Stage 2)", v, "ok" if v else "missing vector",
                              rc["functional"]))
            per = _per_reference(atk, "logit", e, P, cands, pem, pfz, "GPU bf16", rows, notes, dfw)
            summ.append(_summary(st, Fem, atk, "logit", e, per, "GPU bf16"))
    return rows, summ, notes


def _per_reference(atk, kd, e, P, cands, pem, pfz, precision, rows, notes, df_welch=None):
    per = {}
    for lab, t, model, C, why, func in cands:
        if P is None or C is None:
            notes.append(f"{precision} {atk}/{kd}/{lab}: {why}")
            continue
        pc, tr = None, None
        pc = L.paired_contrast(P, C, TOLS)
        se_t = L.se_from_interval(pc["lo"], pc["hi"])
        tr = L.t_contrast(pc["k"], (se_t ** 2 + pem["s_between"] ** 2) ** 0.5, pem["df"], TOLS)
        tf = L.t_contrast(pc["k"], (se_t ** 2 + pfz["s_between"] ** 2) ** 0.5, pfz["df"], TOLS)
        row = {"precision": precision, "dataset": "sst2", "kd": kd, "trigger": atk, "epoch": e, "reference": lab,
               "reference_teacher_seed": t, "reference_model": model, "reference_functional": func,
               "n": pc["n"], "P_rate": pc["a"], "C_ref_rate": pc["b"], "tango_k": pc["k"], "tango_lo": pc["lo"],
               "tango_hi": pc["hi"], "transported_lo": tr["lo"], "transported_hi": tr["hi"],
               "transported_df": pem["df"],
               **{f"transported_verdict@{t:g}": tr[f"verdict@{t:g}"] for t in TOLS},
               "frozen_pool_lo": tf["lo"], "frozen_pool_hi": tf["hi"],
               **{f"frozen_pool_verdict@{t:g}": tf[f"verdict@{t:g}"] for t in TOLS},
               **{f"claim@{t:g}": "one-seed, descriptive" for t in TOLS}}
        if df_welch is not None:
            tw = L.t_contrast(pc["k"], (se_t ** 2 + pem["s_between"] ** 2) ** 0.5, df_welch, TOLS)
            row.update({"welch_df": df_welch, "welch_lo": tw["lo"], "welch_hi": tw["hi"],
                        **{f"welch_verdict@{t:g}": tw[f"verdict@{t:g}"] for t in TOLS}})
        rows.append(row)
        per[lab] = row
    return per


def _summary(st, Fem, atk, kd, e, per, precision):
    use = [r for lab, r in per.items() if r["reference_functional"] is not False]
    pr = Fem["prim"].get(("bert", "sst2", kd, atk), {})
    fr = st.prim.get(("bert", "sst2", kd, atk), {})
    s = {"precision": precision, "dataset": "sst2", "kd": kd, "trigger": atk, "epoch": e,
         "references": ",".join(r["reference"] for r in use), "n_references": len(use),
         "excluded_floor": ",".join(lab for lab, r in per.items() if r["reference_functional"] is False),
         **{f"k_{r['reference']}": r["tango_k"] for r in use}, **{f"C_{r['reference']}": r["C_ref_rate"] for r in use},
         "em_primary_k": pr.get("k"), "em_primary_lo": pr.get("lo"), "em_primary_hi": pr.get("hi"),
         "frozen_primary_k": fr.get("k")}
    if use:
        s["k_min"] = min(r["tango_k"] for r in use)
        s["k_max"] = max(r["tango_k"] for r in use)
    for t in TOLS:
        vs = [r[f"transported_verdict@{t:g}"] for r in use]
        s[f"verdicts@{t:g}"] = "/".join(v[:3] for v in vs)
        s[f"reference_dependent@{t:g}"] = len(set(vs)) > 1
        s[f"equivalent_changes@{t:g}"] = ("EQUIVALENT" in vs) and len(set(vs)) > 1
        s[f"em_primary_verdict@{t:g}"] = pr.get(f"verdict@{t:g}")
        s[f"em_primary_claim@{t:g}"] = pr.get(f"claim@{t:g}")
    s["reference_dependent"] = ",".join(f"{t:g}" for t in TOLS if s[f"reference_dependent@{t:g}"])
    rates = [r["C_ref_rate"] for r in use]
    if len(rates) >= 2 and pr.get("se"):
        s2r, dfr = L.var(rates), len(rates) - 1
        df = L.satterthwaite2(pr["se"] ** 2, pr["df"], s2r, dfr)
        rob = L.t_contrast(pr["k"], (pr["se"] ** 2 + s2r) ** 0.5, df, TOLS)
        s.update({"a33_s_ref": s2r ** 0.5, "a33_df_ref": dfr, "a33_lo": rob["lo"], "a33_hi": rob["hi"],
                  "a33_df": df, **{f"a33_verdict@{t:g}": rob[f"verdict@{t:g}"] for t in TOLS},
                  "a33_reference_dependent": ",".join(f"{t:g}" for t in TOLS
                                                      if rob[f"verdict@{t:g}"] != pr[f"verdict@{t:g}"])})
        if pr.get("own") is not None:
            o2 = pr["own"]["se"] ** 2
            dfo = L.satterthwaite2(o2, pr["own"]["df"], s2r, dfr)
            ro = L.t_contrast(pr["k"], (o2 + s2r) ** 0.5, dfo, TOLS)
            s.update({"a33_own_lo": ro["lo"], "a33_own_hi": ro["hi"], "a33_own_df": dfo,
                      **{f"a33_own_verdict@{t:g}": ro[f"verdict@{t:g}"] for t in TOLS},
                      **{f"a33_claim@{t:g}": (rob[f"verdict@{t:g}"] if rob[f"verdict@{t:g}"] == ro[f"verdict@{t:g}"]
                                              else "rule-dependent") for t in TOLS}})
        else:
            s.update({f"a33_claim@{t:g}": "one-seed, descriptive" for t in TOLS})
        if pr.get("welch"):
            dfw = L.satterthwaite2(pr["se"] ** 2, pr["welch"]["df_welch"], s2r, dfr)
            rw = L.t_contrast(pr["k"], (pr["se"] ** 2 + s2r) ** 0.5, dfw, TOLS)
            s.update({"a33_welch_df": dfw, **{f"a33_welch_verdict@{t:g}": rw[f"verdict@{t:g}"] for t in TOLS}})
    return s


def a2_rows(st, Fem, x15):
    """The A2 replicate with C'(t43, e5)."""
    rep = AD.load_replicate(st.runs)
    cells = {}
    for c in x15:
        if c["part"] != "b2" or c["attack"] != "atba" or c["kd"] != "logit" or c["dataset"] != "sst2":
            continue
        if c["run"] == L.B2_REF_RUN.format(t=43, e=5) or c["run"].startswith("tifs_x15_b2_a2_t43_e5_s"):
            cells[c["student_seed"]] = c
    kept, excluded, floor_rows = L.apply_floor(list(cells.values()), "b2 A2")
    cells = {c["student_seed"]: c for c in kept}
    out, checks = [], []
    if not cells:
        return [{"note": "no A2 epoch-matched cells"}], floor_rows, [{"check": "A2 cells present", "ok": False,
                                                                       "note": "none"}]
    for s, c in sorted(cells.items()):
        f = rep.get(s)
        checks.append({"check": f"A2 seed {s}: re-evaluated P' equals the frozen replicate P'",
                       "ok": f is not None and c["P"] == f["P"],
                       "note": "" if f is None else f"n_diff {sum(x != y for x, y in zip(c['P'], f['P']))}"})
    d = {s: L.diff(rep[s]["P"] if s in rep else c["P"], c["C"]) for s, c in cells.items()}
    prim_em = Fem["prim"][("bert", "sst2", "logit", "atba")]
    for label, pool in (("epoch-matched BERT SST-2 logit pool", Fem["pools"][("bert", "sst2", "logit")]),
                        ("frozen BERT SST-2 logit pool", st.pools[("bert", "sst2", "logit")])):
        s2p, dfp = pool["s_between"] ** 2, pool["df"]
        if len(d) >= 2:
            k, se, r, v, ks = AD.seed_pooled(d, s2p, dfp)
            method = "seed-pooled"
        else:
            (s, x), = d.items()
            pc = L.paired_contrast(rep[s]["P"], cells[s]["C"], TOLS)
            k, ks, r, v = pc["k"], [pc["k"]], 1, None
            se = (L.se_from_interval(pc["lo"], pc["hi"]) ** 2 + s2p) ** 0.5
            method = "transported"
        res = L.t_contrast(k, se, dfp, TOLS)
        wkey = ("bert", "sst2", "logit")
        dfw = (Fem if label.startswith("epoch") else {"welch": {wkey: L.pool_welch(
            {"pools": st.pools, "att_var": st.att_var}, wkey)}})["welch"][wkey]["df_welch"]
        rw = L.t_contrast(k, se, dfw, TOLS)
        row = {"pool": label, "pool_s": pool["s_between"], "pool_df": dfp, "method": method,
               "welch_df": dfw, "welch_lo": rw["lo"], "welch_hi": rw["hi"],
               **{f"welch_verdict@{t:g}": rw[f"verdict@{t:g}"] for t in TOLS},
               "seeds": ",".join(map(str, sorted(d))), "P_prime": L.mean([L.mean(rep[s]["P"]) for s in d if s in rep]),
               "C_prime_e5": L.mean([L.mean(cells[s]["C"]) for s in d]),
               "C_prime_e1_frozen": L.mean([L.mean(rep[s]["C"]) for s in d if s in rep]),
               "k_em": k, "se": se, "lo": res["lo"], "hi": res["hi"],
               **{f"k_s{s}": x for s, x in zip(sorted(d), ks)},
               **{f"reg_verdict@{t:g}": res[f"verdict@{t:g}"] for t in TOLS},
               **{f"reg_p_pos@{t:g}": res[f"p_pos@{t:g}"] for t in TOLS},
               **{f"reg_p_up@{t:g}": res[f"p_up@{t:g}"] for t in TOLS}}
        own = None
        if r >= 2:
            own = L.t_contrast(k, (L.var(ks) / r + v) ** 0.5, r - 1, TOLS)
            row.update({"own_lo": own["lo"], "own_hi": own["hi"], "own_df": r - 1,
                        **{f"own_verdict@{t:g}": own[f"verdict@{t:g}"] for t in TOLS}})
        for t in TOLS:
            def reading(x):
                return ("replicated" if x[f"verdict@{t:g}"] == "POSITIVE" else
                        "not replicated" if x[f"p_up@{t:g}"] < ALPHA else "inconclusive")
            row[f"reg_reading@{t:g}"] = reading(res)
            row[f"own_reading@{t:g}"] = reading(own) if own else "n/a (one seed)"
            row[f"claim_reading@{t:g}"] = (("one-seed, descriptive" if own is None else
                                            row[f"reg_reading@{t:g}"] if row[f"reg_reading@{t:g}"] == row[f"own_reading@{t:g}"]
                                            else "rule-dependent"))
        if prim_em.get("se"):
            dd = L.t_contrast(prim_em["k"] - k, (prim_em["se"] ** 2 + se ** 2) ** 0.5, dfp, (0.05,))
            row.update({"k_em_t42": prim_em["k"], "build_diff_em": prim_em["k"] - k, "build_diff_lo": dd["lo"],
                        "build_diff_hi": dd["hi"]})
        fr = L.read_csv(L.OFFICIAL["replicate"])[0]
        row.update({"k_prime_frozen": float(fr["k"]), "dsel_prime": float(fr["k"]) - k,
                    **{f"frozen_reading@{t:g}": fr[f"reading@{t:g}"] for t in TOLS}})
        for t in TOLS:
            row[f"epoch_dependent_reading@{t:g}"] = fr[f"reading@{t:g}"] != row[f"reg_reading@{t:g}"]
        out.append(row)
    # determinism: C'(t43, e5) seed 42 on the ATBA inputs (atba tag) vs the BadNets tag's ATBA-phrase control
    b = next((x for x in x15 if x["run"] == L.B2_REF_RUN.format(t=43, e=5) and x["attack"] == "badnets"
              and x["kd"] == "logit"), None)
    if b is not None and 42 in cells:
        ctl = b["controls"].get("atba_phrase_sst2")
        if ctl is not None and ctl.get("c") is not None:
            n = sum(x != y for x, y in zip(cells[42]["C"], ctl["c"]))
            checks.append({"check": "A2 s42: C'(t43, e5) on the ATBA inputs (atba tag) = its ATBA-phrase control "
                                    "(badnets tag), same student", "ok": n == 0, "note": f"{n} inputs differ"})
    checks.append({"check": "A2 epoch-matched seeds 42-44 present, none floor-excluded",
                   "ok": sorted(cells) == [42, 43, 44] and not excluded,
                   "note": L.missing_note([f"s{s}" for s in (42, 43, 44) if s not in cells]
                                          + [f"floor-excluded {x}" for x in excluded])})
    return out, floor_rows, checks


def main():
    ns = L.args(extra=[("--hits-dir", {"default": None, "help": "CPU fp32 hits of score_em.py "
                                                               "(default <out>/hits_em)"})])
    st, F0 = L.state(ns.runs)
    out = ns.out
    hits_dir = ns.hits_dir or os.path.join(out, "hits_em")
    x15 = L.load_x15(ns.runs)
    em_cells, info = part_b1.build_em(st, x15, ns.runs)
    Fem = L.frame(st, em_cells, "em")
    refs = ref_cells(st, x15)
    ref_rows = [{"teacher_seed": t, "epoch": e, "run": c["run"], "teacher_ca_null": c.get("teacher_ca_null"),
                 "ca_clean": c.get("ca_c"), "floor": L.floor_of("sst2"), "functional_teacher": c["functional_teacher"],
                 "functional_student": c["functional_student"], "used": c["functional"],
                 "C_badnets_rate": L.mean(c["C"]),
                 "C_atba_phrase_rate": L.mean(c["controls"]["atba_phrase_sst2"]["c"])
                 if c["controls"].get("atba_phrase_sst2") else None,
                 "frozen_ob_selected_epoch": L.REF_EPOCH_FROZEN.get(t)}
                for (t, e), c in sorted(refs.items())]
    tw = part_b1.twin_rows(ns.runs, [({"teacher_seed": t}, f"ob_bert-large_sst-2_seed{t}", L.REF_EPOCH_FROZEN[t])
                                     for t in (43, 44, 45, 46)])
    ref_rows += [{"run": "(twin check)", **r} for r in tw]
    L.write_csv(ref_rows, os.path.join(out, "b2_references.csv"), """
        A15-b2 reference students (C' at epoch e of the rebuilt clean reference t; (43, 1) = the frozen C'(t43),
        whose OB-selected epoch is 1): Stage 2 clean accuracy of the reference teacher at that epoch
        (teacher_ca_null) and of the C' student (ca_clean) against the functional floor (majority share + 0.10;
        A7 / A8 / A13 G2), used = both above; official C' hit rates on the BadNets inputs and on the ATBA-phrase
        control; frozen_ob_selected_epoch = the epoch of the frozen build of t. Rows run = '(twin check)': the
        twin check of each rebuilt reference t43-t46 (runs/tifs/tifs_x15_twincheck/ob_bert-large_sst-2_seed<t>.json:
        identical = canonical tensor SHA-256 of the rebuilt OB-selected epoch equals the frozen build;
        rebuilt_ob_selected_epoch; frozen_C_epoch = the frozen build's OB-selected epoch). The epoch-1 t43
        reference is the frozen C'(t43) while its epoch-4/5 references come from the rebuilt t43, which is valid
        when the rebuilt t43 is identical to the frozen one and selects epoch 1.""")
    jn = L.part_jobs_note(ns.runs, "b2")
    L.write_csv(L.with_jobs(a33_rows(st, Fem, refs), jn), os.path.join(out, "b2_reference_atba.csv"), """
        A15-b2 A3.3 with epoch-matched references (epoch 1, the ATBA teacher's): frozen
        addendum.reference_atba_table on a proxy state = the A15-b1 frame's groups and primary rows (C_t42 =
        the epoch-matched battery C on the ATBA-triggered inputs) and the epoch-1 reference cells (C_t43..C_t46 =
        their C' on the atba_phrase_sst2 control); columns of results/tifs/addendum/reference_atba.csv (k, lo,
        hi, verdicts = the epoch-matched primary row; ref_* = k +- t sqrt(SE^2 + s2_ref), Satterthwaite df).
        ref_own_* = the same with the A3.1 own-variance SE (df r-1 in the Satterthwaite combination);
        ref_claim@d = ref verdict if ref_own agrees, else rule-dependent; ref_welch_* = the ref interval at the
        Satterthwaite df of SE^2 (on the pool's Welch df) and s2_ref (sensitivity only, A15 E3);
        sentence@d = the A13 G2 sentences;
        frozen_ref_verdict@d = the frozen A3.3 (3 references) / A13 G2 (5 references) verdict,
        epoch_dependent_ref@d = differs.""")
    rows, summ, notes = item7_rows(st, Fem, refs, hits_dir)
    L.write_csv(L.with_jobs(rows, jn), os.path.join(out, "b2_reference_k.csv"), """
        A15-b2 A14 item 7 with references at e(P): per BERT SST-2 base-trainer trigger, KD and reference, k = P -
        C_ref on the trigger's inputs (P = the battery P student, seed 42; C_ref = t42: the A15-b1 C_e student
        of the same KD; t43..t46: the logit clean-base C' at e(P); CPU fp32 hits from score_em.py and the A14
        file, or the official GPU bf16 Stage 2 vectors for BadNets / ATBA). tango_* = paired Tango interval;
        transported_* = SE^2 = SE_Tango^2 + s2_pool with the epoch-matched BERT SST-2 logit pool and its df;
        frozen_pool_* = with the frozen pool; welch_* = the transported interval at the Welch df of the
        epoch-matched pool (sensitivity); claim@d = 'one-seed, descriptive' (one student seed per reference, so the
        claim rule makes no claim from these rows). reference_functional = the reference passed the floor.""")
    L.write_csv(L.with_jobs(summ, jn), os.path.join(out, "b2_reference_k_summary.csv"), """
        A15-b2 item 7 summary per (precision, trigger, KD): k per reference, reference_dependent@d = the
        transported verdict is not the same for every (functional) reference; equivalent_changes@d =
        EQUIVALENT under some reference only; em_primary_* = the epoch-matched primary row (A15-b1);
        a33_* = A3.3-style interval: the references' hit-rate variance (df J-1) added to the epoch-matched
        primary SE (Satterthwaite df), a33_reference_dependent = deltas where its verdict differs from the
        epoch-matched primary verdict; a33_own_* = the same with the A3.1 own-variance SE, a33_claim@d = a33
        verdict if a33_own agrees, else rule-dependent ('one-seed, descriptive' for 'feature'); a33_welch_* = with
        the pool's Welch df (sensitivity).""")
    # A14 item 7 reproduction: the epoch-1 t43 reference is the frozen C'(t43); its rows must equal A14's.
    g6 = {(r["kd"], r["trigger"], r["reference"]): r
          for r in L.read_csv(os.path.join(L.G6REF, "reference_k_all_triggers.csv")) if r["dataset"] == "sst2"}
    nb, nn = 0, 0
    for r in rows:
        if r["precision"] != "CPU fp32" or r["reference"] != "t43" or r["epoch"] != 1:
            continue
        o = g6[(r["kd"], r["trigger"], "t43 C'")]
        nn += 1
        for mine, theirs in (("tango_k", "tango_k"), ("tango_lo", "tango_lo"), ("tango_hi", "tango_hi"),
                             ("frozen_pool_lo", "transported_lo"), ("frozen_pool_hi", "transported_hi"),
                             *[(f"frozen_pool_verdict@{t:g}", f"transported_verdict@{t:g}") for t in TOLS]):
            nb += str(r[mine]) != o[theirs]
    item7_check = {"check": "item 7 rows of the frozen epoch-1 reference C'(t43) reproduce A14 item 7 "
                            "(g6-review0 reference_k_all_triggers.csv; tango k/lo/hi, frozen-pool interval and "
                            "verdicts; exact text)", "ok": nb == 0 and nn == 4, "note": f"{nn} rows, {nb} mismatches"}
    a2, a2_floor, a2_checks = a2_rows(st, Fem, x15)
    L.write_csv(L.with_jobs(a2, jn), os.path.join(out, "b2_a2.csv"), """
        A15-b2 A2 with the epoch-matched C': k'_em = P'(t43, e5) - C'(t43, e5), seeds 42-44. Registered A2 rule
        (addendum.replicate_table form): seed-pooled k with SE^2 = s2_pool/r + v at the pool df, pool = the
        epoch-matched BERT SST-2 logit pool (first row) or the frozen pool (second row); reg_reading@d =
        replicated (POSITIVE) / not replicated (p_up < 0.025) / inconclusive; own_* = own variance (df r-1);
        claim_reading = the reading if both rules agree, else rule-dependent; welch_* = the registered k and SE at
        the pool's Welch df (sensitivity). build_diff_em = k_em(t42 ATBA,
        A15-b1) - k'_em with sqrt(SE^2 + SE'^2) at the pool df. k_prime_frozen / frozen_reading = replicate.csv
        (C'(t43) at epoch 1); dsel_prime = k'_frozen - k'_em = C'(e5) - C'(e1) on the ATBA inputs.""")
    t43 = next((r for r in tw if r["teacher_seed"] == 43), {})
    t43_ok = bool(t43.get("present")) and t43.get("identical") is True and t43.get("rebuilt_ob_selected_epoch") == 1
    checks = a2_checks + [item7_check,
        {"check": "rebuilt t43 identical to frozen t43 and selected epoch 1 (frozen C'(t43) valid as epoch-1 "
                  "reference; info: A15 says report and proceed)", "ok": t43_ok, "kind": "info",
         "note": f"present {t43.get('present')}, identical {t43.get('identical')}, rebuilt_ob_selected_epoch "
                 f"{t43.get('rebuilt_ob_selected_epoch')}" + (f", max_abs_diff {t43.get('max_abs_diff')}"
                                                              if t43.get("identical") is False else "")},
        {"check": "every b2 twin check (t43-t46) present and identical (info: A15 says report and proceed)",
         "ok": all(r["present"] and r.get("identical") is True for r in tw), "kind": "info",
         "note": ";".join(f"t{r['teacher_seed']}: present {r['present']} identical {r.get('identical')} epoch "
                          f"{r.get('rebuilt_ob_selected_epoch')}" for r in tw)},
        {"check": "item 7 em: every reference vector present (CPU fp32 and bf16)", "ok": not notes,
         "note": "; ".join(notes[:10])},
        {"check": "b2 reference runs present (11 runs + frozen t43 e1)",
         "ok": len(refs) == 12, "note": L.missing_note([f"t{t} e{e}" for t in (43, 44, 45, 46) for e in (1, 4, 5)
                                                        if (t, e) not in refs])}]
    checks += L.pool_identity_rows({"frozen": F0, "em": Fem})
    checks += L.excluded_rows(ns.runs)
    L.write_csv(checks, os.path.join(out, "check_b2.csv"), "A15-b2 self-checks ('present' rows are False while "
                "b2 jobs or score_em.py outputs are missing).")
    L.write_csv(a2_floor + [dict(r, source="b2 reference") for r in L.student_rows(
        [c for c in x15 if c["part"] == "b2" and c["block"] == "AA"], "b2 reference")],
        os.path.join(out, "b2_floor.csv"), "A8 student floor of the A15-b2 students (references: C' used only "
                                            "if teacher and student pass; A2: run excluded otherwise).")
    print(f"part b2: references {len(refs)}; item 7 rows {len(rows)}; notes {len(notes)}")


if __name__ == "__main__":
    main()
