#!/usr/bin/env python3
"""A15-b1: teacher-epoch-matched batteries (the 13 SST-2 base-trainer cells), the selection effect and the
registered b3 condition.

A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Registered (docs/tifs_addendum.md A15-b1, 10/26 19:15 KST; judgement hierarchy and claim rule of the 10/27
20:58 KST amendment):
  * k_em = P - C_e(P): P = the frozen P student's hit vector (frozen preds preds_obs; the b1 run's
    re-evaluated preds_obs must be identical), C_e(P) = the b1 run's clean student distilled from the
    rebuilt clean twin's epoch e(P) (preds_null). BERT e = 1 (ATBA, AddSent), 4 (BadNets), 5 (SynBkd); logit
    seeds 42-44, 'feature' seed 42; GPT-2 e = 5 (AddSent; BadNets e 2 = the frozen C epoch keeps the frozen
    C); Qwen3 e = 3. The epoch-matched pair is the cell's teacher pair (teacher_asr_null / teacher_ca_null of
    the b1 result, i.e. the frozen gate on (P, C_e)).
  * Frame 'em' = the frozen cells with those cells replaced: the frozen analyze.primary_table recomputes
    every pool from the epoch-matched C (so cells that only borrow an SST-2 pool, e.g. Qwen3 AG News and
    BERT SST-2 SOS, change through the pool), analyze.primary_battery the battery decisions,
    addendum.global_holm_table the 31-cell Holm with the 13 cells replaced.
  * Verdicts at every delta under the registered rule, the A3.1 own-variance rule and the claim rule;
    Welch-df sensitivity; 'epoch-dependent' where the em verdict differs from the frozen one (per rule).
    H2: the em verdicts set the interpretation (Finding, abstract); the frozen verdicts stay in the table.
  * Selection effect Delta_sel = C_e(P) - C_sel (the OB-selected frozen C) on the cell's triggered inputs
    = k_frozen - k_em (descriptive; the registered decision on it is b3's).
  * b3 condition (fixed before data): 95% upper bound of |C_e4 - C_e3| on the BERT BadNets triggered inputs,
    seed-pooled, > 0.01 -> b3 would be triggered. Printed; nothing is enabled here.
  * Reproduction checks: the twin checks' identical flags (runs/tifs/tifs_x15_twincheck/*.json), the b1
    runs' re-evaluated P (and P teacher) vectors equal to the frozen ones, the C teacher epoch recorded in
    the preds provenance = e(P), the A8 student floor.

Outputs: b1_cells.csv, b1_battery.csv, b1_holm.csv, b1_pools.csv, b1_selection.csv, b3_condition.csv,
b1_repro.csv, b1_claims.csv, b1_floor.csv, check_b1.csv.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

import a15lib as L  # noqa: E402

A, AD, TOLS, ALPHA = L.A, L.AD, L.TOLS, L.ALPHA
TWINS = {"bert": "ob_bert-large_sst-2_seed42", "gpt2": "ob_gpt2-xl_sst-2_seed42",
         "qwen3": "ob_Qwen3-1.7B_sst-2_seed42"}
TWIN_FIELDS = ("identical", "files_identical", "selected_epoch_matches_frozen", "rebuilt_ob_selected_epoch",
               "frozen_log_selected_epoch", "expect_epoch", "dev_identical", "record_consistent")


def expected_em():
    """{(fam, ds, kd, attack, seed): epoch} of every registered b1 cell (GPT-2 BadNets excluded: e(P) =
    frozen C epoch)."""
    out = {}
    for (fam, atk), e in L.P_EPOCH.items():
        if e == L.C_EPOCH_FROZEN[fam]:
            continue
        for kd in L.B1_KDS[fam]:
            for s in (L.SEEDS if kd == "logit" else (42,)):
                out[(fam, "sst2", kd, atk, s)] = e
    return out


def provenance_epoch(c):
    """Epoch of the clean teacher the C student was distilled from, from the preds meta.provenance
    (clean_teacher path .../epochs/epoch<e>); None if not recorded."""
    p = (c.get("meta") or {}).get("provenance") or {}
    ct = p.get("clean_teacher")
    if not ct:
        return None
    base = os.path.basename(os.path.normpath(ct))
    return int(base[5:]) if base.startswith("epoch") and base[5:].isdigit() else None


def build_em(st, x15, runs):
    """(em cells, info): the frozen cells with every available b1 cell's C replaced by C_e(P)."""
    frozen = {(c["fam"], c["dataset"], c["kd"], c["attack"], c["seed"]): c for c in st.cells
              if c["block"] == "G" and c["corpus"] == "clean" and not c["single"]}
    exp = expected_em()
    b1 = [c for c in x15 if c["part"] == "b1"]
    kept, excluded, floor_rows = L.apply_floor(b1, "b1")
    em, repro, unexpected = {}, [], []
    for c in kept:
        key = (c["fam"], c["dataset"], c["kd"], c["attack"], c["seed"])
        if key not in exp:
            unexpected.append(f"{c['run']}:{c['tag']}")
            continue
        f = frozen[key]
        e = exp[key]
        pe = provenance_epoch(c)
        row = {"run": c["run"], "tag": c["tag"], "fam": key[0], "dataset": key[1], "kd": key[2],
               "attack": key[3], "seed": key[4], "e_P": e, "run_epoch": c.get("epoch"),
               "provenance_clean_teacher_epoch": pe,
               "provenance_epoch_ok": (pe == e) if pe is not None else None,
               "run_epoch_ok": c.get("epoch") == e,
               "P_equal_frozen": c["P"] == f["P"],
               "P_n_diff": sum(x != y for x, y in zip(c["P"], f["P"])),
               "Pc_equal_frozen": c["Pc"] == f["Pc"],
               "TP_equal_frozen": c["TP"] == f["TP"],
               "teacher_asr_equal_frozen": c["teacher_asr"] == f["teacher_asr"],
               "teacher_ca_equal_frozen": c["teacher_ca"] == f["teacher_ca"],
               "ca_P_equal_frozen": c["ca_p"] == f["ca_p"],
               "n_eval_equal": len(c["C"]) == len(f["C"]),
               "C_n_diff_vs_frozen_C": sum(x != y for x, y in zip(c["C"], f["C"])),
               "TC_n_diff_vs_frozen_TC": (sum(x != y for x, y in zip(c["TC"], f["TC"]))
                                          if c["TC"] and f["TC"] else None),
               "teacher_ca_null_em": c["teacher_ca_null"], "teacher_ca_null_frozen": f["teacher_ca_null"],
               "teacher_asr_null_em": c["teacher_asr_null"], "teacher_asr_null_frozen": f["teacher_asr_null"],
               "teacher_qualified_em": A.qualified(c), "teacher_qualified_frozen": A.qualified(f)}
        repro.append(row)
        new = dict(c)
        # P side: the frozen P (registered); C side: C_e(P) and its teacher from the b1 run.
        for k in ("P", "Pc", "TP", "TPc", "teacher_asr", "teacher_ca", "ca_p"):
            new[k] = f[k]
        new["em_source"] = f"C_e{e} (b1 {c['run']})"
        new["frozen_C"] = f["C"]
        em[key] = new
    # Seed-level gaps (some seeds of a cell epoch-matched, others missing or floor-excluded): the cell is judged on
    # its finished seeds only (A13 practice, r reported), never on a mix of C_e and the frozen C. A cell with no
    # epoch-matched seed at all keeps the frozen C and is listed as missing (em_source says so).
    with_em = {k[:4] for k in em}
    drop = {k for k in exp if k not in em and k[:4] in with_em}
    cells = []
    for c in st.cells:
        key = (c["fam"], c["dataset"], c["kd"], c["attack"], c["seed"])
        if c["block"] == "G" and c["corpus"] == "clean" and not c["single"]:
            if key in drop:
                continue
            c = em.get(key, c)
        cells.append(c)
    missing = sorted(f"{k[0]}/{k[2]}/{k[3]}/s{k[4]}(e{v})" for k, v in exp.items() if k not in em)
    info = {"em": em, "repro": repro, "floor_rows": floor_rows, "excluded": excluded, "missing": missing,
            "unexpected": unexpected, "expected": exp, "frozen": frozen,
            "dropped_seeds": sorted(f"{k[0]}/{k[2]}/{k[3]}/s{k[4]}" for k in drop)}
    return cells, info


def twin_rows(runs, twins=None):
    """One row per twin check (runs/tifs/tifs_x15_twincheck/<teacher>.json): ``twins`` = [(label fields,
    teacher, frozen OB-selected epoch)]; default the three seed-42 clean twins of A15-b1."""
    if twins is None:
        twins = [({"fam": fam}, teacher, L.C_EPOCH_FROZEN[fam]) for fam, teacher in TWINS.items()]
    out = []
    for lab, teacher, fe in twins:
        p = os.path.join(runs, "tifs_x15_twincheck", f"{teacher}.json")
        row = {**lab, "twin": teacher, "file": L.rel(p), "present": os.path.exists(p), "frozen_C_epoch": fe}
        if row["present"]:
            X = L.load_json(p)
            row.update({k: X.get(k) for k in TWIN_FIELDS})
            if X.get("identical") is False:
                row["max_abs_diff"] = (X.get("diff_vs_selected") or {}).get("max_abs_diff")
        out.append(row)
    return out


def selection_rows(st, Fem, info):
    """Delta_sel = C_e(P) - C_sel per registered cell (seeds pooled where r >= 2)."""
    rows = []
    by = {}
    for key, c in info["em"].items():
        by.setdefault(key[:4], {})[key[4]] = c
    for k4, seeds in sorted(by.items()):
        fam, ds, kd, atk = k4
        src = Fem["prim"][k4].get("variance_from")
        pe = Fem["pools"].get(tuple(src.split("/"))) if src else None
        pf = st.pools.get(tuple(src.split("/"))) if src else None
        d = {s: L.diff(c["C"], c["frozen_C"]) for s, c in seeds.items()}
        row = {"fam": fam, "dataset": ds, "kd": kd, "attack": atk, "e_P": info["expected"][k4 + (min(seeds),)],
               "C_sel_epoch": L.C_EPOCH_FROZEN[fam], "seeds": ",".join(map(str, sorted(seeds))),
               "C_em_rate": L.mean([L.mean(c["C"]) for c in seeds.values()]),
               "C_sel_rate": L.mean([L.mean(c["frozen_C"]) for c in seeds.values()]),
               "pool_em": src, "registered_b3_quantity": (fam, kd, atk) == ("bert", "logit", "badnets")}
        for s, c in sorted(seeds.items()):
            pc = L.paired_contrast(c["C"], c["frozen_C"], TOLS)
            row[f"dsel_s{s}"] = pc["k"]
            row[f"dsel_tango_lo_s{s}"] = pc["lo"]
            row[f"dsel_tango_hi_s{s}"] = pc["hi"]
            row[f"n_inputs_C_differs_s{s}"] = pc["n10"] + pc["n01"]
            if c.get("Cc") and info["frozen"][k4 + (s,)].get("Cc"):
                row[f"untrig_dsel_s{s}"] = L.mean(c["Cc"]) - L.mean(info["frozen"][k4 + (s,)]["Cc"])
            if c.get("TC") and info["frozen"][k4 + (s,)].get("TC"):
                row[f"teacher_trig_dsel_s{s}"] = L.mean(c["TC"]) - L.mean(info["frozen"][k4 + (s,)]["TC"])
            if c.get("TCc") and info["frozen"][k4 + (s,)].get("TCc"):
                row[f"teacher_untrig_dsel_s{s}"] = L.mean(c["TCc"]) - L.mean(info["frozen"][k4 + (s,)]["TCc"])
        if len(d) >= 2:
            for name, pool in (("em_pool", pe), ("frozen_pool", pf)):
                if pool is None:
                    continue
                k, se, r, v, ks = AD.seed_pooled(d, pool["s_between"] ** 2, pool["df"])
                res = L.t_contrast(k, se, pool["df"], TOLS)
                row.update({"dsel": k, f"dsel_{name}_se": se, f"dsel_{name}_df": pool["df"],
                            f"dsel_{name}_lo": res["lo"], f"dsel_{name}_hi": res["hi"],
                            f"abs_dsel_{name}_upper95": max(abs(res["lo"]), abs(res["hi"]))})
            k, _, r, v, ks = AD.seed_pooled(d, 0.0, 1.0)
            own = L.t_contrast(k, (L.var(ks) / r + v) ** 0.5, r - 1, TOLS)
            row.update({"dsel_own_se": own["se"], "dsel_own_df": r - 1, "dsel_own_lo": own["lo"],
                        "dsel_own_hi": own["hi"], "abs_dsel_own_upper95": max(abs(own["lo"]), abs(own["hi"]))})
        else:
            (s, c), = seeds.items()
            pc = L.paired_contrast(c["C"], c["frozen_C"], TOLS)
            row.update({"dsel": pc["k"], "dsel_tango_lo": pc["lo"], "dsel_tango_hi": pc["hi"],
                        "abs_dsel_tango_upper95": max(abs(pc["lo"]), abs(pc["hi"]))})
            if pe is not None:
                se = (L.se_from_interval(pc["lo"], pc["hi"]) ** 2 + pe["s_between"] ** 2) ** 0.5
                res = L.t_contrast(pc["k"], se, pe["df"], TOLS)
                row.update({"dsel_em_pool_lo": res["lo"], "dsel_em_pool_hi": res["hi"],
                            "dsel_em_pool_df": pe["df"],
                            "abs_dsel_em_pool_upper95": max(abs(res["lo"]), abs(res["hi"]))})
        row["k_frozen"] = st.prim[k4]["k"]
        row["k_em"] = Fem["prim"][k4]["k"]
        # the identity holds on the same seeds: k_frozen over the epoch-matched cell's seeds
        kf_same = L.mean([L.mean(info["frozen"][k4 + (s,)]["P"]) - L.mean(info["frozen"][k4 + (s,)]["C"])
                          for s in sorted(seeds)])
        row["k_frozen_same_seeds"] = kf_same
        row["identity_k_frozen_minus_k_em_equals_dsel"] = abs((kf_same - row["k_em"]) - row["dsel"]) < 1e-12
        rows.append(row)
    return rows


def b3_rows(sel):
    """The registered b3 condition: BERT BadNets logit (seed-pooled) 95% upper bound of |C_e4 - C_e3| > 0.01.

    Decision column (fixed here, before any A15 output was read): the seed-pooled estimate k = mean_s d_s,
    d_s = C_e4,s - C_e3,s, with its OWN between-seed variance (SE^2 = s2_d / r + v, t at df r - 1; the A3.1
    own-variance form of the frozen seed-pooled formula). Reason: the battery pool s2_pool is the between-seed
    variance of k = P - C; with it the bound is at least t_{0.975,df} s_pool / sqrt(r) (about 0.033 for the
    BERT SST-2 logit pool) whatever C_e4 and C_e3 are, so the registered threshold 0.01 would be exceeded by
    construction. The pooled-formula bounds (epoch-matched pool and frozen pool) are shown beside it and
    'rule-dependent' is written when they disagree. Bound = max(|lo|, |hi|) of the two-sided 95% interval of
    C_e4 - C_e3."""
    rows = []
    for r in sel:
        if (r["fam"], r["attack"]) != ("bert", "badnets"):
            continue
        row = {"fam": r["fam"], "dataset": r["dataset"], "kd": r["kd"], "attack": r["attack"], "seeds": r["seeds"],
               "contrast": "C_e4 - C_e3 (C_e3 = the frozen OB-selected clean twin's student)",
               "C_e4_rate": r["C_em_rate"], "C_e3_rate": r["C_sel_rate"], "dsel": r["dsel"], "threshold": 0.01,
               "registered": r["kd"] == "logit"}
        for name in ("em_pool", "frozen_pool", "own", "tango"):
            b = r.get(f"abs_dsel_{name}_upper95")
            if b is None:
                continue
            row[f"{name}_lo"] = r.get(f"dsel_{name}_lo")
            row[f"{name}_hi"] = r.get(f"dsel_{name}_hi")
            row[f"{name}_abs_upper95"] = b
            row[f"{name}_would_trigger"] = b > 0.01
        if r["kd"] == "logit":
            dec = row.get("own_would_trigger")
            alts = [row.get(f"{n}_would_trigger") for n in ("em_pool", "frozen_pool") if f"{n}_would_trigger" in row]
            row["pool_bound_floor"] = (row.get("em_pool_abs_upper95") - abs(row["dsel"])
                                       if row.get("em_pool_abs_upper95") is not None else None)
            row["b3_would_be_triggered"] = dec
            row["rule_agreement"] = "agree" if all(a == dec for a in alts) else "rule-dependent"
            row["action"] = ("b3 condition met: A15_B3_ENABLED and tifs_x15_b3_go are NOT set by this script "
                             "(operator decision)" if dec else "b3 condition not met: b3 stays disabled")
        else:
            row["b3_would_be_triggered"] = ""
            row["action"] = "descriptive ('feature', seed 42; not the registered quantity)"
        rows.append(row)
    return rows


def pool_rows(F0, Fem):
    out = []
    for key in sorted(set(F0["pools"]) | set(Fem["pools"])):
        row = {"pool": "/".join(key)}
        for pre, F in (("frozen_", F0), ("em_", Fem)):
            p = F["pools"].get(key)
            if p is None:
                continue
            w = F["welch"][key]
            row.update({f"{pre}attacks": p["attacks"], f"{pre}seeds": p["seeds"], f"{pre}s_pool": p["s_between"],
                        f"{pre}df": p["df"], f"{pre}df_welch": w["df_welch"], f"{pre}ms_seed": p["ms_seed"],
                        f"{pre}ms_res": p["ms_res"],
                        **{f"{pre}s_{a}": v ** 0.5 for a, v in sorted(w["s2a"].items())}})
        row["changed"] = (F0["pools"].get(key) or {}).get("s_between") != (Fem["pools"].get(key) or {}).get("s_between")
        out.append(row)
    return out


def main():
    ns = L.args()
    st, F0 = L.state(ns.runs)
    x15 = L.load_x15(ns.runs)
    cells, info = build_em(st, x15, ns.runs)
    Fem = L.frame(st, cells, "em")
    out = ns.out
    changed_k = {k[:4] for k in info["em"]}
    em_pools = {"/".join(k[:3]) for k in changed_k}

    def extra(key, a, b):
        fam, ds, kd, atk = key
        e = L.P_EPOCH.get((fam, atk)) if ds == "sst2" else None
        srcs = sorted({info["em"][k]["em_source"] for k in info["em"] if k[:4] == key})
        # missing epoch-matched seeds of this very cell (full key incl. dataset: b1 is SST-2 only, so an AG News
        # cell with the same fam/kd/attack never matches)
        miss = sorted(f"{fam}/{kd}/{atk}/s{k[4]}(e{v})" for k, v in info["expected"].items()
                      if k[:4] == key and k not in info["em"])
        if key in changed_k:
            src = "; ".join(srcs) + (" (partial: judged on the finished seeds; missing " + ",".join(miss) + ")"
                                     if miss else "")
        elif e is not None and e == L.C_EPOCH_FROZEN.get(fam) and kd in L.B1_KDS.get(fam, ()):
            src = f"frozen C (e(P) = {e} = frozen C epoch)"
        elif miss:
            src = "frozen C (b1 missing: " + ",".join(miss) + ")"
        else:
            src = "not a base-trainer SST-2 cell (unchanged)"
        pool_changed = (a is not None and b is not None and a.get("variance_from")
                        and F0["pools"].get(tuple(a["variance_from"].split("/")), {}).get("s_between")
                        != Fem["pools"].get(tuple(b["variance_from"].split("/")), {}).get("s_between"))
        # in scope by design (not by whether a number moved): C replaced, a registered b1 cell, or variance
        # from a pool that holds replaced cells (recomputed from the epoch-matched C). A registered b1 cell whose
        # C_e(P) is missing for every seed keeps the frozen C: no A15-b1 reading of its own (sentence n/a).
        in_scope = (key in changed_k or e is not None
                    or any(x is not None and x.get("variance_from") in em_pools for x in (a, b)))
        b1_missing = key not in changed_k and bool(miss)
        scope = "missing" if b1_missing else "in" if in_scope else "out"
        return {"e_P": e, "C_epoch_frozen": L.C_EPOCH_FROZEN.get(fam) if e else None, "em_source": src,
                "b1_scope": scope,
                "k_changed": key in changed_k, "pool_changed": bool(pool_changed),
                "pool_only_change": bool(pool_changed) and key not in changed_k,
                "em_C_rate": b["C_trig"] if b else None, "frozen_C_rate": a["C_trig"] if a else None,
                "em_teacher_qualified": b["teacher_qualified"] if b else None}

    keys = [L.ckey(r) for r in st.primary]
    jn = L.part_jobs_note(ns.runs, "b1")
    b1_cells = L.comparison_rows(F0, Fem, keys, "epoch_dependent", "frozen_", "em_", extra)
    for r in b1_cells:   # the A15-b1 sentence, mirroring part_a: from the registered and claim-rule labels only
        scope = r.pop("b1_scope")
        if scope == "out":
            r["sentence"] = "n/a (outside A15-b1: C and pool not epoch-matched)"
        elif scope == "missing":
            r["sentence"] = "n/a (b1 C_e(P) missing for every seed: frozen C kept)"
        elif not (r["epoch_dependent_reg"] or r["epoch_dependent_claim"]):
            r["sentence"] = "teacher epoch에 강건"
        else:
            r["sentence"] = "epoch-dependent (두 판정을 나란히 쓴다)"
    L.write_csv(L.with_jobs(b1_cells, jn), os.path.join(out, "b1_cells.csv"), """
        A15-b1, one row per primary cell (31). frozen_* = the frozen frame (= results/tifs/report/primary.csv);
        em_* = the epoch-matched frame: the frozen analyze.primary_table on the frozen cells whose C is replaced
        by C_e(P) for the 13 SST-2 base-trainer cells (P frozen; pools recomputed from the epoch-matched C).
        Per frame: method, seeds, variance_from, k, se, df, lo/hi; reg_* = registered rule (verdict, one-sided
        p_up: H0 k >= delta, p_pos: H0 k <= delta); own_* = A3.1 own variance (SE^2 = s2_a/r + v, df r-1);
        claim@d = claim rule (registered verdict if own agrees, else rule-dependent; one-seed cells 'one-seed,
        descriptive'; structural nulls 'structural (counted apart)', their Tango verdict in reg_*); pmax_* =
        max(p_reg, p_own) (structural: Tango p; one-seed: 1); welch_* = registered k and SE at the Welch df of
        the pool (sensitivity only);
        F31/F19_<side>_<rule>(_holm)@d = Holm over the 31 qualified cells / 19 logit cells of the one-sided p
        of that side (notransfer: p_up, transfer: p_pos) under reading reg (registered p) / own (A3.1 p; one-seed
        cells p_reg) / claimA (p_max, one-seed cells at p_reg: their rejections descriptive, not counted) /
        claimB (p_max, one-seed cells at p = 1) / welch, flag = adjusted p < 0.025. epoch_dependent_<rule> = the deltas where
        the em verdict differs from the frozen one under that rule. e_P = the P teacher's OB-selected epoch,
        C_epoch_frozen = the frozen C twin's epoch, em_source = where the em C comes from, k_changed = C
        replaced, pool_only_change = only the borrowed pool changed; sentence = the A15-b1 sentence rule, as in
        a_cells.csv, from the registered and claim-rule labels only: "teacher epoch에 강건" unless
        epoch_dependent_reg or epoch_dependent_claim is set, else "epoch-dependent (두 판정을 나란히 쓴다)"; n/a
        for cells outside A15-b1 (C not replaced, not a registered b1 cell, variance not from a pool that holds
        replaced cells) and for a registered b1 cell whose C_e(P) is missing for every seed (frozen C kept); the
        own variance and Welch columns (epoch_dependent_own, epoch_dependent_welch) are sensitivity columns and
        set no label or sentence; a15_jobs = status of the part's queue v9 jobs (markers; the cut / blocked /
        missing jobs).""")
    bkeys = [(b["fam"], b["dataset"], b["kd"]) for b in F0["battery"]]
    L.write_csv(L.with_jobs(L.battery_comparison(F0, Fem, bkeys, "epoch_dependent", "frozen_", "em_"), jn),
                os.path.join(out, "b1_battery.csv"), """
        A15-b1, one row per primary battery. reg_decision = the frozen analyze.primary_battery decision (NO
        TRANSFER: every qualified attack's p_up < 0.025; TRANSFERS: a within-battery Holm p_pos < 0.025; else
        INCONCLUSIVE), reg_transfers = Holm-flagged attacks, transfers_global = the 31-cell Holm flags;
        own_/claim_/welch_decision = the frozen analyze.decide on the own-variance p (one-seed cells p = 1), on
        the claim-rule p_max, on the registered p at the Welch df; 'one-seed, descriptive' when every cell of
        the battery has one seed. frozen_* vs em_* frames as in b1_cells.csv; epoch_dependent_<rule> = deltas
        where the decision differs.""")
    L.write_csv(L.with_jobs(L.holm_rows(F0, Fem, "epoch_dependent", "frozen_", "em_"), jn),
                os.path.join(out, "b1_holm.csv"), """
        A15-b1 31-cell Holm with the 13 SST-2 base-trainer cells replaced. holm_global / transfers_global =
        the frozen addendum.global_holm_table (Holm of p_pos over the qualified cells) of each frame; F31 / F19
        family columns as in b1_cells.csv; epoch_dependent_<rule> lists family:side@delta flags that differ.""")
    L.write_csv(pool_rows(F0, Fem), os.path.join(out, "b1_pools.csv"), """
        Between-seed pools of the frozen and epoch-matched frames (analyze.pool_seed_variance via
        primary_table): attacks, seeds, s_pool, Satterthwaite df, Welch df (A15 E3), MS_seed, MS_res, per-attack
        SD; changed = the pool SD differs.""")
    sel = selection_rows(st, Fem, info)
    L.write_csv(sel, os.path.join(out, "b1_selection.csv"), """
        Selection effect per registered b1 cell (descriptive): dsel = C_e(P) - C_sel on the cell's triggered
        inputs (C_sel = the frozen OB-selected clean twin's student; per seed and pooled), = k_frozen - k_em.
        dsel_em_pool / dsel_frozen_pool = the frozen seed-pooled formula (SE^2 = s2_pool/r + v, pool df) with
        the epoch-matched / frozen battery pool; dsel_own = own variance (df r-1); one-seed cells: Tango and
        the transported interval. untrig_dsel / teacher_*_dsel = the same difference of rates on the untriggered
        inputs and for the teachers (C_e teacher vs selected twin). abs_*_upper95 = max(|lo|, |hi|).""")
    L.write_csv(b3_rows(sel), os.path.join(out, "b3_condition.csv"), """
        Registered b3 condition (A15, fixed before data): b3 is triggered when the 95% upper bound of |C_e4 -
        C_e3| on the BERT BadNets triggered inputs (seed-pooled, logit seeds 42-44) exceeds 0.01. Decision
        column b3_would_be_triggered = own bound (seed-pooled k = mean_s(C_e4,s - C_e3,s), SE^2 = s2_d/r + v, t
        at df r-1; fixed before data, author to confirm); em_pool / frozen_pool bounds (frozen seed-pooled
        formula with the battery pool, floor about t*s_pool/sqrt(r) = 0.033) beside it; rule_agreement = whether
        they agree. Nothing is enabled by this script (A15_B3_ENABLED and the tifs_x15_b3_go marker are operator
        actions).""")
    rep = info["repro"]
    tw = twin_rows(ns.runs)
    L.write_csv(rep + [{"run": "(twin check)", **t} for t in tw], os.path.join(out, "b1_repro.csv"), """
        A15-b1 reproduction checks. Per b1 cell: P_equal_frozen (the b1 run's re-evaluated preds_obs equals the
        frozen file's), Pc / TP / teacher_asr / teacher_ca / ca_P equal frozen, provenance_clean_teacher_epoch
        (epoch of the clean teacher in the preds meta.provenance) = e_P, C_n_diff_vs_frozen_C (inputs where
        C_e differs from the frozen C), teacher_*_null em vs frozen, the frozen gate on both pairs. Twin rows:
        the twin check JSON fields (identical = canonical tensor SHA-256 of the rebuilt OB-selected epoch equals
        the frozen twin; selected epoch; dev metrics identical) for the three seed-42 twins.""")
    L.write_csv(L.claim_summary(F0, "frozen") + L.claim_summary(Fem, "em"), os.path.join(out, "b1_claims.csv"), """
        Claim-rule counts per frame and delta over the qualified cells: claimable (seed-pooled) cells by
        claim verdict (rule-dependent counted as INCONCLUSIVE in claim_INCONCLUSIVE_incl_rule_dependent), by
        registered and by own verdict; one-seed and structural cells apart; family claims F31 / F19 per side
        (notransfer = Holm p_up < 0.025, transfer = Holm p_pos < 0.025) under reg / own / claimA / claimB /
        welch, with the cells (claimA: one-seed rejections listed apart as descriptive, not counted);
        *_a4_counted_n = claims among the A4-counted cells (counted + boundary).""")
    L.write_csv(info["floor_rows"], os.path.join(out, "b1_floor.csv"), """
        A8 student floor for every b1 student (P re-evaluated, C_e): Stage 2 clean accuracy > majority share +
        0.10 (SST-2 0.6008); a run with any student at or below the floor is left out; its seeds are dropped
        from cells that have other epoch-matched seeds, and a cell with no epoch-matched seed keeps the frozen
        C (em_source).""")
    checks = []

    def chk(name, ok, note=""):
        checks.append({"check": name, "ok": bool(ok), "note": note})

    chk("every b1 cell's re-evaluated P equals the frozen P", all(r["P_equal_frozen"] for r in rep),
        ";".join(f"{r['run']}:{r['tag']} n_diff {r['P_n_diff']}" for r in rep if not r["P_equal_frozen"]))
    chk("every b1 cell's C teacher epoch (provenance) = e(P) (recorded: every b1 config sets record_provenance)",
        all(r["provenance_epoch_ok"] is True for r in rep),
        ";".join(f"{r['run']}:{r['tag']} " + ("not recorded" if r["provenance_epoch_ok"] is None else
                                               f"epoch {r['provenance_clean_teacher_epoch']}")
                 for r in rep if r["provenance_epoch_ok"] is not True))
    chk("every b1 run's epoch (run id) = e(P)", all(r["run_epoch_ok"] for r in rep))
    chk("every twin check present and identical", all(t["present"] and t.get("identical") for t in tw),
        ";".join(f"{t['twin']}: present {t['present']} identical {t.get('identical')}" for t in tw))
    chk("no unexpected b1 tags", not info["unexpected"], ";".join(info["unexpected"]))
    chk("frame identity: k_frozen - k_em = Delta_sel for every registered cell",
        all(r["identity_k_frozen_minus_k_em_equals_dsel"] for r in sel))
    chk("every frozen primary row outside the SST-2 pools' users is unchanged in the em frame",
        all(Fem["prim"][k][f] == F0["prim"][k][f] for k in F0["prim"] for f in ("k", "lo", "hi")
            if k[1] != "sst2" and k[:2] != ("qwen3", "agnews")))
    chk("b1 inputs complete", not info["missing"] and not info["excluded"],
        L.missing_note(info["missing"] + [f"floor-excluded {r}" for r in info["excluded"]]))
    fs = L.family_size_row(F0, Fem, "em", kind="info")
    fs["check"] += " (info: the em gate is re-evaluated on the C_e teacher; a smaller family is reported, not fixed)"
    checks.append(fs)
    checks += L.pool_identity_rows({"frozen": F0, "em": Fem})
    checks += L.excluded_rows(ns.runs)
    L.write_csv(checks, os.path.join(out, "check_b1.csv"), "A15-b1 self-checks (ok = True expected; 'b1 inputs "
                "complete' is False while b1 jobs are missing or cut).")
    L.dump_json({"missing": info["missing"], "excluded": info["excluded"], "dropped_seeds": info["dropped_seeds"],
                 "b3": b3_rows(sel)}, os.path.join(out, "logs", "b1_summary.json"))
    for r in b3_rows(sel):
        if r["kd"] == "logit":
            print(f"  b3 condition: |C_e4 - C_e3| upper95 (own) = {r.get('own_abs_upper95')} -> "
                  f"{'TRIGGERED' if r['b3_would_be_triggered'] else 'not triggered'} ({r['rule_agreement']}); "
                  "nothing enabled")
    bad = [c for c in checks if not c["ok"] and c.get("kind") != "info" and c["check"] != "b1 inputs complete"]
    print(f"part b1: {len(info['em'])} epoch-matched cells, missing {len(info['missing'])}; checks "
          f"{'ok' if not bad else 'FAILED: ' + '; '.join(c['check'] for c in bad)}")


if __name__ == "__main__":
    main()
