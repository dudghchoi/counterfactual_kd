#!/usr/bin/env python3
"""Write <out>/README.md: status, data snapshot, key findings, definitions and key-number
tables. Every number is read from this group's CSVs (or the official inputs), never typed
by hand; each lookup asserts that exactly one row matches. Post hoc / registered auxiliary
(docs/tifs_addendum.md A11); changes no verdict.

    PYTHONDONTWRITEBYTECODE=1 python make_readme.py [--out DIR]
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as K  # noqa: E402

OUT = K.OUT_DEFAULT
TOLS = ("0.02", "0.05", "0.1")
C_SD = {"0.5": "0.5", "0.7071067811865476": "0.707", "1.0": "1", "1.224744871391589": "1.225",
        "1.4142135623730951": "1.414", "1.5": "1.5", "2.0": "2"}
C1, C15, C2 = "1.0", "1.5", "2.0"
CV15, CV2 = repr(math.sqrt(1.5)), repr(math.sqrt(2.0))
POOLED = ("BERT AG News logit", "BERT SST-2 logit", "Qwen3 SST-2 logit", "GPT-2 SST-2 logit")
TARGETS = ("BERT AG News feature", "BERT SST-2 feature", "Qwen3 AG News logit")
LBL = {"BERT AG News logit": "BERT AG News", "BERT SST-2 logit": "BERT SST-2", "Qwen3 SST-2 logit": "Qwen3 SST-2",
       "GPT-2 SST-2 logit": "GPT-2 SST-2", "BERT AG News feature": "BERT AG News feature",
       "BERT SST-2 feature": "BERT SST-2 feature", "Qwen3 AG News logit": "Qwen3 AG News"}
KEY_OF = {"BERT SST-2": ("bert", "sst2", "logit"), "BERT AG News": ("bert", "agnews", "logit"),
          "Qwen3 SST-2": ("qwen3", "sst2", "logit"), "GPT-2 SST-2": ("gpt2", "sst2", "logit")}
ATK = {"addsent": "AddSent", "atba": "ATBA", "badnets": "BadNets", "ep": "EP", "sos": "SOS", "synbkd": "SynBkd"}
RULES = ("R1_raw_asr", "R1h_raw_asr_high", "R2_abs_point", "R3_signed_point", "R4_ci_excludes_0", "R5_audit",
         "R5b_audit_binary")
EMP_RULES = RULES + ("R5t_audit_transported",)
SHORT = {"R1_raw_asr": "R1", "R1h_raw_asr_high": "R1h", "R2_abs_point": "R2", "R3_signed_point": "R3",
         "R4_ci_excludes_0": "R4", "R5_audit": "R5", "R5b_audit_binary": "R5b", "R5t_audit_transported": "R5t"}


def rd(name):
    return K.read_csv(os.path.join(OUT, name))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def p4(x):
    return f"{f(x):.4f}"


def pm(x, se):
    return f"{f(x):.4f} ± {f(se):.4f}"


def one(rows, **kw):
    rs = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    assert len(rs) == 1, (kw, len(rs))
    return rs[0]


def dl(d):
    return {"0.02": "0.02", "0.05": "0.05", "0.1": "0.10"}[d]


def a31_label(r):
    """A3.1 'pooling-dependent' δ of an attack_variance.csv row with the reading(s) that differ from the primary
    verdict at each δ (max = max(s²_a, s²_pool) with the pool df; own = s²_a with df r − 1)."""
    lab = r.get("pooling_dependent") or ""
    out = []
    for d in [x for x in lab.split(",") if x]:
        t = f"{float(d):g}"
        rd_ = [x for x in ("max", "own") if r[f"{x}_verdict@{t}"] != r[f"verdict@{t}"]]
        out.append(f"{dl(t)} ({' and '.join(rd_)})")
    return ", ".join(out)


class Data:
    def __init__(self):
        self.summ = rd("s1c_summary.csv")
        self.cells = rd("s1c_cells.csv")
        self.batt = rd("s1c_battery.csv")
        self.jobs = rd("s1c_jobs.csv")
        self.fit = rd("s1c_fit.csv")
        self.blocks = rd("s1c_blocks.csv")
        self.chk_s1b = rd("s1c_check_s1b.csv")
        self.s9s = rd("s9_s1_summary.csv")
        self.s9b = rd("s9_s1_by_scenario.csv")
        self.emp = rd("s9_known_truth_summary.csv")
        self.empc = rd("s9_known_truth_cells.csv")
        self.counts = rd("s9_clean_corpus_counts.csv")
        self.s1x = rd("s1_extra_summary.csv")
        self.checks = rd("checks.csv")
        self.prim = K.read_csv(os.path.join(K.REPORT, "primary.csv"))
        self.av = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r
                   for r in K.read_csv(os.path.join(K.ADDENDUM, "attack_variance.csv"))}

    def sm(self, rule, battery, config, c_sd="", delta="0.02"):
        return one(self.summ, rule=rule, battery=battery, config=config, c_sd=c_sd, delta=delta)

    def cell(self, rule, battery, config, attack, c_sd=""):
        return one(self.cells, rule=rule, battery=battery, config=config, attack=attack, c_sd=c_sd)

    def s9(self, rule, delta, cls, sset, n, grouping="truth"):
        tc = f"truth {cls}" if grouping == "truth" else cls
        return one(self.s9s, grouping=grouping, rule=rule, delta=delta, truth_class=tc, scenario_set=sset, n=n)

    def s9_opt(self, rule, delta, cls, sset, n, grouping="truth"):
        tc = f"truth {cls}" if grouping == "truth" else cls
        rs = [r for r in self.s9s if r["grouping"] == grouping and r["rule"] == rule and r["delta"] == delta
              and r["truth_class"] == tc and r["scenario_set"] == sset and r["n"] == n]
        return rs[0] if rs else None

    def em(self, level, cond, sub, delta, rule):
        return one(self.emp, level=level, condition=cond, subcondition=sub, delta=delta, rule=rule)


# ── key findings (every number from the CSVs) ────────────────────────


def kf_s1c(D):
    L = []
    # 1. pooled, config E
    parts, ok = [], True
    for b in POOLED:
        r = D.sm("pooled", b, "E")
        parts.append(f"{p4(r['coverage_min'])}–{p4(r['coverage_max'])} ({LBL[b]})")
        ok &= f(r["coverage_min"]) >= 0.95 - 2 * f(r["coverage_min_mcse"])
    fb = max((D.sm("pooled", b, "E") for b in POOLED), key=lambda r: f(r["false_below_max"]))
    ok &= f(fb["false_below_max"]) <= 0.025 + 2 * f(fb["false_below_max_mcse"])
    ft = max((D.sm("pooled", b, "E", delta=d) for b in POOLED for d in TOLS), key=lambda r: f(r["false_transfers_all0"]))
    L.append("S1c, pooled (seed-pooled) rule, homogeneous config E, 20,000 reps: "
             + ("calibrated or conservative in every battery (within MC error). " if ok else "")
             + "Coverage " + "; ".join(parts) + ". The largest false 'below δ' at μ = δ is "
             f"{pm(fb['false_below_max'], fb['false_below_max_mcse'])} ({LBL[fb['battery']]} "
             f"{ATK[fb['false_below_max_attack']]}); false TRANSFERS with all μ = 0 is at most "
             f"{pm(ft['false_transfers_all0'], ft['false_transfers_all0_mcse'])} (registered rates; nominal 0.025).")
    # 2. pooled, config H, every F_a > 1 cell
    rows = []
    for r in D.fit:
        if r["part"] != "pooled battery" or r["in_pool"] != "True" or f(r["F_obs"]) <= 1:
            continue
        key = KEY_OF[r["battery"]] + (r["attack"],)
        lab = a31_label(D.av[key])
        cs = {g: D.cell("pooled", r["battery"] + " logit", g, r["attack"]) for g in ("E", "H", "H05")}
        rows.append((f(r["F_obs"]), r["battery"], r["attack"], lab, cs))
    rows.sort(key=lambda t: -t[0])
    txt = []
    for F, b, a, lab, cs in rows:
        txt.append(f"{b} {ATK[a]} (F {F:.3f}): {pm(cs['H']['coverage'], cs['H']['coverage_mcse'])} "
                   f"[{p4(cs['H05']['coverage'])}] / {p4(cs['H']['false_below_at_mu_eq_delta'])}, "
                   + (f"A3.1 δ {lab}" if lab else "no A3.1 label"))
    e_cov = [f(cs["E"]["coverage"]) for _, _, _, _, cs in rows]
    labelled = [t for t in rows if t[3]]
    unl = [f"{b} {ATK[a]}" for _, b, a, lab, _ in rows if not lab]
    extra = []
    for key, r in sorted(D.av.items()):
        if not r["pooling_dependent"] or key[2] != "logit":
            continue
        F = (f(r["s_attack"]) / f(r["s_pool"])) ** 2
        if F <= 1:
            name = {v: k for k, v in KEY_OF.items()}[key[:3]]
            extra.append(f"{name} {ATK[key[3]]}: {a31_label(r)}")
    own_only = all("max" not in x for x in extra)
    lab_d = [(b, a, d) for _, b, a, lab, _ in rows for d in (lab.split(", ") if lab else [])]
    own_all = all("own" in d for _, _, d in lab_d)
    max_at = [f"{b} {ATK[a]} δ {d.split(' ')[0]}" for b, a, d in lab_d if "max" in d]
    L.append("S1c, pooled rule, measured heterogeneity (config H, fitted w; H05 = w 0.5 in brackets): the shared "
             "pool under-covers the cells whose own seed variance exceeds the pool's (F_a > 1). All "
             f"{len(rows)} F_a > 1 cells, coverage ± MC SE [H05] / false 'below δ' at μ = δ under H, with the A3.1 "
             "'pooling-dependent' δ and the A3.1 reading that differs there (max = max(s²_a, s²_pool) at the pool df; "
             "own = s²_a at df r − 1): " + "; ".join(txt) + f". Under E the same cells cover "
             f"{min(e_cov):.4f}–{max(e_cov):.4f}. {len(labelled)} of the {len(rows)} carry an A3.1 label"
             + (f" ({', '.join(unl)} does not)" if unl else "")
             + (f"; the own reading differs at every one of their {len(lab_d)} labelled δ" if own_all else "")
             + (f", the max reading only at {', '.join(max_at)}" if max_at else ", the max reading at none")
             + f". A3.1 also labels {len(extra)} cells with F_a ≤ 1 (" + "; ".join(extra) + ")"
             + ("; those labels come from the own reading only, which S1c does not simulate" if own_only else "") + ".")
    # 3. battery level, config H
    ag = [D.sm("pooled", "BERT AG News logit", "H", delta=d) for d in TOLS]
    oth = max((D.sm("pooled", b, "H", delta=d) for b in POOLED[1:] for d in TOLS),
              key=lambda r: f(r["false_transfers_all0"]))
    agE = [D.sm("pooled", "BERT AG News logit", "E", delta=d) for d in TOLS]
    nt = max((D.sm("pooled", b, "H", delta=d) for b in POOLED for d in TOLS),
             key=lambda r: f(r["false_no_transfer_one_at_d_max"]))
    same_h = len({r["false_transfers_alld"] for r in ag}) == 1 and len({r["false_transfers_alld"] for r in agE}) == 1
    alld = (f"{pm(ag[0]['false_transfers_alld'], ag[0]['false_transfers_alld_mcse'])} at every δ (E: "
            f"{pm(agE[0]['false_transfers_alld'], agE[0]['false_transfers_alld_mcse'])}; with μ = δ for every cell and "
            "no structural row the rate does not depend on δ)" if same_h else
            " / ".join(p4(r["false_transfers_alld"]) for r in ag) + " at δ 0.02 / 0.05 / 0.10 (E: "
            + " / ".join(p4(r["false_transfers_alld"]) for r in agE) + ")")
    L.append("S1c battery level, config H. Registered false TRANSFERS with all μ = 0 for BERT AG News: "
             + ", ".join(f"{pm(r['false_transfers_all0'], r['false_transfers_all0_mcse'])} at δ {dl(r['delta'])}"
                         for r in ag)
             + f"; at most {pm(oth['false_transfers_all0'], oth['false_transfers_all0_mcse'])} in every other "
             f"battery. With every simulated μ = δ, the Holm TRANSFERS rate for BERT AG News is " + alld
             + ". The largest false NO TRANSFER with one "
             f"attack at μ = δ is {pm(nt['false_no_transfer_one_at_d_max'], nt['false_no_transfer_one_at_d_max_mcse'])} "
             f"({LBL[nt['battery']]} {ATK[nt['false_no_transfer_one_at_d_argmax']]}, δ {dl(nt['delta'])}).")
    # 4. NO TRANSFER all 0 and df
    parts = []
    for b in POOLED:
        e = [p4(D.sm("pooled", b, "E", delta=d)["no_transfer_all0"]) for d in TOLS]
        h = [p4(D.sm("pooled", b, "H", delta=d)["no_transfer_all0"]) for d in TOLS]
        parts.append(f"{LBL[b]} {'/'.join(e)} (E) and {'/'.join(h)} (H)")
    dfp = []
    for b in POOLED:
        meds = {g: f(D.sm("pooled", b, g)["sim_df_median"]) for g in ("E", "H", "H05")}
        dfp.append(f"{LBL[b]} E {meds['E']:.2f}, H {meds['H']:.2f}, H05 {meds['H05']:.2f} "
                   f"(measured {f(D.sm('pooled', b, 'E')['real_df']):.2f})")
    L.append("S1c, P(NO TRANSFER | all μ = 0) at δ 0.02/0.05/0.10: " + "; ".join(parts)
             + ". Simulated Satterthwaite df medians: " + "; ".join(dfp) + ".")
    # 5. transported, full simulation, E, c = 1
    parts = []
    for t in TARGETS:
        r = D.sm("transported", t, "E", C1)
        parts.append(f"{LBL[t]} coverage {p4(r['coverage_min'])}–{p4(r['coverage_max'])}, false TRANSFERS @0.02 "
                     f"{pm(r['false_transfers_all0'], r['false_transfers_all0_mcse'])}")
    L.append("S1c transported rule, full simulation (source pool and its df simulated), config E, c = 1, 10,000 reps: "
             + "; ".join(parts) + ".")
    # 6. misspecification (E)
    ag = lambda c: D.sm("transported", "BERT AG News feature", "E", c)
    L.append("S1c transported rule, variance misspecification, config E (BERT AG News feature): coverage "
             f"{p4(ag(CV15)['coverage_min'])}–{p4(ag(CV15)['coverage_max'])} at c² = 1.5 and "
             f"{p4(ag(CV2)['coverage_min'])}–{p4(ag(CV2)['coverage_max'])} at c² = 2 (variance reading); "
             f"{p4(ag(C15)['coverage_min'])}–{p4(ag(C15)['coverage_max'])} at c = 1.5 and "
             f"{p4(ag(C2)['coverage_min'])}–{p4(ag(C2)['coverage_max'])} at c = 2 (SD reading). False TRANSFERS "
             f"@0.02: {pm(ag(CV2)['false_transfers_all0'], ag(CV2)['false_transfers_all0_mcse'])} at c² = 2, "
             f"{pm(ag(C15)['false_transfers_all0'], ag(C15)['false_transfers_all0_mcse'])} at c = 1.5, "
             f"{pm(ag(C2)['false_transfers_all0'], ag(C2)['false_transfers_all0_mcse'])} at c = 2; at c = 2 BERT "
             "SST-2 feature "
             + pm(D.sm("transported", "BERT SST-2 feature", "E", C2)["false_transfers_all0"],
                  D.sm("transported", "BERT SST-2 feature", "E", C2)["false_transfers_all0_mcse"])
             + " and Qwen3 AG News "
             + pm(D.sm("transported", "Qwen3 AG News logit", "E", C2)["false_transfers_all0"],
                  D.sm("transported", "Qwen3 AG News logit", "E", C2)["false_transfers_all0_mcse"]) + ".")
    # 7. config H transported, c = 1, 40,000 reps
    h = {t: D.sm("transported", t, "H", C1) for t in TARGETS}
    h05 = D.sm("transported", "BERT AG News feature", "H", C1, "0.05")
    blk = sorted((b for b in D.blocks if b["rule"] == "transported" and b["battery"] == "BERT AG News feature"
                  and b["config"] == "H" and b["c_sd"] == C1 and b["delta"] == "0.02"), key=lambda b: int(b["block"]))
    L.append(f"S1c transported rule, config H (c·√F_a), c = 1, {int(h['BERT AG News feature']['reps']):,} reps (4 seed "
             f"blocks): BERT AG News feature {ATK[h['BERT AG News feature']['coverage_min_attack']]} coverage "
             f"{pm(h['BERT AG News feature']['coverage_min'], h['BERT AG News feature']['coverage_min_mcse'])}, "
             "false TRANSFERS "
             f"{pm(h['BERT AG News feature']['false_transfers_all0'], h['BERT AG News feature']['false_transfers_all0_mcse'])}"
             f" at δ 0.02 ({f(h['BERT AG News feature']['false_transfers_all0']) / 0.025:.1f} times the nominal 0.025)"
             f" and {pm(h05['false_transfers_all0'], h05['false_transfers_all0_mcse'])} at 0.05 (the four "
             "blocks give " + ", ".join(p4(b["false_transfers_all0"]) for b in blk) + " at 0.02; block 1 is the "
             "pre-promotion 10,000-rep value); BERT SST-2 feature "
             f"{ATK[h['BERT SST-2 feature']['coverage_min_attack']]} "
             f"{pm(h['BERT SST-2 feature']['coverage_min'], h['BERT SST-2 feature']['coverage_min_mcse'])}; Qwen3 AG News "
             f"{ATK[h['Qwen3 AG News logit']['coverage_min_attack']]} "
             f"{pm(h['Qwen3 AG News logit']['coverage_min'], h['Qwen3 AG News logit']['coverage_min_mcse'])}.")
    # 8. realdf vs full (cell by cell; every stated direction is asserted on the CSVs)
    parts = {}
    for t in TARGETS:
        rr, ff = D.sm("transported_realdf", t, "E", C1), D.sm("transported", t, "E", C1)
        parts[t] = (f"{LBL[t]} {p4(rr['coverage_min'])}–{p4(rr['coverage_max'])} vs "
                    f"{p4(ff['coverage_min'])}–{p4(ff['coverage_max'])}")
    st = realdf_stats(D)
    rH = {t: D.sm("transported_realdf", t, "H", C1) for t in TARGETS}
    rH_ag = D.cell("transported_realdf", "BERT AG News feature", "H", "ep", C1)
    fH_ag = D.cell("transported", "BERT AG News feature", "H", "ep", C1)
    rH_s = D.cell("transported_realdf", "BERT SST-2 feature", "H", "atba", C1)
    fH_s = D.cell("transported", "BERT SST-2 feature", "H", "atba", C1)
    bert = ("BERT AG News feature", "BERT SST-2 feature")
    q = "Qwen3 AG News logit"
    # stated directions: H optimistic for the BERT targets, not for Qwen3 AG News; Qwen3 lower under E at c = 1
    assert all(st[(t, "H")]["min"] >= 0 and st[(t, "H")]["ft_le"] == st[(t, "H")]["ft_n"] for t in bert), st
    assert st[(q, "H")]["min"] < 0 and st[(q, "E")]["max1"] < 0, st
    qfrac, qdf = qwen_df_below(D)
    import re
    cf = []
    for r in D.checks:
        if r["group"] != "S1c transported_realdf vs closed form":
            continue
        m = re.match(r"^(.+?): max \|z\| over (\d+) cells \(mean z ([+-]?[\d.]+), share \|z\|>2 ([\d.]+)\)$", r["check"])
        assert m, r["check"]
        cf.append(f"{m.group(1)}: max |z| {f(r['ours']):.2f}, mean z {m.group(3)} over {m.group(2)} cells")
    L.append("S1c realdf reading (S1b's df 12 replaced by the measured pool df; s²_pool = E[s²_pool]·χ²_df/df), "
             "compared cell by cell with the full simulation (realdf minus full coverage). Config E, c = 1: close for "
             "the two BERT 'feature' batteries (coverage realdf vs full: " + "; ".join(parts[t] for t in bert)
             + f"; cell differences {rng(st[(bert[0], 'E')], 1)} and {rng(st[(bert[1], 'E')], 1)}) and lower for "
             f"Qwen3 AG News ({parts[q].replace(LBL[q] + ' ', '')}; cell differences {rng(st[(q, 'E')], 1)}). Config "
             "H: optimistic for the two BERT 'feature' "
             "batteries, so the full simulation is the relevant reading there: over every c the realdf coverage is at "
             f"or above the full simulation in every cell ({rng(st[(bert[0], 'H')])} and {rng(st[(bert[1], 'H')])}) and "
             "its false TRANSFERS at δ 0.02 and 0.05 are at or below it in every row; at c = 1, BERT AG News feature EP "
             f"coverage {pm(rH_ag['coverage'], rH_ag['coverage_mcse'])} (realdf) vs "
             f"{pm(fH_ag['coverage'], fH_ag['coverage_mcse'])} (full), BERT SST-2 feature ATBA "
             f"{p4(rH_s['coverage'])} vs {p4(fH_s['coverage'])}, false TRANSFERS @0.02 for BERT AG News feature "
             f"{pm(rH['BERT AG News feature']['false_transfers_all0'], rH['BERT AG News feature']['false_transfers_all0_mcse'])}"
             f" vs {pm(h['BERT AG News feature']['false_transfers_all0'], h['BERT AG News feature']['false_transfers_all0_mcse'])}."
             f" For Qwen3 AG News realdf is not optimistic under H ({rng(st[(q, 'H')])} over every c, as under E): the "
             f"simulated Qwen3 SST-2 source pool has a Satterthwaite df below the measured {qdf:.2f} in "
             f"{min(qfrac):.1%}–{max(qfrac):.1%} of draws, so the full rule's t quantile is wider in most draws."
             " The realdf cells match a closed-form 1-D χ² integral (checks.csv; "
             + "; ".join(cf) + ").")
    return L


def realdf_stats(D):
    """transported_realdf minus the full transported simulation, per target battery and config: cell coverage
    differences (min/max over every c and cell, and at c = 1) and the number of (c, δ ∈ {0.02, 0.05}) battery rows
    whose false-TRANSFERS rate is at or below / below the full simulation's."""
    out = {}
    c_all = sorted({c["c_sd"] for c in D.cells if c["rule"] == "transported"}, key=float)
    for t in TARGETS:
        for g in ("E", "H"):
            dif, dif1, ft = [], [], []
            for c in c_all:
                rr = {x["attack"]: x for x in D.cells if x["rule"] == "transported_realdf" and x["battery"] == t
                      and x["config"] == g and x["c_sd"] == c}
                ff = {x["attack"]: x for x in D.cells if x["rule"] == "transported" and x["battery"] == t
                      and x["config"] == g and x["c_sd"] == c}
                assert rr and set(rr) == set(ff), (t, g, c)
                for a in rr:
                    dif.append(f(rr[a]["coverage"]) - f(ff[a]["coverage"]))
                    if c == C1:
                        dif1.append(dif[-1])
                for dd in ("0.02", "0.05"):
                    br = one(D.batt, rule="transported_realdf", battery=t, config=g, c_sd=c, delta=dd)
                    bf = one(D.batt, rule="transported", battery=t, config=g, c_sd=c, delta=dd)
                    ft.append(f(br["false_transfers_all0"]) - f(bf["false_transfers_all0"]))
            out[(t, g)] = {"min": min(dif), "max": max(dif), "min1": min(dif1), "max1": max(dif1),
                           "ft_le": sum(x <= 0 for x in ft), "ft_n": len(ft)}
    return out


def rng(s, at_c1=False):
    lo, hi = (s["min1"], s["max1"]) if at_c1 else (s["min"], s["max"])
    return f"{lo:+.4f} to {hi:+.4f}"


def qwen_df_below(D):
    """Share of simulated source-pool df below the measured df in the Qwen3 AG News transported jobs."""
    js = [r for r in D.jobs if r["rule"] == "transported" and r["battery"] == "Qwen3 AG News logit"]
    assert js and len({r["real_df"] for r in js}) == 1
    return [f(r["frac_sim_df_below_real"]) for r in js], f(js[0]["real_df"])


def kf_s9(D):
    L = []
    # S1 generator, R5 by delta
    g02 = D.s9("R5_audit", "0.02", "N", "grid+mirror", "all")
    b912 = D.s9("R5_audit", "0.02", "boundary (k = delta)", "a11.12", "912", "truth class")
    b5700 = D.s9("R5_audit", "0.02", "boundary (k = delta)", "a11.12", "5700", "truth class")
    gb = D.s9_opt("R5_audit", "0.02", "boundary (k = delta)", "grid+mirror", "all", "truth class")
    assert gb is None   # the grid has no k = 0.02 scenario
    bs = [r for r in D.s9b if r["rule"] == "R5_audit" and r["delta"] == "0.02" and r["n"] == "912"
          and r["set"].startswith("a11.12") and r["truth_class"] == "boundary (k = delta)"]
    assert len(bs) == int(b912["n_scenarios"])
    sp = [f(r["p_wrong"]) for r in bs if f(r["discordance"]) <= 0.03 + 1e-12]
    npairs = sorted({912 * f(r["discordance"]) for r in bs if f(r["discordance"]) <= 0.03 + 1e-12})
    above = f(b912["p_wrong_mean"]) - 2 * f(b912["p_wrong_mean_mcse"]) > 0.025 and min(sp) > 0.025
    txt = (f"S9 on the S1 generator (20,000 reps per scenario), audit R5, false transfer when the truth is N. At δ 0.02 "
           "the S1 grid has no k = δ scenario; its worst case there is "
           f"{pm(g02['p_wrong_max'], g02['worst_p_wrong_mcse'])} at k = {f(g02['worst_true_k']):g} "
           f"({g02['worst_scenario']}). The δ 0.02 boundary exists only in the A11.12 scenarios. At k = δ = 0.02 and "
           f"n = 912, R5 claims transfer with probability {pm(b912['p_wrong_mean'], b912['p_wrong_mean_mcse'])} on "
           f"average over the {b912['n_scenarios']} scenarios ({p4(b912['p_wrong_min'])}–{p4(b912['p_wrong_max'])}); in "
           f"the {len(sp)} sparse tables (discordance ≤ 0.03, about {npairs[0]:.0f}–{npairs[-1]:.0f} discordant pairs) it is "
           f"{min(sp):.4f}–{max(sp):.4f} (mean {sum(sp) / len(sp):.4f}), the largest "
           f"{pm(b912['p_wrong_max'], b912['worst_p_wrong_mcse'])} ({b912['worst_scenario']}, a selected maximum). At "
           f"n = 5,700 it is {pm(b5700['p_wrong_mean'], b5700['p_wrong_mean_mcse'])} over {b5700['n_scenarios']} "
           f"scenarios (max {p4(b5700['p_wrong_max'])})."
           + (" So at the δ 0.02 boundary R5's false-transfer rate is above the nominal 0.025, about 0.03 in sparse "
              "tables." if above else ""))
    for d in ("0.05", "0.1"):
        parts = []
        for n in ("211", "570", "912", "5700"):
            b = D.s9("R5_audit", d, "boundary (k = delta)", "grid+mirror", n, "truth class")
            parts.append(f"n = {int(n):,} {pm(b['p_wrong_mean'], b['p_wrong_mean_mcse'])} (max {p4(b['p_wrong_max'])})")
        w = D.s9("R5_audit", d, "N", "all", "all")
        txt += (f" At δ {dl(d)} the worst case over every set is {pm(w['p_wrong_max'], w['worst_p_wrong_mcse'])} "
                f"({w['worst_scenario']}, {w['worst_truth_class']}); grid boundary means " + ", ".join(parts) + ".")
    L.append(txt)
    # other rules: worst false transfer and worst miss over every scenario
    parts = []
    for rule in ("R1_raw_asr", "R1h_raw_asr_high", "R2_abs_point", "R3_signed_point", "R4_ci_excludes_0"):
        vals = []
        for d in TOLS:
            w = D.s9(rule, d, "N", "all", "all")
            vals.append(p4(w["p_wrong_max"]))
        parts.append(f"{SHORT[rule]} {' / '.join(vals)}")
    r3b = {d: D.s9("R3_signed_point", d, "boundary (k = delta)", "all", "all", "truth class") for d in TOLS}
    r4n = D.s9("R4_ci_excludes_0", "0.02", "null (k = 0)", "all", "all", "truth class")
    miss = []
    for rule in ("R5_audit", "R5b_audit_binary", "R4_ci_excludes_0", "R3_signed_point", "R2_abs_point"):
        vals = [D.s9(rule, d, "T", "all", "all") for d in TOLS]
        miss.append(f"{SHORT[rule]} " + " / ".join(p4(v["p_wrong_max"]) for v in vals))
    ab = []
    for d in TOLS:
        a = [f(r["p_abstain"]) for r in D.s9b if r["rule"] == "R5_audit" and r["delta"] == d
             and r["set"] in ("grid", "grid_mirror") and r["n"] in ("912",)]
        b = [f(r["p_abstain"]) for r in D.s9b if r["rule"] == "R5_audit" and r["delta"] == d
             and r["set"] in ("grid", "grid_mirror") and r["n"] in ("5700",)]
        ab.append(f"{sum(a) / len(a):.3f} / {sum(b) / len(b):.3f}")
    L.append("S9 on the S1 generator, the other stages: worst false-transfer probability over every scenario set and "
             "n, at δ 0.02 / 0.05 / 0.10: " + "; ".join(parts) + ". At k = δ, R3 claims transfer "
             + " / ".join(p4(r3b[d]["p_wrong_mean"]) for d in TOLS) + " of the time on average; R4 at k = 0 "
             f"{p4(r4n['p_wrong_mean'])} on average (max {p4(r4n['p_wrong_max'])}). Worst miss when the truth is T "
             "(δ 0.02 / 0.05 / 0.10): " + "; ".join(miss) + ". R5's mean abstention on the grid (n = 912 / 5,700) is "
             + "; ".join(f"{a} at δ {dl(d)}" for a, d in zip(ab, TOLS)) + "; reading the audit as binary (R5b) turns "
             "abstentions into misses.")
    # S1 cells set
    parts = []
    for d in TOLS:
        w = D.s9("R5_audit", d, "N", "s1_cells", "all")
        m = D.s9("R5_audit", d, "T", "s1_cells", "all")
        r4 = D.s9("R4_ci_excludes_0", d, "N", "s1_cells", "all")
        parts.append(f"δ {dl(d)}: R5 {p4(w['p_wrong_max'])} ({w['worst_scenario']}), R4 {p4(r4['p_wrong_max'])}, "
                     f"R5 worst miss {p4(m['p_wrong_max'])}")
    n_sc = D.s9("R5_audit", "0.05", "N", "s1_cells", "all")["n_scenarios"]
    n_t = D.s9("R5_audit", "0.05", "T", "s1_cells", "all")["n_scenarios"]
    L.append("S9 on S1's cell set (results/tifs/s1/cells.csv: plug-in, null and edge twins of the 48 v1 cells, n = 211 "
             f"/ 5,700, plus the edge twins k = 0.03 and 0.15; {int(n_sc) + int(n_t)} scenarios), worst false transfer "
             "and worst miss: " + "; ".join(parts) + ".")
    # empirical
    def line(level, cond, sub, d, rules=EMP_RULES):
        out = []
        for rule in rules:
            r = D.em(level, cond, sub, d, rule)
            if int(r["not_applicable"]) == int(r["n_tests"]):
                continue
            s = f"{SHORT[rule]} {r['wrong']}/{r['n_tests']}"
            if cond == "A/A backdoor-free swap":
                s += f" ({r['items_wrong']}/{r['n_items']} pairs)"
            if rule in ("R5_audit", "R5t_audit_transported") and int(r["abstain"]):
                s += f" [{r['abstain']} abstain]"
            out.append(s)
        return ", ".join(out)
    ep = [D.em("condition", "EP SST-2 structural null", "(all)", d, rule) for d in TOLS for rule in RULES]
    epw = max(int(r["wrong"]) for r in ep)
    ep_ab = [D.em("condition", "EP SST-2 structural null", "(all)", d, "R5t_audit_transported")["abstain"] for d in TOLS]
    pcs = [D.em("condition", "positive control", "(all)", d, rule) for d in TOLS for rule in EMP_RULES]
    pcw = max(int(r["wrong"]) for r in pcs)
    pck = sorted(f(r["k"]) for r in D.empc if r["condition"] == "positive control" and r["rule"] == "R5_audit"
                 and r["delta"] == "0.05")
    r1 = {d: D.em("condition", "R1 contaminated corpus, no teacher", "(all)", d, "R1_raw_asr") for d in TOLS}
    r1h = {d: D.em("condition", "R1 contaminated corpus, no teacher", "(all)", d, "R1h_raw_asr_high") for d in TOLS}
    r1v = [f(r["suspect_rate"]) for r in D.empc if r["condition"] == "R1 contaminated corpus, no teacher"
           and r["rule"] == "R1_raw_asr" and r["delta"] == "0.05"]
    r1lo = [f(r["suspect_wilson_lo"]) for r in D.empc if r["condition"] == "R1 contaminated corpus, no teacher"
            and r["rule"] == "R1_raw_asr" and r["delta"] == "0.05"]
    aa = "A/A backdoor-free swap"
    subs = ["clean corpus, BadNets trigger", "contaminated corpus (badnets10), BadNets trigger",
            "clean corpus, ATBA phrase", "contaminated corpus (badnets10), ATBA phrase"]
    atba_k = {s: sorted({abs(f(r["k"])) for r in D.empc if r["condition"] == aa and r["subcondition"] == s
                          and r["rule"] == "R5_audit" and r["delta"] == "0.05"}) for s in subs}
    txt = [f"S9 empirical known-truth items, wrong claims / tests (A/A also wrong pairs / pairs; both orientations of a "
           f"pair are tests). EP SST-2 structural null (logit, 'feature'): no rule is ever wrong (max {epw} of 2; R5t "
           f"abstains {' / '.join(ep_ab)} of 2 at δ 0.02 / 0.05 / 0.10). Positive controls (k "
           f"{min(pck):.4f}–{max(pck):.4f}): no rule misses (max {pcw} of 4) at any δ. Contaminated-corpus CE-only "
           f"R1 items (raw ASR {min(r1v):.4f}–{max(r1v):.4f}, Wilson lower bound ≥ {min(r1lo):.4f}): R1 "
           f"{r1['0.05']['wrong']}/{r1['0.05']['n_tests']} and R1h {r1h['0.05']['wrong']}/{r1h['0.05']['n_tests']} "
           "wrong at every δ; the matched rules are not applicable."]
    for s in subs:
        n_pairs = D.em("subcondition", aa, s, "0.05", "R5_audit")["n_items"]
        ks = atba_k[s]
        txt.append(f"A/A, {s} ({n_pairs} pairs, |k| {ks[0]:.4f}–{ks[-1]:.4f}) at δ 0.05: " + line("subcondition", aa, s, "0.05")
                   + "; R5 " + ", ".join(f"{D.em('subcondition', aa, s, d, 'R5_audit')['wrong']}/"
                                         f"{D.em('subcondition', aa, s, d, 'R5_audit')['n_tests']} at δ {dl(d)}"
                                         for d in ("0.02", "0.1"))
                   + ".")
    both = "ATBA phrase (clean + contaminated corpus)"
    rr = {d: D.em("trigger", aa, both, d, "R5_audit") for d in TOLS}
    bn = {d: D.em("trigger", aa, "BadNets trigger (clean + contaminated corpus)", d, "R5_audit") for d in TOLS}
    txt.append("ATBA phrase over both corpora, R5 wrong: " + ", ".join(
        f"{rr[d]['wrong']}/{rr[d]['n_tests']} tests ({rr[d]['items_wrong']}/{rr[d]['n_items']} pairs) at δ {dl(d)}"
        for d in TOLS) + ". These are real differences between independent clean teacher builds on the ATBA phrase "
        "(reference dependence, A3.3), not a backdoor, and they count as false transfer claims. On the BadNets "
        "trigger over both corpora R5 is wrong " + ", ".join(
            f"{bn[d]['wrong']}/{bn[d]['n_tests']} ({bn[d]['items_wrong']}/{bn[d]['n_items']} pairs) at δ {dl(d)}"
            for d in TOLS) + ".")
    L.append(" ".join(txt))
    # clean-corpus counts
    parts = []
    pos = {d: sum(1 for r in D.prim if r[f"verdict@{float(d):g}"] == "POSITIVE") for d in TOLS}
    match = True
    for d in TOLS:
        cs = {r["rule"]: r for r in D.counts if r["delta"] == d}
        match &= int(cs["R5_audit"]["claim_T"]) == pos[d]
        parts.append(f"δ {dl(d)}: " + ", ".join(
            f"{SHORT[r]} {cs[r]['claim_T']}" + (f" (+{cs[r]['abstain']} INCONCLUSIVE)" if r == "R5_audit" else "")
            for r in RULES if r != "R5b_audit_binary"))
    L.append("S9 clean-corpus primary cells (not scored), number of the 31 cells each rule calls transfer: "
             + "; ".join(parts) + (". R5's counts equal the frozen primary POSITIVE cells." if match else "."))
    return L


def kf_s1x(D):
    L = []
    bd = [r for r in D.s1x if r["set"] in ("boundary", "boundary+sparse")]
    cov = [f(r["tango_k.coverage"]) for r in bd]
    eq = [f(r["tango_k.equivalent@0.02"]) for r in bd]
    pos = [f(r["tango_k.positive@0.02"]) for r in bd]
    by_d = {}
    for r in bd:
        by_d.setdefault(round(f(r["discordance"]), 6), []).append(f(r["tango_k.positive@0.02"]))
    se_pos_mean = math.sqrt(sum(p * (1 - p) / int(r["reps"]) for p, r in zip(pos, bd))) / len(bd)
    ncb = max(f(r["newcombe_k.equivalent@0.02"]) for r in bd)
    btb = max(f(r["boot_k.equivalent@0.02"]) for r in bd)
    se_typ = K.mc_se(0.03, int(bd[0]["reps"]))
    sparse_d = [(d, v) for d, v in sorted(by_d.items()) if d <= 0.03]
    above = all(min(v) > 0.025 for _, v in sparse_d)
    L.append(f"S1 extra rows (frozen run_scenario, {int(bd[0]['reps']):,} reps): n = 912 at k = δ = 0.02 "
             f"({len(bd)} scenarios). Tango coverage {min(cov):.4f}–{max(cov):.4f}; P(EQUIVALENT@0.02) "
             f"{min(eq):.4f}–{max(eq):.4f}" + (" (at most 0.025 in every scenario)" if all(x <= 0.025 for x in eq) else "")
             + f"; P(POSITIVE@0.02) {min(pos):.4f}–{max(pos):.4f}, mean over the scenarios {sum(pos) / len(pos):.4f} ± "
             f"{se_pos_mean:.4f}. In the sparse tables it is " + " and ".join(
                 f"{min(v):.4f}–{max(v):.4f} at d = {d:g} (about {912 * d:.0f} discordant pairs)" for d, v in sparse_d)
             + (", every value above the nominal 0.025" if above else "")
             + f" (MC SE about {se_typ:.4f} per scenario; maxima are selected and biased upward). Newcombe and "
             f"the k-bootstrap give false equivalence up to {ncb:.4f} and {btb:.4f}.")
    sp = [r for r in D.s1x if r["set"] in ("sparse", "boundary+sparse") and f(r["true_k"]) != 0]
    c912 = [f(r["tango_k.coverage"]) for r in sp if r["n"] == "912"]
    c5700 = [f(r["tango_k.coverage"]) for r in sp if r["n"] == "5700"]
    b912 = [f(r["boot_k.coverage"]) for r in sp if r["n"] == "912"]
    half = [f(r["tango_k.equivalent@0.02"]) for r in sp if r["n"] == "912" and abs(f(r["true_k"]) - 0.01) < 1e-12
            and abs(f(r["discordance"]) - 0.02) < 1e-12]
    full = [f(r["tango_k.equivalent@0.02"]) for r in sp if r["n"] == "912" and abs(f(r["true_k"]) - 0.01) < 1e-12
            and abs(f(r["discordance"]) - 0.01) < 1e-12]
    real = [r for r in D.s1x if r["set"] == "real"]
    L.append("S1 extra rows, sparse discordance with k ≠ 0: Tango coverage "
             f"{min(c912):.4f}–{max(c912):.4f} at n = 912 and {min(c5700):.4f}–{max(c5700):.4f} at n = 5,700 "
             f"(k-bootstrap {min(b912):.4f}–{max(b912):.4f} at n = 912). At n = 912 with k = 0.01, "
             f"P(EQUIVALENT@0.02) is {min(half):.4f}–{max(half):.4f} at d = 0.02 (k = d/2) and "
             f"{min(full):.4f}–{max(full):.4f} at d = 0.01 (k = d). The {len(real)} measured sparse plug-ins "
             f"(discordance ≤ 0.03, n10 ≠ n01) have Tango coverage {min(f(r['tango_k.coverage']) for r in real):.4f}–"
             f"{max(f(r['tango_k.coverage']) for r in real):.4f} and P(EQUIVALENT@0.02) "
             f"{min(f(r['tango_k.equivalent@0.02']) for r in real):.4f}–"
             f"{max(f(r['tango_k.equivalent@0.02']) for r in real):.4f}.")
    return L


# ── tables ───────────────────────────────────────────────────────────


def s1c_tables(D):
    fit = D.fit
    L = ["### Item 10 — S1c", "",
         "**Fit (s1c_fit.csv).** Pooled batteries as simulated (cells = the frozen pool's attacks plus SOS SST-2, "
         "which borrows the BERT SST-2 pool; EP SST-2 enters the battery decision as its fixed seed-42 Tango row):", "",
         "| battery | n | pool attacks (m) | simulated cells | F_a range (attack) | w fitted H / E | pool s measured / model | pool df measured |",
         "|---|---|---|---|---|---|---|---|"]
    bats = []
    for r in fit:
        if r["part"] == "pooled battery" and r["battery"] not in bats:
            bats.append(r["battery"])
    for b in bats:
        rs = [r for r in fit if r["battery"] == b and r["part"] == "pooled battery"]
        pool = [r for r in rs if r["in_pool"] == "True"]
        lo = min(pool, key=lambda r: f(r["F_obs"]))
        hi = max(pool, key=lambda r: f(r["F_obs"]))
        st = [r for r in fit if r["battery"] == b and r["part"].startswith("pooled battery (structural")]
        L.append(f"| {b} | {rs[0]['n']} | {len(pool)} | {len(rs)}{' + ' + st[0]['attack'] + ' (structural)' if st else ''} | "
                 f"{f(lo['F_obs']):.4f} ({lo['attack']}) – {f(hi['F_obs']):.3f} ({hi['attack']}) | "
                 f"{f(rs[0]['w_fit_H']):.3f} / {f(rs[0]['w_fit_E']):.3f} | {f(rs[0]['pool_s_obs']):.5f} / "
                 f"{f(rs[0]['pool_s_model']):.5f} | {f(rs[0]['pool_df_obs']):.2f} |")
    L += ["", "**Pooled (seed-pooled) rule, r = 3, 20,000 reps (s1c_summary.csv, rule = pooled).** Coverage of the 95% "
          "interval (min–max over the battery's simulated cells), the largest false 'below δ' at μ = δ (registered; "
          "nominal 0.025), the largest false 'above δ' at μ = δ, and the battery's P(false TRANSFERS | all μ = 0) "
          "(registered) and P(NO TRANSFER | all μ = 0) per δ (MC SE binomial):", "",
          "| battery | config | w | sim df median (measured) | coverage min (attack) – max | false below max (attack) | false above max (attack) | false TRANSFERS all-0 @0.02 / @0.05 / @0.10 | NO TRANSFER all-0 @0.02 / @0.05 / @0.10 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in D.summ:
        if r["rule"] != "pooled" or r["delta"] != "0.02":
            continue
        ds = [x for x in D.summ if x["rule"] == "pooled" and x["battery"] == r["battery"] and x["config"] == r["config"]]
        ft = " / ".join(pm(x["false_transfers_all0"], x["false_transfers_all0_mcse"]) for x in ds)
        nt = " / ".join(p4(x["no_transfer_all0"]) for x in ds)
        L.append(f"| {r['battery']} | {r['config']} | {f(r['w']):.3f} | {f(r['sim_df_median']):.2f} ({f(r['real_df']):.2f}) | "
                 f"{p4(r['coverage_min'])} ({r['coverage_min_attack']}) – {p4(r['coverage_max'])} | "
                 f"{pm(r['false_below_max'], r['false_below_max_mcse'])} ({r['false_below_max_attack']}) | "
                 f"{pm(r['false_above_max'], r['false_above_max_mcse'])} ({r['false_above_max_attack']}) | {ft} | {nt} |")
    for rule, title in (("transported", "simulated source pool and its Satterthwaite df (the full rule)"),
                        ("transported_realdf", "s2_pool = E[s2_pool]·χ²_df/df at the measured df")):
        L += ["", f"**Transported rule, {title} (s1c_summary.csv, rule = {rule}).** c = SD ratio of the target's "
              "true training SD to √E[s2_pool] (c² = variance ratio); config E: same c for every cell, config H: "
              "c·√F_a (F_a of the same attack in the family's logit battery).", "",
              "| battery | config | c (c²) | reps | coverage min (attack) – max | false below max (attack) | false TRANSFERS all-0 @0.02 / @0.05 / @0.10 | NO TRANSFER all-0 @0.05 |",
              "|---|---|---|---|---|---|---|---|"]
        for r in D.summ:
            if r["rule"] != rule or r["delta"] != "0.02":
                continue
            ds = [x for x in D.summ if x["rule"] == rule and x["battery"] == r["battery"] and x["config"] == r["config"]
                  and x["c_sd"] == r["c_sd"]]
            ft = " / ".join(pm(x["false_transfers_all0"], x["false_transfers_all0_mcse"]) for x in ds)
            nt05 = [x for x in ds if x["delta"] == "0.05"][0]
            L.append(f"| {r['battery']} | {r['config']} | {C_SD.get(r['c_sd'], r['c_sd'])} ({f(r['c_var']):.2f}) | "
                     f"{int(r['reps']):,} | {p4(r['coverage_min'])} ({r['coverage_min_attack']}) – {p4(r['coverage_max'])} | "
                     f"{pm(r['false_below_max'], r['false_below_max_mcse'])} ({r['false_below_max_attack']}) | {ft} | "
                     f"{p4(nt05['no_transfer_all0'])} |")
        if rule == "transported":
            L += ["", "Config H rows pool 4 seed blocks of 10,000 reps (s1c_blocks.csv lists each block; block 1 "
                  "reproduces the pre-promotion 10,000-rep run)."]
        else:
            st = realdf_stats(D)
            qfrac, qdf = qwen_df_below(D)
            bert = ("BERT AG News feature", "BERT SST-2 feature")
            q = "Qwen3 AG News logit"
            assert all(st[(t, "H")]["min"] >= 0 and st[(t, "H")]["ft_le"] == st[(t, "H")]["ft_n"] for t in bert)
            assert st[(q, "H")]["min"] < 0 and st[(q, "E")]["max1"] < 0
            L += ["", "Realdf minus the full simulation, cell coverage (s1c_cells.csv). Config E, c = 1: "
                  f"{rng(st[(bert[0], 'E')], 1)} (BERT AG News feature), {rng(st[(bert[1], 'E')], 1)} (BERT SST-2 "
                  f"feature), {rng(st[(q, 'E')], 1)} (Qwen3 AG News). Config H, every c: {rng(st[(bert[0], 'H')])} and "
                  f"{rng(st[(bert[1], 'H')])} for the two BERT 'feature' batteries, whose realdf false TRANSFERS at δ "
                  "0.02 and 0.05 are also at or below the full simulation's in every row: realdf is optimistic there, "
                  "so the full-simulation table above is the relevant H reading. For Qwen3 AG News it is "
                  f"{rng(st[(q, 'H')])}: realdf is not optimistic there (the simulated source-pool df is below the "
                  f"measured {qdf:.2f} in {min(qfrac):.1%}–{max(qfrac):.1%} of draws, so the full rule's t quantile "
                  "is wider in most draws)."]
    L += ["", "**Pooled rule, every cell whose measured seed variance exceeds the pool (F_a > 1)** (s1c_cells.csv), beside "
          "its frozen primary verdicts (results/tifs/report/primary.csv) and the A3.1 'pooling-dependent' δ "
          "(results/tifs/addendum/attack_variance.csv) with the A3.1 reading that differs from the primary verdict at "
          "that δ (max = max(s²_a, s²_pool) at the pool df; own = s²_a at df r − 1):", "",
          "| battery | attack | F_a | k (primary) | primary verdict @0.02/0.05/0.10 | A3.1 pooling-dependent δ (reading) | coverage E / H / H05 | false below at μ=δ E / H / H05 | false above at μ=δ E / H / H05 |",
          "|---|---|---|---|---|---|---|---|---|"]
    prim = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in D.prim}
    for r in fit:
        if r["part"] != "pooled battery" or r["in_pool"] != "True" or f(r["F_obs"]) <= 1:
            continue
        key = KEY_OF[r["battery"]] + (r["attack"],)
        pr, a3 = prim[key], D.av.get(key, {})
        lab = a31_label(a3) if a3 else ""
        cs = {c["config"]: c for c in D.cells if c["rule"] == "pooled" and c["battery"] == r["battery"] + " logit"
              and c["attack"] == r["attack"]}
        trio = lambda col: " / ".join(f"{f(cs[g][col]):.4f}" for g in ("E", "H", "H05"))
        L.append(f"| {r['battery']} | {r['attack']} | {f(r['F_obs']):.3f} | {f(pr['k']):.4f} | "
                 f"{pr['verdict@0.02']} / {pr['verdict@0.05']} / {pr['verdict@0.1']} | {lab or '-'} | "
                 f"{trio('coverage')} | {trio('false_below_at_mu_eq_delta')} | {trio('false_above_at_mu_eq_delta')} |")
    det = [r for r in D.chk_s1b if r["check"].startswith("frozen pool_seed_variance")][0]
    zs = [(abs(f(r["z_diff"])), r) for r in D.chk_s1b if r.get("z_diff") not in (None, "")]
    zmax, zr = max(zs, key=lambda t: t[0])
    L += ["", f"**Validation (s1c_check_s1b.csv).** On S1b's own draws the frozen `analyze.pool_seed_variance` "
          f"reproduces S1b's pooled SE and Satterthwaite df (max |diff| SE {f(det['max_abs_diff_se']):.1e}, df "
          f"{f(det['max_abs_diff_df']):.1e}). This simulator run in S1b mode (homogeneous τ, w = 0.5, equicorrelated "
          f"input noise) against the archived results/tifs/s1/s1b.csv pooled rows: {len(zs)} metric comparisons, "
          f"max |z| = {zmax:.2f} ({zr['setting']}, {zr['metric']}: {f(zr['this_sim']):.5f} vs {f(zr['s1b_csv']):.5f})."]
    rp = os.path.join(OUT, "logs", "recheck_s1b.json")
    if os.path.exists(rp):
        rj = json.load(open(rp))
        L[-1] += (f" Re-running that setting with two fresh seeds (recheck_s1b.py, 2 × 20,000 reps, logs/recheck_s1b.json) "
                  f"gives {rj['pooled_new_40000']:.6f} (z = {rj['z_new_vs_s1b']:.2f} against S1b's {rj['s1b_csv']:.5f}); "
                  f"all 60,000 reps together {rj['all_60000']:.6f} (z = {rj['z_all_vs_s1b']:.2f}), consistent with chance.")
    L.append("")
    return L


def s9_tables(D):
    L = ["### Item 11 — S9", "",
         "**S1 generator: worst-case probability of a wrong claim** (s9_s1_summary.csv, grouping = truth, 20,000 reps "
         "per scenario). Truth N: P(claim T) = false transfer; truth T: P(claim N) = miss. Columns give the maximum "
         "over the scenarios of a set at each n ('–' = no scenario of that truth); the last column is the maximum over "
         "every set and n, with its scenario and MC SE.", ""]
    cols = [("grid+mirror", "211"), ("grid+mirror", "570"), ("grid+mirror", "912"), ("grid+mirror", "5700"),
            ("a11.12", "912"), ("a11.12", "5700"), ("s1_cells", "211"), ("s1_cells", "5700")]
    head = " | ".join(f"{s} n={n}" for s, n in cols)
    for truth, title in (("N", "Truth N (false transfer)"), ("T", "Truth T (miss)")):
        L += [f"{title}:", "", f"| rule | δ | {head} | all: max (scenario) |", "|---|---|" + "---|" * (len(cols) + 1)]
        for rule in RULES:
            for d in TOLS:
                vals = []
                for s, n in cols:
                    r = D.s9_opt(rule, d, truth, s, n)
                    vals.append(p4(r["p_wrong_max"]) if r else "–")
                w = D.s9(rule, d, truth, "all", "all")
                L.append(f"| {SHORT[rule]} | {dl(d)} | " + " | ".join(vals)
                         + f" | {pm(w['p_wrong_max'], w['worst_p_wrong_mcse'])} ({w['worst_scenario']}) |")
        L.append("")
    L += ["**The boundary k = δ (truth N)**: R5's probability of claiming transfer, mean ± MC SE of the mean over the "
          "boundary scenarios of each set and n (min–max). The grid has boundary scenarios at δ 0.05 and 0.10 only; "
          "the δ 0.02 boundary exists only in the A11.12 scenarios.", "",
          "| δ | set | n | scenarios | mean ± MC SE | min – max | worst scenario |", "|---|---|---|---|---|---|---|"]
    for d in TOLS:
        for s in ("grid+mirror", "a11.12", "s1_cells"):
            for n in ("211", "570", "912", "5700"):
                r = D.s9_opt("R5_audit", d, "boundary (k = delta)", s, n, "truth class")
                if r is None:
                    continue
                L.append(f"| {dl(d)} | {s} | {n} | {r['n_scenarios']} | {pm(r['p_wrong_mean'], r['p_wrong_mean_mcse'])} | "
                         f"{p4(r['p_wrong_min'])} – {p4(r['p_wrong_max'])} | {r['worst_scenario']} |")
    L += ["", "**R5 mean abstention** (P(INCONCLUSIVE), mean over scenarios; other rules never abstain):", "",
          "| δ | " + " | ".join(f"{s} n={n}" for s, n in cols) + " |", "|---|" + "---|" * len(cols)]
    for d in TOLS:
        vals = []
        for s, n in cols:
            v = [f(r["p_abstain"]) for r in D.s9b if r["rule"] == "R5_audit" and r["delta"] == d and r["n"] == n
                 and (r["set"] in ("grid", "grid_mirror") if s == "grid+mirror" else r["set"].startswith(s))]
            vals.append(f"{sum(v) / len(v):.3f}" if v else "–")
        L.append(f"| {dl(d)} | " + " | ".join(vals) + " |")
    L += ["", "MC SE of a single rate p with 20,000 reps is √(p(1−p)/20000) (0.0011 at p = 0.025).", "",
          "**Empirical known-truth items (s9_known_truth_summary.csv).** Cells: wrong tests / tests, then for A/A wrong "
          "pairs / pairs in brackets (both orientations of a pair are separate tests; a pair is wrong if a wrong claim "
          "occurs in either orientation); abstentions of R5 and R5t in square brackets; n/a = rule not applicable. "
          "Rows 'A/A, both corpora' are level = trigger.", ""]
    subs = []
    for r in D.emp:
        if r["level"] == "subcondition" and (r["condition"], r["subcondition"]) not in subs:
            subs.append((r["condition"], r["subcondition"]))
    trig = sorted({r["subcondition"] for r in D.emp if r["level"] == "trigger"})
    for d in TOLS:
        L += [f"δ = {dl(d)}:", "", "| condition | sub-condition | truth | items | " + " | ".join(SHORT[r] for r in EMP_RULES) + " |",
              "|---|---|---|---|" + "---|" * len(EMP_RULES)]
        rows = [("subcondition", c, s) for c, s in subs] + [("trigger", "A/A backdoor-free swap", t) for t in trig]
        for level, cond, sub in rows:
            rs = {r["rule"]: r for r in D.emp if r["level"] == level and r["condition"] == cond
                  and r["subcondition"] == sub and r["delta"] == d}
            cells = []
            for rule in EMP_RULES:
                r = rs[rule]
                if int(r["not_applicable"]) == int(r["n_tests"]):
                    cells.append("n/a")
                    continue
                s = f"{r['wrong']}/{r['n_tests']}"
                if cond == "A/A backdoor-free swap":
                    s += f" ({r['items_wrong']}/{r['n_items']})"
                if rule in ("R5_audit", "R5t_audit_transported"):
                    s += f" [{r['abstain']}]"
                cells.append(s)
            first = rs[RULES[0]]
            name = cond if level == "subcondition" else "A/A, both corpora"
            L.append(f"| {name} | {sub} | {first['truth']} | {first['n_items']} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["**Clean-corpus primary cells (s9_clean_corpus_counts.csv; truth unknown, not scored).** Number of the 31 "
          "cells each rule would call 'transfer' (T); R1h at the primary level = HIGH flag in every seed of the cell:", "",
          "| δ | " + " | ".join(SHORT[r] for r in RULES) + " |", "|---|" + "---|" * len(RULES)]
    for d in TOLS:
        cs = {r["rule"]: r for r in D.counts if r["delta"] == d}
        L.append(f"| {dl(d)} | " + " | ".join(cs[r]["claim_T"] + (f" (+{cs[r]['abstain']} A)" if r == "R5_audit" else "")
                                            for r in RULES) + " |")
    L.append("")
    return L


def s1x_table(D):
    L = ["### Item 12 — S1 extra rows (frozen run_scenario, 10,000 reps, bootstrap R = 2,000)", "",
         "| set | scenario | expected discordant pairs | true k | Tango coverage | Tango P(EQUIV@0.02) | Tango P(POS@0.02) | Newcombe P(EQUIV@0.02) / P(POS@0.02) | k-bootstrap P(EQUIV@0.02) / P(POS@0.02) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(D.s1x, key=lambda r: (r["set"], r["scenario"])):
        L.append(f"| {r['set']} | {r['scenario']} | {f(r['expected_discordant_pairs']):.1f} | {f(r['true_k']):+.5f} | "
                 f"{pm(r['tango_k.coverage'], r['tango_k.coverage_mcse'])} | "
                 f"{pm(r['tango_k.equivalent@0.02'], r['tango_k.equivalent@0.02_mcse'])} | "
                 f"{pm(r['tango_k.positive@0.02'], r['tango_k.positive@0.02_mcse'])} | "
                 f"{p4(r['newcombe_k.equivalent@0.02'])} / {p4(r['newcombe_k.positive@0.02'])} | "
                 f"{p4(r['boot_k.equivalent@0.02'])} / {p4(r['boot_k.positive@0.02'])} |")
    L.append("")
    return L


def checks_section(D):
    groups = {}
    for r in D.checks:
        groups.setdefault(r["group"], [0, 0])
        groups[r["group"]][0] += 1
        groups[r["group"]][1] += r["pass"] == "True"
    L = ["## Self-checks (checks.csv)", "",
         f"{sum(v[1] for v in groups.values())} of {sum(v[0] for v in groups.values())} checks pass.", "",
         "| group | checks | pass |", "|---|---|---|"]
    for g, (n, p) in groups.items():
        L.append(f"| {g} | {n} | {p} |")
    fails = [r for r in D.checks if r["pass"] != "True"]
    if fails:
        L += ["", "Failed checks:"] + [f"- {r['group']}: {r['check']} (ours {r['ours']}, reference {r['reference']}) "
                                       f"{r['note']}" for r in fails]
    L.append("")
    return L


def caveats(D):
    fit = D.fit
    clipped = [f"{r['battery']} {r['attack']}" for r in fit if r["part"] == "pooled battery" and f(r["T_H"]) == 0.0]
    inflated = sorted({f"{r['battery']} (pool s model {f(r['pool_s_model']):.5f} vs measured {f(r['pool_s_obs']):.5f})"
                       for r in fit if r["part"] == "pooled battery" and r["in_pool"] == "True"
                       and f(r["pool_s_model"]) > f(r["pool_s_obs"]) * 1.001})
    w0 = sorted({r["battery"] for r in fit if r["part"] == "pooled battery" and f(r["w_fit_H"]) == 0.0})
    vd = K.read_csv(os.path.join(K.ADDENDUM, "variance_diag.csv"))
    ag = {r["attack"]: [f(r[k]) for k in r if k.startswith("signflip_p_")] for r in vd
          if (r["fam"], r["dataset"], r["kd"]) == ("bert", "agnews", "logit")}
    act = [a for a in sorted(ag) if max(ag[a]) < 0.05]
    quiet = [a for a in sorted(ag) if min(ag[a]) >= 0.05]
    act_p = [p for a in act for p in ag[a]]
    quiet_p = [p for a in quiet for p in ag[a]]
    L = ["## Caveats", "",
         "- Everything here is post hoc and auxiliary (A11); none of it changes a primary verdict, battery decision or "
         "the paper-wide Holm. The S1c and S9/S1 generator results are model-based (normal approximation for S1c, "
         "multinomial paired tables for S9/S1) and inherit their assumptions: Σ is the measured input covariance taken "
         "as the truth, v_a is treated as known, training noise is Gaussian with one shared seed factor.",
         "- S1c config H treats the measured per-attack seed variances (2 df each) as the true heterogeneity; config E "
         "treats them as homogeneous. For BERT AG News logit the A3.2 sign-flip p-values (results/tifs/addendum/"
         f"variance_diag.csv) of every seed pair are < 0.05 for {', '.join(act)} (max {max(act_p):.3g}) and >= 0.05 for "
         f"{', '.join(quiet)} (min {min(quiet_p):.3g}): there the seed-to-seed shifts of the active attacks are not "
         "input noise, so the heterogeneity is real and H is the closer description.",
         f"- T_a is clipped at 0 for {', '.join(clipped)} (s_obs below the input-noise SD √ψ); the model's E[s²_pool] then "
         f"exceeds the measured pool for {', '.join(inflated) if inflated else 'no battery'}.",
         "- The fitted shared fraction w rests on one measured ms_seed/ms_res ratio (2 and 2(m−1) df); it is clipped at "
         f"w = 0 for {', '.join(w0)}. H05 (w = 0.5) is reported beside it.",
         "- S9 A/A truth is 'no teacher-attributable backdoor': a POSITIVE on an A/A swap is a real teacher-swap effect "
         "between two clean builds (reference dependence, A3.3), counted as a false transfer claim. A/A tests count both "
         "orientations of each pair, so tests are not independent; the pair counts are given beside them. At most one "
         "orientation can be T for R3, R4, R5, R5b and R5t; R2 gives the same claim in both orientations.",
         "- R1's 0.5 point threshold is a choice of this table (the plan's S9 registers no threshold); R1h uses the "
         "frozen audit_verdict HIGH flag (Wilson 95% lower bound ≥ 0.5) and is a sensitivity row.",
         "- Positive controls: the truth is the existence of a teacher-mediated channel; their observed k are far beyond "
         "every δ, so scoring T at every δ is safe, but no magnitude is known a priori.",
         "- R1 items: only the raw-ASR rules are applicable; the matched rules need a teacher swap, which a CE-only "
         "student does not have. The logit P1 − C1 of the same corpus is listed as context only (truth unknown).",
         "- S1's cell set comes from the v1 (Zenodo v1.1) seed-42 predictions (n = 211 SST-2, 5,700 AG News), as archived "
         "in results/tifs/s1/cells.csv; its truth is the generator's k, like every S1 scenario.",
         "- MC SEs are binomial per rate; maxima over cells or scenarios are selected maxima (biased upward); a rate "
         "printed as 0 had no event (95% upper bound about 3/reps).",
         "- The A3.4 cross-scored null swaps (results/tifs/addendum/cross_hits_bert_sst2.json, cross_hits_bert_agnews.json) "
         "are not a registered S9 truth condition and are not used; nor are the A2 tifs_rep_*, A8 tifs_hid_* and "
         "A10-a tifs_rcc_* runs (the frozen load_cells does not parse them). A10-b did not run.", ""]
    return L


def cost_line():
    p = os.path.join(OUT, "logs", "cpu_seconds.csv")
    if not os.path.exists(p):
        return ""
    rows = [r for r in K.read_csv(p) if r["script"] != "make_readme.py"]
    wall = sum(f(r["wall_s"]) for r in rows)
    cpu = sum(f(r["cpu_s"]) for r in rows)
    per = ", ".join(f"{r['script']} {f(r['cpu_s']) / 60:.1f}" for r in rows)
    return (f"Run cost of the scripts before this README (logs/cpu_seconds.csv): {wall / 60:.1f} min wall, "
            f"{cpu / 60:.1f} CPU-min ({per} CPU-min).")


HEAD = """# A11 items 10–12 (group g4): S1c, S9, S1 extra rows

**Status.** Post hoc / registered auxiliary tables (docs/tifs_addendum.md A11, registered 2026-10-24 18:05 KST,
items 10, 11 and 12). They do **not** change any primary verdict, battery decision or the paper-wide Holm
(31 cells). They read the existing Stage 2 predictions and the archived S1/S1b CSVs on CPU and call the frozen
functions unchanged: `analyze.load_cells / primary_table / pool_seed_variance / decide` (through `addendum.State`),
`analysis.tifs.paired_contrast / t_contrast / se_from_interval / holm`, `metrics.paired.wilson_interval /
tango_paired_ci / audit_verdict`, `analysis.calibration.Scenario / scenario_from_rates / run_scenario / tango_ci /
verdict / classify`, `scripts/tifs/s1_calibration.grid_scenarios / cell_scenarios` and
`scripts/tifs/s1b_seed_noise.simulate` (imported, never run as main). Every CSV starts with a `# ...` line that
states its definition and this status. Code: `scripts/tifs/aux/g4-s1c-s9-s1extra/` (run order in its README.md
and run_all.sh). Generated {now}.

**Data snapshot.** {n_preds} prediction files of {n_runs} runs parsed by the frozen `load_cells`; newest
{newest} (`{newest_path}`). The A2 `tifs_rep_*`, A8 `tifs_hid_*` and A10-a `tifs_rcc_*` runs are not parsed by it
and are not used{newest_any}; A10-b did not run (its registered gate fired: runs/tifs/_status/tifs_a10b_go.failed). Official
inputs: results/tifs/report/{{primary,variance,aa,secondary,factorial,cells}}.csv (frozen analyze.py) and
results/tifs/addendum/{{attack_variance,variance_diag}}.csv (addendum.py), whose SHA-256 are listed in checks.csv;
the 31 primary cells recomputed from the predictions equal results/tifs/report/primary.csv (checks.csv).
{cost}

**Revision note (promotion to the repository, 2026-10-25).** These tables replace a pre-promotion run of the same
code on the same data (2026-10-24, same seeds), after an independent verification. Changes: (1) S9 adds the
A11.12 scenarios and S1's cell set to the S1-generator tables and states the δ 0.02 boundary result (the 'all'
rows of s9_s1_summary.csv at n = 211, 5,700 and all n now include S1's cell set); (2) the A/A
ATBA-phrase rows are reported for the contaminated corpus and for both corpora together; (3) pair-level A/A counts
beside the test counts; (4) the R1 threshold is stated and the frozen HIGH-flag variant R1h added; (5) config-H
transported S1c jobs use 40,000 reps (4 seed blocks) instead of 10,000; (6) wording: the realdf reading is
compared with the full simulation cell by cell and per config (optimistic under config H for the two BERT 'feature'
batteries only, not for Qwen3 AG News), every F_a > 1 cell is listed, the k = 0.01 sparse rows are split by
discordance. Every other number is unchanged (same seeds).
"""

DEFS = """## Reproduce

CPU only; `bash scripts/tifs/aux/g4-s1c-s9-s1extra/run_all.sh [OUT]` (see that folder's README.md). Environment:
`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1`, `/data/Youngho/envs/blade/bin/python`
(the HF offline variables are exported but no tokenizer or model is loaded). Order: s1_extra.py → s1c.py → s9.py →
recheck_s1b.py → checks.py → make_readme.py. Monte Carlo seeds are fixed in the scripts: s1c 20261025–20261123 in
job-creation order (blocks 2–4 of the config-H transported jobs: seed + 1,000,000·b); s9 110000 + scenario index;
s1_extra 120000 + scenario index; recheck 777001, 777002. In the S1c loops `scipy.stats.t` is routed to
`common.ExactFastT`, which calls the same `scipy.special.stdtr / stdtrit` that scipy's t.cdf / t.sf / t.ppf call;
the frozen `t_contrast` is unchanged and its output is bit-identical (checks.csv, 'exact fast t').

## Files

| file | content |
|---|---|
| s1c_fit.csv | Item 10: plug-in model per battery and attack (n, discordance, cross-seed input correlation, √v, √ψ, s_obs, F_obs, T_H, T_E, fitted w, model vs measured pool SD, df) |
| s1c_cells.csv | Item 10: per rule × battery × config × c × cell: coverage, false 'below δ' and 'above δ' at μ = δ, P(EQUIVALENT@δ \\| μ = 0), MC SE |
| s1c_battery.csv | Item 10: per rule × battery × config × c × δ: P(false TRANSFERS \\| all μ = 0), P(NO TRANSFER \\| all μ = 0), P(false TRANSFERS \\| all μ = δ), max_j P(NO TRANSFER \\| μ_j = δ), MC SE |
| s1c_summary.csv | Item 10: compact join of the two tables above (cell extremes + battery rates) |
| s1c_jobs.csv | Item 10: simulation jobs, seed blocks and seeds, simulated df distribution vs measured df, simulated vs model vs measured s2_pool |
| s1c_blocks.csv | Item 10: per seed block of the config-H transported jobs, battery rates and cell extremes |
| s1c_check_s1b.csv | Item 10 validation against S1b (deterministic and Monte Carlo) |
| s9_s1_by_scenario.csv | Item 11: per S1-generator scenario × δ × rule: P(T), P(N), P(abstain), P(correct), P(wrong), MC SE |
| s9_s1_summary.csv | Item 11: mean (with MC SE) / min / max over scenarios by truth class or truth, scenario set and n; worst scenario |
| s9_known_truth_cells.csv | Item 11: every empirical known-truth test (item × orientation × δ × rule): value, claim, outcome |
| s9_known_truth_summary.csv | Item 11: counts by condition / sub-condition / A/A trigger (both corpora) × δ × rule, per test and per pair |
| s9_clean_corpus_side_by_side.csv | Item 11: 31 clean-corpus primary cells (primary and per-seed) and contaminated P1 − C1 cells, the judgements side by side (not scored) |
| s9_clean_corpus_counts.csv | Item 11: how many of the 31 primary cells each rule calls T (not scored) |
| s1_extra.csv | Item 12: full frozen `run_scenario` output (columns as results/tifs/s1/grid.csv) |
| s1_extra_summary.csv | Item 12: coverage, width, verdict frequencies (Tango, Newcombe, k-bootstrap; δ 0.02/0.05/0.10) with MC SE |
| checks.csv | self-checks and input fingerprints (see below) |
| logs/ | run logs, cpu_seconds.csv (wall and CPU time per script), recheck_s1b.json, key_findings.json (the key findings above as JSON) |

## Definitions

### Item 10 — S1c: calibration of the actual primary rule

*Model* (S1b's normal approximation, fitted to the measured batteries). For a seed-pooled battery with simulated
cells a and student seeds s = 1..3:
k[a,s] = μ_a + T_a(√w·z_s + √(1−w)·z_as) + e[a,s], z ~ N(0,1) iid, e ~ N(0, Σ), where Σ is the measured covariance
(ddof 1) / n of the stacked per-input d = P − C vectors of every (attack, seed) — real n, discordance, cross-seed and
cross-attack input correlation. z_s is the seed factor shared by the attacks of a seed (S1b: w = 0.5).
- Heterogeneous training SD (config **H**, the real F_a range): T_a² = max(0, s²_a,obs − ψ_a) with
  ψ_a = (tr Σ_aa − 1ᵀΣ_aa1/r)/(r − 1), the between-seed sample variance that input noise alone gives; then
  E[s²_a] = s²_a,obs (unless clipped at 0) and the model's F_a = s²_a/s²_pool equals the measured F_a.
- Homogeneous (config **E**, as S1b): T_a² = mean of the pooled attacks' T_a² (SOS SST-2 keeps its own T).
- w is fitted so that the model's E[MS_seed]/E[MS_res] equals the measured ms_seed/ms_res of the frozen pool
  (clipped to [0, 1]); config **H05** = H with S1b's w = 0.5.
- Battery shapes: BERT SST-2 4 pooled attacks (AddSent, ATBA, BadNets, SynBkd) + SOS (seed-pooled, borrows the pool)
  + EP (structural null, fixed seed-42 Tango row); BERT AG News 6 pooled; Qwen3 SST-2 3; GPT-2 SST-2 2; r = 3.

*Analyst* = the frozen code on each simulated battery: s²_pool and Satterthwaite df from `analyze.pool_seed_variance`
over the pool's attacks; SE² = s²_pool/r + v_a with v_a = 1ᵀΣ_aa1/r² (= the measured var_i(d̄)/n, treated as known as
in S1b); interval, verdicts and one-sided p from `t_contrast`; battery decision from `analyze.decide`
(Holm over p_pos via `holm`: TRANSFERS iff a Holm-adjusted p_pos < 0.025; NO TRANSFER iff every one-sided upper bound
< δ) over the real battery's qualified cells.

*Transported batteries* (one seed: BERT SST-2 'feature' ← BERT SST-2 logit pool; BERT AG News 'feature' ← BERT AG News
logit pool; Qwen3 AG News logit ← Qwen3 SST-2 logit pool): k'_a = μ_a + T'_a(√w·z' + √(1−w)·z'_a) + e'_a with e' from the
measured seed-42 covariance; SE² = SE²_Tango,a + s²_pool with the cell's real Tango SE (`se_from_interval`).
Two readings of the borrowed pool: **transported** — s²_pool and its Satterthwaite df from an independently simulated
source battery (the full rule); **transported_realdf** — S1b's transported rule with its assumed df_τ = 12 replaced
by the measured pool df: s²_pool = E[s²_pool]·χ²_df/df at df = 4.67 (BERT SST-2), 9.83 (BERT AG News), 5.89 (Qwen3 SST-2).
*Variance misspecification c*: T'_a = c·√E[s²_pool] (E) or c·√(F_a·E[s²_pool]) (H, F_a of the same attack in the
family's logit battery; Qwen3 AG News uses the Qwen3 SST-2 F_a). c is the SD ratio (as `addendum.critical_ratio`);
c ∈ {0.5, 0.707, 1, 1.225, 1.414, 1.5, 2} covers the SD reading c ∈ {0.5, 1, 1.5, 2} and the variance reading
c² ∈ {0.5, 1, 1.5, 2}.

*Metrics* (μ is a pure shift, so one noise draw serves every μ): coverage P(lo ≤ μ_a ≤ hi); false 'below δ' at
μ_a = δ, P(hi < δ) (= P(hi < μ_a), the same for every δ; registered); false 'above δ' at μ_a = δ, P(lo > δ);
P(EQUIVALENT@δ | μ = 0); battery: P(TRANSFERS | all μ = 0) (registered false TRANSFERS), P(NO TRANSFER | all μ = 0),
P(Holm p_pos < 0.025 | all simulated μ = δ) and max_j P(NO TRANSFER | μ_j = δ, others 0). Reps: pooled 20,000;
transported config H 40,000 (4 blocks of 10,000); transported config E and transported_realdf 10,000; S1b check
20,000. MC SE = √(p(1−p)/reps); a rate shown as 0.0000 ± 0.0000 had no event in its reps (95% upper bound ≈ 3/reps).

### Item 11 — S9: stage-wise judgements scored only against known truth

Each rule maps a (suspect, reference) pair to a claim about teacher-attributable backdoor transfer beyond δ:
T (transfer), N (no transfer) or A (abstain).
- **R1 raw ASR**: T iff the suspect's triggered target rate ≥ 0.5 ("backdoor present"), else N (no reference, no δ).
  The 0.5 point threshold is this table's choice; docs/tifs_plan.md S9 names the raw-ASR stage without a threshold.
- **R1h raw ASR, HIGH flag** (sensitivity): T iff the frozen `audit_verdict` exposure flag is HIGH, i.e. the Wilson 95%
  lower bound of the suspect's rate ≥ 0.5. At the clean-corpus primary level: HIGH in every seed of the cell.
- **R2 |P − C|**: T iff |k̂| ≥ δ (unsigned point estimate), else N.
- **R3 signed P − C**: T iff k̂ ≥ δ, else N.
- **R4 CI excludes 0**: T iff the 95% Tango interval has lo > 0 (signed + CI), else N (the two-sided flag lo > 0 or
  hi < 0 is reported, not scored).
- **R5 audit** (equivalence-aware, frozen verdict@δ on the 95% Tango interval): POSITIVE → T; EQUIVALENT or NEGATIVE → N;
  INCONCLUSIVE → A. **R5b**: the same verdict read as POSITIVE → T, otherwise → N. **R5t** (empirical single-seed items
  only, sensitivity): R5 on the transported interval (SE² = SE²_Tango + s²_pool of the family's logit pool, else SST-2;
  pool df), as `analyze.secondary_table`.
Outcome = correct if the claim equals the truth, wrong if it is the other definite claim, abstain for A.

*Truth, only where known*:
- S1 generator scenarios: the generator's k; at δ, truth T iff k > δ, else N (classes null k = 0, within 0 < |k| ≤ δ,
  boundary k = δ, beyond k > δ, negative beyond k < −δ). Sets: the S1 grid (120 scenarios of
  `s1_calibration.grid_scenarios`, n ∈ {211, 570, 912, 5700}) with its sign-mirrored k < 0 twins ('grid+mirror'); the
  item-12 scenarios ('a11.12'); S1's cell set ('s1_cells': `s1_calibration.cell_scenarios(0.05)`, the 142 plug-in,
  null-twin and edge-twin (k = 0.075) scenarios of results/tifs/s1/cells.csv, from the 48 v1 seed-42 cells, n = 211 /
  5,700, plus the edge twins of tolerances 0.02 and 0.10, k = 0.03 and 0.15). Tables ~ Multinomial(n, Scenario.probs);
  intervals from the frozen vectorised `tango_ci`, verdicts from `verdict`. 20,000 reps per scenario.
- EP SST-2 structural null (logit and 'feature', seed 42): P and C hit vectors identical by construction → truth N.
- Positive controls kl01, kl10 (BERT SST-2 and AG News, logit): label-kept triggered corpus rows, the only path to the
  target mapping is the poisoned teacher → truth T (existence).
- Contaminated-corpus CE-only R1 (no teacher; BadNets 10% BERT/Qwen3 SST-2/AG News, AddSent 10% BERT): truth N for any
  teacher-transfer claim; only R1 and R1h are applicable (no teacher swap exists, R2–R5 are n/a).
- A/A backdoor-free swaps: every pair of independent clean teacher builds (t42 = battery/grid seed-42 C, t43/t44 =
  A/A C'), same corpus, student seed 42, same triggered inputs, both orientations; truth N (neither teacher is
  poisoned). Sub-conditions: clean or contaminated (badnets10) corpus × BadNets trigger or ATBA phrase (t42: the
  battery's ATBA C on the ATBA-triggered inputs; t43/t44: the A/A C' on the `atba_phrase_<ds>` control); level
  'trigger' pools the two corpora. Counts per test and per pair (items_wrong = pairs with a wrong claim in some
  orientation).
- Clean-corpus battery cells and contaminated P1 − C1 cells: listed side by side, **not scored** (truth unknown).

### Item 12 — S1 extra rows

Frozen `run_scenario` (reps 10,000, bootstrap R 2,000, tol_h 0.2, tols_k (0.02, 0.05, 0.10), as results/tifs/s1/meta.json),
scenario seeds 120000 + index:
- **boundary**: n = 912, k = δ = 0.02 (`scenario_from_rates`), clean rate c ∈ {0.05, 0.10, 0.50, 0.90}, discordance
  d ∈ {0.02, 0.03, 0.06, 0.10, 0.20}, feasible combinations ({n_boundary}). Read P(EQUIVALENT@0.02) as the
  false-equivalence rate and P(POSITIVE@0.02) as the false-positive rate at the boundary (each nominally ≤ 0.025);
  `classify()`'s false_* columns depend on the floating-point sign of k − 0.02 here, so read equivalent@0.02 /
  positive@0.02 directly.
- **sparse**: n ∈ {912, 5700}, c ∈ {0.05, 0.50}, d ∈ {0.005, 0.01, 0.02}, k ∈ {d/2, d} (k = d: every discordant pair in
  one direction); the two n = 912, d = k = 0.02 scenarios are also boundary scenarios (set "boundary+sparse").
- **real**: plug-in tables of the measured clean-corpus battery cells (every seed, logit and 'feature') with
  discordance ≤ 0.03 and n10 ≠ n01 (truth = the observed table).
"""


def newest_any(snap):
    """The newest prediction file under runs/tifs whether or not the frozen load_cells parses it."""
    paths = glob.glob(os.path.join(K.RUNS, "tifs_*", "artifacts", "results", "*.preds.json"))
    p = max(paths, key=os.path.getmtime)
    if os.path.relpath(p, K.R) == snap["newest_path"]:
        return ""
    ts = datetime.datetime.fromtimestamp(os.path.getmtime(p), datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (f" (the newest of all {len(paths)} prediction files under runs/tifs, {ts}, belongs to "
            f"{p.split(os.sep)[-4]})")


def keep_last_readme_cost(out):
    """README regenerations replace make_readme.py's earlier row in logs/cpu_seconds.csv (they do not accumulate)."""
    p = os.path.join(out, "logs", "cpu_seconds.csv")
    if not os.path.exists(p):
        return
    with open(p) as fh:
        lines = fh.readlines()
    keep = [ln for ln in lines if not ln.startswith("make_readme.py,")]
    if len(keep) != len(lines):
        with open(p, "w") as fh:
            fh.writelines(keep)


def write_logs_readme(snap):
    """logs/README.md: what the run logs are (mirrors group g1's logs/README.md)."""
    d = os.path.join(OUT, "logs")
    logs = sorted(x for x in os.listdir(d) if x != "README.md")
    L = ["# Run logs (A11 items 10, 11, 12; group g4)", "",
         "Standard output and error of the scripts in scripts/tifs/aux/g4-s1c-s9-s1extra/, one .log file per run_all.sh "
         "step, plus run bookkeeping: cpu_seconds.csv (wall and CPU seconds per script), run_start.txt, "
         "recheck_s1b.json (the S1b-mode recheck) and key_findings.json (the README's key findings). They belong to the "
         "A11 auxiliary (post hoc) tables of the parent folder, which **change no primary verdict, battery decision or "
         "the paper-wide Holm**.", "",
         f"**Data snapshot.** Newest parsed preds file {snap['newest_utc']} ({snap['n_preds']} preds files in "
         f"{snap['n_runs']} runs).", "", "Files: " + ", ".join(logs) + ".", ""]
    with open(os.path.join(d, "README.md"), "w") as fh:
        fh.write("\n".join(L))


def main():
    global OUT
    ap = K.add_out_arg(argparse.ArgumentParser())
    args = ap.parse_args()
    OUT = args.out
    t_start = time.time()
    D = Data()
    snap = K.snapshot()
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
    head = (HEAD.replace("{now}", now).replace("{n_preds}", str(snap["n_preds"])).replace("{n_runs}", str(snap["n_runs"]))
            .replace("{newest}", snap["newest_utc"]).replace("{newest_path}", snap["newest_path"])
            .replace("{newest_any}", newest_any(snap))
            .replace("{cost}", cost_line()).replace("{{", "{").replace("}}", "}"))
    n_boundary = sum(1 for r in D.s1x if r["set"] in ("boundary", "boundary+sparse"))
    defs = DEFS.replace("{n_boundary}", str(n_boundary))
    kf = kf_s1c(D) + kf_s9(D) + kf_s1x(D)
    L = [head, "## Key findings", ""] + [f"{i}. {t}" for i, t in enumerate(kf, 1)] + ["", defs, "## Key numbers", ""]
    L += s1c_tables(D)
    L += s9_tables(D)
    L += s1x_table(D)
    L += checks_section(D)
    L += caveats(D)
    with open(os.path.join(OUT, "README.md"), "w") as fh:
        fh.write("\n".join(L))
    with open(os.path.join(OUT, "logs", "key_findings.json"), "w") as fh:
        json.dump(kf, fh, indent=1, ensure_ascii=False)
    write_logs_readme(snap)
    print("wrote", os.path.join(OUT, "README.md"), "and logs/README.md")
    keep_last_readme_cost(OUT)
    K.log_cost(OUT, "make_readme.py", t_start)


if __name__ == "__main__":
    main()
