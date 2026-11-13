#!/usr/bin/env python3
"""A11 item 11 — S9 (registered): stage-wise judgements scored only where the truth is known.

Post hoc / registered auxiliary analysis (docs/tifs_addendum.md A11.11,
docs/tifs_plan.md S9). It does not change any primary verdict, battery decision
or the paper-wide Holm.

Every rule turns a (suspect student, reference student) pair -- or, for R1, the
suspect alone -- into a claim about teacher-attributable backdoor transfer beyond
the tolerance delta: T (transfer), N (no transfer) or A (abstain):

  R1 raw ASR      T iff the suspect's triggered target rate >= 0.5 ("backdoor present"),
                  else N (no delta, no reference). The 0.5 point threshold is this group's
                  choice: docs/tifs_plan.md S9 names the raw-ASR stage but registers no
                  threshold.
  R1h raw ASR,    sensitivity variant of R1: T iff the frozen audit_verdict exposure flag
      HIGH flag   is HIGH, i.e. the Wilson 95% lower bound of the suspect's rate >= 0.5
                  (metrics.paired.wilson_interval), else N
  R2 |P - C|      T iff |k_hat| >= delta, else N           (unsigned point estimate)
  R3 P - C        T iff k_hat >= delta, else N             (signed point estimate)
  R4 CI excl. 0   T iff the 95% Tango interval of k has lo > 0, else N (signed + CI;
                  the two-sided flag lo > 0 or hi < 0 is reported, not scored)
  R5 audit        frozen verdict@delta (Tango 95%): POSITIVE -> T; EQUIVALENT or
                  NEGATIVE -> N; INCONCLUSIVE -> A (abstain)
  R5b audit,      the same verdict collapsed to two outcomes: POSITIVE -> T, otherwise
      binary      (EQUIVALENT / INCONCLUSIVE / NEGATIVE) -> N
  R5t audit,      R5 with the transported interval (SE^2 = SE_Tango^2 + s2_pool of the
      transported family's logit pool, else SST-2; pool df), as analyze.secondary_table
                  (empirical single-seed items only; sensitivity)

Truth (only where it is known):
  S1 generator scenarios: the generator's k; at delta, truth T iff k > delta, else N
      (classes: null k = 0; within 0 < |k| <= delta; boundary k = delta; beyond k > delta;
      negative beyond k < -delta). Sets: the S1 grid (scripts/tifs/s1_calibration.
      grid_scenarios, 120), its sign-mirrored k < 0 twins, the A11.12 scenarios of
      s1_extra.py (n = 912 boundary k = delta = 0.02; sparse discordance with k != 0;
      plug-in tables of the measured sparse cells) and S1's cell set (s1_calibration.
      cell_scenarios(0.05) = results/tifs/s1/cells.csv: plug-in, null twin and edge twin
      k = 0.075 of the 48 v1 cells, n = 211 / 5700) with the edge twins of the other two
      tolerances (k = 0.03, 0.15) added. Tables are drawn from Scenario.probs (frozen
      generator), intervals by the frozen calibration.tango_ci / verdict. Scenario seeds
      110000 + list index; the S1-cell sets come last, so earlier seeds are unchanged.
  EP SST-2 structural null: P and C students identical by construction (k = 0): truth N.
  Positive controls kl01, kl10 (label-kept triggered corpus rows, teacher-mediated channel
      designed in): truth T (existence; observed k 0.92-0.97, far beyond every delta).
  R1 (contaminated corpus, CE-only, no teacher): truth N for teacher attribution; only R1 is
      applicable (no teacher swap exists, so R2-R5 are n/a).
  A/A backdoor-free swaps: two independent clean teacher builds (t42 battery/grid C, A/A C'
      t43, t44; same corpus, student seed 42, same inputs), both orientations: truth N for
      teacher-attributable *backdoor* transfer (neither teacher is poisoned). Sub-conditions:
      clean / contaminated corpus x BadNets trigger / ATBA phrase; level 'trigger' pools the
      two corpora. A teacher-swap effect that is not a backdoor (different clean builds)
      still counts as a false transfer claim. Counts are given per test (both orientations)
      and per pair (items_wrong = pairs with a wrong claim in some orientation).
Clean-corpus battery cells (and contaminated P1 - C1) are listed side by side, not scored.

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" python s9.py [--out DIR]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import common as K  # noqa: E402
from counterfactual_kd.analysis.calibration import (  # noqa: E402  (frozen)
    Scenario, classify, scenario_from_rates, tango_ci, verdict,
)
from counterfactual_kd.metrics.paired import tango_paired_ci, wilson_interval  # noqa: E402

import s1_calibration as S1  # noqa: E402  (frozen script; grid_scenarios only)
import s1_extra as S1X  # noqa: E402  (this group's A11.12 scenario sets)

A = K.A
TOLS = K.TOLS
RAW = 0.5
EPS = 1e-9
HIGH = 0.5          # frozen audit_verdict high_exposure (Wilson lower bound)
RULES = ("R1_raw_asr", "R1h_raw_asr_high", "R2_abs_point", "R3_signed_point", "R4_ci_excludes_0", "R5_audit",
         "R5b_audit_binary")
PAIR_RULES = ("R2_abs_point", "R3_signed_point", "R4_ci_excludes_0", "R5_audit", "R5b_audit_binary",
              "R5t_audit_transported")


def claim_of_verdict(v):
    return "T" if v == "POSITIVE" else ("A" if v == "INCONCLUSIVE" else "N")


def truth_class(k, delta):
    if k > delta + EPS:
        return "beyond (k > delta)", "T"
    if abs(k - delta) <= EPS:
        return "boundary (k = delta)", "N"
    if k < -delta - EPS:
        return "negative beyond (k < -delta)", "N"
    if abs(k) <= 1e-12:
        return "null (k = 0)", "N"
    return "within (0 < |k| <= delta)", "N"


# ── S1 generator scenarios ───────────────────────────────────────────


def s1_scenarios(st):
    grid = S1.grid_scenarios()
    out = [("grid", sc) for sc in grid]
    for sc in grid:
        if sc.k > 1e-12:
            c = sc.pi11 + sc.pi01
            d = sc.pi10 + sc.pi01
            neg = scenario_from_rates(sc.name + "_neg", sc.n, c, -sc.k, d)
            if min(neg.pi11, neg.pi00) >= 0:
                out.append(("grid_mirror", neg))
    for name, (label, sc) in sorted(S1X.scenarios(st).items()):
        out.append((f"a11.12_{label}", sc))
    # S1's cell set as archived (cell_scenarios(0.05) = results/tifs/s1/cells.csv) ...
    for sc in S1.cell_scenarios(0.05):
        out.append(("s1_cells", sc))
    # ... and the edge twins (k = 1.5 tol) of the other two tolerances
    for tol in (0.02, 0.10):
        for sc in S1.cell_scenarios(tol):
            if sc.name.endswith("|edge"):
                out.append(("s1_cells_edge_extra", Scenario(f"{sc.name}@tol{tol:g}", sc.n, sc.pi11, sc.pi10, sc.pi01)))
    return out


def set_group(sset):
    if sset in ("grid", "grid_mirror"):
        return "grid+mirror"
    if sset.startswith("a11.12"):
        return "a11.12"
    if sset.startswith("s1_cells"):
        return "s1_cells"
    raise ValueError(sset)


def wilson_lo_of_counts(x, n):
    """Frozen wilson_interval lower bound for every count in x (computed once per distinct count)."""
    ux, inv = np.unique(x, return_inverse=True)
    lo = np.array([wilson_interval(int(v), int(n))[0] for v in ux])
    return lo[inv.reshape(-1)]


def s1_rows(st, reps=20000, seed0=110000):
    rows = []
    for i, (sset, sc) in enumerate(s1_scenarios(st)):
        rng = np.random.default_rng(seed0 + i)
        tab = rng.multinomial(sc.n, sc.probs, size=reps)
        n11, n10, n01, n00 = tab.T
        n = sc.n
        P = (n11 + n10) / n
        diff = (n10 - n01).astype(float)
        lo, hi = tango_ci(n10, n01, n)
        wl = wilson_lo_of_counts(n11 + n10, n)
        c_rate = sc.pi11 + sc.pi01
        for t in TOLS:
            v = verdict(lo, hi, t)
            claims = {
                "R1_raw_asr": np.where(P >= RAW, "T", "N"),
                "R1h_raw_asr_high": np.where(wl >= HIGH, "T", "N"),
                "R2_abs_point": np.where(np.abs(diff) >= t * n - EPS, "T", "N"),
                "R3_signed_point": np.where(diff >= t * n - EPS, "T", "N"),
                "R4_ci_excludes_0": np.where(lo > 0, "T", "N"),
                "R5_audit": np.where(v == "POSITIVE", "T", np.where(v == "INCONCLUSIVE", "A", "N")),
                "R5b_audit_binary": np.where(v == "POSITIVE", "T", "N"),
            }
            cls, truth = truth_class(sc.k, t)
            for rule, cl in claims.items():
                pT, pN, pA = (float(np.mean(cl == x)) for x in ("T", "N", "A"))
                correct = pT if truth == "T" else pN
                wrong = pN if truth == "T" else pT
                row = {"set": sset, "scenario": sc.name, "n": n, "C_rate": c_rate, "P_rate": c_rate + sc.k,
                       "discordance": sc.pi10 + sc.pi01, "true_k": sc.k, "delta": t,
                       "truth_class": cls, "truth": truth, "rule": rule, "reps": reps,
                       "p_claim_T": pT, "p_claim_N": pN, "p_abstain": pA,
                       "p_correct": correct, "p_wrong": wrong,
                       "p_wrong_mcse": K.mc_se(wrong, reps)}
                if rule == "R5_audit":
                    cf = classify(v, sc.k, t)
                    row.update({f"audit_{k}": cf[k] for k in ("equivalent", "positive", "negative",
                                                               "inconclusive", "false_equiv", "false_dir",
                                                               "power_equiv", "power_dir")})
                if rule == "R4_ci_excludes_0":
                    row["p_two_sided_excludes_0"] = float(np.mean((lo > 0) | (hi < 0)))
                rows.append(row)
    return rows


def s1_summary(rows):
    groups = {}
    for r in rows:
        for sset in (set_group(r["set"]), "all"):
            for nkey in (str(r["n"]), "all"):
                groups.setdefault(("truth class", r["rule"], r["delta"], r["truth_class"], sset, nkey), []).append(r)
                groups.setdefault(("truth", r["rule"], r["delta"], f"truth {r['truth']}", sset, nkey), []).append(r)
    out = []
    for (grouping, rule, t, cls, sset, nkey), rs in sorted(groups.items()):
        row = {"grouping": grouping, "rule": rule, "delta": t, "truth_class": cls, "truth": rs[0]["truth"],
               "scenario_set": sset, "n": nkey, "n_scenarios": len(rs), "reps_each": rs[0]["reps"]}
        for m in ("p_claim_T", "p_claim_N", "p_abstain", "p_wrong"):
            vals = [x[m] for x in rs]
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_min"] = float(np.min(vals))
            row[f"{m}_max"] = float(np.max(vals))
        # MC SE of the mean over scenarios (independent draws per scenario)
        row["p_wrong_mean_mcse"] = math.sqrt(sum(x["p_wrong"] * (1 - x["p_wrong"]) / x["reps"] for x in rs)) / len(rs)
        worst = max(rs, key=lambda x: x["p_wrong"])
        row["worst_scenario"] = worst["scenario"]
        row["worst_p_wrong_mcse"] = worst["p_wrong_mcse"]
        row["worst_true_k"] = worst["true_k"]
        row["worst_truth_class"] = worst["truth_class"]
        out.append(row)
    return out


def tango_vector_check(n_tables=400, seed=5):
    """frozen vectorised calibration.tango_ci vs scalar metrics.paired.tango_paired_ci."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for n in (211, 912, 5700):
        for _ in range(n_tables // 3):
            p = rng.dirichlet([1, 0.3, 0.3, 1])
            n11, n10, n01, n00 = rng.multinomial(n, p)
            lo_v, hi_v = tango_ci(np.array([n10]), np.array([n01]), n)
            a = [1] * n11 + [1] * n10 + [0] * n01 + [0] * n00
            b = [1] * n11 + [0] * n10 + [1] * n01 + [0] * n00
            lo_s, hi_s = tango_paired_ci(a, b)
            worst = max(worst, abs(float(lo_v[0]) - lo_s), abs(float(hi_v[0]) - hi_s))
    return worst


# ── empirical known-truth items ──────────────────────────────────────


def pool_for_transport(st, fam, ds):
    for key in ((fam, ds, "logit"), (fam, "sst2", "logit")):
        if key in st.pools:
            return key, st.pools[key]
    return None, None


def pair_rows(st, cond, sub, item, fam, ds, suspect, reference, truth, orientation="", pair="", note="",
              only_r1=False, extra=None, trigger=""):
    n = len(suspect)
    p_rate = sum(suspect) / n
    lo_w, hi_w = wilson_interval(sum(suspect), n)
    rows = []
    if not only_r1:
        assert len(reference) == n, (item, n, len(reference))
        pc = K.paired_contrast(suspect, reference, TOLS)
        pkey, pool = pool_for_transport(st, fam, ds)
        tr = None
        if pool is not None:
            se = math.sqrt(K.se_from_interval(pc["lo"], pc["hi"]) ** 2 + pool["s_between"] ** 2)
            tr = K.t_contrast(pc["k"], se, pool["df"], TOLS)
    for t in TOLS:
        base = {"condition": cond, "subcondition": sub, "trigger": trigger, "item": item, "pair": pair or item,
                "fam": fam, "dataset": ds, "orientation": orientation,
                "n": n, "suspect_rate": p_rate, "suspect_wilson_lo": lo_w, "suspect_wilson_hi": hi_w,
                "delta": t, "truth": truth, "note": note}
        if extra:
            base.update(extra)
        claims = {"R1_raw_asr": "T" if p_rate >= RAW else "N",
                  "R1h_raw_asr_high": "T" if lo_w >= HIGH else "N"}
        vals = {"R1_raw_asr": p_rate, "R1h_raw_asr_high": lo_w}
        if only_r1:
            for rname in PAIR_RULES:
                claims[rname] = "n/a"
        else:
            base.update(reference_rate=pc["b"], k=pc["k"], lo=pc["lo"], hi=pc["hi"],
                        delta_star=pc["delta_star"], n10=pc["n10"], n01=pc["n01"],
                        two_sided_excludes_0=pc["excludes_zero"],
                        verdict_tango=pc[f"verdict@{t:g}"])
            claims["R2_abs_point"] = "T" if abs(pc["k"]) >= t - 1e-12 else "N"
            claims["R3_signed_point"] = "T" if pc["k"] >= t - 1e-12 else "N"
            claims["R4_ci_excludes_0"] = "T" if pc["lo"] > 0 else "N"
            claims["R5_audit"] = claim_of_verdict(pc[f"verdict@{t:g}"])
            claims["R5b_audit_binary"] = "T" if pc[f"verdict@{t:g}"] == "POSITIVE" else "N"
            vals.update(R2_abs_point=abs(pc["k"]), R3_signed_point=pc["k"], R4_ci_excludes_0=pc["lo"])
            if tr is not None:
                claims["R5t_audit_transported"] = claim_of_verdict(tr[f"verdict@{t:g}"])
                base.update(transported_lo=tr["lo"], transported_hi=tr["hi"],
                            transported_from="/".join(pkey), verdict_transported=tr[f"verdict@{t:g}"])
        for rname, cl in claims.items():
            if cl == "n/a":
                outcome = "n/a"
            elif cl == "A":
                outcome = "abstain"
            else:
                outcome = "correct" if cl == truth else "wrong"
            rows.append(dict(base, rule=rname, value=vals.get(rname), claim=cl, outcome=outcome))
    return rows


def empirical_rows(st):
    rows = []
    cells = st.cells
    # EP SST-2 structural null (k = 0 by construction)
    for kd in ("logit", "feature"):
        c = st.groups.get(("bert", "sst2", kd, "ep"), {}).get(42)
        if c is not None:
            assert c["P"] == c["C"], "structural null with P != C"
            rows += pair_rows(st, "EP SST-2 structural null", "clean corpus, EP trigger",
                              f"bert/sst2/{kd}/ep/s42", "bert", "sst2",
                              c["P"], c["C"], "N", note="P and C students identical by construction",
                              trigger="EP trigger")
    # positive controls
    for c in cells:
        if c["block"] == "PC" and not c["single"] and c["kd"] == "logit":
            r = st.refs.get(("PC", c["fam"], c["dataset"], c["attack"], c["corpus"], c["seed"]))
            rows += pair_rows(st, "positive control", f"{c['corpus']} (label-kept triggered rows)",
                              f"{c['fam']}/{c['dataset']}/{c['corpus']}/{c['attack']}",
                              c["fam"], c["dataset"], c["P"], c["C"], "T",
                              note="label-kept triggered corpus rows; truth = existence (magnitude unknown a priori)",
                              extra={"R_rate": (sum(r["P"]) / len(r["P"])) if r else None},
                              trigger=f"{c['attack']} trigger")
    # R1: contaminated corpus, CE-only student (no teacher)
    for c in cells:
        if c["block"] == "G" and c["corpus"] != "clean" and c["kd"] == "ce" and c["single"]:
            p1 = [x for x in cells if x["block"] == "G" and x["corpus"] == c["corpus"] and x["fam"] == c["fam"]
                  and x["dataset"] == c["dataset"] and x["attack"] == c["attack"] and x["kd"] == "logit"
                  and x["seed"] == c["seed"] and not x["single"]]
            extra = {}
            if p1:
                pc = K.paired_contrast(p1[0]["P"], p1[0]["C"], TOLS)
                extra = {"context_P1_rate": pc["a"], "context_C1_rate": pc["b"], "context_P1_minus_C1": pc["k"],
                         "context_P1_minus_C1_lo": pc["lo"], "context_P1_minus_C1_hi": pc["hi"]}
            rows += pair_rows(st, "R1 contaminated corpus, no teacher", f"{c['corpus']} corpus, CE-only",
                              f"{c['fam']}/{c['dataset']}/{c['corpus']}/{c['attack']}/R1", c["fam"], c["dataset"],
                              c["P"], None, "N", only_r1=True,
                              note="no teacher: any teacher-transfer claim is false; R2-R5 need a teacher swap (n/a); "
                                   "context_* = the logit P1 - C1 of the same corpus (truth unknown, not scored)",
                              extra=extra, trigger=f"{c['attack']} trigger")
    # A/A backdoor-free swaps among independent clean builds
    builds = {}
    for c in cells:
        if c["single"] or c["kd"] != "logit":
            continue
        if c["block"] == "G" and c["seed"] == 42 and c["attack"] == "badnets" and c["teacher_seed"] == 42:
            builds.setdefault((c["fam"], c["dataset"], c["corpus"], "badnets"), {})[42] = c["C"]
        if c["block"] == "AA" and c["attack"] == "badnets":
            builds.setdefault((c["fam"], c["dataset"], c["corpus"], "badnets"), {})[c["teacher_seed"]] = c["C"]
            ctl = c["controls"].get(f"atba_phrase_{c['dataset']}")
            if ctl is not None:
                builds.setdefault((c["fam"], c["dataset"], c["corpus"], "atba phrase"), {})[c["teacher_seed"]] = ctl["c"]
        if (c["block"] == "G" and c["seed"] == 42 and c["attack"] == "atba" and c["corpus"] == "clean"
                and c["teacher_seed"] == 42):
            builds.setdefault((c["fam"], c["dataset"], c["corpus"], "atba phrase"), {})[42] = c["C"]
    for (fam, ds, corpus, trig), by_t in sorted(builds.items()):
        ts = sorted(by_t)
        sub = (("clean corpus" if corpus == "clean" else f"contaminated corpus ({corpus})") + ", "
               + ("BadNets trigger" if trig == "badnets" else "ATBA phrase"))
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = ts[i], ts[j]
                pair = f"{fam}/{ds}/{corpus}/{trig}/C_t{a}~C_t{b}"
                for (s_t, r_t) in ((a, b), (b, a)):
                    rows += pair_rows(st, "A/A backdoor-free swap", sub, pair,
                                      fam, ds, by_t[s_t], by_t[r_t], "N", orientation=f"C_t{s_t} - C_t{r_t}",
                                      pair=pair, trigger=("BadNets trigger" if trig == "badnets" else "ATBA phrase"),
                                      note=("t42 = battery/grid seed-42 C on the ATBA-triggered inputs; t43/t44 = "
                                            "A/A C' on the ATBA phrase control" if trig == "atba phrase" and 42 in (s_t, r_t)
                                            else "clean teacher builds only"))
    return rows


def empirical_summary(rows):
    out = []
    for level in ("condition", "subcondition", "trigger"):
        groups = {}
        for r in rows:
            if level == "trigger":
                if r["condition"] != "A/A backdoor-free swap":
                    continue
                sub = f"{r['trigger']} (clean + contaminated corpus)"
            else:
                sub = r["subcondition"] if level == "subcondition" else "(all)"
            key = (r["condition"], sub, r["delta"], r["rule"])
            groups.setdefault(key, []).append(r)
        for (cond, sub, t, rule), rs in sorted(groups.items()):
            cnt = {x: sum(1 for r in rs if r["claim"] == x) for x in ("T", "N", "A", "n/a")}
            oc = {x: sum(1 for r in rs if r["outcome"] == x) for x in ("correct", "wrong", "abstain", "n/a")}
            pairs = sorted({r["pair"] for r in rs})
            flagged = sorted({r["pair"] for r in rs if r["claim"] == "T"})
            wrong_pairs = sorted({r["pair"] for r in rs if r["outcome"] == "wrong"})
            out.append({"level": level, "condition": cond, "subcondition": sub, "delta": t, "rule": rule,
                        "truth": rs[0]["truth"], "n_tests": len(rs), "n_items": len(pairs),
                        "claim_T": cnt["T"], "claim_N": cnt["N"], "abstain": cnt["A"], "not_applicable": cnt["n/a"],
                        "correct": oc["correct"], "wrong": oc["wrong"],
                        "items_with_a_T_claim": len(flagged), "items_wrong": len(wrong_pairs),
                        "wrong_items": ";".join(sorted({f"{r['item']}[{r['orientation']}]" if r["orientation"] else r["item"]
                                                        for r in rs if r["outcome"] == "wrong"}))})
    return out


# ── clean-corpus side by side (not scored) ───────────────────────────


def _claims(raw, k, lo, verdict_t, t, wilson_lo):
    return {"R1_raw_asr": "T" if raw >= RAW else "N",
            "R1h_raw_asr_high": "T" if wilson_lo >= HIGH else "N",
            "R2_abs_point": "T" if abs(k) >= t - 1e-12 else "N",
            "R3_signed_point": "T" if k >= t - 1e-12 else "N",
            "R4_ci_excludes_0": "T" if lo > 0 else "N",
            "R5_audit": claim_of_verdict(verdict_t),
            "R5b_audit_binary": "T" if verdict_t == "POSITIVE" else "N"}


def _wlo(hits):
    return wilson_interval(sum(hits), len(hits))[0]


def side_by_side(st):
    """Primary level: R1 on the frozen raw_asr; R1h = HIGH flag in every seed of the cell
    (the smallest per-seed Wilson lower bound >= 0.5). Seed level: the seed's own values."""
    rows = []
    for key, pr in sorted(st.prim.items()):
        fam, ds, kd, atk = key
        seeds = [int(x) for x in str(pr["seeds"]).split(",")]
        wl_min = min(_wlo(st.groups[key][s]["P"]) for s in seeds)
        for t in TOLS:
            rows.append({"block": "clean-corpus primary cell", "fam": fam, "dataset": ds, "kd": kd, "attack": atk,
                         "level": "primary", "method": pr["method"], "seeds": pr["seeds"], "delta": t,
                         "raw_asr": pr["raw_asr"], "C_rate": pr["C_trig"], "k": pr["k"], "lo": pr["lo"], "hi": pr["hi"],
                         **_claims(pr["raw_asr"], pr["k"], pr["lo"], pr[f"verdict@{t:g}"], t, wl_min),
                         "audit_verdict": pr[f"verdict@{t:g}"], "suspect_wilson_lo": wl_min, "scored": False})
        for s, c in sorted(st.groups[key].items()):
            pc = K.paired_contrast(c["P"], c["C"], TOLS)
            wl = _wlo(c["P"])
            for t in TOLS:
                rows.append({"block": "clean-corpus primary cell", "fam": fam, "dataset": ds, "kd": kd, "attack": atk,
                             "level": f"seed {s} (Tango)", "method": "per-seed Tango", "seeds": str(s), "delta": t,
                             "raw_asr": pc["a"], "C_rate": pc["b"], "k": pc["k"], "lo": pc["lo"], "hi": pc["hi"],
                             **_claims(pc["a"], pc["k"], pc["lo"], pc[f"verdict@{t:g}"], t, wl),
                             "audit_verdict": pc[f"verdict@{t:g}"], "suspect_wilson_lo": wl, "scored": False})
    for c in st.cells:
        if c["block"] == "G" and c["corpus"] != "clean" and not c["single"]:
            pc = K.paired_contrast(c["P"], c["C"], TOLS)
            wl = _wlo(c["P"])
            for t in TOLS:
                rows.append({"block": "contaminated corpus P1 - C1", "fam": c["fam"], "dataset": c["dataset"],
                             "kd": c["kd"], "attack": c["attack"], "corpus": c["corpus"],
                             "level": f"seed {c['seed']} (Tango)", "method": "per-seed Tango", "seeds": str(c["seed"]),
                             "delta": t, "raw_asr": pc["a"], "C_rate": pc["b"], "k": pc["k"], "lo": pc["lo"],
                             "hi": pc["hi"], **_claims(pc["a"], pc["k"], pc["lo"], pc[f"verdict@{t:g}"], t, wl),
                             "audit_verdict": pc[f"verdict@{t:g}"], "suspect_wilson_lo": wl, "scored": False})
    return rows


def side_by_side_counts(rows):
    """Primary-level claim counts per rule and delta over the 31 cells (descriptive)."""
    out = []
    prim = [r for r in rows if r["level"] == "primary"]
    for t in TOLS:
        rs = [r for r in prim if r["delta"] == t]
        for rule in RULES:
            out.append({"block": "clean-corpus primary cells (31)", "delta": t, "rule": rule, "n_cells": len(rs),
                        "claim_T": sum(r[rule] == "T" for r in rs), "claim_N": sum(r[rule] == "N" for r in rs),
                        "abstain": sum(r[rule] == "A" for r in rs),
                        "cells_T": ";".join(f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']}"
                                            for r in rs if r[rule] == "T")})
    return out


def main():
    ap = K.add_out_arg(argparse.ArgumentParser())
    ap.add_argument("--reps", type=int, default=20000, help="reps per S1-generator scenario (smoke tests only)")
    args = ap.parse_args()
    t_start = time.time()
    st = K.state()
    out = args.out
    worst = tango_vector_check()
    print(f"vectorised Tango vs scalar: max |diff| = {worst:.3e}")
    rows = s1_rows(st, reps=args.reps)
    K.write_csv(rows, os.path.join(out, "s9_s1_by_scenario.csv"),
                f"S9 on the S1 generator (frozen Scenario/scenario_from_rates/tango_ci/verdict; S1 grid + sign-mirrored "
                f"twins + the A11.12 scenarios of s1_extra.py + S1's cell set (results/tifs/s1/cells.csv) with the edge "
                f"twins of tolerances 0.02 and 0.10; 20000 reps each): per scenario, delta and rule the claim "
                f"frequencies T/N/A and the frequency of a correct and a wrong claim against the generator's k (truth T "
                f"iff k > delta). Rules: R1 raw ASR>=0.5 (point; threshold not registered); R1h frozen HIGH flag "
                f"(Wilson 95% lower bound>=0.5; sensitivity); R2 |k_hat|>=delta; R3 k_hat>=delta; R4 Tango lo>0; R5 "
                f"audit POSITIVE->T, EQUIVALENT/NEGATIVE->N, INCONCLUSIVE->A; R5b audit POSITIVE->T else N. Vectorised "
                f"vs scalar Tango max |diff| {worst:.2e}.")
    K.write_csv(s1_summary(rows), os.path.join(out, "s9_s1_summary.csv"),
                "S9 S1-generator summary: per rule, delta, truth class (grouping='truth class') or truth T/N "
                "(grouping='truth'), scenario set (grid+mirror, a11.12, s1_cells, all) and n (and all n) the "
                "mean/min/max over scenarios of P(claim T), P(claim N), P(abstain) and P(wrong claim), the MC SE of "
                "the mean P(wrong), and the worst scenario with its MC SE, k and truth class.")
    emp = empirical_rows(st)
    K.write_csv(emp, os.path.join(out, "s9_known_truth_cells.csv"),
                "S9 empirical known-truth items (EP SST-2 structural null: truth N; positive controls kl01/kl10: truth T "
                "(existence); contaminated-corpus CE-only R1: truth N, only R1/R1h applicable; A/A backdoor-free swaps "
                "among independent clean builds t42/t43/t44, both orientations: truth N): per item, delta and rule the "
                "value, claim T/N/A and outcome. R1h = frozen HIGH flag (Wilson lower bound >= 0.5), R5 = frozen Tango "
                "verdict (A = INCONCLUSIVE), R5b = POSITIVE vs rest, R5t = transported interval (sensitivity).")
    K.write_csv(empirical_summary(emp), os.path.join(out, "s9_known_truth_summary.csv"),
                "S9 empirical known-truth counts per condition (level=condition), sub-condition "
                "(level=subcondition) and, for A/A, trigger over both corpora (level=trigger), per delta and rule: "
                "claims T/N/abstain/n-a and correct/wrong per test (A/A: both orientations of each pair are tests), "
                "items (A/A: unordered pairs) with a T claim in some orientation and items_wrong = items with a wrong "
                "claim in some orientation (pair-level count).")
    sbs = side_by_side(st)
    K.write_csv(sbs, os.path.join(out, "s9_clean_corpus_side_by_side.csv"),
                "S9 clean-corpus battery cells (31 primary cells: primary level and each seed's Tango) and contaminated "
                "P1 - C1 cells: the stage-wise judgements side by side (R1h at the primary level = HIGH flag in every "
                "seed). Truth unknown: NOT scored.")
    K.write_csv(side_by_side_counts(sbs), os.path.join(out, "s9_clean_corpus_counts.csv"),
                "S9 clean-corpus primary cells (31): number of T/N/abstain claims of each stage-wise rule per delta and "
                "the cells claimed T. Truth unknown: descriptive, NOT scored.")
    print("wrote S9 tables to", out)
    K.log_cost(out, "s9.py", t_start)


if __name__ == "__main__":
    main()
