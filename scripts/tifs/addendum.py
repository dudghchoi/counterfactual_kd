#!/usr/bin/env python3
"""Added runs and labelled sensitivity analyses of docs/tifs_addendum.md.

Every table here is a sensitivity analysis (A3) or an added run (A2). The
primary verdicts are those of scripts/tifs/analyze.py, frozen with the rule;
this script imports its functions without changing them and shows the
primary verdict beside each sensitivity verdict. The A2 replicate runs
(tifs_rep_*) and the A8 hidden-state KD runs (tifs_hid_*) are read here
only: analyze.py does not parse them. Works on
partial results; a table whose inputs do not exist yet is left out.

Outputs (``--out``, default results/tifs/addendum/):
  replicate.csv        A2: the ATBA teacher-seed-43 replicate k' (seed-pooled
                       with the BERT SST-2 logit pool, and with its own
                       variance at df r - 1), its verdicts, the replicate
                       teacher's gate, and the build difference k - k'
  attack_variance.csv  A3.1: seed-pooled cells with SE from max(s2_a, s2_pool)
                       (pool df) and from s2_a alone (df r - 1)
  variance_diag.csv    A3.2: s_a, F_a = s2_a / s2_pool and the sign-flip p of
                       every seed-to-seed shift
  reference_atba.csv   A3.3: clean-reference variance at the ATBA phrase (the
                       seed-42 clean student and the A/A clean students) and
                       the ATBA cells with it added
  cross_null.csv       A3.4: per trigger, every student without that
                       trigger's backdoor against the matched C (frozen
                       formula), and the attack's excess over them
  cross_check.csv      A3.4: CPU fp32 re-scoring vs the official bf16 vectors
  decomposition.csv    A3.5: P - R and C - R with transported intervals
                       (s2_pool and 2 s2_pool)
  agnews_plan.csv      A3.6: AG News seed-42 cells transported from the SST-2
                       logit pool (the reading planned before A1), with the
                       battery decision
  seed_check.csv       A3.7: seeds of every pooled attack vs the pool's seeds
  decidability.csv     A3.8: upper bound of a k = 0 cell, S1b power at the
                       nearest tau, and the variance ratio at which a
                       transported verdict changes
  global_holm.csv      A3.9: Holm-adjusted p of every qualified cell over all
                       batteries present
  teacher_function.csv A7: every teacher's clean accuracy against the
                       majority-class share (+ the RIPPLe CPU check)
  hidden_seeds.csv     A8: each hidden seed admitted, incomplete or an
                       objective failure (a student at or below the
                       majority share + 0.10)
  hidden_check.csv     A8: scripts/tifs/hidden_check.py per seed (training,
                       validation accuracy, manipulation check)
  hidden_cells.csv     A8: hidden-state KD cells of the admitted seeds under
                       the frozen primary rule (own pool from two seeds on,
                       else transported from the BERT SST-2 logit pool),
                       student clean accuracy, delta_b, k*, and hidden -
                       logit per attack with Holm across attacks
  hidden_battery.csv   A8: the hidden battery's decision (Holm within the
                       battery; outside the global Holm family), the logit
                       decision on the same attacks and the agreement label
  hidden_variance.csv  A8: the hidden battery's between-seed pool
  hidden_attack_variance.csv, hidden_variance_diag.csv  A8: A3.1 / A3.2 on
                       the hidden cells
  hidden_controls.csv  A8: control-trigger P - C, hidden and logit
  hidden_decomposition.csv  A8: P - R / C - R at seed 42 (R = the frozen
                       battery's CE-only student; the larger of the hidden
                       and logit pools)
  ce_coef.csv          A10-a: the CE-only student trained with L_B = 0.5*CE
                       (tifs_rcc_*) against the frozen battery's R (L_C = CE),
                       per trigger: Tango interval of R_B - R_C (same seed,
                       input term only), clean accuracy, untriggered target rate
  ce_coef_decomposition.csv  A10-a: A3.5 (P - R, C - R) recomputed with R_B,
                       beside the A3.5 verdicts; 'ce_coefficient_dependent'
                       lists the verdicts that differ
  summary.md

    python scripts/tifs/addendum.py [--out DIR] [--cross results/tifs/addendum/cross_hits_bert_sst2.json ...]
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import random
import re
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze as A  # noqa: E402  (frozen; imported, never modified)
from counterfactual_kd.analysis.tifs import (  # noqa: E402
    holm, paired_contrast, se_from_interval, t_contrast,
)

TOLS = A.TOLS
ALPHA = A.ALPHA
PRIMARY_KD = "logit"
REP_RE = "tifs_rep_bert_atba_t43_s"
S1B = os.path.join(ROOT, "results", "tifs", "s1", "s1b.csv")
# Trainer of each attack's clean teacher (counterfactual_kd.registry.attack_trigger
# 'base' vs dedicated trainers), as the student tags name them.
CLEAN_TAG = {"sos": "clean-sos", "ep": "clean-ep", "ripple": "clean-ripples"}


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def var(v):
    m = mean(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1) if len(v) > 1 else float("nan")


def tq(df):
    from scipy.stats import t
    return float(t.ppf(0.975, df))


def verdicts(res, prefix=""):
    return {f"{prefix}verdict@{t:g}": res[f"verdict@{t:g}"] for t in TOLS}


def write_csv(rows, path):
    if rows:
        A.write_csv(rows, path)


# ── shared state ─────────────────────────────────────────────────────


class State:
    """The frozen analysis' cells, pools and primary rows."""

    def __init__(self, runs):
        self.runs = runs
        self.cells = A.load_cells(runs)
        self.structural, self.near_null = A.load_pairs()
        self.ref_var = A.reference_variance(self.cells)
        self.primary, self.var_rows = A.primary_table(
            self.cells, self.structural, self.near_null, ref_var=self.ref_var)
        self.pools = {(v["fam"], v["dataset"], v["kd"]): v
                      for v in self.var_rows if v["attack"] == "(pooled)"}
        self.att_var = {(v["fam"], v["dataset"], v["kd"], v["attack"]): v
                        for v in self.var_rows if v["attack"] != "(pooled)"}
        self.prim = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in self.primary}
        self.groups = defaultdict(dict)
        for c in self.cells:
            if c["block"] == "G" and c["corpus"] == "clean" and not c["single"] and c["kd"] != "ce":
                self.groups[(c["fam"], c["dataset"], c["kd"], c["attack"])][c["seed"]] = c
        self.refs = A.reference_index(self.cells)

    def pool_for(self, fam, ds, kd):
        """The pool the frozen rule would use (analyze.primary_table source_for)."""
        for key in ((fam, ds, kd), (fam, ds, PRIMARY_KD), (fam, "sst2", PRIMARY_KD)):
            if key in self.pools:
                return key, self.pools[key]
        others = [k for k in self.pools if k[1] == "sst2" and k[2] == PRIMARY_KD]
        if not others:
            return None, None
        k = max(others, key=lambda k: self.pools[k]["s_between"])
        return k, self.pools[k]


def seed_pooled(ds_by_seed, s2, df):
    """Frozen seed-pooled estimate: per-seed d vectors -> (k, se, r, v, ks)."""
    seeds = sorted(ds_by_seed)
    d = [ds_by_seed[s] for s in seeds]
    n = len(d[0])
    ks = [mean(x) for x in d]
    dbar = [mean(col) for col in zip(*d)]
    k = mean(ks)
    v = sum((x - k) ** 2 for x in dbar) / (n - 1) / n
    se = (s2 / len(seeds) + v) ** 0.5
    return k, se, len(seeds), v, ks


def diff(a, b):
    return [x - y for x, y in zip(a, b)]


# ── A2: replicate ────────────────────────────────────────────────────


def load_replicate(runs):
    out = {}
    for p in sorted(glob.glob(os.path.join(runs, f"{REP_RE}*", "artifacts", "results", "*.preds.json"))):
        res = p[: -len(".preds.json")] + ".json"
        if not os.path.exists(res):
            continue
        preds, r = json.load(open(p)), json.load(open(res))
        if (preds.get("meta") or {}).get("schema") != 2 or r.get("kd_type") != "logit":
            continue
        out[int(r.get("student_seed", r.get("seed")))] = {"P": preds["preds_obs"], "C": preds["preds_null"],
                                                          "res": r, "run": p.split(os.sep)[-4]}
    return out


def replicate_table(st: State):
    rep = load_replicate(st.runs)
    if not rep:
        return []
    key, pool = ("bert", "sst2", "logit"), st.pools.get(("bert", "sst2", "logit"))
    prim = st.prim.get(("bert", "sst2", "logit", "atba"))
    if pool is None:
        return []
    s2p, dfp = pool["s_between"] ** 2, pool["df"]
    any_r = next(iter(rep.values()))["res"]
    gate = A.qualified({k: any_r.get(k) for k in ("teacher_asr", "teacher_ca", "teacher_ca_null")})
    row = {"attack": "atba", "teacher_seed": 43, "seeds": ",".join(map(str, sorted(rep))),
           "teacher_asr": any_r.get("teacher_asr"), "teacher_asr_null": any_r.get("teacher_asr_null"),
           "teacher_ca": any_r.get("teacher_ca"), "teacher_ca_null": any_r.get("teacher_ca_null"),
           "teacher_qualified": gate,
           "P": mean([mean(v["P"]) for v in rep.values()]),
           "C": mean([mean(v["C"]) for v in rep.values()]),
           "variance_from": "/".join(key), "pool_s": s2p ** 0.5, "pool_df": dfp}
    d = {s: diff(v["P"], v["C"]) for s, v in rep.items()}
    if len(d) >= 2:
        k, se, r, v, ks = seed_pooled(d, s2p, dfp)
        method = "seed-pooled, BERT SST-2 logit pool"
    else:
        (s, x), = d.items()
        pc = paired_contrast(rep[s]["P"], rep[s]["C"], TOLS)
        k, ks, r = pc["k"], [pc["k"]], 1
        se = (se_from_interval(pc["lo"], pc["hi"]) ** 2 + s2p) ** 0.5
        method = "transported, BERT SST-2 logit pool"
    res = t_contrast(k, se, dfp, TOLS)
    row.update(method=method, k=k, se=se, lo=res["lo"], hi=res["hi"],
               **{f"k_s{s}": x for s, x in zip(sorted(d), ks)}, **verdicts(res),
               **{f"p_pos@{t:g}": res[f"p_pos@{t:g}"] for t in TOLS})
    for t in TOLS:
        row[f"reading@{t:g}"] = ("replicated" if res[f"verdict@{t:g}"] == "POSITIVE" else
                                 "not replicated" if res[f"p_up@{t:g}"] < ALPHA else "inconclusive")
    if r >= 2:
        own = t_contrast(k, (var(ks) / r + v) ** 0.5, r - 1, TOLS)
        row.update(own_lo=own["lo"], own_hi=own["hi"], **verdicts(own, "own_"))
    if prim is not None and prim.get("se"):
        dd = t_contrast(prim["k"] - k, (prim["se"] ** 2 + se ** 2) ** 0.5, dfp, (0.05,))
        row.update(k_primary=prim["k"], build_diff=prim["k"] - k, build_diff_lo=dd["lo"],
                   build_diff_hi=dd["hi"])
    # Determinism: the replicate's seed-42 C' and the A/A run's C' share teacher and seed.
    aa = next((c for c in st.cells if c["block"] == "AA" and c["corpus"] == "clean"
               and c["teacher_seed"] == 43 and c["dataset"] == "sst2" and not c["single"]), None)
    if aa is not None and 42 in rep:
        ctl = aa["controls"].get("atba_phrase_sst2")
        if ctl is not None and ctl.get("c") is not None:
            row["c_prime_vs_aa_disagree"] = sum(a != b for a, b in zip(rep[42]["C"], ctl["c"]))
    return [row]


# ── A3.1 / A3.2: per-attack variance ─────────────────────────────────


def attack_variance_table(st: State, prim=None, att_var=None, pools=None):
    """``prim`` / ``att_var`` / ``pools`` default to the frozen rows; A8
    passes the hidden battery's."""
    prim = st.prim if prim is None else prim
    att_var = st.att_var if att_var is None else att_var
    pools = st.pools if pools is None else pools
    rows = []
    for key, pr in sorted(prim.items()):
        if not pr.get("method", "").startswith("seed-pooled"):
            continue
        av = att_var.get(key)
        src = tuple(pr["variance_from"].split("/"))
        pool = pools.get(src)
        if av is None or pool is None:
            continue
        r = len(pr["seeds"].split(","))
        s2a, s2p, v = av["s_between"] ** 2, pool["s_between"] ** 2, av["input_se"] ** 2
        mx = t_contrast(pr["k"], (max(s2a, s2p) / r + v) ** 0.5, pool["df"], TOLS)
        own = t_contrast(pr["k"], (s2a / r + v) ** 0.5, r - 1, TOLS)
        row = {"fam": key[0], "dataset": key[1], "kd": key[2], "attack": key[3], "r": r,
               "k": pr["k"], "s_attack": s2a ** 0.5, "s_pool": s2p ** 0.5, "pool_df": pool["df"],
               "lo": pr["lo"], "hi": pr["hi"], **verdicts(pr),
               "max_lo": mx["lo"], "max_hi": mx["hi"], **verdicts(mx, "max_"),
               "own_lo": own["lo"], "own_hi": own["hi"], **verdicts(own, "own_")}
        row["pooling_dependent"] = ",".join(
            f"{t:g}" for t in TOLS
            if pr[f"verdict@{t:g}"] != mx[f"verdict@{t:g}"] or pr[f"verdict@{t:g}"] != own[f"verdict@{t:g}"])
        rows.append(row)
    return rows


def sign_flip_p(e, B=20000, seed=0):
    """Two-sided sign-flip p of mean(e) = 0 (per-input exchangeable signs)."""
    nz = [x for x in e if x != 0]
    if not nz:
        return 1.0
    obs = abs(sum(nz))
    rng = random.Random(seed)
    hits = 0
    for _ in range(B):
        s = sum(x if rng.random() < 0.5 else -x for x in nz)
        hits += abs(s) >= obs - 1e-12
    return (hits + 1) / (B + 1)


def variance_diag_table(st: State, groups=None, att_var=None, prim=None, pools=None):
    """Defaults as attack_variance_table; ``groups``: {cell key: {seed: cell}}."""
    groups = st.groups if groups is None else groups
    att_var = st.att_var if att_var is None else att_var
    prim = st.prim if prim is None else prim
    pools = st.pools if pools is None else pools
    rows = []
    for key, by_seed in sorted(groups.items()):
        seeds = sorted(by_seed)
        if len(seeds) < 2:
            continue
        av = att_var.get(key)
        src = prim.get(key, {}).get("variance_from")
        pool = pools.get(tuple(src.split("/"))) if src else None
        d = {s: diff(by_seed[s]["P"], by_seed[s]["C"]) for s in seeds}
        row = {"fam": key[0], "dataset": key[1], "kd": key[2], "attack": key[3],
               "seeds": ",".join(map(str, seeds)),
               "s_attack": av["s_between"] if av else None,
               "s_pool": pool["s_between"] if pool else None,
               "F": (av["s_between"] ** 2 / pool["s_between"] ** 2) if (av and pool and pool["s_between"] > 0) else None,
               "in_pool": bool(pool and key[3] in pool["attacks"].split(","))}
        for a, b in itertools.combinations(seeds, 2):
            e = diff(d[b], d[a])
            row[f"shift_{a}_{b}"] = mean(e)
            row[f"signflip_p_{a}_{b}"] = sign_flip_p(e)
        rows.append(row)
    return rows


# ── A3.3: clean-reference variance at the ATBA phrase ────────────────


def reference_atba_table(st: State):
    rows = []
    for ds in ("sst2", "agnews"):
        base = st.groups.get(("bert", ds, "logit", "atba"), {}).get(42)
        if base is None:
            continue
        refs = {"t42": mean(base["C"])}
        for c in st.cells:
            if (c["block"] == "AA" and c["corpus"] == "clean" and c["fam"] == "bert"
                    and c["dataset"] == ds and not c["single"] and c["kd"] == "logit"):
                ctl = c["controls"].get(f"atba_phrase_{ds}")
                if ctl is not None and ctl.get("c") is not None:
                    refs[f"t{c['teacher_seed']}"] = mean(ctl["c"])
        row_ref = {"dataset": ds, **{f"C_{k}": v for k, v in sorted(refs.items())},
                   "n_references": len(refs)}
        if len(refs) < 2:
            rows.append({**row_ref, "note": "A/A clean runs not evaluated yet"})
            continue
        s2r, dfr = var(list(refs.values())), len(refs) - 1
        row_ref.update(s_ref=s2r ** 0.5, df_ref=dfr)
        for kd in ("logit", "feature"):
            pr = st.prim.get(("bert", ds, kd, "atba"))
            if pr is None or not pr.get("se"):
                continue
            se2, dfp = pr["se"] ** 2, pr["df"]
            df = (se2 + s2r) ** 2 / (se2 ** 2 / dfp + s2r ** 2 / dfr)
            rob = t_contrast(pr["k"], (se2 + s2r) ** 0.5, df, TOLS)
            rows.append({**row_ref, "kd": kd, "k": pr["k"], "lo": pr["lo"], "hi": pr["hi"],
                         **verdicts(pr), "ref_lo": rob["lo"], "ref_hi": rob["hi"], "ref_df": df,
                         **verdicts(rob, "ref_"),
                         "reference_dependent": ",".join(f"{t:g}" for t in TOLS
                                                         if pr[f"verdict@{t:g}"] != rob[f"verdict@{t:g}"]),
                         "note": "feature KD uses the logit-KD references" if kd == "feature" else ""})
    return rows


# ── A3.4: operating-point-matched teacher-swap null ──────────────────


def clean_tag(attack):
    return CLEAN_TAG.get(attack, "clean-base")


def cross_tables(st: State, paths):
    null_rows, check_rows = [], []
    for path in paths:
        if not os.path.exists(path):
            continue
        X = json.load(open(path))
        fam, ds = X["meta"]["fam"], X["meta"]["dataset"]
        meta = X["students"]
        for atk, hits in sorted(X["hits"].items()):
            for kd in X["meta"]["kd"]:
                pk, pool = st.pool_for(fam, ds, kd)
                if pool is None:
                    continue
                s2p, dfp = pool["s_between"] ** 2, pool["df"]
                seeds = sorted({m["seed"] for k, m in meta.items() if m["kd"] == kd and k in hits})
                own = {s: f"{kd}/{next((m['tag'] for k, m in meta.items() if m['kd'] == kd and m['seed'] == s and m['role'] == 'poisoned' and m['tag'].split('_')[0] == atk), '')}/poisoned/s{s}" for s in seeds}
                ref = {s: f"{kd}/{clean_tag(atk)}/clean/s{s}" for s in seeds}
                seeds = [s for s in seeds if own[s] in hits and ref[s] in hits]
                if not seeds:
                    continue
                # Official vs CPU for the run's own P and C.
                for s in seeds:
                    cell = st.groups.get((fam, ds, kd, atk), {}).get(s)
                    if cell is not None:
                        check_rows.append({"fam": fam, "dataset": ds, "kd": kd, "attack": atk, "seed": s,
                                           "P_disagree": sum(a != b for a, b in zip(hits[own[s]], cell["P"])),
                                           "C_disagree": sum(a != b for a, b in zip(hits[ref[s]], cell["C"])),
                                           "n": len(cell["P"])})
                others = sorted({k.rsplit("/s", 1)[0] for k in hits
                                 if meta[k]["kd"] == kd and meta[k]["role"] in ("poisoned", "clean")
                                 and k.rsplit("/s", 1)[0] not in (own[seeds[0]].rsplit("/s", 1)[0],
                                                                  ref[seeds[0]].rsplit("/s", 1)[0])})
                excess = {}
                for s in seeds:
                    o = [mean(hits[f"{b}/s{s}"]) for b in others if f"{b}/s{s}" in hits]
                    excess[s] = mean(hits[own[s]]) - mean(o) if o else float("nan")
                for b in [own[seeds[0]].rsplit("/s", 1)[0]] + others:
                    ss = [s for s in seeds if f"{b}/s{s}" in hits]
                    if not ss:
                        continue
                    d = {s: diff(hits[f"{b}/s{s}"], hits[ref[s]]) for s in ss}
                    if len(ss) >= 2:
                        k, se, r, _, ks = seed_pooled(d, s2p, dfp)
                        method = "seed-pooled"
                    else:
                        s, = ss
                        pc = paired_contrast(hits[f"{b}/s{s}"], hits[ref[s]], TOLS)
                        k, ks, r = pc["k"], [pc["k"]], 1
                        se = (se_from_interval(pc["lo"], pc["hi"]) ** 2 + s2p) ** 0.5
                        method = "transported"
                    res = t_contrast(k, se, dfp, TOLS)
                    is_own = b == own[seeds[0]].rsplit("/s", 1)[0]
                    null_rows.append({
                        "fam": fam, "dataset": ds, "kd": kd, "trigger": atk,
                        "student": b.split("/", 1)[1], "role": "own P" if is_own else "null",
                        "vs": f"{clean_tag(atk)} C", "seeds": ",".join(map(str, ss)), "method": method,
                        "variance_from": "/".join(pk), "k": k, "lo": res["lo"], "hi": res["hi"],
                        **{f"k_s{s}": x for s, x in zip(ss, ks)}, **verdicts(res),
                        **({f"excess_s{s}": e for s, e in excess.items()} if is_own else {}),
                        **({"excess_mean": mean(list(excess.values())), "n_null_students": len(others)}
                           if is_own else {})})
    return null_rows, check_rows


# ── A3.5: P - R and C - R ────────────────────────────────────────────


def decomposition_table(st: State, cells=None, pool_for=None):
    """``cells`` / ``pool_for`` default to the frozen cells and pools (A3.5);
    A8 passes its hidden cells and their pools."""
    rows = []
    for c in (st.cells if cells is None else cells):
        if c["single"] or c["kd"] == "ce" or c["block"] != "G" or c["corpus"] != "clean" or c["seed"] != 42:
            continue
        r = st.refs.get((c["block"], c["fam"], c["dataset"], c["attack"], c["corpus"], c["seed"]))
        if r is None:
            continue
        pk, pool = (pool_for or st.pool_for)(c["fam"], c["dataset"], c["kd"])
        if pool is None:
            continue
        s2p, dfp = pool["s_between"] ** 2, pool["df"]
        row = {"fam": c["fam"], "dataset": c["dataset"], "kd": c["kd"], "attack": c["attack"],
               "P": mean(c["P"]), "C": mean(c["C"]), "R": mean(r["P"]), "variance_from": "/".join(pk)}
        for name, (a, b) in {"P-C": (c["P"], c["C"]), "P-R": (c["P"], r["P"]),
                             "C-R": (c["C"], r["P"])}.items():
            pc = paired_contrast(a, b, TOLS)
            se_t = se_from_interval(pc["lo"], pc["hi"])
            for mult in (1, 2):
                res = t_contrast(pc["k"], (se_t ** 2 + mult * s2p) ** 0.5, dfp, TOLS)
                tag = name if mult == 1 else f"{name}[2x]"
                row.update({tag: pc["k"], f"{tag}_lo": res["lo"], f"{tag}_hi": res["hi"],
                            **{f"{tag}_verdict@{t:g}": res[f"verdict@{t:g}"] for t in TOLS}})
        k = row["P-C"]
        if abs(k) > 1e-9:
            row["share_injection"] = row["P-R"] / k
            row["share_lost_suppression"] = -row["C-R"] / k
        # Labelled only where there is an effect to split (|P - C| >= 0.02).
        row["mostly"] = ("-" if abs(k) < 0.02 else
                         "injection (P-R)" if abs(row["P-R"]) >= abs(row["C-R"])
                         else "loss of clean-teacher suppression (R-C)")
        rows.append(row)
    return rows


# ── A10-a: CE coefficient check ──────────────────────────────────────

RCC_RX = re.compile(r"^tifs_rcc_(?P<fam>[a-z0-9]+)_(?P<bds>[a-z0-9]+)$")


def load_rcc(runs):
    """A10-a CE-only students trained with L_B = 0.5*CE, by (fam, dataset, attack)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(runs, "tifs_rcc_*", "artifacts", "results", "*.preds.json"))):
        run = p.split(os.sep)[-4]
        m = RCC_RX.match(run)
        res = p[: -len(".preds.json")] + ".json"
        if m is None or not os.path.exists(res):
            continue
        preds, r = json.load(open(p)), json.load(open(res))
        if (preds.get("meta") or {}).get("schema") != 2 or r.get("kd_type") != "ce":
            continue
        out[(m["fam"], r["dataset"], r["attack_method"])] = {
            "R": preds["preds_obs"], "Rc": preds.get("preds_p_clean"), "ca": r.get("ca_poisoned"),
            "seed": int(r.get("student_seed", r.get("seed", -1))), "run": run}
    return out


def ce_coef_tables(st: State):
    """A10-a. (1) R_B - R_C on each trigger's inputs: Tango, input term only (the
    two students share seed, initialization and data order). (2) A3.5 with R_B
    in place of R_C, beside the A3.5 verdicts; a cell whose verdict differs at
    any delta (s2_pool interval) is 'CE-coefficient-dependent'."""
    rcc = {k: v for k, v in load_rcc(st.runs).items() if v["seed"] == 42}
    if not rcc:
        return [], []
    rows = []
    for (fam, ds, atk), b in sorted(rcc.items()):
        rc = st.refs.get(("G", fam, ds, atk, "clean", 42))
        if rc is None:
            continue
        pc = paired_contrast(b["R"], rc["P"], TOLS)
        row = {"fam": fam, "dataset": ds, "attack": atk, "run": b["run"],
               "R_B": mean(b["R"]), "R_C": mean(rc["P"]),
               "RB-RC": pc["k"], "RB-RC_lo": pc["lo"], "RB-RC_hi": pc["hi"],
               **{f"RB-RC_verdict@{t:g}": pc[f"verdict@{t:g}"] for t in TOLS},
               "n_disagree": sum(x != y for x, y in zip(b["R"], rc["P"])), "n": len(b["R"]),
               "ca_R_B": b["ca"], "ca_R_C": rc.get("ca_p")}
        if b["Rc"] is not None and rc.get("Pc") is not None:
            pu = paired_contrast(b["Rc"], rc["Pc"], TOLS)
            row.update({"RB-RC_untriggered": pu["k"], "RB-RC_untriggered_lo": pu["lo"],
                        "RB-RC_untriggered_hi": pu["hi"]})
        rows.append(row)
    all_eq = bool(rows) and all(r["RB-RC_verdict@0.02"] == "EQUIVALENT" for r in rows)
    for r in rows:
        r["all_cells_equivalent@0.02"] = all_eq
    base = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in decomposition_table(st)}
    saved = st.refs
    st.refs = dict(saved)
    for (fam, ds, atk), b in rcc.items():
        key = ("G", fam, ds, atk, "clean", 42)
        if key in saved:
            st.refs[key] = {**saved[key], "P": b["R"]}
    try:
        alt = decomposition_table(st)
    finally:
        st.refs = saved
    drows = []
    for a in alt:
        if (a["fam"], a["dataset"], a["attack"]) not in rcc:
            continue
        o = base.get((a["fam"], a["dataset"], a["kd"], a["attack"]), {})
        row = {"fam": a["fam"], "dataset": a["dataset"], "kd": a["kd"], "attack": a["attack"],
               "P": a["P"], "C": a["C"], "R_B": a["R"], "R_C": o.get("R"),
               "variance_from": a["variance_from"]}
        changed = []
        for name in ("P-R", "C-R"):
            for tag in (name, f"{name}[2x]"):
                row.update({f"{tag}_B": a[tag], f"{tag}_B_lo": a[f"{tag}_lo"], f"{tag}_B_hi": a[f"{tag}_hi"],
                            f"{tag}_C": o.get(tag)})
                for t in TOLS:
                    vb, vc = a[f"{tag}_verdict@{t:g}"], o.get(f"{tag}_verdict@{t:g}")
                    row[f"{tag}_B_verdict@{t:g}"], row[f"{tag}_C_verdict@{t:g}"] = vb, vc
                    if tag == name and vb != vc:
                        changed.append(f"{name}@{t:g}")
        row.update(mostly_B=a["mostly"], mostly_C=o.get("mostly"),
                   ce_coefficient_dependent=",".join(changed))
        drows.append(row)
    return rows, drows


# ── A8: hidden-state KD battery ──────────────────────────────────────

HID_RX = re.compile(r"^tifs_hid_(?P<fam>[a-z0-9]+)_(?P<bds>[a-z0-9]+)_s(?P<sseed>\d+)$")
HIDDEN_CHECK = os.path.join(ROOT, "results", "tifs", "addendum", "hidden_check.json")
DIFF_ALPHA = 0.05   # hidden - logit: two-sided, Holm over the battery's attacks
# The registered A8 attacks (configs/tifs/tifs_hid_bert_sst2_s*.yaml).
HID_ATTACKS = {("bert", "sst2"): ("addsent", "atba", "badnets", "sos", "synbkd")}


def load_hidden_cells(runs):
    """A8 cells (tifs_hid_*), read by the frozen loader.

    analyze.load_cells parses run ids with the module-level RUN_RE, which
    does not match tifs_hid_*. Swapping in one battery-style pattern for
    the call gives these runs the frozen cell format (block G, corpus
    clean); analyze.py itself is not edited and RUN_RE is restored at once.
    """
    saved = A.RUN_RE
    A.RUN_RE = [("B", HID_RX)]
    try:
        cells = A.load_cells(runs)
    finally:
        A.RUN_RE = saved
    return [c for c in cells if c["kd"] == "hidden"]


def hidden_seed_status(hid):
    """A8 seed rule, one row per (family, dataset, seed).

    A hidden seed enters the battery only if it is complete (every
    registered attack, HID_ATTACKS, has a Stage 2 cell) and every P and C
    student of it is functional: Stage 2 clean accuracy above the
    majority-class share + FUNCTION_MARGIN (A7's floor, applied to
    students). A collapsed P/C pair would otherwise read as EQUIVALENT /
    NO TRANSFER, and a seed cut in the middle of Stage 2 would be decided on
    fewer attacks. A missing accuracy counts as below the floor.
    """
    by = defaultdict(lambda: defaultdict(list))
    for c in hid:
        by[(c["fam"], c["dataset"])][c["student_seed"]].append(c)
    rows = []
    for (fam, ds), seeds in sorted(by.items()):
        attacks = set(HID_ATTACKS.get((fam, ds), ())).union(*({c["attack"] for c in cs} for cs in seeds.values()))
        floor = MAJORITY[ds] + FUNCTION_MARGIN
        for seed, cs in sorted(seeds.items()):
            accs = [(c["attack"], side, acc) for c in cs
                    for side, acc in (("P", c["ca_p"]), ("C", c["ca_c"]))]
            low = sorted({f"{a}/{side}" for a, side, acc in accs if acc is None or acc <= floor})
            missing = sorted(attacks - {c["attack"] for c in cs})
            known = [acc for _, _, acc in accs if acc is not None]
            rows.append({"fam": fam, "dataset": ds, "kd": "hidden", "seed": seed,
                         "status": "objective failure" if low else "incomplete" if missing else "admitted",
                         "attacks": ",".join(sorted(c["attack"] for c in cs)),
                         "missing": ",".join(missing), "below_floor": ",".join(low),
                         "floor": floor, "min_ca": min(known) if known else None})
    return rows


def agreement(h, lg):
    """A8 reading rule: 'same' only for two NO TRANSFER decisions, or two
    TRANSFERS decisions whose Holm-flagged attacks are the same; 'both
    undecided' for two INCONCLUSIVE ones; else 'objective-dependent'."""
    (hd, ht), (ld, lt) = h, lg
    if hd == ld == "INCONCLUSIVE":
        return "both undecided"
    if hd == ld == "NO TRANSFER" or (hd == ld == "TRANSFERS" and set(ht.split(",")) == set(lt.split(","))):
        return "same"
    return "objective-dependent"


def kstar_mean(by_seed):
    """(delta_b, k*) averaged over seeds; None without clean-input vectors."""
    cs = list(by_seed.values())
    if not cs or not all(c["Pc"] and c["Cc"] for c in cs):
        return None, None
    db = mean([mean(c["Pc"]) - mean(c["Cc"]) for c in cs])
    return db, mean([mean(c["P"]) - mean(c["C"]) for c in cs]) - db


def control_rows(fam, ds, kd, atk, by_seed):
    """Every control trigger's P - C, averaged over seeds (descriptive)."""
    rows = []
    cs = list(by_seed.values())
    names = sorted(set().union(*(c["controls"] for c in cs))) if cs else []
    for name in names:
        ctl = [c["controls"][name] for c in cs if name in c["controls"]]
        rows.append({"fam": fam, "dataset": ds, "kd": kd, "attack": atk, "control": name,
                     "seeds": ",".join(str(s) for s, c in sorted(by_seed.items()) if name in c["controls"]),
                     "P_ctrl": mean([mean(x["p"]) for x in ctl]), "C_ctrl": mean([mean(x["c"]) for x in ctl]),
                     "k_ctrl": mean([mean(x["p"]) - mean(x["c"]) for x in ctl])})
    return rows


def hidden_check_rows(path=HIDDEN_CHECK):
    """scripts/tifs/hidden_check.py: training, function on the GLUE SST-2
    validation split, and the manipulation check, per hidden seed. Reported
    only: seed admission uses the Stage 2 rule (hidden_seed_status)."""
    if not os.path.exists(path):
        return []
    hc = json.load(open(path))
    rows = []
    for seed, r in sorted(hc.get("seeds", {}).items(), key=lambda kv: int(kv[0])):
        accs = [v["val_acc"] for v in r.get("students", {}).values()]
        rows.append({"seed": int(seed), "run": r.get("run"), "status": r.get("status", "checked"),
                     "training_ok": r.get("training_ok"),
                     "functional_ok": r.get("functional_ok"), "min_val_acc": min(accs) if accs else None,
                     "val_floor": hc.get("floor"), "logit_c_val_acc": r.get("logit_c_val_acc"),
                     "M_hidden": r.get("M_hidden"), "M_logit": r.get("M_logit"), "M_init": hc.get("M_init"),
                     "ratio": r.get("ratio"), "max_ratio": hc.get("max_ratio"),
                     "manipulation_ok": r.get("manipulation_ok"), "pass": r.get("pass")})
    return rows


def hidden_tables(st: State, check_path=HIDDEN_CHECK):
    """A8: the frozen primary rule on the 'hidden' battery. Returns {name: rows}.

    Seeds: only seeds admitted by hidden_seed_status enter; the others are
    listed, not decided on (no admitted seed: UNDECIDABLE). primary_table
    runs on the frozen cells plus the admitted hidden cells, so a hidden
    cell takes its own battery's pool once two admitted seeds exist (r = 2
    included), and before that the BERT SST-2 logit pool (transported), in
    source_for's order. The battery decision (Holm within the battery)
    comes from primary_battery on the hidden rows alone: the hidden cells
    never enter the frozen global Holm family.

    Against logit KD:
    - the logit decision over the hidden battery's attacks (EP is not one
      of them), next to the frozen logit decision, and an agreement label
      (``agreement``);
    - per attack, hidden - logit (SEs added as if independent, df = the
      smaller pool df), its two-sided p with Holm over the battery's
      attacks, and the k* = k - delta_b difference: 'hidden larger/smaller'
      only if Holm p < DIFF_ALPHA and the k* difference has the same sign;
    - delta_b, k*, every control trigger's P - C (descriptive) and the
      students' clean accuracy.
    A3.1/A3.2 run on the hidden rows. P - R / C - R (seed 42; R = the frozen
    battery's CE-only student) use the larger of the hidden and the logit
    pool: R's own seed noise is in neither.
    """
    check = hidden_check_rows(check_path)
    hid_all = load_hidden_cells(st.runs)
    if not hid_all:
        return {"check": check} if check else {}
    seeds = hidden_seed_status(hid_all)
    ok = {(r["fam"], r["dataset"], r["seed"]) for r in seeds if r["status"] == "admitted"}
    hid = [c for c in hid_all if (c["fam"], c["dataset"], c["student_seed"]) in ok]
    out = {"seeds": seeds, "check": check}
    if not hid:
        out["battery"] = [{"fam": fam, "dataset": ds, "kd": "hidden", "n_attacks": 0,
                           **{f"decision@{t:g}": "UNDECIDABLE (no admitted seed)" for t in TOLS},
                           "global_holm": "not in the family (A8)"}
                          for fam, ds in sorted({(r["fam"], r["dataset"]) for r in seeds})]
        return out
    base = [c for c in st.cells if c["kd"] != "hidden"]
    rows, var_rows = A.primary_table(base + hid, st.structural, st.near_null, ref_var=st.ref_var)
    hrows = [r for r in rows if r["kd"] == "hidden"]
    pools = {(v["fam"], v["dataset"], v["kd"]): v for v in var_rows if v["attack"] == "(pooled)"}
    att_var = {(v["fam"], v["dataset"], v["kd"], v["attack"]): v for v in var_rows
               if v["attack"] != "(pooled)" and v["kd"] == "hidden"}
    groups = defaultdict(dict)
    for c in hid:
        if c["block"] == "G" and c["corpus"] == "clean" and not c["single"]:
            groups[(c["fam"], c["dataset"], c["kd"], c["attack"])][c["student_seed"]] = c

    def ca(fam, ds, kd, atk):
        cs = [c for c in base + hid if c["block"] == "G" and c["corpus"] == "clean" and not c["single"]
              and (c["fam"], c["dataset"], c["kd"], c["attack"]) == (fam, ds, kd, atk)]
        return (mean([c["ca_p"] for c in cs if c["ca_p"] is not None]),
                mean([c["ca_c"] for c in cs if c["ca_c"] is not None]))

    from scipy.stats import t as t_dist
    cells, controls = [], []
    for r in hrows:
        fam, ds, atk = r["fam"], r["dataset"], r["attack"]
        row = dict(r)
        row["ca_p"], row["ca_c"] = ca(fam, ds, "hidden", atk)
        row["ca_p_logit"], row["ca_c_logit"] = ca(fam, ds, PRIMARY_KD, atk)
        lg = st.prim.get((fam, ds, PRIMARY_KD, atk))
        src = pools.get(tuple(r.get("variance_from", "").split("/")))
        lsrc = pools.get(tuple(lg.get("variance_from", "").split("/"))) if lg else None
        if lg is not None and lg.get("se") and r.get("se") and src and lsrc:
            df = min(src["df"], lsrc["df"])
            se = (r["se"] ** 2 + lg["se"] ** 2) ** 0.5
            dd = t_contrast(r["k"] - lg["k"], se, df, TOLS)
            row.update(k_logit=lg["k"], diff_vs_logit=dd["k"], diff_lo=dd["lo"], diff_hi=dd["hi"],
                       diff_df=df, diff_p_two=float(2 * t_dist.sf(abs(dd["k"]) / se, df)),
                       **verdicts(dd, "diff_"))
        lgroup = st.groups.get((fam, ds, PRIMARY_KD, atk), {})
        row["delta_b"], row["k_star"] = kstar_mean(groups[(fam, ds, "hidden", atk)])
        row["delta_b_logit"], row["k_star_logit"] = kstar_mean(lgroup)
        if row["k_star"] is not None and row["k_star_logit"] is not None:
            row["k_star_diff"] = row["k_star"] - row["k_star_logit"]
        cells.append(row)
        controls += control_rows(fam, ds, "hidden", atk, groups[(fam, ds, "hidden", atk)])
        controls += control_rows(fam, ds, PRIMARY_KD, atk, lgroup)
    tested = [r for r in cells if r.get("diff_p_two") is not None]
    for r, p in zip(tested, holm([r["diff_p_two"] for r in tested])):
        sign = (r["diff_vs_logit"] > 0) - (r["diff_vs_logit"] < 0)
        ks = r.get("k_star_diff")
        same = ks is not None and ((ks > 0) - (ks < 0)) == sign
        r["diff_p_holm"] = p
        r["diff_claim"] = ("hidden larger" if sign > 0 else "hidden smaller") if p < DIFF_ALPHA and same else "none"

    battery = A.primary_battery(hrows)
    frozen = {(b["fam"], b["dataset"]): b for b in A.primary_battery(st.primary) if b["kd"] == PRIMARY_KD}
    for b in battery:
        for t in TOLS:
            b.pop(f"transfers_global@{t:g}", None)
        b["global_holm"] = "not in the family (A8)"
        atks = {r["attack"] for r in hrows if (r["fam"], r["dataset"]) == (b["fam"], b["dataset"])}
        lrows = [r for k, r in sorted(st.prim.items())
                 if k[:3] == (b["fam"], b["dataset"], PRIMARY_KD) and k[3] in atks]
        same = A.primary_battery(lrows)[0] if lrows else {}
        fz = frozen.get((b["fam"], b["dataset"]), {})
        b["logit_same_attacks_m"] = same.get("n_qualified")
        b["logit_frozen_m"] = fz.get("n_qualified")
        for t in TOLS:
            key, tk = f"decision@{t:g}", f"transfers@{t:g}"
            b[f"logit_same_attacks@{t:g}"] = same.get(key)
            b[f"logit_same_attacks_transfers@{t:g}"] = same.get(tk, "")
            b[f"logit_frozen@{t:g}"] = fz.get(key)
            b[f"agreement@{t:g}"] = (agreement((b.get(key), b.get(tk, "")), (same.get(key), same.get(tk, "")))
                                     if same else None)

    hprim = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in hrows}

    def guard_pool(fam, ds, kd):
        keys = [k for k in ((fam, ds, kd), (fam, ds, PRIMARY_KD), (fam, "sst2", PRIMARY_KD)) if k in pools]
        opts = [k for k in keys if k[2] == kd][:1] + [k for k in keys if k[2] == PRIMARY_KD][:1]
        if not opts:
            return None, None
        k = max(opts, key=lambda k: pools[k]["s_between"])
        return k, pools[k]

    out.update(cells=cells, battery=battery, controls=controls,
               variance=[v for v in var_rows if v["kd"] == "hidden"],
               attack_variance=attack_variance_table(st, prim=hprim, att_var=att_var, pools=pools),
               variance_diag=variance_diag_table(st, groups=groups, att_var=att_var, prim=hprim, pools=pools),
               decomposition=decomposition_table(st, cells=hid, pool_for=guard_pool))
    return out


# ── A3.6: AG News as planned before A1 ───────────────────────────────


def agnews_plan_table(st: State):
    pool = st.pools.get(("bert", "sst2", "logit"))
    if pool is None:
        return [], []
    s2p, dfp = pool["s_between"] ** 2, pool["df"]
    rows = []
    for (fam, ds, kd, atk), by_seed in sorted(st.groups.items()):
        if fam != "bert" or ds != "agnews" or 42 not in by_seed:
            continue
        c = by_seed[42]
        pr = st.prim.get((fam, ds, kd, atk), {})
        if (atk, ds) in st.structural:
            continue
        pc = paired_contrast(c["P"], c["C"], TOLS)
        res = t_contrast(pc["k"], (se_from_interval(pc["lo"], pc["hi"]) ** 2 + s2p) ** 0.5, dfp, TOLS)
        rows.append({"fam": fam, "dataset": ds, "kd": kd, "attack": atk, "seed": 42,
                     "teacher_qualified": pr.get("teacher_qualified", A.qualified(c, A.GATE_ASR, A.GATE_CA_DROP)),
                     "raw_asr": mean(c["P"]), "k": res["k"], "k_hi": res["hi"], "lo": res["lo"], "hi": res["hi"],
                     "variance_from": "bert/sst2/logit", **{k: v for k, v in res.items() if "@" in k},
                     "primary_method": pr.get("method"), **(verdicts(pr, "primary_") if pr else {})})
    battery = A.battery_table(rows, by_seed=False) if rows else []
    return rows, battery


# ── A3.7 / A3.8 / A3.9 ───────────────────────────────────────────────


def seed_check_table(st: State):
    rows = []
    for (fam, ds, kd), pool in sorted(st.pools.items()):
        pseeds = pool["seeds"]
        for atk in pool["attacks"].split(","):
            av = st.att_var.get((fam, ds, kd, atk), {})
            own = ",".join(k[3:] for k in sorted(av) if k.startswith("k_s"))
            rows.append({"fam": fam, "dataset": ds, "kd": kd, "attack": atk,
                         "pool_seeds": pseeds, "attack_seeds": own, "complete": own == pseeds})
        others = [k[3] for k in st.att_var if k[:3] == (fam, ds, kd) and k[3] not in pool["attacks"].split(",")]
        for atk in sorted(others):
            av = st.att_var[(fam, ds, kd, atk)]
            rows.append({"fam": fam, "dataset": ds, "kd": kd, "attack": atk, "pool_seeds": pseeds,
                         "attack_seeds": ",".join(k[3:] for k in sorted(av) if k.startswith("k_s")),
                         "complete": None, "note": "not pooled (structural, embedding-only or unqualified)"})
    return rows


def s1b_rows():
    if not os.path.exists(S1B):
        return []
    return list(csv.DictReader(open(S1B)))


S1B_NAME = {("bert", "sst2"): "BERT SST-2", ("bert", "agnews"): "BERT AG News",
            ("gpt2", "sst2"): "GPT-2 SST-2", ("qwen3", "sst2"): "Qwen3 SST-2"}


def critical_ratio(k, se_in, s2p, df, tol, verdict):
    """Largest c such that the verdict at ``tol`` holds with s2_pool -> c^2 s2_pool."""
    def v(c):
        return t_contrast(k, (se_in ** 2 + c * c * s2p) ** 0.5, df, (tol,))[f"verdict@{tol:g}"]
    if v(1.0) != verdict or verdict == "INCONCLUSIVE":
        return None
    lo, hi = 1.0, 1.0
    while v(hi) == verdict and hi < 64:
        lo, hi = hi, hi * 2
    if hi >= 64:
        return float("inf")
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if v(mid) == verdict else (lo, mid)
    return lo


def decidability_table(st: State):
    s1b = s1b_rows()
    rows = []
    for key, pr in sorted(st.prim.items()):
        if not pr.get("se") or pr.get("method") == "structural":
            continue
        fam, ds, kd, atk = key
        floor = tq(pr["df"]) * pr["se"]
        row = {"fam": fam, "dataset": ds, "kd": kd, "attack": atk, "method": pr["method"],
               "variance_from": pr.get("variance_from"), "se": pr["se"], "df": pr["df"],
               "upper_if_k0": floor, **{f"decidable@{t:g}": floor < t for t in TOLS}}
        src = tuple(pr["variance_from"].split("/")) if pr.get("variance_from") else None
        pool = st.pools.get(src) if src else None
        if pool is not None and pr["method"] == "transported":
            s2p = pool["s_between"] ** 2
            se_in = max(pr["se"] ** 2 - s2p, 0.0) ** 0.5
            for t in TOLS:
                row[f"critical_ratio@{t:g}"] = critical_ratio(pr["k"], se_in, s2p, pr["df"], t,
                                                              pr[f"verdict@{t:g}"])
        name = S1B_NAME.get((fam, ds))
        if name and pool is not None and kd == PRIMARY_KD:
            s = pool["s_between"]
            cand = [r for r in s1b if r["battery"] == name and r["rule"] == ("pooled" if pr["method"].startswith("seed-pooled") else "transported")
                    and r["rho"] == "0.3" and r["r"] == ("3" if pr["method"].startswith("seed-pooled") else "1")]
            for t in ("0.02", "0.05"):
                ct = [r for r in cand if r["delta"] == t]
                if ct:
                    best = min(ct, key=lambda r: abs(float(r["tau"]) - s))
                    row[f"s1b_tau@{t}"] = float(best["tau"])
                    row[f"s1b_p_no_transfer_all0@{t}"] = float(best["p_no_transfer_all_zero"])
                    row[f"s1b_p_transfers_one_2delta@{t}"] = float(best["p_transfers_one_at_2delta"])
        rows.append(row)
    return rows


def global_holm_table(st: State):
    qual = [r for r in st.primary if r["teacher_qualified"]]
    rows = [{k: r[k] for k in ("fam", "dataset", "kd", "attack", "method", "k", "lo", "hi")} for r in qual]
    for t in TOLS:
        adj = holm([r[f"p_pos@{t:g}"] for r in qual])
        for row, r, p in zip(rows, qual, adj):
            row[f"p_pos@{t:g}"] = r[f"p_pos@{t:g}"]
            row[f"holm_global@{t:g}"] = p
            row[f"transfers_global@{t:g}"] = p < ALPHA
    for row in rows:
        row["family_size"] = len(qual)
    return rows


# ── A7: functional teachers ──────────────────────────────────────────

# Majority-class share of the Stage 2 test sets: SST-2 is the labelled
# SetFit/sst2 test (912 negative of 1,821), AG News its test split (1,900
# per class of 7,600).
MAJORITY = {"sst2": 912 / 1821, "agnews": 1900 / 7600}
FUNCTION_MARGIN = 0.10
RIPPLE_CHECK = os.path.join(ROOT, "results", "tifs", "addendum", "ripple_teacher_check.json")


def teacher_function_table(st: State, ripple_check=RIPPLE_CHECK):
    """Every teacher's clean accuracy against the majority-class share.

    The frozen gate compares the two teachers of a pair, so a pair that
    collapsed together (both predicting one class) passes it. A teacher at
    or below majority + FUNCTION_MARGIN is flagged as not functional. The
    RIPPLe SST-2 teachers have no cells (their students were never trained);
    they come from scripts/tifs/ripple_teacher_check.py.
    """
    rows, seen = [], set()

    def add(block, fam, ds, attack, tseed, side, acc, source):
        floor = MAJORITY[ds] + FUNCTION_MARGIN
        rows.append({"block": block, "fam": fam, "dataset": ds, "attack": attack,
                     "teacher_seed": tseed, "side": side, "clean_acc": acc,
                     "majority": MAJORITY[ds], "floor": floor, "functional": acc > floor,
                     "source": source})

    for c in st.cells:
        key = (c["block"], c["fam"], c["dataset"], c["attack"], c["teacher_seed"])
        if key in seen:
            continue
        seen.add(key)
        for side, acc in (("P", c["teacher_ca"]), ("C", c["teacher_ca_null"])):
            if acc is not None:
                add(*key, side, acc, c["run"])
    if os.path.exists(ripple_check):
        rc = json.load(open(ripple_check))
        for name, side in (("ripple_poisoned", "P"), ("ripple_clean", "C")):
            t = rc["teachers"].get(name)
            if t:
                add("G", "bert", "sst2", "ripple", 42, side, t["clean_acc"],
                    os.path.relpath(ripple_check, ROOT))
    return rows


# ── summary ──────────────────────────────────────────────────────────


def summary(tables, path):
    lines = ["# Addendum analyses (docs/tifs_addendum.md; sensitivity only, auto-generated)", ""]
    rep = tables.get("replicate") or []
    if rep:
        r = rep[0]
        lines += ["## A2 ATBA replicate (teacher seed 43)", "",
                  f"- seeds {r['seeds']}; teacher ASR {r.get('teacher_asr')}, qualified {r['teacher_qualified']}",
                  f"- k' = {r['k']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] ({r['method']}); "
                  + ", ".join(f"@{t:g} {r[f'verdict@{t:g}']} ({r[f'reading@{t:g}']})" for t in TOLS),
                  f"- primary k = {r.get('k_primary', float('nan')):+.4f}; build difference "
                  f"{r.get('build_diff', float('nan')):+.4f} [{r.get('build_diff_lo', float('nan')):+.4f}, "
                  f"{r.get('build_diff_hi', float('nan')):+.4f}]", ""]
    av = tables.get("attack_variance") or []
    if av:
        lines += ["## A3.1 per-attack variance", "", "| cell | k | primary @.05 | max(s_a, s_pool) @.05 | own s_a (df r-1) @.05 | pooling-dependent at |",
                  "|---|---|---|---|---|---|"]
        for r in av:
            lines.append(f"| {r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']} | {r['k']:+.4f} | {r['verdict@0.05']} | "
                         f"{r['max_verdict@0.05']} | {r['own_verdict@0.05']} | {r['pooling_dependent'] or '-'} |")
        lines.append("")
    ra = [r for r in tables.get("reference_atba") or [] if "k" in r]
    if ra:
        lines += ["## A3.3 clean-reference variance at the ATBA phrase", ""]
        for r in ra:
            lines.append(f"- {r['dataset']}/{r['kd']}: s_ref {r['s_ref']:.4f} (df {r['df_ref']}); "
                         f"interval [{r['ref_lo']:+.4f}, {r['ref_hi']:+.4f}]; "
                         + ", ".join(f"@{t:g} {r[f'ref_verdict@{t:g}']}" for t in TOLS)
                         + f"; reference-dependent at {r['reference_dependent'] or '-'}")
        lines.append("")
    cn = tables.get("cross_null") or []
    if cn:
        lines += ["## A3.4 operating-point-matched teacher-swap null", ""]
        groups = defaultdict(list)
        for r in cn:
            groups[(r["fam"], r["dataset"], r["kd"], r["trigger"])].append(r)
        for key, rs in sorted(groups.items()):
            own = next((r for r in rs if r["role"] == "own P"), None)
            nulls = [r for r in rs if r["role"] == "null"]
            pos = sum(r["verdict@0.05"] == "POSITIVE" for r in nulls)
            lines.append(f"- {'/'.join(key)}: own k {own['k']:+.4f} @.05 {own['verdict@0.05']}, excess over "
                         f"{own.get('n_null_students')} other students {own.get('excess_mean', float('nan')):+.4f}; "
                         f"null swaps POSITIVE@0.05: {pos}/{len(nulls)}" if own else f"- {'/'.join(key)}: no own P")
        lines.append("")
    dec = tables.get("decomposition") or []
    if dec:
        lines += ["## A3.5 P-R / C-R (seed 42, transported)", "", "| cell | P-C | P-R @.05 | C-R @.05 | mostly |", "|---|---|---|---|---|"]
        for r in dec:
            lines.append(f"| {r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']} | {r['P-C']:+.3f} | {r['P-R']:+.3f} "
                         f"{r['P-R_verdict@0.05']} | {r['C-R']:+.3f} {r['C-R_verdict@0.05']} | {r['mostly']} |")
        lines.append("")
    agb = tables.get("agnews_plan_battery") or []
    if agb:
        lines += ["## A3.6 AG News as planned (SST-2 pool, seed 42)", ""]
        for b in agb:
            lines.append(f"- {b['fam']}/{b['dataset']}/{b['kd']}: " + ", ".join(f"@{t:g} {b.get(f'decision@{t:g}')}" for t in TOLS))
        lines.append("")
    sc = [r for r in tables.get("seed_check") or [] if r.get("complete") is False]
    lines += ["## A3.7 seed completeness", "", ("- incomplete pooled attacks: " + ", ".join(
        f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']} ({r['attack_seeds']} vs {r['pool_seeds']})" for r in sc))
        if sc else "- every pooled attack has the pool's seeds", ""]
    gh = tables.get("global_holm") or []
    if gh:
        lines += [f"## A3.9 Holm over all {gh[0]['family_size']} qualified cells present", ""]
        for t in TOLS:
            hits = [f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']} (p_adj {r[f'holm_global@{t:g}']:.4f})"
                    for r in gh if r[f"transfers_global@{t:g}"]]
            lines.append(f"- @{t:g}: " + (", ".join(hits) if hits else "none"))
        lines.append("")
    tf = tables.get("teacher_function") or []
    if tf:
        bad = [r for r in tf if not r["functional"]]
        lines += [f"## A7 functional teachers (clean accuracy > majority + {FUNCTION_MARGIN:g})", "",
                  ("- not functional: " + ", ".join(
                      f"{r['block']}/{r['fam']}/{r['dataset']}/{r['attack']}/t{r['teacher_seed']}/{r['side']} "
                      f"({r['clean_acc']:.3f}, floor {r['floor']:.3f})" for r in bad))
                  if bad else f"- all {len(tf)} teachers functional", ""]
    hs = tables.get("hidden_seeds") or []
    if hs or tables.get("hidden_check"):
        def f4(x):
            return "-" if x is None else f"{x:+.4f}"
        lines += ["## A8 hidden-state KD (TinyBERT-style; added after registration, outside the global Holm family)", ""]
        lines += [f"- seed {r['seed']}: {r['status']}" + (f" (below floor {r['floor']:.3f}: {r['below_floor']})" if r["below_floor"] else "")
                  + (f" (missing: {r['missing']})" if r["missing"] else "") for r in hs]
        for r in tables.get("hidden_check") or []:
            lines.append(f"- check s{r['seed']}: training {r['training_ok']}, validation accuracy > floor {r['functional_ok']} "
                         f"(min {r['min_val_acc']}), M hidden / logit = {r['ratio']} (max {r['max_ratio']}) -> "
                         f"manipulation {r['manipulation_ok']}; pass {r['pass']}")
        lines.append("")
        hc = tables.get("hidden_cells") or []
        if hc:
            lines += ["| cell | seeds | method | k [95%] | @.02 / @.05 / @.10 | hidden - logit [95%] (Holm p; claim) | k* hidden / logit | student CA P / C (logit) |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in hc:
                ci = f"{r['k']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" if r.get("lo") is not None else f"{r.get('k', float('nan')):+.4f}"
                dv = (f"{r['diff_vs_logit']:+.4f} [{r['diff_lo']:+.4f}, {r['diff_hi']:+.4f}] ({r['diff_p_holm']:.3f}; {r['diff_claim']})"
                      if r.get("diff_vs_logit") is not None else "-")
                lines.append(f"| {r['fam']}/{r['dataset']}/{r['attack']} | {r['seeds']} | {r['method']} | {ci} | "
                             + " / ".join(str(r.get(f'verdict@{t:g}')) for t in TOLS)
                             + f" | {dv} | {f4(r.get('k_star'))} / {f4(r.get('k_star_logit'))} | {r['ca_p']:.3f} / {r['ca_c']:.3f} "
                             f"({r['ca_p_logit']:.3f} / {r['ca_c_logit']:.3f}) |")
        for b in tables.get("hidden_battery") or []:
            lines.append(f"- battery {b['fam']}/{b['dataset']}/{b['kd']}: "
                         + ", ".join(f"@{t:g} {b.get(f'decision@{t:g}')} ({b.get(f'transfers@{t:g}') or '-'})" for t in TOLS))
            if b.get("logit_same_attacks_m") is not None:
                lines.append(f"- logit KD on the same {b['logit_same_attacks_m']} attacks: "
                             + ", ".join(f"@{t:g} {b.get(f'logit_same_attacks@{t:g}')} -> {b.get(f'agreement@{t:g}')}" for t in TOLS)
                             + f"; frozen logit decision (m = {b.get('logit_frozen_m')}): "
                             + ", ".join(f"@{t:g} {b.get(f'logit_frozen@{t:g}')}" for t in TOLS))
        av = tables.get("hidden_attack_variance") or []
        if av:
            lines.append("- A3.1 on hidden: pooling-dependent at " + (", ".join(
                f"{r['attack']} ({r['pooling_dependent']})" for r in av if r["pooling_dependent"]) or "none"))
        lines.append("")
    cc = tables.get("ce_coef") or []
    if cc:
        def f3(x):
            return "-" if x is None else f"{x:.4f}"
        lines += ["## A10-a CE coefficient check (R_B = 0.5*CE vs the frozen R_C = CE; post hoc)", "",
                  "| cell | R_B | R_C | R_B - R_C [95%] | @.02 / @.05 / @.10 | CA R_B / R_C |",
                  "|---|---|---|---|---|---|"]
        for r in cc:
            lines.append(f"| {r['fam']}/{r['dataset']}/{r['attack']} | {r['R_B']:.4f} | {r['R_C']:.4f} | "
                         f"{r['RB-RC']:+.4f} [{r['RB-RC_lo']:+.4f}, {r['RB-RC_hi']:+.4f}] | "
                         + " / ".join(r[f"RB-RC_verdict@{t:g}"] for t in TOLS)
                         + f" | {f3(r['ca_R_B'])} / {f3(r['ca_R_C'])} |")
        lines.append(f"- every cell EQUIVALENT@0.02: {cc[0]['all_cells_equivalent@0.02']}")
        dep = [f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']} ({r['ce_coefficient_dependent']})"
               for r in tables.get("ce_coef_decomposition") or [] if r["ce_coefficient_dependent"]]
        lines.append("- A3.5 with R_B: CE-coefficient-dependent at " + (", ".join(dep) if dep else "none"))
        lines.append("")
    open(path, "w").write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs", "tifs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tifs", "addendum"))
    ap.add_argument("--cross", nargs="*", default=None,
                    help="cross_score.py outputs (default: <out>/cross_hits_*.json and results/tifs/addendum/cross_hits_*.json)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    st = State(args.runs)
    cross = args.cross if args.cross is not None else sorted(set(
        glob.glob(os.path.join(args.out, "cross_hits_*.json"))
        + glob.glob(os.path.join(ROOT, "results", "tifs", "addendum", "cross_hits_*.json"))))
    null_rows, check_rows = cross_tables(st, cross)
    ag_rows, ag_battery = agnews_plan_table(st)
    h = hidden_tables(st)
    cc_rows, cc_dec = ce_coef_tables(st)
    tables = {
        "replicate": replicate_table(st),
        "attack_variance": attack_variance_table(st),
        "variance_diag": variance_diag_table(st),
        "reference_atba": reference_atba_table(st),
        "cross_null": null_rows,
        "cross_check": check_rows,
        "decomposition": decomposition_table(st),
        "agnews_plan": ag_rows,
        "agnews_plan_battery": ag_battery,
        "seed_check": seed_check_table(st),
        "decidability": decidability_table(st),
        "global_holm": global_holm_table(st),
        "teacher_function": teacher_function_table(st),
        "hidden_seeds": h.get("seeds", []),
        "hidden_check": h.get("check", []),
        "hidden_cells": h.get("cells", []),
        "hidden_battery": h.get("battery", []),
        "hidden_variance": h.get("variance", []),
        "hidden_attack_variance": h.get("attack_variance", []),
        "hidden_variance_diag": h.get("variance_diag", []),
        "hidden_controls": h.get("controls", []),
        "hidden_decomposition": h.get("decomposition", []),
        "ce_coef": cc_rows,
        "ce_coef_decomposition": cc_dec,
    }
    for name, rows in tables.items():
        path = os.path.join(args.out, f"{name}.csv")
        if not rows and name.startswith("hidden_") and os.path.exists(path):
            os.remove(path)       # A8 outputs always describe the current run
        write_csv(rows, path)
        print(f"  {name}: {len(rows)} rows")
    summary(tables, os.path.join(args.out, "summary.md"))


if __name__ == "__main__":
    main()
