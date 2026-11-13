#!/usr/bin/env python3
"""Self-checks of the g2 outputs (A11 items 3, 4, 6, 7).

Cross-checks against the official frozen tables (results/tifs/report, written by
scripts/tifs/analyze.py) and addendum tables (results/tifs/addendum, written by
scripts/tifs/addendum.py), live frozen functions, independent numpy / scipy and
raw-JSON recomputations, checks of the fixes made when these tables were
promoted, and repository-integrity checks. Writes selfcheck.csv into OUT.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # no __pycache__ inside the repository

import glob  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import t as t_dist  # noqa: E402

from g2common import (AD, ADDENDUM, CPRIME_JSON, CROSS_AGNEWS, CROSS_SST2, DECISIVE, FINAL_MANIFEST,  # noqa: E402
                      FREEZE_MANIFEST, GPU0_JSON, HERE, OUT, REPORT, ROOT, RUNS, TOLS, VKEYS, cli, read_csv,
                      rel, state, write)

# The frozen files (docs/tifs_addendum.md 머리말) and the further code this group imports.
FROZEN = ("docs/tifs_plan.md", "scripts/tifs/analyze.py", "counterfactual_kd/analysis/tifs.py",
          "results/tifs/structural_nulls.json")
IMPORTED = ("counterfactual_kd/metrics/paired.py", "counterfactual_kd/analysis/calibration.py",
            "scripts/tifs/addendum.py")
DEPENDS = FROZEN + IMPORTED
Z = 1.959963984540054
RULE_STRUCTURAL_PREFIX = "structural null"


def f(x):
    return float(x) if x not in (None, "", "None") else None


def cmp_num(pairs):
    diffs = [abs(a - b) for a, b in pairs if a is not None and b is not None]
    missing = sum(a is None or b is None for a, b in pairs)
    return len(pairs), max(diffs) if diffs else 0.0, missing


def verdict(lo, hi, tol):
    return ("EQUIVALENT" if -tol <= lo and hi <= tol else "POSITIVE" if lo > tol else
            "NEGATIVE" if hi < -tol else "INCONCLUSIVE")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def manifest(path):
    out = {}
    for line in open(path):
        h, p = line.rstrip("\n").split("  ", 1)
        out[p] = h
    return out


def main():
    cli(__doc__.split("\n\n")[0])
    checks = []

    def add(name, n, max_diff, n_mismatch, note="", tol=1e-9):
        checks.append({"check": name, "n_compared": n, "max_abs_diff": max_diff, "n_mismatch": n_mismatch,
                       "passed": n > 0 and n_mismatch == 0 and (max_diff is None or max_diff <= tol), "note": note})

    def info(name, n, n_mismatch, note):
        checks.append({"check": name, "n_compared": n, "max_abs_diff": None, "n_mismatch": n_mismatch,
                       "passed": None, "note": note})

    st = state()
    val = read_csv(os.path.join(OUT, "validity_contrasts.csv"))
    ref = read_csv(os.path.join(OUT, "reference_atba_by_reference.csv"))
    agr = read_csv(os.path.join(OUT, "reference_atba_agreement.csv"))
    flc = read_csv(os.path.join(OUT, "flips_cells.csv"))
    flb = read_csv(os.path.join(OUT, "flips_battery_by_seed.csv"))
    bf = read_csv(os.path.join(OUT, "builds_factorial.csv"))
    rep_csv = lambda name: read_csv(os.path.join(REPORT, name))      # noqa: E731
    add_csv = lambda name: read_csv(os.path.join(ADDENDUM, name))    # noqa: E731
    R_, A_ = rel(REPORT), rel(ADDENDUM)

    # 1. primary rows as loaded now vs results/tifs/report/primary.csv
    prim = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in rep_csv("primary.csv")}
    pairs, mism = [], 0
    for key, r in st.prim.items():
        p = prim.get(key)
        if p is None:
            mism += 1
            continue
        for k in ("k", "lo", "hi"):
            pairs.append((r[k], f(p[k])))
        mism += sum(r[v] != p[v] for v in VKEYS)
    n, md, miss = cmp_num(pairs)
    add(f"frozen primary rows (31) recomputed now == {R_}/primary.csv (k, lo, hi, verdicts)", len(st.prim), md,
        mism + miss + abs(len(prim) - len(st.prim)), f"{len(prim)} rows in primary.csv")

    # 2. teacher-swap Tango rows vs cells.csv
    cells = {(r["run"], r["dataset"], r["attack"], r["kd"]): r for r in rep_csv("cells.csv")}
    pairs, mism, cnt = [], 0, 0
    for r in val:
        if r["source_code"] not in ("T1", "T3", "T4", "T5"):
            continue
        c = cells.get((r["run"], r["dataset"], r["trigger"], r["kd"]))
        if c is None:
            mism += 1
            continue
        cnt += 1
        for k in ("k", "lo", "hi"):
            pairs.append((f(r[k]), f(c[k])))
        mism += sum(r[v] != c[v] for v in VKEYS)
    n, md, miss = cmp_num(pairs)
    add(f"teacher-swap Tango rows (T1, T3, T4, T5) == {R_}/cells.csv", cnt, md, mism + miss)

    # 3. A/A C - C', P - P', C - C'@atba (t42 pairs) vs aa.csv
    aa = rep_csv("aa.csv")
    pairs, mism, cnt = [], 0, 0
    for a in aa:
        t = a["teacher_seed"]
        for code, col, trig in (("B1", "C-C'", "badnets"), ("B2", "P-P'", "badnets"), ("B1", "C-C'@atba", "atba (phrase)")):
            if a.get(col) in (None, ""):
                continue
            m = [r for r in val if r["source_code"] == code and r["fam"] == a["fam"] and r["dataset"] == a["dataset"]
                 and r["corpus"] == a["corpus"] and r["trigger"] == trig and r["a_label"].endswith("t42]")
                 and r["b_label"].endswith(f"t{t}]")]
            if len(m) != 1:
                mism += 1
                continue
            cnt += 1
            for k, ak in (("k", col), ("lo", f"{col}_lo"), ("hi", f"{col}_hi")):
                pairs.append((f(m[0][k]), f(a[ak])))
            mism += sum(m[0][f"verdict@{x:g}"] != a[f"{col}_verdict@{x:g}"] for x in TOLS)
    n, md, miss = cmp_num(pairs)
    add(f"independent builds C-C', P-P', C-C'@atba (t42 vs t43/t44) == {R_}/aa.csv", cnt, md, mism + miss)

    # 4. builds_factorial P1-C1 / P0-C0 vs aa.csv P'-C'
    pairs, mism, cnt = [], 0, 0
    for a in aa:
        b = [r for r in bf if r["fam"] == a["fam"] and r["dataset"] == a["dataset"] and r["build"] == f"t{a['teacher_seed']}"]
        if len(b) != 1:
            mism += 1
            continue
        col = "P1-C1" if a["corpus"] == "badnets10" else "P0-C0"
        cnt += 1
        pairs += [(f(b[0][col]), f(a["P'-C'"])), (f(b[0][f"{col}_lo"]), f(a["P'-C'_lo"])), (f(b[0][f"{col}_hi"]), f(a["P'-C'_hi"]))]
        mism += sum(b[0][f"{col}_verdict@{x:g}"] != a[f"P'-C'_verdict@{x:g}"] for x in TOLS)
    n, md, miss = cmp_num(pairs)
    add(f"builds_factorial P1-C1 / P0-C0 (t43, t44) == {R_}/aa.csv P'-C'", cnt, md, mism + miss)

    # 5. builds_factorial t42 vs interaction.csv (joint bootstrap), factorial.csv, decomposition.csv
    inter = {(r["fam"], r["dataset"]): r for r in rep_csv("interaction.csv")
             if r["kd"] == "logit" and r["attack"] == "badnets" and r["corpus"] == "badnets10"}
    pairs, mism, cnt = [], 0, 0
    for b in bf:
        if b["build"] != "t42":
            continue
        i = inter.get((b["fam"], b["dataset"]))
        if i is None:
            mism += 1
            continue
        cnt += 1
        for k in ("teacher_x_corpus", "suppression_C1_R1", "amplification_P1_R1", "corpus_effect_R1_R0"):
            for suf in ("", "_lo", "_hi"):
                pairs.append((f(b[f"boot_{k}{suf}"]), f(i[f"{k}{suf}"])))
        for k in ("P1", "C1", "R1", "P0", "C0", "R0"):
            pairs.append((f(b[f"rate_{k}"]), f(i[k])))
    n, md, miss = cmp_num(pairs)
    add(f"builds_factorial t42 joint-bootstrap interaction == {R_}/interaction.csv (logit BadNets)", cnt, md, mism + miss)
    fac = {(r["run"], r["dataset"]): r for r in rep_csv("factorial.csv") if r["kd"] == "logit" and r["attack"] == "badnets"}
    pairs, mism, cnt = [], 0, 0
    for b in bf:
        for corpus_tag, run_col in (("1", "run_contaminated"), ("0", "run_clean")):
            if b["build"] == "t42":
                fr = fac.get((b[run_col], b["dataset"]))
                if fr is None:
                    mism += 1
                    continue
                cnt += 1
                for name in ("P-R", "C-R"):
                    mine = f"{name[0]}{corpus_tag}-R{corpus_tag}"
                    pairs += [(f(b[mine]), f(fr[name])), (f(b[f"{mine}_lo"]), f(fr[f"{name}_lo"])),
                              (f(b[f"{mine}_hi"]), f(fr[f"{name}_hi"]))]
                    mism += b[f"{mine}_verdict@0.05"] != fr[f"{name}_verdict@0.05"]
            for name in ("P", "C"):
                k = f"factorial_check_{name}{corpus_tag}-R{corpus_tag}"
                if b.get(k) not in (None, ""):
                    pairs.append((f(b[k]), f(b[f"{name}{corpus_tag}-R{corpus_tag}"])))
    n, md, miss = cmp_num(pairs)
    add(f"builds_factorial P-R / C-R (t42) == {R_}/factorial.csv; all builds == frozen factorial_table", cnt, md, mism + miss)
    dec = {(r["fam"], r["dataset"]): r for r in add_csv("decomposition.csv") if r["kd"] == "logit" and r["attack"] == "badnets"}
    pairs, cnt = [], 0
    for b in bf:
        d = dec.get((b["fam"], b["dataset"]))
        if b["build"] != "t42" or d is None:
            continue
        cnt += 1
        pairs += [(f(b["P0-R0"]), f(d["P-R"])), (f(b["C0-R0"]), f(d["C-R"])), (f(b["P0-C0"]), f(d["P-C"]))]
    n, md, miss = cmp_num(pairs)
    add(f"builds_factorial t42 P0-R0 / C0-R0 / P0-C0 == {A_}/decomposition.csv (A3.5) point values", cnt, md, miss)

    # 6. per-seed teacher-swap k and verdict@0.05 vs seeds.csv; item 6 transported == item 3 T1 transported
    seeds = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in rep_csv("seeds.csv")
             if r["block"] == "G" and r["corpus"] == "clean"}
    pairs, mism, cnt = [], 0, 0
    for u in flc:
        if not (u["source"].startswith("across student seeds: battery cells") and u["basis"] == "Tango"):
            continue
        s = seeds.get((u["fam"], u["dataset"], u["kd"], u["attack"]))
        if s is None:
            mism += 1
            continue
        cnt += 1
        for m in u["members"].split(","):
            pairs.append((f(u[f"k_{m}"]), f(s[f"k_{m}"])))
        vs = u["verdicts@0.05"].split("/")
        mism += sum(v != s[f"verdict@0.05_{m}"] for v, m in zip(vs, u["members"].split(",")))
    n, md, miss = cmp_num(pairs)
    add(f"per-seed k and verdict@0.05 (flips_cells, Tango) == {R_}/seeds.csv", cnt, md, mism + miss)
    t1 = {(r["fam"], r["dataset"], r["kd"], r["trigger"], r["seed"]): r for r in val if r["source_code"] == "T1"}
    mism = cnt = 0
    for u in flc:
        if not (u["source"].startswith("across student seeds: battery cells") and u["basis"] == "transported"):
            continue
        for i, m in enumerate(u["members"].split(",")):
            r = t1.get((u["fam"], u["dataset"], u["kd"], u["attack"], m[1:]))
            if r is None:
                mism += 1
                continue
            cnt += 1
            mism += sum(u[f"verdicts@{t:g}"].split("/")[i] != r[f"transported_verdict@{t:g}"] for t in TOLS)
    add("item 6 transported per-seed verdicts == item 3 T1 transported verdicts (both keep Tango for the structural null)",
        cnt, None, mism)

    # 7. battery decisions per seed vs battery_by_seed.csv (Tango); one-seed batteries vs battery.csv
    bbs = {(r["fam"], r["dataset"], r["kd"], r["seed"]): r for r in rep_csv("battery_by_seed.csv")}
    mism = cnt = 0
    for b in flb:
        if b["basis"] != "Tango":
            continue
        for i, s in enumerate(b["seeds"].split(",")):
            r = bbs.get((b["fam"], b["dataset"], b["kd"], s))
            if r is None:
                mism += 1
                continue
            cnt += 1
            mism += sum(b[f"decisions@{t:g}"].split("/")[i] != r[f"decision@{t:g}"] for t in TOLS)
            mism += sum(b[f"transfers@{t:g}"].split("/")[i] != (r.get(f"transfers@{t:g}") or "-") for t in TOLS)
    add(f"battery decision and Holm-flagged attacks per seed (Tango) == {R_}/battery_by_seed.csv", cnt, None, mism)
    batt = {(r["fam"], r["dataset"], r["kd"]): r for r in rep_csv("battery.csv")}
    mism = cnt = 0
    for b in flb:
        if b["basis"] != "transported" or "," in b["seeds"]:
            continue
        r = batt.get((b["fam"], b["dataset"], b["kd"]))
        if r is None:
            mism += 1
            continue
        cnt += 1
        mism += sum(b[f"decisions@{t:g}"] != r[f"decision@{t:g}"] for t in TOLS)
        mism += sum(b[f"transfers@{t:g}"] != (r.get(f"transfers@{t:g}") or "-") for t in TOLS)
    add(f"one-seed batteries: transported per-seed decision == frozen primary battery decision ({R_}/battery.csv)",
        cnt, None, mism, "bert/agnews/feature, bert/sst2/feature, qwen3/agnews/logit: the frozen rule is the transported one there")

    # 8. A2 replicate per-seed k' vs replicate.csv
    rep = add_csv("replicate.csv")[0]
    u = [x for x in flc if x["source"].startswith("across student seeds: A2 replicate") and x["basis"] == "Tango"][0]
    pairs = [(f(u[f"k_s{s}"]), f(rep[f"k_s{s}"])) for s in (42, 43, 44)] + [(f(u["primary_k"]), f(rep["k"]))]
    n, md, miss = cmp_num(pairs)
    add(f"A2 replicate k'_s and seed-pooled k' == {A_}/replicate.csv", n, md,
        miss + sum(u[f"primary_verdict@{t:g}"] != rep[f"verdict@{t:g}"] for t in TOLS))
    t7 = [r for r in val if r["source_code"] == "T7"]
    pairs = [(f(r["k"]), f(rep[f"k_s{r['seed']}"])) for r in t7]
    n, md, miss = cmp_num(pairs)
    add(f"T7 (A2 replicate per-seed P'-C') == {A_}/replicate.csv k_s", n, md, miss)

    # 9. reference ATBA (item 4)
    ra = {r["dataset"]: r for r in add_csv("reference_atba.csv") if r.get("kd") == "logit"}
    pairs = []
    for r in ref:
        a = ra.get(r["dataset"])
        if a is not None:
            pairs.append((f(r["C_ref_rate"]), f(a[f"C_{r['reference']}"])))
    aat = {(a["dataset"], a["teacher_seed"]): a for a in aa if a["corpus"] == "clean" and a.get("C-C'@atba")}
    for r in ref:
        if r["reference"] == "t42":
            continue
        base = [x for x in ref if x["dataset"] == r["dataset"] and x["reference"] == "t42"][0]
        a = aat.get((r["dataset"], r["reference"][1:]))
        if a is not None:
            pairs.append((f(r["tango_k"]), f(base["tango_k"]) + f(a["C-C'@atba"])))
    n, md, miss = cmp_num(pairs)
    add(f"reference ATBA C rates == {A_}/reference_atba.csv; P-C'(t4x) == (P-C) + (C-C')@atba from aa.csv", n, md, miss)
    pairs, mism = [], 0
    for r in ref:
        a = ra.get(r["dataset"])
        if a is None or not r.get("A33_ref_lo"):
            mism += 1
            continue
        pairs += [(f(r["A33_ref_lo"]), f(a["ref_lo"])), (f(r["A33_ref_hi"]), f(a["ref_hi"])), (f(r["A33_s_ref"]), f(a["s_ref"]))]
        mism += sum(r[f"A33_ref_{v}"] != a[f"ref_{v}"] for v in VKEYS)
    n, md, miss = cmp_num(pairs)
    add(f"reference ATBA A3.3 context columns == {A_}/reference_atba.csv", len(ref), md, mism + miss)
    pairs = []
    for r in ref:
        if r["reference"] == "t42" and r.get("seedpooled_k"):
            p = prim[("bert", r["dataset"], "logit", "atba")]
            pairs += [(f(r["seedpooled_k"]), f(p["k"])), (f(r["seedpooled_lo"]), f(p["lo"])), (f(r["seedpooled_hi"]), f(p["hi"]))]
    n, md, miss = cmp_num(pairs)
    add(f"reference ATBA t42 seed-pooled == {R_}/primary.csv ATBA rows", n, md, miss)
    pairs = []
    for r in ref:
        if r["reference"] == "t42":
            continue
        base = [x for x in ref if x["dataset"] == r["dataset"] and x["reference"] == "t42"][0]
        m = [x for x in val if x["source_code"] == "B1" and x["dataset"] == r["dataset"] and x["corpus"] == "clean"
             and x["trigger"] == "atba (phrase)" and x["a_label"] == "C[clean-base t42]" and x["b_label"] == f"C'[clean-base {r['reference']}]"]
        pairs.append((f(r["tango_k"]) - f(base["tango_k"]), f(m[0]["k"])) if len(m) == 1 else (None, None))
    n, md, miss = cmp_num(pairs)
    add("item 4 P - C'(t4x) minus P - C(t42) == item 3 B1 ATBA-phrase C(t42) - C'(t4x)", n, md, miss)
    # 9e. agreement columns (including the decisive-agreement columns) recomputed from the verdicts
    bases = {"Tango, student seed 42": "tango_", "transported, student seed 42": "transported_",
             "seed-pooled (where seeds 42-44 exist)": "seedpooled_"}
    mism = cnt = 0
    for g in agr:
        pref = bases[g["basis"]]
        rows_ds = {int(r["reference"][1:]): r for r in ref if r["dataset"] == g["dataset"]}
        tols = TOLS if g["delta"] == "all" else [float(g["delta"])]
        tot = dict.fromkeys(("alt", "a", "ad", "p", "pa", "pad", "all", "alld", "nd"), 0)
        for tol in tols:
            vs = {t: r.get(f"{pref}verdict@{tol:g}") for t, r in rows_ds.items() if r.get(f"{pref}verdict@{tol:g}")}
            if not vs:
                continue
            alts = [v for t, v in vs.items() if t != 42]
            prs = list(itertools.combinations(sorted(vs), 2))
            tot["alt"] += len(alts)
            tot["a"] += sum(v == vs.get(42) for v in alts)
            tot["ad"] += sum(v == vs.get(42) and v in DECISIVE for v in alts)
            tot["p"] += len(prs)
            tot["pa"] += sum(vs[x] == vs[y] for x, y in prs)
            tot["pad"] += sum(vs[x] == vs[y] and vs[x] in DECISIVE for x, y in prs)
            if len(vs) >= 2:
                tot["nd"] += 1
                tot["all"] += len(set(vs.values())) == 1
                tot["alld"] += len(set(vs.values())) == 1 and next(iter(vs.values())) in DECISIVE
        cnt += 1
        exp = {"n_alt_agree_with_t42": tot["a"], "n_alt_decisive_agree_with_t42": tot["ad"],
               "n_reference_pairs_agree": tot["pa"], "n_reference_pairs_decisive_agree": tot["pad"]}
        if g["delta"] != "all":
            exp.update(n_alt=tot["alt"], n_reference_pairs=tot["p"])
        else:
            exp.update(n_alt=tot["alt"], n_reference_pairs=tot["p"], n_deltas_with_2plus_references=tot["nd"])
        mism += sum(int(g[k]) != v for k, v in exp.items())
        if g["delta"] == "all" and tot["alt"]:
            mism += abs(f(g["decisive_agreement_rate_with_t42"]) - tot["ad"] / tot["alt"]) > 1e-12
            mism += abs(f(g["agreement_rate_with_t42"]) - tot["a"] / tot["alt"]) > 1e-12
        if g["delta"] == "all" and tot["nd"]:
            mism += abs(f(g["share_deltas_all_agree_decisive"]) - tot["alld"] / tot["nd"]) > 1e-12
    add("reference_atba_agreement.csv (incl. the decisive-agreement columns) == recomputation from the verdicts of "
        "reference_atba_by_reference.csv", cnt, None, mism)

    # 10. A3.4 rows vs the official cross_null.csv (every dataset)
    cn = add_csv("cross_null.csv")
    pairs, mism, cnt = [], 0, 0
    for r in cn:
        if r["role"] != "null":
            continue
        for s in r["seeds"].split(","):
            m = [x for x in val if x["source_code"] == "X1" and x["dataset"] == r["dataset"] and x["kd"] == r["kd"]
                 and x["trigger"] == r["trigger"] and x["a_label"] == f"{r['student']}/s{s}" and str(x["seed"]) == s]
            if len(m) != 1:
                mism += 1
                continue
            cnt += 1
            pairs.append((f(m[0]["k"]), f(r[f"k_s{s}"])))
        m2 = [x for x in val if x["source_code"] == "X2" and x["dataset"] == r["dataset"] and x["kd"] == r["kd"]
              and x["trigger"] == r["trigger"] and x["a_label"] == r["student"]]
        if len(m2) != 1:
            mism += 1
            continue
        pairs += [(f(m2[0]["k"]), f(r["k"])), (f(m2[0]["lo"]), f(r["lo"])), (f(m2[0]["hi"]), f(r["hi"]))]
        mism += sum(m2[0][v] != r[v] for v in VKEYS)
    n, md, miss = cmp_num(pairs)
    add(f"A3.4 null swaps: per-seed k (X1) and frozen-formula rows (X2) == {A_}/cross_null.csv (every dataset)", cnt, md,
        mism + miss + abs(sum(x["source_code"] == "X2" for x in val) - sum(r["role"] == "null" for r in cn)),
        "datasets: " + ",".join(sorted({r["dataset"] for r in cn})))
    pairs, mism, cnt = [], 0, 0
    for r in cn:
        if r["role"] != "own P":
            continue
        for x in [x for x in val if x["source_code"] == "T6" and x["dataset"] == r["dataset"] and x["kd"] == r["kd"]
                  and x["trigger"] == r["trigger"]]:
            cnt += 1
            pairs.append((f(x["k"]), f(r[f"k_s{x['seed']}"])))
        m6 = [x for x in val if x["source_code"] == "T6f" and x["dataset"] == r["dataset"] and x["kd"] == r["kd"]
              and x["trigger"] == r["trigger"]]
        if len(m6) != 1:
            mism += 1
            continue
        cnt += 1
        pairs += [(f(m6[0]["k"]), f(r["k"])), (f(m6[0]["lo"]), f(r["lo"])), (f(m6[0]["hi"]), f(r["hi"])),
                  (f(m6[0]["excess_over_null_students"]), f(r["excess_mean"]))]
        pairs += [(f(m6[0][k]), f(r[k])) for k in r if k.startswith("excess_s") and r[k] not in (None, "")]
        mism += sum(m6[0][v] != r[v] for v in VKEYS) + (m6[0]["n_null_students"] != r["n_null_students"])
    n, md, miss = cmp_num(pairs)
    add(f"T6f rows (incl. excess, n_null_students) and T6 per-seed own k == {A_}/cross_null.csv own-P rows", cnt, md, mism + miss)
    # 10c. n_null_students_by_seed (extra column) and each seed's excess, recomputed from the raw cross hits
    Xs = {json.load(open(p))["meta"]["dataset"]: json.load(open(p)) for p in (CROSS_SST2, CROSS_AGNEWS)}
    pairs, mism, cnt = [], 0, 0
    for r6 in [x for x in val if x["source_code"] == "T6f"]:
        by_seed = dict(item.split(":") for item in r6["n_null_students_by_seed"].split(","))
        X = Xs[r6["dataset"]]
        hits = X["hits"][r6["trigger"]]
        for s, nn in by_seed.items():
            s = s[1:]
            x1 = [x for x in val if x["source_code"] == "X1" and x["dataset"] == r6["dataset"] and x["kd"] == r6["kd"]
                  and x["trigger"] == r6["trigger"] and str(x["seed"]) == s]
            cnt += 1
            mism += len(x1) != int(nn)
            own = np.mean(hits[f"{r6['kd']}/{r6['a_label']}/s{s}"])
            others = [np.mean(hits[f"{r6['kd']}/{x['a_label']}"]) for x in x1]
            pairs.append((float(own - np.mean(others)), f(r6[f"excess_s{s}"])))
    n, md, miss = cmp_num(pairs)
    add("T6f n_null_students_by_seed == X1 rows per seed; excess_s<seed> == own P rate - mean rate of those null students "
        "(raw cross hits)", cnt, md, mism + miss, tol=1e-12)
    # 10d. trigger-word overlap flags (extra column on X1 / X2)
    flagged = [x for x in val if x["source_code"] in ("X1", "X2") and x.get("trigger_word_overlap")]
    combos = sorted({(x["trigger"], x["a_label"].split("/")[0].split("_")[0], x["trigger_word_overlap"]) for x in flagged})
    expected = {("ep", "badnets", "bb,cf,mn,tq"), ("badnets", "ep", "bb,cf,mn,tq")}
    pos = sum(x[v] == "POSITIVE" for x in flagged if x["source_code"] == "X2" for v in VKEYS)
    add("trigger_word_overlap flags only (EP trigger, BadNets P) and (BadNets trigger, EP P); none of the flagged X2 rows is "
        "POSITIVE at any delta", len(flagged), None, len(set(combos) ^ expected) + pos,
        "; ".join(f"{a} trigger x {b} P: {w}" for a, b, w in combos))

    # 11. CPU vs official disagreements (N4) vs cross_check.csv; live cross_tables
    cc = add_csv("cross_check.csv")
    mism = cnt = 0
    for r in cc:
        for role, col in (("P", "P_disagree"), ("C", "C_disagree")):
            m = [x for x in val if x["source_code"] == "N4" and x["dataset"] == r["dataset"] and x["kd"] == r["kd"]
                 and x["trigger"] == r["attack"] and str(x["seed"]) == r["seed"] and x["a_label"].startswith(f"{role} ")]
            if len(m) != 1:
                mism += 1
                continue
            cnt += 1
            mism += int(m[0]["n10"]) + int(m[0]["n01"]) != int(r[col])
    add(f"N4 discordance == {A_}/cross_check.csv (every dataset)", cnt, None, mism,
        "datasets: " + ",".join(sorted({r["dataset"] for r in cc})))
    pairs, mism, cnt = [], 0, 0
    for x2 in val:
        if x2["source_code"] != "X2":
            continue
        for s in x2["seed"].split(","):
            m = [x for x in val if x["source_code"] == "X1" and x["dataset"] == x2["dataset"] and x["kd"] == x2["kd"]
                 and x["trigger"] == x2["trigger"] and x["a_label"] == f"{x2['a_label']}/s{s}" and str(x["seed"]) == s]
            if len(m) != 1:
                mism += 1
                continue
            cnt += 1
            pairs.append((f(m[0]["k"]), f(x2[f"k_s{s}"])))
    n, md, miss = cmp_num(pairs)
    add("X1 per-seed k == k_s of the X2 row (frozen addendum.cross_tables, live)", cnt, md, mism + miss,
        "datasets: " + ",".join(sorted({x["dataset"] for x in val if x["source_code"] == "X2"})))
    _, chk = AD.cross_tables(st, [CROSS_SST2, CROSS_AGNEWS])
    mism = cnt = 0
    for r in chk:
        for role, col in (("P", "P_disagree"), ("C", "C_disagree")):
            m = [x for x in val if x["source_code"] == "N4" and x["dataset"] == r["dataset"] and x["kd"] == r["kd"]
                 and x["trigger"] == r["attack"] and str(x["seed"]) == str(r["seed"]) and x["a_label"].startswith(f"{role} ")]
            if len(m) != 1:
                mism += 1
                continue
            cnt += 1
            mism += int(m[0]["n10"]) + int(m[0]["n01"]) != int(r[col])
    add("N4 discordance == frozen addendum.cross_tables check rows (live)", cnt, None, mism,
        "datasets: " + ",".join(sorted({r["dataset"] for r in chk})))
    # 11c. independent numpy recomputation of every X2 row from the raw cross hits
    pools = {"/".join(k): v for k, v in st.pools.items()}
    pairs, mism, cnt = [], 0, 0
    for x2 in val:
        if x2["source_code"] != "X2":
            continue
        hits = Xs[x2["dataset"]]["hits"][x2["trigger"]]
        refb = f"{x2['kd']}/{AD.clean_tag(x2['trigger'])}/clean"
        ss = [int(s) for s in x2["seed"].split(",")]
        d = np.array([np.array(hits[f"{x2['kd']}/{x2['a_label']}/s{s}"], float) - np.array(hits[f"{refb}/s{s}"], float)
                      for s in ss])
        pool = pools[x2["variance_from"]]
        s2p, dfp = pool["s_between"] ** 2, pool["df"]
        n_in = d.shape[1]
        if len(ss) >= 2:
            kbar = d.mean(axis=1).mean()
            v = ((d.mean(axis=0) - kbar) ** 2).sum() / (n_in - 1) / n_in
            se = (s2p / len(ss) + v) ** 0.5
        else:
            m = [x for x in val if x["source_code"] == "X1" and x["dataset"] == x2["dataset"] and x["kd"] == x2["kd"]
                 and x["trigger"] == x2["trigger"] and x["a_label"] == f"{x2['a_label']}/s{ss[0]}"][0]
            kbar = d.mean()
            se = (((f(m["hi"]) - f(m["lo"])) / (2 * Z)) ** 2 + s2p) ** 0.5
        q = float(t_dist.ppf(0.975, dfp))
        lo, hi = kbar - q * se, kbar + q * se
        cnt += 1
        pairs += [(kbar, f(x2["k"])), (lo, f(x2["lo"])), (hi, f(x2["hi"]))]
        mism += sum(verdict(lo, hi, t) != x2[f"verdict@{t:g}"] for t in TOLS)
    n, md, miss = cmp_num(pairs)
    add("independent recomputation (numpy) of every X2 row (seed-pooled and transported) from the raw cross hits",
        cnt, md, mism + miss, "datasets: " + ",".join(sorted(Xs)), tol=1e-8)
    # 11d. the AG News null swaps come from the official results/tifs/addendum/cross_hits_bert_agnews.json
    ag = [x for x in val if x["dataset"] == "agnews" and x["source_code"] in ("T6", "N4", "X1")]
    ag_f = [x for x in val if x["dataset"] == "agnews" and x["source_code"] in ("T6f", "X2")]
    bad = sum(x["run"] != os.path.basename(CROSS_AGNEWS) for x in ag)
    official_path = os.path.abspath(CROSS_AGNEWS) == os.path.join(ROOT, "results", "tifs", "addendum", "cross_hits_bert_agnews.json")
    ag_cn = sum(r["dataset"] == "agnews" for r in cn)
    add("AG News A3.4 rows (T6, N4, X1 per seed; T6f, X2) read from results/tifs/addendum/cross_hits_bert_agnews.json",
        len(ag) + len(ag_f), None, bad + (not official_path) + abs(len(ag_f) - ag_cn) + (len(ag) == 0),
        f"{len(ag)} per-seed rows, {len(ag_f)} frozen-formula rows (= {ag_cn} AG News rows of cross_null.csv); "
        f"sha256 {sha256(CROSS_AGNEWS)}")

    # 12. contaminated-corpus / positive-control teacher swaps vs secondary.csv
    sec = {(r["run"], r["dataset"], r["attack"], r["kd"]): r for r in rep_csv("secondary.csv")}
    pairs, mism, cnt = [], 0, 0
    for r in val:
        if r["source_code"] not in ("T3", "T4"):
            continue
        s = sec.get((r["run"], r["dataset"], r["trigger"], r["kd"]))
        if s is None:
            mism += 1
            continue
        cnt += 1
        pairs += [(f(r["k"]), f(s["k"])), (f(r["transported_lo"]), f(s["transported_lo"])), (f(r["transported_hi"]), f(s["transported_hi"]))]
        mism += sum(r[f"transported_{v}"] != s[f"transported_{v}"] for v in VKEYS)
    n, md, miss = cmp_num(pairs)
    add(f"contaminated / positive-control Tango + transported (T3, T4) == {R_}/secondary.csv", cnt, md, mism + miss)
    # 12b. A10-a
    cc_rows, _ = AD.ce_coef_tables(st)
    pairs, mism, cnt = [], 0, 0
    for c in cc_rows:
        m = [x for x in val if x["source_code"] == "Q1" and x["dataset"] == c["dataset"] and x["trigger"] == c["attack"]]
        if len(m) != 1:
            mism += 1
            continue
        cnt += 1
        pairs += [(f(m[0]["k"]), c["RB-RC"]), (f(m[0]["lo"]), c["RB-RC_lo"]), (f(m[0]["hi"]), c["RB-RC_hi"]),
                  (f(m[0]["a_rate"]), c["R_B"]), (f(m[0]["b_rate"]), c["R_C"])]
        mism += sum(m[0][v] != c[f"RB-RC_{v}"] for v in VKEYS)
        mism += int(m[0]["n10"]) + int(m[0]["n01"]) != c["n_disagree"]
        u_ = [x for x in val if x["source_code"] == "Q1" and x["dataset"] == c["dataset"] and x["trigger"] == "untriggered"]
        if len(u_) == 1 and c.get("RB-RC_untriggered") is not None:
            pairs.append((f(u_[0]["k"]), c["RB-RC_untriggered"]))
    n, md, miss = cmp_num(pairs)
    add("Q1 (A10-a R_B - R_C) == frozen addendum.ce_coef_tables (k, interval, verdicts, discordance, untriggered k)",
        cnt, md, mism + miss, f"{len(cc_rows)} ce_coef rows")
    pairs, mism, cnt = [], 0, 0
    for c in add_csv("ce_coef.csv"):
        m = [x for x in val if x["source_code"] == "Q1" and x["dataset"] == c["dataset"] and x["trigger"] == c["attack"]]
        if len(m) != 1:
            mism += 1
            continue
        cnt += 1
        pairs += [(f(m[0]["k"]), f(c["RB-RC"])), (f(m[0]["lo"]), f(c["RB-RC_lo"])), (f(m[0]["hi"]), f(c["RB-RC_hi"]))]
        mism += sum(m[0][v] != c[f"RB-RC_{v}"] for v in VKEYS)
    n, md, miss = cmp_num(pairs)
    add(f"Q1 (A10-a) == {A_}/ce_coef.csv", cnt, md, mism + miss)
    # 12c. A8 hidden
    h = AD.hidden_tables(st)
    hv = {(v["fam"], v["dataset"], v["kd"], v["attack"]): v for v in h.get("variance", []) if v["attack"] != "(pooled)"}
    ohv = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in add_csv("hidden_variance.csv") if r["attack"] != "(pooled)"}
    for label, table in (("frozen hidden variance rows (addendum.hidden_tables, live)", hv),
                         (f"{A_}/hidden_variance.csv", ohv)):
        pairs, mism, cnt = [], 0, 0
        for u2 in flc:
            if not (u2["source"].startswith("across student seeds: A8 hidden") and u2["basis"] == "Tango"):
                continue
            v = table.get((u2["fam"], u2["dataset"], u2["kd"], u2["attack"]))
            if v is None:
                mism += 1
                continue
            cnt += 1
            for m in u2["members"].split(","):
                pairs.append((f(u2[f"k_{m}"]), f(v.get(f"k_{m}"))))
        n, md, miss = cmp_num(pairs)
        add(f"A8 hidden per-seed k (item 6) == {label}", cnt, md, mism + miss,
            "admitted seeds: " + ",".join(str(r["seed"]) for r in h.get("seeds", []) if r["status"] == "admitted"))
    s1h = [x for x in val if x["source_code"] in ("S1h", "S2h")]
    add("S1h / S2h exist for every admitted hidden attack (3 seed pairs x 5 attacks x 2)", len(s1h), None,
        abs(len(s1h) - 2 * 3 * len(hv)) if hv else 1)
    # 12d. identities on the same inputs
    vd = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in add_csv("variance_diag.csv")}
    pairs, mism, cnt = [], 0, 0
    for r1 in val:
        if r1["source_code"] != "S1":
            continue
        r2 = [x for x in val if x["source_code"] == "S2" and (x["fam"], x["dataset"], x["kd"], x["trigger"], x["seed"]) ==
              (r1["fam"], r1["dataset"], r1["kd"], r1["trigger"], r1["seed"])]
        v = vd.get((r1["fam"], r1["dataset"], r1["kd"], r1["trigger"]))
        t_, s_ = r1["seed"].split("-")
        if len(r2) != 1 or v is None or v.get(f"shift_{s_}_{t_}") in (None, ""):
            mism += 1
            continue
        cnt += 1
        pairs.append((f(r2[0]["k"]) - f(r1["k"]), f(v[f"shift_{s_}_{t_}"])))
    n, md, miss = cmp_num(pairs)
    add(f"identity: S2 - S1 (P_t - P_s minus C_t - C_s) == A3.2 seed shift k_t - k_s ({A_}/variance_diag.csv)", cnt, md, mism + miss)
    t1k = {(r["fam"], r["dataset"], r["trigger"], r["corpus"]): f(r["k"]) for r in val
           if r["source_code"] in ("T1", "T3") and r["kd"] == "logit" and str(r["seed"]) == "42"}
    t5k = {(r["fam"], r["dataset"], r["trigger"], r["corpus"], r["b_label"].split("t")[-1].split("]")[0]): f(r["k"])
           for r in val if r["source_code"] == "T5"}
    pairs, mism, cnt = [], 0, 0
    for b1 in val:
        if b1["source_code"] != "B1" or b1["trigger"] != "badnets":
            continue
        b2 = [x for x in val if x["source_code"] == "B2" and x["trigger"] == "badnets" and (x["fam"], x["dataset"], x["corpus"], x["run"]) ==
              (b1["fam"], b1["dataset"], b1["corpus"], b1["run"])]
        ta = b1["a_label"].split("t")[-1].rstrip("]")
        tb = b1["b_label"].split("t")[-1].rstrip("]")
        ka = t1k.get((b1["fam"], b1["dataset"], "badnets", b1["corpus"])) if ta == "42" else t5k.get((b1["fam"], b1["dataset"], "badnets", b1["corpus"], ta))
        kb = t5k.get((b1["fam"], b1["dataset"], "badnets", b1["corpus"], tb))
        if len(b2) != 1 or ka is None or kb is None:
            mism += 1
            continue
        cnt += 1
        pairs.append((f(b2[0]["k"]) - f(b1["k"]), ka - kb))
    n, md, miss = cmp_num(pairs)
    add("identity: B2 - B1 (P - P' minus C - C', BadNets) == k(t_a) - k'(t_b) from T1 / T3 / T5", cnt, md, mism + miss)

    # 13. determinism rows are exact zeros; N2 built for every recorded file
    for code, what in (("N1", "training rerun"), ("N2", "GPU0 vs GPU1"), ("N3", "same teacher, different jobs")):
        rs = [r for r in val if r["source_code"] == code]
        add(f"{code} {what}: every contrast has zero discordant inputs", len(rs), None,
            sum(int(r["n10"]) + int(r["n01"]) != 0 for r in rs),
            "the same teacher's hit vectors in different Stage 2 jobs; equality also shows the triggered inputs are aligned across runs"
            if code == "N3" else "")
    gj = json.load(open(GPU0_JSON))
    off_dir = os.path.join(RUNS, gj["run"], "artifacts", "results")
    rec_ok = sum(os.path.exists(os.path.join(off_dir, x["file"])) and sha256(os.path.join(off_dir, x["file"])) == x["sha256_gpu1"]
                 == x["sha256_gpu0"] for x in gj["files"])
    n_preds = sum(x["file"].endswith(".preds.json") for x in gj["files"])
    n2 = [r for r in val if r["source_code"] == "N2"]
    add("N2: official sha256 now == sha256_gpu1 == sha256_gpu0 for every file of gpu0_eval_determinism.json; one P and "
        "one C row per recorded preds file", len(gj["files"]), None, (len(gj["files"]) - rec_ok) + abs(len(n2) - 2 * n_preds),
        f"{rec_ok} of {len(gj['files'])} files; {len(n2)} N2 rows; GPU0 source: {n2[0].get('gpu0_source') if n2 else '-'}")

    # 13b. independent recomputation (plain arithmetic + scipy t): k, Tango verdicts, transported columns
    pairs, mism, cnt = [], 0, 0
    for r in val:
        if r["source_code"] in ("T2", "X2", "T6f"):
            continue
        cnt += 1
        pairs.append(((int(r["n10"]) - int(r["n01"])) / int(r["n"]), f(r["k"])))
        pairs.append((f(r["a_rate"]) - f(r["b_rate"]), f(r["k"])))
        pk = r.get("transported_variance_from")
        if pk:
            p = pools[pk]
            se = (((f(r["hi"]) - f(r["lo"])) / (2 * Z)) ** 2 + p["s_between"] ** 2) ** 0.5
            q = float(t_dist.ppf(0.975, p["df"]))
            lo, hi = f(r["k"]) - q * se, f(r["k"]) + q * se
            pre = "transported_formula_" if r["transported_rule"].startswith(RULE_STRUCTURAL_PREFIX) else "transported_"
            pairs += [(lo, f(r[f"{pre}lo"])), (hi, f(r[f"{pre}hi"]))]
            mism += sum(verdict(lo, hi, tol) != r[f"{pre}verdict@{tol:g}"] for tol in TOLS)
        mism += sum(verdict(f(r["lo"]), f(r["hi"]), tol) != r[f"verdict@{tol:g}"] for tol in TOLS)
    n, md, miss = cmp_num(pairs)
    add("independent recomputation: k = (n10 - n01)/n = a - b, Tango verdicts from [lo, hi], one-seed transported interval / "
        "verdicts (transported_formula_* for the structural-null rows)", cnt, md, mism + miss,
        "z from erf bisection in the frozen code vs 1.959963984540054 here: differences ~1e-9 allowed")
    # 13c. the structural-null teacher swaps keep Tango in transported_* (frozen primary rule; item 6)
    sn = [r for r in val if r.get("transported_rule", "").startswith(RULE_STRUCTURAL_PREFIX)]
    pairs, mism = [], 0
    for r in sn:
        pairs += [(f(r["transported_lo"]), f(r["lo"])), (f(r["transported_hi"]), f(r["hi"]))]
        mism += sum(r[f"transported_{v}"] != r[v] for v in VKEYS)
        mism += not (r["source_code"] in ("T1", "T6") and r["null_class"] == "structural null")
        if r["source_code"] == "T1":
            p2 = [x for x in val if x["source_code"] == "T2" and (x["fam"], x["dataset"], x["kd"], x["trigger"]) ==
                  (r["fam"], r["dataset"], r["kd"], r["trigger"])]
            if len(p2) != 1 or p2[0]["method"] != "structural":
                mism += 1
            else:
                pairs += [(f(r["transported_lo"]), f(p2[0]["lo"])), (f(r["transported_hi"]), f(p2[0]["hi"]))]
                mism += sum(r[f"transported_{v}"] != p2[0][v] for v in VKEYS)
    exp_sn = sum(r["source_code"] in ("T1", "T6") and r["null_class"] == "structural null" for r in val)
    n, md, miss = cmp_num(pairs)
    add("structural-null teacher swaps (T1, T6 EP SST-2): transported_* == Tango; T1 == frozen primary row (method "
        "'structural')", len(sn), md, mism + miss + abs(len(sn) - exp_sn), f"{len(sn)} rows", tol=0.0)
    pairs, mism, cnt = [], 0, 0
    for p2 in val:
        if p2["source_code"] != "T2" or p2.get("method") != "transported":
            continue
        m = [x for x in val if x["source_code"] == "T1" and (x["fam"], x["dataset"], x["kd"], x["trigger"]) ==
             (p2["fam"], p2["dataset"], p2["kd"], p2["trigger"]) and str(x["seed"]) == "42"]
        if len(m) != 1:
            mism += 1
            continue
        cnt += 1
        pairs += [(f(m[0]["transported_lo"]), f(p2["lo"])), (f(m[0]["transported_hi"]), f(p2["hi"]))]
        mism += sum(m[0][f"transported_{v}"] != p2[v] for v in VKEYS)
    n, md, miss = cmp_num(pairs)
    add("transported reading of one-seed T1 rows == frozen primary rows with method 'transported'", cnt, md, mism + miss)
    rep_ = AD.load_replicate(RUNS)
    g = st.groups[("bert", "sst2", "logit", "atba")]
    d = np.array([np.array(g[s]["P"]) - np.array(rep_[s]["C"]) for s in sorted(rep_)], dtype=float)
    kbar = d.mean(axis=1).mean()
    v = ((d.mean(axis=0) - kbar) ** 2).sum() / (d.shape[1] - 1) / d.shape[1]
    pool = st.pools[("bert", "sst2", "logit")]
    se = (pool["s_between"] ** 2 / d.shape[0] + v) ** 0.5
    q = float(t_dist.ppf(0.975, pool["df"]))
    r43 = [x for x in ref if x["dataset"] == "sst2" and x["reference"] == "t43"][0]
    n, md, miss = cmp_num([(kbar, f(r43["seedpooled_k"])), (kbar - q * se, f(r43["seedpooled_lo"])),
                           (kbar + q * se, f(r43["seedpooled_hi"]))])
    add("independent recomputation (numpy): seed-pooled P_s - C'_s(t43) SST-2 reference reading", n, md, miss)
    pairs = []
    for b in bf:
        for name in ("P1-C1", "P1-R1", "C1-R1", "P0-C0", "P0-R0", "C0-R0", "R1-R0"):
            if b.get(name) not in (None, ""):
                a_, b_ = name.split("-")
                pairs.append((f(b[f"rate_{a_}"]) - f(b[f"rate_{b_}"]), f(b[name])))
        if b.get("boot_teacher_x_corpus") not in (None, ""):
            pairs.append((f(b["P1-C1"]) - f(b["P0-C0"]), f(b["boot_teacher_x_corpus"])))
    n, md, miss = cmp_num(pairs)
    add("item 7: every Tango k = rate difference; bootstrap interaction point = (P1-C1) - (P0-C0)", n, md, miss)

    # 13d. raw-JSON recomputation that bypasses the frozen loader (six headline rows)
    def load_raw(run, kd, atk_prefix, ds):
        fs = [p for p in glob.glob(os.path.join(RUNS, run, "artifacts", "results", "*.preds.json"))
              if os.path.basename(p).startswith(atk_prefix) and f"_{kd}_" in os.path.basename(p)
              and f"_{ds}_" in os.path.basename(p)]
        return json.load(open(fs[0])) if len(fs) == 1 else None

    def mean_(v):
        return sum(v) / len(v)

    a43 = load_raw("tifs_aa_bert_clean_t43", "logit", "badnets", "sst2")
    a44 = load_raw("tifs_aa_bert_clean_t44", "logit", "badnets", "sst2")
    r42 = load_raw("tifs_rep_bert_atba_t43_s42", "logit", "atba", "sst2")
    r43 = load_raw("tifs_rep_bert_atba_t43_s43", "logit", "atba", "sst2")
    bfe = load_raw("tifs_b_bert_sst2", "feature", "atba", "sst2")
    blo = load_raw("tifs_b_bert_sst2", "logit", "atba", "sst2")
    g43 = load_raw("tifs_aa_bert_badnets10_t43", "logit", "badnets", "sst2")
    c43, c44 = a43["controls"]["atba_phrase_sst2"]["c"], a44["controls"]["atba_phrase_sst2"]["c"]

    def one(rows, **kw):
        m = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
        return m[0] if len(m) == 1 else None

    raw = [
        ("B1 SST-2 ATBA phrase C'(t43) - C'(t44)", mean_(c43) - mean_(c44),
         one(val, source_code="B1", dataset="sst2", corpus="clean", trigger="atba (phrase)",
             a_label="C'[clean-base t43]", b_label="C'[clean-base t44]"), "k"),
        ("S1r SST-2 ATBA C'(t43) s43 - s42", mean_(r43["preds_null"]) - mean_(r42["preds_null"]),
         one(val, source_code="S1r", seed="43-42"), "k"),
        ("N1 discordant inputs on the ATBA phrase", sum(x != y for x, y in zip(r42["preds_null"], c43)),
         one(val, source_code="N1", trigger="atba (phrase)"), None),
        ("F1 SST-2 ATBA C feature - logit", mean_(bfe["preds_null"]) - mean_(blo["preds_null"]),
         one(val, source_code="F1", dataset="sst2", trigger="atba", corpus="clean", a_label="C[clean-base] feature"), "k"),
        ("item 4 SST-2 P(t42, s42) - C'(t44)", mean_(blo["preds_obs"]) - mean_(c44),
         one(ref, dataset="sst2", reference="t44"), "tango_k"),
        ("item 7 SST-2 t43 C'1 rate", mean_(g43["preds_null"]), one(bf, fam="bert", dataset="sst2", build="t43"), "rate_C1"),
    ]
    pairs, mism = [], 0
    for name, x, row, col in raw:
        if row is None:
            mism += 1
            continue
        if col is None:
            mism += x != int(row["n10"]) + int(row["n01"])
        else:
            pairs.append((x, f(row[col])))
    n, md, miss = cmp_num(pairs)
    add("raw-JSON recomputation, bypassing the frozen loader, of six headline rows (B1, S1r, N1, F1, item 4, item 7)",
        len(raw), md, mism + miss, "; ".join(f"{nm} = {x if isinstance(x, int) else round(x, 4)}" for nm, x, _, _ in raw),
        tol=1e-12)
    # 13e. the A2 seed-42 C' and the A/A t43 C' are one model (one of the 18 ATBA-phrase readings is a duplicate)
    cpr = json.load(open(CPRIME_JSON))
    add("A2 seed-42 C' (tifs_rep_bert_atba_t43_s42) == A/A t43 C' (tifs_aa_bert_clean_t43): sha256-identical weights "
        "(cprime_determinism.json) and identical ATBA-phrase hit vectors", 1, None,
        int(not cpr.get("identical")) + int(r42["preds_null"] != c43))

    # 14. repository integrity
    fm, am = manifest(FREEZE_MANIFEST), manifest(FINAL_MANIFEST)
    add(f"frozen files sha256 == freeze manifest ({os.path.basename(FREEZE_MANIFEST)})", len(FROZEN), None,
        sum(sha256(os.path.join(ROOT, p)) != fm.get(p) for p in FROZEN), ", ".join(FROZEN))
    mods = sorted({os.path.relpath(os.path.abspath(m.__file__), ROOT) for m in list(sys.modules.values())
                   if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(ROOT + os.sep)
                   and not os.path.abspath(m.__file__).startswith(HERE + os.sep)})
    off = [p for p in mods if sha256(os.path.join(ROOT, p)) != am.get(p)]
    add(f"every repository module imported by this run (outside this script folder) sha256 == final-analysis snapshot "
        f"manifest ({os.path.basename(FINAL_MANIFEST)})", len(mods), None, len(off),
        f"{len(mods)} modules incl. {', '.join(p for p in mods if p in DEPENDS)}" + (f"; differ: {', '.join(off)}" if off else ""))
    start = os.environ.get("G2_SHA256_START")
    if start:
        st_h = dict(item.split("=", 1) for item in start.split(";") if item)   # "path=hash;..." -> {path: hash}
        add("frozen files and imported code unchanged during this run (sha256 at the start of run_all.sh)", len(DEPENDS), None,
            sum(sha256(os.path.join(ROOT, p)) != st_h.get(p) for p in DEPENDS), ", ".join(DEPENDS))
    else:
        info("info: run outside run_all.sh; the 'unchanged during this run' check needs G2_SHA256_START", 0, 0, "")
    t0 = os.environ.get("G2_RUN_START_EPOCH")
    if t0:
        pyc = subprocess.run(["find", ROOT, "-path", os.path.join(ROOT, "runs"), "-prune", "-o", "-newermt", f"@{t0}",
                              "(", "-name", "*.pyc", "-o", "-name", "__pycache__", ")", "-print"],
                             capture_output=True, text=True).stdout.split("\n")
        pyc = [rel(x) for x in pyc if x]
        add("no __pycache__ / .pyc anywhere in the repository (outside runs/) newer than this run's start", 1, None, len(pyc),
            "; ".join(pyc))
        newer = subprocess.run(["find", ROOT, "-path", os.path.join(ROOT, "runs"), "-prune", "-o", "-path",
                                os.path.join(ROOT, "logs"), "-prune", "-o", "-path", os.path.join(ROOT, ".git"), "-prune",
                                "-o", "-newermt", f"@{t0}", "-type", "f", "-print"], capture_output=True, text=True).stdout.split("\n")
        aux_dirs = (os.path.join(ROOT, "scripts", "tifs", "aux") + os.sep, os.path.join(ROOT, "results", "tifs", "addendum", "aux") + os.sep)
        mine = [x for x in newer if x and x.startswith(OUT + os.sep)]
        other_aux = [x for x in newer if x and x.startswith(aux_dirs) and not x.startswith(OUT + os.sep)]
        rest = [rel(x) for x in newer if x and not x.startswith(aux_dirs) and not x.startswith(OUT + os.sep)]
        info("info: repository files written since this run's start (outside runs/, logs/, .git/)", len(mine) + len(other_aux) + len(rest),
             len(rest), f"{len(mine)} in this group's output folder; {len(other_aux)} elsewhere under scripts/tifs/aux/ or "
             f"results/tifs/addendum/aux/ (other A11 groups); anywhere else: {'; '.join(rest) or 'none'}")
    else:
        info("info: run outside run_all.sh; the timestamp checks need G2_RUN_START_EPOCH", 0, 0, "")
    bad_writes = []
    for fn in sorted(os.listdir(HERE)):
        if fn.endswith(".py"):
            for i, line in enumerate(open(os.path.join(HERE, fn)), 1):
                if re.search(r"open\([^)]*,\s*[\"'][wa]", line) and "OUT" not in line and "path" not in line:
                    bad_writes.append(f"{fn}:{i}")
    add("static: every open(..., 'w'/'a') in this group's scripts writes a path under OUT", 1, None, len(bad_writes),
        "; ".join(bad_writes))

    write(checks, "selfcheck.csv")
    for c in checks:
        print(("INFO" if c["passed"] is None else "PASS" if c["passed"] else "FAIL"), c["check"], "| n", c["n_compared"],
              "| max diff", c["max_abs_diff"], "| mismatches", c["n_mismatch"], "|", c["note"][:200])
    if any(c["passed"] is False for c in checks):
        raise SystemExit("selfcheck: at least one check FAILED (see selfcheck.csv)")


if __name__ == "__main__":
    main()
