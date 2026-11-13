#!/usr/bin/env python3
"""A13 adversarial self-review: the four headline readings recomputed by an independent code path.

A13 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

This file imports neither the frozen modules (scripts/tifs/analyze.py, addendum.py,
counterfactual_kd/*) nor this group's common.py. It reads the prediction / result JSON files directly,
re-implements the registered formulas with numpy / scipy (two-way seed x unit ANOVA pool, Satterthwaite
df, seed-pooled SE, t intervals, Holm, IUT), and compares with the CSVs the main path wrote:

  1. G1: Qwen3 AG News logit battery decision at delta 0.05 with its own seed pool (seeds 42-44).
  2. G2: 5-reference A3.3 interval for the ATBA cell, BERT SST-2 and AG News, logit KD.
  3. G3: the smallest dose rate POSITIVE at delta 0.05 (cell verdict and Holm over the four rates).
  4. G4: P1 - R1 at the lowest rate (0.1 %), with R1.
Writes independent_check.csv (main vs independent, absolute difference, agreement).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import t as tdist  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RUNS = os.path.join(ROOT, "runs", "tifs")
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(
    ROOT, "results", "tifs", "addendum", "aux", "g7-a13")
A = 0.025
DELTA = 0.05
TOL = 1e-10


def jload(p):
    with open(p) as f:
        return json.load(f)


def cells_of(run):
    """[(result json, preds json)] of one run (schema 2 only)."""
    out = []
    for p in sorted(glob.glob(os.path.join(RUNS, run, "artifacts", "results", "*.preds.json"))):
        q = p[: -len(".preds.json")] + ".json"
        if not os.path.exists(q):
            continue
        d = jload(p)
        if (d.get("meta") or {}).get("schema") != 2:
            continue
        out.append((jload(q), d))
    return out


def find(run, attack=None, kd="logit", ds=None):
    hits = [(r, d) for r, d in cells_of(run) if (attack is None or r["attack_method"] == attack)
            and r["kd_type"] == kd and (ds is None or r["dataset"] == ds)]
    if len(hits) != 1:
        raise ValueError(f"{run} {attack} {kd} {ds}: {len(hits)} cells")
    return hits[0]


def pool2way(K):
    """K: units x seeds array of k. Returns s2 (pooled), df (Satterthwaite) via the two-way decomposition."""
    K = np.asarray(K, float)
    m, r = K.shape
    g = K.mean()
    ms_seed = m * ((K.mean(0) - g) ** 2).sum() / (r - 1)
    resid = K - K.mean(1, keepdims=True) - K.mean(0, keepdims=True) + g
    ms_res = (resid ** 2).sum() / ((m - 1) * (r - 1))
    comb = ms_seed + (m - 1) * ms_res
    s2_alt = K.var(axis=1, ddof=1).mean()          # identity: mean within-unit variance
    s2 = comb / m
    assert abs(s2 - s2_alt) < 1e-14, (s2, s2_alt)
    return s2, comb ** 2 * (r - 1) / (ms_seed ** 2 + (m - 1) * ms_res ** 2)


def seed_pooled(D, s2, df):
    """D: seeds x n array of per-input differences. k, se (frozen form), interval at df."""
    D = np.asarray(D, float)
    r, n = D.shape
    k = D.mean(1).mean()
    dbar = D.mean(0)
    v = ((dbar - k) ** 2).sum() / (n - 1) / n
    se = np.sqrt(s2 / r + v)
    q = tdist.ppf(0.975, df)
    return k, se, k - q * se, k + q * se


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def read(path):
    with open(path) as f:
        return list(csv.DictReader([ln for ln in f if not ln.startswith("# definition")]))


def vec(d, key="preds_obs"):
    return np.asarray(d[key], float)


def main():
    rows = []

    def cmp(item, quantity, main, ind):
        if isinstance(ind, (str, bool)):
            ok = str(main) == str(ind)
            dd = ""
        else:
            dd = abs(float(main) - float(ind))
            ok = dd < TOL
        rows.append({"item": item, "quantity": quantity, "main_path": main, "independent": ind,
                     "abs_diff": dd, "agree": ok})

    # ── 1. G1 ──
    runs = {42: "tifs_b_qwen3_agnews", 43: "tifs_x13_qwen3_agnews_s43", 44: "tifs_x13_qwen3_agnews_s44"}
    atks = ("addsent", "badnets")
    D = {a: {} for a in atks}
    qual = {}
    for s, run in runs.items():
        for a in atks:
            r, d = find(run, a, "logit", "agnews")
            assert int(r["student_seed"]) == s and int(r["teacher_seed"]) == 42
            D[a][s] = vec(d, "preds_obs") - vec(d, "preds_null")
            if s == 42:
                qual[a] = r["teacher_asr"] >= 0.90 and r["teacher_ca_null"] - r["teacher_ca"] <= 0.05 + 1e-9
    Kmat = [[D[a][s].mean() for s in (42, 43, 44)] for a in atks]
    s2, df = pool2way(Kmat)
    pu, pp, ks = [], [], {}
    for a in atks:
        k, se, lo, hi = seed_pooled([D[a][s] for s in (42, 43, 44)], s2, df)
        ks[a] = (k, lo, hi)
        pu.append(tdist.cdf((k - DELTA) / se, df))
        pp.append(tdist.sf((k - DELTA) / se, df))
    dec = ("TRANSFERS" if (holm(pp) < A).any() else "NO TRANSFER" if max(pu) < A else "INCONCLUSIVE")
    assert all(qual.values())
    g1 = {r["delta"]: r for r in read(os.path.join(OUT, "g1_battery.csv"))}["0.05"]
    g1c = {r["attack"]: r for r in read(os.path.join(OUT, "g1_cells.csv"))}
    g1p = read(os.path.join(OUT, "g1_pool.csv"))[0]
    cmp("1 G1", "own pool s", g1p["s_pool"], s2 ** 0.5)
    cmp("1 G1", "own pool Satterthwaite df", g1p["df_satterthwaite"], df)
    for a in atks:
        cmp("1 G1", f"{a} k", g1c[a]["own_pool_k"], ks[a][0])
        cmp("1 G1", f"{a} 95% upper", g1c[a]["own_pool_hi"], ks[a][2])
    cmp("1 G1", "max one-sided p_up@0.05", g1["own_pool_max_p_up"], max(pu))
    cmp("1 G1", "battery decision@0.05 (own pool)", g1["own_pool_decision"], dec)

    # ── 2. G2 ──
    sn = jload(os.path.join(ROOT, "results", "tifs", "structural_nulls.json"))["pairs"]
    out_pool = {(p["attack"], p["dataset"]) for p in sn if p.get("structural_null") or p.get("only_input_embeddings")}
    g2 = {(r["dataset"], r["kd"]): r for r in read(os.path.join(OUT, "g2_reference_atba5.csv"))}
    for ds in ("sst2", "agnews"):
        bruns = {42: f"tifs_b_bert_{ds}", 43: f"tifs_b_bert_{ds}_s43", 44: f"tifs_b_bert_{ds}_s44"}
        Dd = {}
        for s, run in bruns.items():
            for r, d in cells_of(run):
                if r["kd_type"] != "logit" or d["meta"].get("single_student"):
                    continue
                a = r["attack_method"]
                q = r["teacher_asr"] >= 0.90 and r["teacher_ca_null"] - r["teacher_ca"] <= 0.05 + 1e-9
                if (a, ds) in out_pool or not q:
                    continue
                Dd.setdefault(a, {})[s] = vec(d, "preds_obs") - vec(d, "preds_null")
        units = sorted(a for a, v in Dd.items() if set(v) == {42, 43, 44})
        s2p, dfp = pool2way([[Dd[a][s].mean() for s in (42, 43, 44)] for a in units])
        k, se, _, _ = seed_pooled([Dd["atba"][s] for s in (42, 43, 44)], s2p, dfp)
        _, d42 = find(bruns[42], "atba", "logit", ds)
        refs = [vec(d42, "preds_null").mean()]
        for run in ("tifs_aa_bert_clean_t43", "tifs_aa_bert_clean_t44", "tifs_x13_bert_ref_t45", "tifs_x13_bert_ref_t46"):
            _, d = find(run, "badnets", "logit", ds)
            refs.append(np.mean(d["controls"][f"atba_phrase_{ds}"]["c"]))
        s2r = np.var(refs, ddof=1)
        dfx = (se ** 2 + s2r) ** 2 / (se ** 4 / dfp + s2r ** 2 / 4)
        q = tdist.ppf(0.975, dfx)
        lo, hi = k - q * np.sqrt(se ** 2 + s2r), k + q * np.sqrt(se ** 2 + s2r)
        verdict = ("EQUIVALENT" if -DELTA <= lo and hi <= DELTA else "POSITIVE" if lo > DELTA
                   else "NEGATIVE" if hi < -DELTA else "INCONCLUSIVE")
        m = g2[(ds, "logit")]
        cmp("2 G2", f"{ds} pool units", "addsent,atba,badnets,synbkd" if ds == "sst2" else
            "addsent,atba,badnets,ep,sos,synbkd", ",".join(units))
        cmp("2 G2", f"{ds} ATBA primary k", m["k"], k)
        cmp("2 G2", f"{ds} reference SD (5 refs)", m["s_ref"], s2r ** 0.5)
        cmp("2 G2", f"{ds} 5-ref interval lower", m["ref5_lo"], lo)
        cmp("2 G2", f"{ds} 5-ref interval upper", m["ref5_hi"], hi)
        cmp("2 G2", f"{ds} 5-ref Satterthwaite df", m["ref5_df"], dfx)
        cmp("2 G2", f"{ds} 5-ref verdict@0.05", m["ref5_verdict@0.05"], verdict)

    # ── 3. G3 ──
    rates = ("0001", "0003", "0010", "0030")
    D3 = {}
    for rc in rates:
        for s in (42, 43, 44):
            r, d = find(f"tifs_x13_pcb_r{rc}_s{s}", "badnets", "logit", "sst2")
            assert int(r["student_seed"]) == s
            D3.setdefault(rc, {})[s] = vec(d, "preds_obs") - vec(d, "preds_null")
    s2, df = pool2way([[D3[rc][s].mean() for s in (42, 43, 44)] for rc in rates])
    res3 = {}
    for rc in rates:
        k, se, lo, hi = seed_pooled([D3[rc][s] for s in (42, 43, 44)], s2, df)
        res3[rc] = (k, lo, hi, tdist.sf((k - DELTA) / se, df))
    adj = dict(zip(rates, holm([res3[rc][3] for rc in rates])))
    pct = {rc: f"{int(rc) / 100:g}%" for rc in rates}
    small_cell = next((pct[rc] for rc in rates if res3[rc][1] > DELTA), "none")
    small_holm = next((pct[rc] for rc in rates if res3[rc][1] > DELTA and adj[rc] < A), "none")
    g3r = {r["delta"]: r for r in read(os.path.join(OUT, "g3_readings.csv"))}["0.05"]
    g3 = {r["rate_percent"]: r for r in read(os.path.join(OUT, "g3_rates.csv"))}
    cmp("3 G3", "four-rate pool s", read(os.path.join(OUT, "g3_pool.csv"))[0]["s_pool"], s2 ** 0.5)
    for rc in rates:
        cmp("3 G3", f"{pct[rc]} k", g3[pct[rc]]["pooled_k"], res3[rc][0])
        cmp("3 G3", f"{pct[rc]} 95% lower", g3[pct[rc]]["pooled_lo"], res3[rc][1])
    cmp("3 G3", "smallest POSITIVE@0.05 rate (cell)", g3r["smallest_positive_rate_cell"], small_cell)
    cmp("3 G3", "smallest POSITIVE@0.05 rate (Holm over 4 rates)", g3r["smallest_positive_rate_holm"], small_holm)

    # ── 4. G4 ──
    g4rates = ("0010", "0030", "0100")
    P, R = {}, {}
    for rc in g4rates:
        for s in (42, 43, 44):
            run = f"tifs_x13_gb_r{rc}_s{s}"
            _, dl = find(run, "badnets", "logit", "sst2")
            _, dc = find(run, "badnets", "ce", "sst2")
            P.setdefault(rc, {})[s] = vec(dl, "preds_obs")
            R.setdefault(rc, {})[s] = vec(dc, "preds_obs")
    DPR = {rc: {s: P[rc][s] - R[rc][s] for s in (42, 43, 44)} for rc in g4rates}
    s2, df = pool2way([[DPR[rc][s].mean() for s in (42, 43, 44)] for rc in g4rates])
    k, se, lo, hi = seed_pooled([DPR["0010"][s] for s in (42, 43, 44)], s2, df)
    R1 = np.mean([R["0010"][s].mean() for s in (42, 43, 44)])
    g4 = {(r["rate_percent"], r["contrast"]): r for r in read(os.path.join(OUT, "g4_contrasts.csv"))}[("0.1%", "P1-R1")]
    cmp("4 G4", "0.1% R1", g4["R1"], R1)
    cmp("4 G4", "0.1% P1-R1 k", g4["pooled_k"], k)
    cmp("4 G4", "0.1% P1-R1 95% lower", g4["pooled_lo"], lo)
    cmp("4 G4", "0.1% P1-R1 95% upper", g4["pooled_hi"], hi)
    cmp("4 G4", "0.1% P1-R1 pool df", g4["pool_df"], df)
    cmp("4 G4", "0.1% amplification (R1<0.90 and lower>0.05)", "False", str(bool(R1 < 0.90 and lo > DELTA)))

    with open(os.path.join(OUT, "independent_check.csv"), "w", newline="") as f:
        f.write("# definition: A13 adversarial self-review: each headline quantity from the main path (the CSVs "
                "written through the frozen functions) against an independent re-implementation that imports no "
                "frozen module and no common.py (direct JSON reads, numpy / scipy); numbers agree when |diff| < "
                "1e-10, labels when equal. Status: A13 post hoc analysis; changes no primary verdict, battery "
                "decision or the paper-wide Holm.\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["agree"] for r in rows)
    print(f"  independent check: {n_ok} / {len(rows)} agree")
    for r in rows:
        if not r["agree"]:
            print("  DISAGREE:", r)
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
