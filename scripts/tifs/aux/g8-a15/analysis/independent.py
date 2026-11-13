#!/usr/bin/env python3
"""Independent re-computation of the key A15 numbers (no frozen module, no a15lib; numpy / scipy only).

A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Reads the raw prediction files directly (frozen runs, A15 runs, ws re-scores) and recomputes, with code written
separately from the analysis scripts, then compares with their CSVs (|difference| < 1e-9, or equal text):
  1. b1: k_em per cell (mean over seeds of mean(P_frozen) - mean(C_e)), the epoch-matched pools (s_pool and the
     Satterthwaite df from MS_seed / MS_res), the registered and own-variance intervals of the seed-pooled
     cells, the claim-rule verdicts, and the Holm-adjusted p_pos over the 31 cells (transfer side, registered);
  2. b3: the own-variance 95% bound of |C_e4 - C_e3| (BERT BadNets logit);
  3. a: k of every decoder cell on the ws inputs;
  4. d: the BERT AG News logit pool over the available seeds (s_pool, df);
  5. e / e2: k per alpha and attack;
  6. b2 A2: k'_em.
Writes independent_check.csv. A comparison whose inputs are missing is listed with ok = '' (not applicable).
Like the analysis, it reads an A15 run only when its evaluation job is terminal-ok (runs/tifs/_status/
<run>__eval.ok) and a re-score only when tifs_x15_ws__<run>.ok exists (checked here with its own code), and a
decoder cell's ws k is the mean over its ws seeds only when every seed has a ws re-score (else the official k).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
PLAN = os.path.join(ROOT, "scripts", "tifs", "aux", "g8-a15", "plan_check.json")
TOLS = (0.02, 0.05, 0.1)
POOLS = {("bert", "sst2"): ("addsent", "atba", "badnets", "synbkd"), ("gpt2", "sst2"): ("addsent", "badnets"),
         ("qwen3", "sst2"): ("addsent", "badnets", "synbkd"),
         ("bert", "agnews"): ("addsent", "atba", "badnets", "ep", "sos", "synbkd")}
STATUS = "A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm"


def jl(p):
    with open(p) as f:
        return json.load(f)


def rcsv(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader([ln for ln in f if not ln.startswith("# definition")]))


def run_cells(runs, run):
    """{(dataset, attack, kd, seed): (P, C)} of a run's Stage 2 files."""
    out = {}
    for p in sorted(glob.glob(os.path.join(runs, run, "artifacts", "results", "*.preds.json"))):
        r = p[: -len(".preds.json")] + ".json"
        if not os.path.exists(r):
            continue
        d, m = jl(p), jl(r)
        out[(m["dataset"], m["attack_method"], m["kd_type"], int(m.get("seed", -1)))] = (d["preds_obs"], d["preds_null"])
    return out


def eval_ok(runs, run):
    return os.path.isfile(os.path.join(runs, "_status", run + "__eval.ok"))


def ws_ok(runs, run):
    return os.path.isfile(os.path.join(runs, "_status", "tifs_x15_ws__" + run + ".ok"))


def x15_cells(runs, run):
    """run_cells of an A15 run whose evaluation job is terminal-ok, else {}."""
    return run_cells(runs, run) if eval_ok(runs, run) else {}


DECODER_RUNS = ("tifs_b_gpt2_sst2", "tifs_b_gpt2_sst2_s43", "tifs_b_gpt2_sst2_s44", "tifs_b_qwen3_sst2",
                "tifs_b_qwen3_sst2_s43", "tifs_b_qwen3_sst2_s44", "tifs_b_qwen3_agnews")


def frozen_run(fam, ds, s):
    return f"tifs_b_{fam}_{ds}" + ("" if s == 42 else f"_s{s}")


def pool(ks):
    """{attack: {seed: k}} -> (s2, df) over the common seeds (two-way decomposition, Satterthwaite)."""
    seeds = sorted(set.intersection(*(set(v) for v in ks.values())))
    atks = sorted(ks)
    K = np.array([[ks[a][s] for s in seeds] for a in atks], dtype=float)
    m, r = K.shape
    g = K.mean()
    row, col = K.mean(axis=1), K.mean(axis=0)
    ms_seed = m * ((col - g) ** 2).sum() / (r - 1)
    ms_res = ((K - row[:, None] - col[None, :] + g) ** 2).sum() / ((m - 1) * (r - 1)) if m > 1 else 0.0
    comb = ms_seed + (m - 1) * ms_res
    den = ms_seed ** 2 + (m - 1) * ms_res ** 2
    return comb / m, (comb ** 2 * (r - 1) / den if den > 0 else m * (r - 1)), seeds


def verdict(lo, hi, t):
    return ("EQUIVALENT" if -t <= lo and hi <= t else "POSITIVE" if lo > t else
            "NEGATIVE" if hi < -t else "INCONCLUSIVE")


def interval(k, se, df):
    q = stats.t.ppf(0.975, df)
    return k - q * se, k + q * se


def holm(ps):
    m = len(ps)
    order = np.argsort(ps, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs", "tifs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tifs", "addendum", "aux", "g8-a15"))
    a = ap.parse_args()
    runs, out = a.runs, a.out
    rows = []

    def cmp(name, expected, got, tol=1e-9):
        if expected in (None, "") or got is None:
            rows.append({"check": name, "csv": expected, "independent": got, "ok": ""})
            return
        try:
            ok = abs(float(expected) - float(got)) < tol
        except (TypeError, ValueError):
            ok = str(expected) == str(got)
        rows.append({"check": name, "csv": expected, "independent": got, "ok": ok})

    plan = jl(PLAN)
    # ── 1. b1 ──
    em = {}      # (fam, kd, atk) -> {seed: (P_frozen, C_e)}
    for r in plan:
        if not r["run"].startswith("tifs_x15_b1_"):
            continue
        p = os.path.join(runs, r["run"], "artifacts", "results", r["tag"] + ".preds.json")
        if not os.path.exists(p) or not eval_ok(runs, r["run"]):
            continue
        fz = jl(os.path.join(ROOT, r["aligned_frozen_preds"]))
        fam = r["run"].split("_")[3]
        em.setdefault((fam, r["kd"], r["attack"]), {})[r["seed"]] = (fz["preds_obs"], jl(p)["preds_null"])
    b1 = {(x["fam"], x["dataset"], x["kd"], x["attack"]): x for x in rcsv(os.path.join(out, "b1_cells.csv"))}
    ks_em = {}
    stats_em = {}
    for (fam, kd, atk), by in sorted(em.items()):
        ks = {s: float(np.mean(P)) - float(np.mean(C)) for s, (P, C) in by.items()}
        k = float(np.mean(list(ks.values())))
        cmp(f"b1 k_em {fam}/sst2/{kd}/{atk} (seeds {sorted(by)})", b1.get((fam, "sst2", kd, atk), {}).get("em_k"), k)
        if kd == "logit":
            ks_em.setdefault((fam, "sst2"), {})[atk] = ks
            D = np.array([np.array(P) - np.array(C) for P, C in by.values()], dtype=float)
            dbar = D.mean(axis=0)
            stats_em[(fam, atk)] = {"k": k, "v": dbar.var(ddof=1) / len(dbar), "ks": list(ks.values())}
    # GPT-2 BadNets keeps the frozen C (e(P) = frozen C epoch)
    if ("gpt2", "sst2") in ks_em:
        ks, Ds = {}, []
        for s in (42, 43, 44):
            c = run_cells(runs, frozen_run("gpt2", "sst2", s)).get(("sst2", "badnets", "logit", s))
            if c:
                ks[s] = float(np.mean(c[0])) - float(np.mean(c[1]))
                Ds.append(np.array(c[0]) - np.array(c[1]))
        ks_em[("gpt2", "sst2")]["badnets"] = ks
        dbar = np.mean(Ds, axis=0)
        stats_em[("gpt2", "badnets")] = {"k": float(np.mean(list(ks.values()))), "v": dbar.var(ddof=1) / len(dbar),
                                          "ks": list(ks.values())}
    pools_csv = {x["pool"]: x for x in rcsv(os.path.join(out, "b1_pools.csv"))}
    pv = {}
    for (fam, ds), atks in POOLS.items():
        if ds != "sst2" or (fam, ds) not in ks_em or set(ks_em[(fam, ds)]) != set(atks):
            continue
        s2, df, seeds = pool(ks_em[(fam, ds)])
        pv[fam] = (s2, df, len(seeds))
        pc = pools_csv.get(f"{fam}/{ds}/logit", {})
        cmp(f"b1 em pool {fam}/{ds}/logit s_pool", pc.get("em_s_pool"), math.sqrt(s2))
        cmp(f"b1 em pool {fam}/{ds}/logit df", pc.get("em_df"), df)
    ps_pos = {}
    for (fam, atk), x in sorted(stats_em.items()):
        if fam not in pv:
            continue
        s2, df, _ = pv[fam]
        r = len(x["ks"])                      # the cell's own seeds (frozen rule), the pool over common seeds
        row = b1.get((fam, "sst2", "logit", atk), {})
        se = math.sqrt(s2 / r + x["v"])
        lo, hi = interval(x["k"], se, df)
        cmp(f"b1 em registered lo {fam}/{atk}", row.get("em_lo"), lo)
        cmp(f"b1 em registered hi {fam}/{atk}", row.get("em_hi"), hi)
        s2a = float(np.var(x["ks"], ddof=1))
        seo = math.sqrt(s2a / len(x["ks"]) + x["v"])
        olo, ohi = interval(x["k"], seo, len(x["ks"]) - 1)
        cmp(f"b1 em own lo {fam}/{atk}", row.get("em_own_lo"), olo)
        cmp(f"b1 em own hi {fam}/{atk}", row.get("em_own_hi"), ohi)
        for t in TOLS:
            v_reg, v_own = verdict(lo, hi, t), verdict(olo, ohi, t)
            cmp(f"b1 em claim {fam}/{atk}@{t:g}", row.get(f"em_claim@{t:g}"),
                v_reg if v_reg == v_own else "rule-dependent")
            ps_pos.setdefault(t, {})[(fam, "sst2", "logit", atk)] = float(stats.t.sf((x["k"] - t) / se, df))
    # Holm over the 31 cells (transfer side, registered): seed-pooled em cells recomputed, the rest from the CSV
    if b1 and ps_pos:
        for t in TOLS:
            keys = list(b1)
            ps = [ps_pos[t].get(k, float(b1[k][f"em_reg_p_pos@{t:g}"])) for k in keys]
            adj = holm(np.array(ps))
            for k, a_ in zip(keys, adj):
                if k in ps_pos[t]:
                    cmp(f"b1 em F31 Holm p_pos {'/'.join(k)}@{t:g}", b1[k].get(f"em_F31_transfer_reg_holm@{t:g}"), a_)
    # ── 2. b3 ──
    b3 = next((x for x in rcsv(os.path.join(out, "b3_condition.csv")) if x.get("kd") == "logit"), None)
    if ("bert", "logit", "badnets") in em and b3:
        frz = {s: run_cells(runs, frozen_run("bert", "sst2", s)).get(("sst2", "badnets", "logit", s)) for s in (42, 43, 44)}
        D = []
        for s, (P, Ce) in em[("bert", "logit", "badnets")].items():
            D.append(np.array(Ce) - np.array(frz[s][1]))
        D = np.array(D, dtype=float)
        ks = D.mean(axis=1)
        dbar = D.mean(axis=0)
        k = float(ks.mean())
        se = math.sqrt(float(np.var(ks, ddof=1)) / len(ks) + dbar.var(ddof=1) / len(dbar))
        lo, hi = interval(k, se, len(ks) - 1)
        cmp("b3 own-variance 95% bound of |C_e4 - C_e3|", b3.get("own_abs_upper95"), max(abs(lo), abs(hi)))
    # ── 3. a ──
    a_cells = {(x["fam"], x["dataset"], x["attack"]): x for x in rcsv(os.path.join(out, "a_cells.csv"))}
    acc, off = {}, {}
    for run in DECODER_RUNS:
        fam = run.split("_")[2]
        for (ds, atk, kd, seed), (P, C) in run_cells(runs, run).items():
            if kd == "logit":
                off.setdefault((fam, ds, atk), {})[seed] = float(np.mean(P)) - float(np.mean(C))
        d = os.path.join(runs, "tifs_x15_ws", run)
        if not ws_ok(runs, run):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.ws.preds.json"))):
            m = jl(p[: -len(".ws.preds.json")] + ".ws.json")
            if m["kd_type"] != "logit":
                continue
            x = jl(p)
            acc.setdefault((fam, m["dataset"], m["attack_method"]), {})[int(m.get("seed", -1))] = (
                float(np.mean(x["preds_obs"])) - float(np.mean(x["preds_null"])))
    for key, ks_off in sorted(off.items()):
        ks = acc.get(key, {})
        if not ks:
            continue
        if set(ks) == set(ks_off):
            cmp(f"a ws k {'/'.join(key)} ({len(ks)} seeds, ws complete)", a_cells.get(key, {}).get("ws_k"),
                float(np.mean(list(ks.values()))))
        else:
            cmp(f"a ws k {'/'.join(key)} (ws seeds {sorted(ks)} of {sorted(ks_off)}: official k kept)",
                a_cells.get(key, {}).get("ws_k"), float(np.mean(list(ks_off.values()))))
    # ── 4. d ──
    dk = {}
    for s in (42, 43, 44, 45, 46):
        run = frozen_run("bert", "agnews", s) if s <= 44 else f"tifs_x15_d_bert_agnews_s{s}"
        for (ds, atk, kd, seed), (P, C) in (run_cells(runs, run) if s <= 44 else x15_cells(runs, run)).items():
            if kd == "logit":
                dk.setdefault(atk, {})[seed] = float(np.mean(P)) - float(np.mean(C))
    dpools = {x["frame"]: x for x in rcsv(os.path.join(out, "d_pools.csv"))}
    if dk and "r5" in dpools:
        s2, df, seeds = pool({a_: v for a_, v in dk.items() if a_ in POOLS[("bert", "agnews")]})
        cmp(f"d pool bert/agnews/logit s_pool (seeds {seeds})", dpools["r5"].get("s_pool"), math.sqrt(s2))
        cmp(f"d pool bert/agnews/logit df (seeds {seeds})", dpools["r5"].get("df_satterthwaite"), df)
    # ── 5. e / e2 ──
    ecsv = rcsv(os.path.join(out, "e_cells.csv"))
    for x in ecsv:
        if x.get("part") == "e":
            code = "08" if float(x["alpha"]) == 0.8 else "10"
            ks = []
            for s in (42, 43, 44):
                c = x15_cells(runs, f"tifs_x15_e_bert_agnews_a{code}_s{s}").get(("agnews", x["attack"], "logit", s))
                if c:
                    ks.append(float(np.mean(c[0])) - float(np.mean(c[1])))
            if ks:
                cmp(f"e k alpha {x['alpha']} {x['attack']} ({len(ks)} seeds)", x.get(f"a{float(x['alpha']):g}_k"),
                    float(np.mean(ks)))
        elif x.get("part") == "e2":
            e = 2 if x["attack"] == "badnets" else 5
            ks = []
            for s in (42, 43, 44):
                c = x15_cells(runs, f"tifs_x15_e2_gpt2_sst2_a10_e{e}_s{s}").get(("sst2", x["attack"], "logit", s))
                if c:
                    ks.append(float(np.mean(c[0])) - float(np.mean(c[1])))
            if ks:
                cmp(f"e2 k alpha 1 {x['attack']} ({len(ks)} seeds)", x.get("a1_k"), float(np.mean(ks)))
    # ── 6. A2 ──
    a2 = rcsv(os.path.join(out, "b2_a2.csv"))
    if a2 and "k_em" in a2[0]:
        ks = []
        for s in (42, 43, 44):
            run = "tifs_x15_b2_bert_ref_t43_e5" if s == 42 else f"tifs_x15_b2_a2_t43_e5_s{s}"
            c = x15_cells(runs, run).get(("sst2", "atba", "logit", s))
            rep = run_cells(runs, f"tifs_rep_bert_atba_t43_s{s}").get(("sst2", "atba", "logit", s))
            if c and rep:
                ks.append(float(np.mean(rep[0])) - float(np.mean(c[1])))
        if ks:
            cmp(f"b2 A2 k'_em ({len(ks)} seeds)", a2[0].get("k_em"), float(np.mean(ks)))
    path = os.path.join(out, "independent_check.csv")
    keys = ["check", "csv", "independent", "ok"]
    with open(path, "w", newline="") as f:
        f.write("# definition: Independent re-computation (numpy / scipy, no frozen module, no a15lib) of key A15 "
                "numbers from the raw prediction files, against the analysis CSVs; ok = |difference| < 1e-9 (or equal "
                "text); empty ok = inputs missing (not applicable). Status: " + STATUS + ".\n")
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["ok"] is False]
    print(f"independent: {len(rows)} comparisons, {sum(r['ok'] is True for r in rows)} ok, {len(bad)} differ, "
          f"{sum(r['ok'] == '' for r in rows)} not applicable")
    for r in bad[:10]:
        print("  DIFFERS:", r["check"], r["csv"], r["independent"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
