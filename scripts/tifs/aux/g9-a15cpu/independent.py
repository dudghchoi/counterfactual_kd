#!/usr/bin/env python3
"""Adversarial self-review of the A15 CPU items: the key numbers recomputed by an independent code path.

A15 post hoc CPU analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Nothing here imports the frozen analysis (analyze.py, addendum.py, analysis/tifs.py, metrics/paired.py),
this folder's common.py or the corpus construction code. It reads the raw files with its own parsers and
re-implements every step with numpy / scipy (own run-id regex and gate, own Tango score interval via
brentq, own pools, Satterthwaite and Welch df, own Holm and battery rule, own null-swap bookkeeping, own
transmission shares, own corpus draw from the HF dataset cache), then compares with the item outputs:
  (a) the frozen-df verdicts of all 31 primary cells (vs primary.csv) and every verdict that changes under
      the Welch df (cells, batteries, transfers_global, 31-cell Holm flags; vs welch_changes.csv)
  (b) the corrected null-swap POSITIVE counts, every rule and scheme (vs nullswap_summary.csv)
  (c) the transmission shares and gap with its interval (vs transmission_cells.csv)
  (d) the G3 / anchor / G4 triggered and non-target triggered row counts (vs g3_dose.csv)
Output: independent_check.csv (one row per compared quantity, ok = agreement).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
from scipy import optimize, stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RUNS = os.path.join(ROOT, "runs", "tifs")
ADD = os.path.join(ROOT, "results", "tifs", "addendum")
DELTAS = (0.02, 0.05, 0.1)
A = 0.025
Z = stats.norm.ppf(0.975)
STATUS = "A15 post hoc CPU analysis; changes no primary verdict, battery decision or the paper-wide Holm"
RX = re.compile(r"^tifs_br?_([a-z0-9]+)_([a-z0-9]+?)(?:_s(\d+))?$")


def outdir():
    d = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(ADD, "aux", "g9-a15cpu")
    os.makedirs(d, exist_ok=True)
    return d


def rd(path):
    with open(path) as f:
        return list(csv.DictReader([ln for ln in f if not ln.startswith("# definition")]))


# ── own statistics ───────────────────────────────────────────────────


def tango_score(n10, n01, n, d):
    """Tango (1998) score for H0: p10 - p01 = d, with the restricted MLE of p01."""
    qa = 2.0 * n
    qb = -(n10 + n01) + (2.0 * n - n10 + n01) * d
    qc = -n01 * d * (1.0 - d)
    p01 = (-qb + np.sqrt(max(qb * qb - 4 * qa * qc, 0.0))) / (2 * qa)
    v = n * (2 * p01 + d * (1 - d))
    num = n10 - n01 - n * d
    if v <= 0:
        return 0.0 if num == 0 else np.copysign(np.inf, num)
    return num / np.sqrt(v)


def tango(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    k = (n10 - n01) / n
    e = 1e-12
    f = lambda d, tgt: tango_score(n10, n01, n, d) - tgt  # noqa: E731
    lo = -1.0 if f(-1 + e, Z) <= 0 else optimize.brentq(f, -1 + e, k, args=(Z,), xtol=1e-15, rtol=1e-15, maxiter=500) \
        if f(k, Z) < 0 else k
    hi = 1.0 if f(1 - e, -Z) >= 0 else optimize.brentq(f, k, 1 - e, args=(-Z,), xtol=1e-15, rtol=1e-15, maxiter=500) \
        if f(k, -Z) > 0 else k
    pv = {}
    for t in DELTAS:
        zh = tango_score(n10, n01, n, min(t, 1 - 1e-12))
        zl = tango_score(n10, n01, n, max(-t, -1 + 1e-12))
        pv[t] = {"p_pos": 1 - stats.norm.cdf(zh), "p_up": stats.norm.cdf(zh), "p_low": 1 - stats.norm.cdf(zl)}
    return k, lo, hi, pv


def verdict(lo, hi, t):
    if -t <= lo and hi <= t:
        return "EQUIVALENT"
    if lo > t:
        return "POSITIVE"
    if hi < -t:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def tjudge(k, se, df):
    q = stats.t.ppf(0.975, df)
    lo, hi = k - q * se, k + q * se
    out = {"lo": lo, "hi": hi}
    for t in DELTAS:
        out[t] = {"v": verdict(lo, hi, t), "p_pos": stats.t.sf((k - t) / se, df), "p_up": stats.t.cdf((k - t) / se, df)}
    return out


def holm_adj(p):
    p = np.asarray(p, float)
    o = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def welch(s2, r):
    s2 = np.asarray(s2)
    return s2.sum() ** 2 * (r - 1) / (s2 ** 2).sum()


def anova_df(K):
    """Satterthwaite df of (MS_seed + (m-1) MS_res)/m for an attack x seed table K (m x r)."""
    m, r = K.shape
    row, col, g = K.mean(1, keepdims=True), K.mean(0, keepdims=True), K.mean()
    ms_seed = m * ((col - g) ** 2).sum() / (r - 1)
    ms_res = ((K - row - col + g) ** 2).sum() / ((m - 1) * (r - 1)) if m > 1 else 0.0
    comb, den = ms_seed + (m - 1) * ms_res, ms_seed ** 2 + (m - 1) * ms_res ** 2
    return comb ** 2 * (r - 1) / den


# ── (a) primary cells, Welch df ──────────────────────────────────────


def load_battery():
    cells = {}
    for p in glob.glob(os.path.join(RUNS, "tifs_b*_*", "artifacts", "results", "*.preds.json")):
        run = p.split(os.sep)[-4]
        if not RX.match(run):
            continue
        rj = p[:-11] + ".json"
        if not os.path.exists(rj):
            continue
        d = json.load(open(p))
        if (d.get("meta") or {}).get("schema") != 2:
            continue
        r = json.load(open(rj))
        cells[(RX.match(run).group(1), r["dataset"], r["kd_type"], r["attack_method"], int(r["seed"]))] = {"d": d, "r": r}
    return cells


def primary_independent(cells):
    sn = json.load(open(os.path.join(ROOT, "results", "tifs", "structural_nulls.json")))["pairs"]
    structural = {(p["attack"], p["dataset"]) for p in sn if p.get("structural_null")}
    emb = {(p["attack"], p["dataset"]) for p in sn if p.get("only_input_embeddings")}
    by = defaultdict(dict)
    for (fam, ds, kd, atk, s), c in cells.items():
        if kd != "ce" and not c["d"]["meta"].get("single_student"):
            by[(fam, ds, kd, atk)][s] = c
    st = {}
    for key, ss in by.items():
        seeds = sorted(ss)
        D = np.array([np.array(ss[s]["d"]["preds_obs"]) - np.array(ss[s]["d"]["preds_null"]) for s in seeds], float)
        ks = D.mean(1)
        r0 = ss[seeds[0]]["r"]
        q = (r0["teacher_asr"] >= 0.90 and r0["teacher_ca_null"] - r0["teacher_ca"] <= 0.05 + 1e-9)
        st[key] = {"seeds": seeds, "ks": ks, "n": D.shape[1], "v": D.mean(0).var(ddof=1) / D.shape[1],
                   "s2": ks.var(ddof=1) if len(seeds) > 1 else None, "q": q, "c42": ss.get(42, ss[seeds[0]])}
    mem = defaultdict(dict)
    for (fam, ds, kd, atk), x in st.items():
        if x["s2"] is not None and x["q"] and (atk, ds) not in structural and (atk, ds) not in emb:
            mem[(fam, ds, kd)][atk] = dict(zip(x["seeds"], x["ks"]))
    pools = {}
    for key, m in mem.items():
        seeds = sorted(set.intersection(*(set(v) for v in m.values())))
        atks = sorted(m)
        K = np.array([[m[a][s] for s in seeds] for a in atks])
        s2a = K.var(1, ddof=1)
        pools[key] = {"s2": s2a.mean(), "df": anova_df(K), "welch": welch(s2a, len(seeds))}

    def src(fam, ds, kd):
        for k in ((fam, ds, kd), (fam, ds, "logit"), (fam, "sst2", "logit")):
            if k in pools:
                return k
        o = [k for k in pools if k[1] == "sst2" and k[2] == "logit"]
        return max(o, key=lambda k: pools[k]["s2"]) if o else None

    rows = {}
    for key, x in sorted(st.items()):
        fam, ds, kd, atk = key
        c42 = x["c42"]["d"]
        k42, lo42, hi42, pv42 = tango(c42["preds_obs"], c42["preds_null"])
        row = {"q": x["q"], "battery": key[:3]}
        if (atk, ds) in structural:
            row["structural"] = True
            for var_ in ("frozen", "welch"):
                row[var_] = {"lo": lo42, "hi": hi42, **{t: {"v": verdict(lo42, hi42, t), "p_pos": pv42[t]["p_pos"],
                                                            "p_up": pv42[t]["p_up"]} for t in DELTAS}}
            rows[key] = row
            continue
        sk = src(fam, ds, kd)
        P = pools[sk]
        r = len(x["seeds"])
        if r >= 2:
            k, se = x["ks"].mean(), np.sqrt(P["s2"] / r + x["v"])
        else:
            k, se = k42, np.sqrt(((hi42 - lo42) / (2 * Z)) ** 2 + P["s2"])
        row.update(k=k, se=se, pool=sk, frozen=tjudge(k, se, P["df"]), welch=tjudge(k, se, P["welch"]),
                   df=P["df"], dfw=P["welch"])
        rows[key] = row
    return rows, pools


def decisions(rows, var_):
    qual = [(k, r) for k, r in sorted(rows.items()) if r["q"]]
    out = {"cell": {}, "battery": {}, "global": {}}
    for k, r in qual:
        for t in DELTAS:
            out["cell"][(k, t)] = r[var_][t]["v"]
    bats = defaultdict(list)
    for k, r in qual:
        bats[r["battery"]].append((k, r))
    for b, items in bats.items():
        for t in DELTAS:
            adj = holm_adj([r[var_][t]["p_pos"] for _, r in items])
            tr = [k[3] for (k, _), a in zip(items, adj) if a < A]
            below = max(r[var_][t]["p_up"] for _, r in items) < A
            out["battery"][(b, t)] = ("TRANSFERS" if tr else "NO TRANSFER" if below else "INCONCLUSIVE", ",".join(tr))
    for t in DELTAS:
        adj = holm_adj([r[var_][t]["p_pos"] for _, r in qual])
        for (k, _), a in zip(qual, adj):
            out["global"][(k, t)] = bool(a < A)
    return out


# ── (b) null swaps ───────────────────────────────────────────────────

CTAG = {"sos": "clean-sos", "ep": "clean-ep", "ripple": "clean-ripples"}
PAIR = {"clean-ep/clean", "clean-sos/clean"}
FAMILY = {"ep_cf_mn/poisoned", "clean-ep/clean", "sos_friends_weekend/poisoned", "clean-sos/clean"}
DROP = {("ep", "badnets_cf_mn/poisoned"), ("badnets", "ep_cf_mn/poisoned")}


def nullswaps(pools):
    res = {}
    for ds in ("sst2", "agnews"):
        X = json.load(open(os.path.join(ADD, f"cross_hits_bert_{ds}.json")))
        for atk, H in X["hits"].items():
            for kd in X["meta"]["kd"]:
                pk = next(k for k in (("bert", ds, kd), ("bert", ds, "logit"), ("bert", "sst2", "logit")) if k in pools)
                P = pools[pk]
                ref = f"{kd}/{CTAG.get(atk, 'clean-base')}/clean"
                studs = defaultdict(dict)
                for key in H:
                    m = re.match(r"^([a-z]+)/(.+)/(poisoned|clean|ce)/s(\d+)$", key)
                    if m and m.group(1) == kd and m.group(3) in ("poisoned", "clean"):
                        studs[f"{m.group(2)}/{m.group(3)}"][int(m.group(4))] = np.array(H[key], float)
                own = next(s for s in studs if s.endswith("/poisoned") and s.split("_")[0] == atk)
                refv = studs[ref.split("/", 1)[1]]
                nulls = [s for s in studs if s not in (own, ref.split("/", 1)[1])]
                verd = {}
                for s in nulls:
                    seeds = sorted(set(studs[s]) & set(refv))
                    Dm = np.array([studs[s][x] - refv[x] for x in seeds])
                    out = {}
                    if len(seeds) >= 2:
                        ks = Dm.mean(1)
                        k, v, r = ks.mean(), Dm.mean(0).var(ddof=1) / Dm.shape[1], len(seeds)
                        out["frozen"] = tjudge(k, np.sqrt(P["s2"] / r + v), P["df"])
                        out["welch"] = tjudge(k, np.sqrt(P["s2"] / r + v), P["welch"])
                        out["own"] = tjudge(k, np.sqrt(ks.var(ddof=1) / r + v), r - 1)
                        out["max"] = tjudge(k, np.sqrt(max(ks.var(ddof=1), P["s2"]) / r + v), P["df"])
                    else:
                        k, lo, hi, _ = tango(studs[s][seeds[0]], refv[seeds[0]])
                        se = np.sqrt(((hi - lo) / (2 * Z)) ** 2 + P["s2"])
                        out["frozen"], out["welch"] = tjudge(k, se, P["df"]), tjudge(k, se, P["welch"])
                    verd[s] = out
                for scheme in ("A14", "registered", "family"):
                    units = defaultdict(list)
                    for s in nulls:
                        if scheme != "A14" and (atk, s) in DROP:
                            continue
                        u = s
                        if ds == "sst2" and scheme == "registered" and s in PAIR:
                            u = "pair"
                        if ds == "sst2" and scheme == "family" and s in FAMILY:
                            u = "family"
                        units[u].append(s)
                    res[(ds, kd, atk, scheme, "n")] = len(units)
                    for rule in ("frozen", "own", "max", "welch"):
                        for t in DELTAS:
                            for how in ("any", "all"):
                                cnt, nrep = 0, 0
                                for u, mem in units.items():
                                    vs = [verd[s][rule][t]["v"] == "POSITIVE" for s in mem if rule in verd[s]]
                                    if not vs:
                                        continue
                                    nrep += 1
                                    cnt += any(vs) if how == "any" else all(vs)
                                res[(ds, kd, atk, scheme, rule, t, how)] = cnt if (rule in ("frozen", "welch") or nrep) else "n/a"
    return res


# ── (c) transmission ─────────────────────────────────────────────────


def transmission(cells, rows):
    out = {}
    for key in rows:
        fam, ds, kd, atk = key
        seeds = sorted(s for (f, d, k, a, s) in cells if (f, d, k, a) == key)
        fP, mC, dd = [], [], []
        for s in seeds:
            x = cells[key + (s,)]["d"]
            TP, TC = np.array(x["teacher_p_clean"]), np.array(x["teacher_c_clean"])
            Pc, Cc = np.array(x["preds_p_clean"]), np.array(x["preds_c_clean"])
            D = TP != TC
            if D.sum() == 0:
                break
            a, b = (Pc[D] == TP[D]).astype(float), (Cc[D] == TP[D]).astype(float)
            fP.append(a.mean())
            mC.append(b.mean())
            dd.append(a - b)
        if not fP:
            out[key] = None
            continue
        dd = np.array(dd)
        gap_s = dd.mean(1)
        rec = {"follow_P": np.mean(fP), "match_C": np.mean(mC), "gap": np.mean(gap_s)}
        if len(seeds) >= 2:
            se = np.sqrt(gap_s.var(ddof=1) / len(seeds) + dd.mean(0).var(ddof=1) / dd.shape[1])
            q = stats.t.ppf(0.975, len(seeds) - 1)
            rec.update(lo=rec["gap"] - q * se, hi=rec["gap"] + q * se)
        else:
            x = cells[key + (seeds[0],)]["d"]
            TP = np.array(x["teacher_p_clean"])
            D = TP != np.array(x["teacher_c_clean"])
            a = (np.array(x["preds_p_clean"])[D] == TP[D]).astype(int)
            b = (np.array(x["preds_c_clean"])[D] == TP[D]).astype(int)
            _, lo, hi, _ = tango(a, b)
            rec.update(lo=lo, hi=hi)
        out[key] = rec
    return out


# ── (d) corpus draw ──────────────────────────────────────────────────


def dose():
    import datasets
    tr = datasets.load_dataset("glue", "sst2", split="train")
    lab = np.array(tr["label"])
    N = len(lab)
    out = {}
    for rate in (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1):
        random.seed(42)
        idx = list(range(N))
        random.shuffle(idx)
        sel = idx[: int(rate * N)]
        out[rate] = (N, len(sel), int((lab[sel] != 1).sum()))
    return out


def main():
    out = outdir()
    checks = []

    def chk(name, exp, got, ok=None, tol=None):
        if ok is None:
            ok = (abs(float(exp) - float(got)) <= tol) if tol is not None else (str(exp) == str(got))
        checks.append({"check": name, "item_output": exp, "independent": got, "ok": bool(ok)})

    cells = load_battery()
    rows, pools = primary_independent(cells)
    # (a0) frozen-df reproduction of primary.csv
    prim = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in rd(os.path.join(ROOT, "results", "tifs", "report", "primary.csv"))}
    nv, mx = 0, 0.0
    for key, r in rows.items():
        o = prim[key]
        for t in DELTAS:
            nv += o[f"verdict@{t:g}"] != r["frozen"][t]["v"]
            mx = max(mx, abs(float(o[f"p_pos@{t:g}"]) - r["frozen"][t]["p_pos"]))
        mx = max(mx, abs(float(o["lo"]) - r["frozen"]["lo"]), abs(float(o["hi"]) - r["frozen"]["hi"]))
    chk("(a0) 31 primary cells, frozen df: verdict mismatches vs primary.csv", 0, nv)
    chk("(a0) 31 primary cells, frozen df: max |diff| of lo, hi, p_pos vs primary.csv", 0.0, mx, tol=1e-9)
    wp = {r["pool"]: r for r in rd(os.path.join(out, "welch_pools.csv"))}
    for key, p in pools.items():
        w = wp["/".join(key)]
        chk(f"(a1) pool {'/'.join(key)}: frozen df", w["df_frozen"], p["df"], tol=1e-9)
        chk(f"(a1) pool {'/'.join(key)}: Welch df", w["df_welch"], p["welch"], tol=1e-9)
    # (a2) every change under the Welch df
    F, Wd = decisions(rows, "frozen"), decisions(rows, "welch")
    mine = set()
    for (k, t), v in F["cell"].items():
        if Wd["cell"][(k, t)] != v:
            mine.add(("cell verdict", "/".join(k), f"{t:g}", v, Wd["cell"][(k, t)]))
    for (b, t), (dec, tr) in F["battery"].items():
        d2, t2 = Wd["battery"][(b, t)]
        if dec != d2:
            mine.add(("battery decision", "/".join(b), f"{t:g}", dec, d2))
        if tr != t2:
            mine.add(("battery transfers", "/".join(b), f"{t:g}", tr or "(none)", t2 or "(none)"))
    for (k, t), v in F["global"].items():
        if Wd["global"][(k, t)] != v:
            mine.add(("global Holm flag (31 cells)", "/".join(k), f"{t:g}", str(v), str(Wd["global"][(k, t)])))
    theirs = {(c["scope"], c["item"], f"{float(c['delta']):g}", c["frozen"], c["new"])
              for c in rd(os.path.join(out, "welch_changes.csv"))
              if c["variant"] == "welch" and c["scope"] in ("cell verdict", "battery decision", "battery transfers",
                                                              "global Holm flag (31 cells)")}
    chk("(a2) Welch df: set of changed cell verdicts / battery decisions / battery transfers / 31-cell Holm flags "
        "(item 1 vs independent)", f"{len(theirs)} changes", f"{len(mine)} changes, symmetric difference "
        f"{len(theirs ^ mine)}", ok=theirs == mine)
    for c in sorted(mine):
        checks.append({"check": "(a2) change: " + " | ".join(c), "item_output": c in theirs, "independent": True,
                       "ok": c in theirs})
    # (b) null swaps
    ns = nullswaps(pools)
    summ = rd(os.path.join(out, "nullswap_summary.csv"))
    nb, ncmp = 0, 0
    for r in summ:
        ds, kd, atk = r["dataset"], r["kd"], r["trigger"]
        for scheme in ("A14", "registered", "family"):
            ncmp += 1
            nb += str(ns[(ds, kd, atk, scheme, "n")]) != r[f"n_null_{scheme}"]
            for rule in ("frozen", "own", "max", "welch"):
                for t in DELTAS:
                    if scheme == "A14":
                        ncmp += 1
                        nb += str(ns[(ds, kd, atk, scheme, rule, t, "any")]) != r[f"POS_{rule}_A14@{t:g}"]
                    else:
                        for how in ("any", "all"):
                            ncmp += 1
                            nb += str(ns[(ds, kd, atk, scheme, rule, t, how)]) != r[f"POS_{rule}_{scheme}_{how}@{t:g}"]
    chk(f"(b) null-swap unit counts and POSITIVE counts, 16 groups x 3 schemes x 4 rules x 3 deltas "
        f"({ncmp} numbers): mismatches", 0, nb)
    key_b = [("agnews", "logit", "atba"), ("sst2", "logit", "atba"), ("agnews", "feature", "atba")]
    for ds, kd, atk in key_b:
        r = next(x for x in summ if (x["dataset"], x["kd"], x["trigger"]) == (ds, kd, atk))
        chk(f"(b) {ds}/{kd}/{atk} registered: POSITIVE@0.05 frozen/own/max/welch of n",
            "/".join(str(r[f"POS_{ru}_registered_any@0.05"]) for ru in ("frozen", "own", "max", "welch"))
            + f" of {r['n_null_registered']}",
            "/".join(str(ns[(ds, kd, atk, 'registered', ru, 0.05, 'any')]) for ru in ("frozen", "own", "max", "welch"))
            + f" of {ns[(ds, kd, atk, 'registered', 'n')]}")
    # (c) transmission
    tc = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in rd(os.path.join(out, "transmission_cells.csv"))}
    tr = transmission(cells, rows)
    md = 0.0
    for key, rec in tr.items():
        o = tc[key]
        if rec is None:
            chk(f"(c) {'/'.join(key)}: no teacher disagreement on untriggered inputs", o["n_D_untrig"], 0)
            continue
        for f_, g_ in (("follow_P", "follow_P_untrig"), ("match_C", "match_C_untrig"), ("gap", "gap_untrig"),
                       ("lo", "gap_lo"), ("hi", "gap_hi")):
            md = max(md, abs(rec[f_] - float(o[g_])))
    chk("(c) transmission follow_P, match_C, gap, gap interval (27 cells with D): max |diff|", 0.0, md, tol=1e-9)
    lg = [rec["gap"] for k, rec in tr.items() if rec is not None and k[2] == "logit"]
    s = next(r for r in rd(os.path.join(out, "transmission_summary.csv")) if r["group"] == "all logit")
    chk("(c) mean untriggered gap over the logit cells", s["gap_mean"], float(np.mean(lg)), tol=1e-12)
    # (d) corpus
    g3 = rd(os.path.join(out, "g3_dose.csv"))
    dz = dose()
    for r in g3:
        N, n, nt = dz[float(r["rate"])]
        chk(f"(d) {r['block']} rate {r['rate']}: corpus rows / triggered / non-target triggered",
            f"{r['corpus_rows']}/{r['triggered_rows']}/{r['triggered_source_non_target_label']}", f"{N}/{n}/{nt}")
    with open(os.path.join(out, "independent_check.csv"), "w", newline="") as f:
        f.write("# definition: Adversarial self-review: item outputs (item_output) against an independent "
                "recomputation (own parsers, numpy/scipy, own Tango/Holm/pools/null-swap bookkeeping/corpus draw; "
                "no frozen module, no common.py). ok = agreement (exact for verdicts and counts, 1e-9 for "
                "numbers). Status: " + STATUS + ".\n")
        w = csv.DictWriter(f, fieldnames=["check", "item_output", "independent", "ok"], lineterminator="\n")
        w.writeheader()
        w.writerows(checks)
    bad = [c for c in checks if not c["ok"]]
    print(f"independent: {len(checks)} checks, {len(bad)} disagree")
    for c in bad:
        print("  DISAGREE:", c)


if __name__ == "__main__":
    main()
