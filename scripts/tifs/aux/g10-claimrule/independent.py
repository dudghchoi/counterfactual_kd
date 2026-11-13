#!/usr/bin/env python3
"""g10 independent — the key g10 numbers recomputed by separate code (adversarial self-review).

A15 post hoc auxiliary analysis (claim-rule calibration); changes no primary verdict, battery decision or the
paper-wide Holm.

The reproduction gate (g10_common.state: the official tables reproduced exactly through the frozen loader) runs
first, as in frozen_claims.py and sim.py, and the script stops on a mismatch. Apart from that gate no frozen analysis
function is called here (no analyze.py, addendum.py, analysis/tifs.py, g4 s1c.py or g10 sim.py code path in the
recomputations): scipy.stats.t, numpy and a separate Holm implementation only.
  (a) frozen data: from the raw text of results/tifs/report/primary.csv and variance.csv, the own-variance
      intervals (s2_a / r + input_se^2, df r - 1), the claim-rule verdicts, p_max and the family Holm (F31, F19,
      readings reg / own / claimA / claimB, both sides) -> compared with frozen_claim_counts.csv and
      frozen_claim_family.csv (exact counts) and frozen_claim_cells.csv (intervals to 1e-12).
  (b) Monte Carlo: a vectorised re-simulation of the same registered model with fresh seeds. The model plug-ins
      (Sigma, T_a, w, v_a, Tango SE, F_a, E[s2_pool], structural rows) are read from the harness objects (they
      define the registered model and are not what is tested); everything downstream is re-implemented: the noise
      draw (eigen square root, not Cholesky), the two-way ANOVA pool and its Satterthwaite df, own variance, t
      intervals, verdicts, p-values, the claim rule, p_max and the family Holm. Three settings (config H with the
      measured Sigma at r = 3; config E with the exchangeable Sigma at r = 5 for every seed-pooled battery; config H
      in the a15d setting, BERT AG News logit exchangeable at r = 5 and the other batteries measured at r = 3),
      --reps each; rates compared with sim_cells.csv / sim_family.csv by the two-sided Fisher exact test
      (hypergeometric; pass p >= 6.33e-5), expected correct-claim counts by z over both MC SEs (pass |z| < 4).
Output: independent_check.csv.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TOLS = (0.02, 0.05, 0.10)
ALPHA = 0.025
P_EXACT_MIN = 6.334e-05


def rd(path):
    with open(path) as f:
        return list(csv.DictReader(ln for ln in f if not ln.startswith("#")))


def holm_ind(p):
    """Holm step-down adjusted p (independent implementation; ties by input order)."""
    p = np.asarray(p, float)
    m = p.shape[-1]
    idx = sorted(range(m), key=lambda i: (p[i], i))
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def holm_rows(P):
    """Holm on each row of P (reps x m), independent of sim.holm_vec (loop over ranks)."""
    reps, m = P.shape
    order = np.argsort(P, axis=1, kind="mergesort")      # stable: ties keep the input order
    Ps = np.take_along_axis(P, order, axis=1)
    adj_s = np.zeros_like(Ps)
    run = np.zeros(reps)
    for rank in range(m):
        run = np.maximum(run, np.minimum(1.0, (m - rank) * Ps[:, rank]))
        adj_s[:, rank] = run
    adj = np.empty_like(P)
    adj[np.arange(reps)[:, None], order] = adj_s
    return adj


def verdict(lo, hi, t):
    if -t <= lo and hi <= t:
        return "EQUIVALENT"
    if lo > t:
        return "POSITIVE"
    if hi < -t:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def tint(k, se, df):
    q = stats.t.ppf(0.975, df)
    return k - q * se, k + q * se


def pv(k, se, df, t):
    return {"p_up": float(stats.t.cdf((k - t) / se, df)), "p_pos": float(stats.t.sf((k - t) / se, df))}


# ── (a) frozen data ──────────────────────────────────────────────────


def part_a(out_dir):
    prim = rd(os.path.join(ROOT, "results", "tifs", "report", "primary.csv"))
    var = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in rd(os.path.join(ROOT, "results", "tifs", "report", "variance.csv"))}
    cells = [r for r in prim if r["teacher_qualified"] == "True"]
    assert len(cells) == 31
    rows = []
    recs = []
    for r in cells:
        key = (r["fam"], r["dataset"], r["kd"], r["attack"])
        cls = "structural" if r["method"] == "structural" else ("seed-pooled" if r["method"].startswith("seed-pooled") else "one-seed")
        k = float(r["k"])
        rec = {"id": "/".join(key), "cls": cls, "f19": r["kd"] == "logit", "reg": {}, "own": {}}
        for t in TOLS:
            rec["reg"][t] = {"v": r[f"verdict@{t:g}"], "p_up": float(r[f"p_up@{t:g}"]), "p_pos": float(r[f"p_pos@{t:g}"])}
        if cls == "seed-pooled":
            v = var[key]
            nr = len(r["seeds"].split(","))
            se = math.sqrt(float(v["s_between"]) ** 2 / nr + float(v["input_se"]) ** 2)
            lo, hi = tint(k, se, nr - 1)
            rec["own_lohi"] = (lo, hi)
            rec["reg_lohi"] = (float(r["lo"]), float(r["hi"]))
            for t in TOLS:
                rec["own"][t] = dict(v=verdict(lo, hi, t), **pv(k, se, nr - 1, t))
        recs.append(rec)
    # claim counts
    mine = {}
    for t in TOLS:
        lab = []
        for rec in recs:
            if rec["cls"] != "seed-pooled":
                continue
            a, b = rec["reg"][t]["v"], rec["own"][t]["v"]
            lab.append(a if a == b else "rule-dependent")
        mine[t] = {"claim_POS": lab.count("POSITIVE"), "claim_EQU": lab.count("EQUIVALENT"),
                   "claim_NEG": lab.count("NEGATIVE"), "claim_INC_agreed": lab.count("INCONCLUSIVE"),
                   "claim_rule_dependent": lab.count("rule-dependent")}
        for fam in ("F31", "F19"):
            members = [rec for rec in recs if fam == "F31" or rec["f19"]]
            for reading in ("reg", "own", "claimA", "claimB"):
                for side, pc in (("nt", "p_up"), ("tr", "p_pos")):
                    ps = []
                    for rec in members:
                        preg = rec["reg"][t][pc]
                        if rec["cls"] == "seed-pooled":
                            pown = rec["own"][t][pc]
                            p = {"reg": preg, "own": pown}.get(reading, max(preg, pown))
                        elif rec["cls"] == "one-seed" and reading == "claimB":
                            p = 1.0
                        else:
                            p = preg
                        ps.append(p)
                    adj = holm_ind(ps)
                    cnt = sum(1 for rec, a in zip(members, adj) if a < ALPHA and rec["cls"] != "structural"
                              and (rec["cls"] == "seed-pooled" or not reading.startswith("claim")))
                    mine[t][f"{fam}_{reading}_{side}_counted"] = cnt
    off = {float(r["delta"]): r for r in rd(os.path.join(out_dir, "frozen_claim_counts.csv"))}
    nb, nt = 0, 0
    bad = []
    for t in TOLS:
        for kname, v in mine[t].items():
            nt += 1
            if str(v) != off[t][kname]:
                nb += 1
                bad.append(f"{t}:{kname} {v} vs {off[t][kname]}")
    rows.append({"check": "(a) frozen data: claim-rule cell counts and family-wise counted claims (F31, F19 x reg, own, "
                          "claimA, claimB x no-transfer, transfer x 3 deltas) recomputed from the raw primary.csv / "
                          "variance.csv text with scipy only == frozen_claim_counts.csv",
                 "expected": f"{nt} equal", "got": f"{nt - nb} equal", "ok": nb == 0, "note": "; ".join(bad[:5])})
    cc = rd(os.path.join(out_dir, "frozen_claim_cells.csv"))
    md = 0.0
    for rec in recs:
        if rec["cls"] != "seed-pooled":
            continue
        for x in cc:
            if x["cell"] == rec["id"]:
                md = max(md, abs(float(x["own_lo"]) - rec["own_lohi"][0]), abs(float(x["own_hi"]) - rec["own_lohi"][1]))
    rows.append({"check": "(a) frozen data: own-variance intervals (scipy t) == frozen_claim_cells.csv own_lo / own_hi",
                 "expected": "max |diff| < 1e-12", "got": f"{md:.3g}", "ok": md < 1e-12})
    return rows


# ── (b) Monte Carlo ──────────────────────────────────────────────────


def sqrtm_psd(S):
    vals, vecs = np.linalg.eigh(S)
    return vecs * np.sqrt(np.clip(vals, 0, None))


def anova_pool(K):
    """K: (reps, m, r) -> s2_pool (mean within-attack between-seed variance), Satterthwaite df."""
    reps, m, r = K.shape
    row = K.mean(axis=2, keepdims=True)
    col = K.mean(axis=1, keepdims=True)
    g = K.mean(axis=(1, 2), keepdims=True)
    ms_seed = m * ((col - g) ** 2).sum(axis=(1, 2)) / (r - 1)
    ms_res = ((K - row - col + g) ** 2).sum(axis=(1, 2)) / ((m - 1) * (r - 1)) if m > 1 else np.zeros(reps)
    comb = ms_seed + (m - 1) * ms_res
    den = ms_seed ** 2 + (m - 1) * ms_res ** 2
    df = np.where(den > 0, comb ** 2 * (r - 1) / np.where(den > 0, den, 1.0), m * (r - 1))
    return comb / m, df


def vcode(lo, hi, t):
    out = np.full(lo.shape, 3, dtype=np.int8)
    out[hi < -t] = 2
    out[lo > t] = 1
    out[(-t <= lo) & (hi <= t)] = 0
    return out


def part_b(out_dir, reps, seed):
    sys.path.insert(0, HERE)
    import g10_common as G  # only for the model plug-ins (registered model objects) and the cell layout
    S, HP = G.S, G.HP
    st = G.state()                       # gated (already run in main; cached)
    models = {key: S.pooled_model(st, *key) for key in S.POOLED}
    targets = {key: S.target_model(st, *key, models) for key in S.TARGETS}
    fam = G.family_cells(st)
    sim_cells = rd(os.path.join(out_dir, "sim_cells.csv"))
    sim_fam = rd(os.path.join(out_dir, "sim_family.csv"))
    rows = []
    for si, (config, mode, r) in enumerate((("H", "measured", 3), ("E", "exchangeable", 5), ("H", "a15d", 5))):
        rng = np.random.default_rng(seed + 7919 * si)
        # draw every pooled battery
        est, se_reg, df_reg, se_own, df_own, kind = {}, {}, {}, {}, {}, {}
        pools = {}
        for key in S.POOLED:
            b = models[key]
            if mode == "a15d":                # A15-d: BERT AG News logit at r = 5, the others measured at r = 3
                rb = 5 if key == ("bert", "agnews") else 3
                mdl = HP.exchangeable(b, 5) if rb == 5 else b
            else:
                rb = r
                mdl = b if mode == "measured" else HP.exchangeable(b, r)
            assert mdl["r"] == rb
            T, w = (b["T_H"], b["w_fit"]) if config == "H" else (b["T_E"], b["w_fit_E"])
            m = b["m_sim"]
            L = sqrtm_psd(mdl["Sigma"])
            e = (rng.standard_normal((reps, m * rb)) @ L.T).reshape(reps, m, rb)
            zs = rng.standard_normal((reps, 1, rb))
            za = rng.standard_normal((reps, m, rb))
            Kd = T[None, :, None] * (math.sqrt(w) * zs + math.sqrt(1 - w) * za) + e
            pidx = mdl["pool_idx"]
            s2p, dfp = anova_pool(Kd[:, pidx, :])
            pools[key] = (s2p, dfp)
            for j, a in enumerate(b["sim_attacks"]):
                cid = f"{key[0]}/{key[1]}/logit/{a}"
                est[cid] = Kd[:, j, :].mean(axis=1)
                v = mdl["v_arr"][j]
                se_reg[cid] = np.sqrt(s2p / rb + v)
                df_reg[cid] = dfp
                se_own[cid] = np.sqrt(Kd[:, j, :].var(axis=1, ddof=1) / rb + v)
                df_own[cid] = rb - 1
                kind[cid] = "seed-pooled"
        for tkey in S.TARGETS:
            t_ = targets[tkey]
            src = models[t_["src_key"]]
            mt = len(t_["sim_attacks"])
            T_t = (math.sqrt(src["E_s2"]) * np.ones(mt) if config == "E" else np.sqrt(t_["F"] * src["E_s2"]))
            w_t = src["w_fit_E"] if config == "E" else src["w_fit"]
            L = sqrtm_psd(t_["Sigma"])
            e = rng.standard_normal((reps, mt)) @ L.T
            z = rng.standard_normal((reps, 1))
            za = rng.standard_normal((reps, mt))
            kt = T_t[None, :] * (math.sqrt(w_t) * z + math.sqrt(1 - w_t) * za) + e
            s2p, dfp = pools[t_["src_key"]]
            for j, a in enumerate(t_["sim_attacks"]):
                cid = f"{tkey[0]}/{tkey[1]}/{tkey[2]}/{a}"
                est[cid] = kt[:, j]
                se_reg[cid] = np.sqrt(t_["seT"][j] ** 2 + s2p)
                df_reg[cid] = dfp
                kind[cid] = "one-seed"
        # cell metrics
        ncomp, nbad, zs_all, pmins = 0, 0, [], []
        kobs = {c["id"]: c["row"]["k"] for c in fam}

        def cmp(mine_p, ref_row, col):
            nonlocal ncomp, nbad
            rv, rn = float(ref_row[col]), int(ref_row["reps"])
            x1, x2 = int(round(mine_p * reps)), int(round(rv * rn))
            p = (1.0 if x1 + x2 in (0, reps + rn) else
                 float(stats.fisher_exact([[x1, reps - x1], [x2, rn - x2]]).pvalue))
            pb = (x1 + x2) / (reps + rn)
            z = (mine_p - rv) / math.sqrt(pb * (1 - pb) * (1 / reps + 1 / rn)) if 0 < pb < 1 else 0.0
            ncomp += 1
            nbad += p < P_EXACT_MIN
            zs_all.append(z)
            pmins.append(p)

        for cid, k0 in est.items():
            q_reg = stats.t.ppf(0.975, df_reg[cid])
            lo_r, hi_r = k0 - q_reg * se_reg[cid], k0 + q_reg * se_reg[cid]
            ivs = {"reg": (lo_r, hi_r)}
            if kind[cid] == "seed-pooled":
                q_own = stats.t.ppf(0.975, df_own[cid])
                ivs["own"] = (k0 - q_own * se_own[cid], k0 + q_own * se_own[cid])
                ivs["claim"] = (np.minimum(ivs["reg"][0], ivs["own"][0]), np.maximum(ivs["reg"][1], ivs["own"][1]))
            for rule, (lo, hi) in ivs.items():
                for ti, t in enumerate(TOLS):
                    ref = [x for x in sim_cells if x["config"] == config and x["sigma_mode"] == mode and int(x["r"]) == r
                           and x["cell"] == cid and x["rule"] == rule and abs(float(x["delta"]) - t) < 1e-12]
                    assert len(ref) == 1, (cid, rule, t)
                    ref = ref[0]
                    if ti == 0:
                        cmp(float(np.mean((lo <= 0) & (0 <= hi))), ref, "coverage")
                    cmp(float(np.mean(vcode(lo + t, hi + t, t) == 0)), ref, "false_equivalent_at_mu_eq_delta")
                    cmp(float(np.mean(vcode(lo + t, hi + t, t) == 1)), ref, "false_positive_at_mu_eq_delta")
                    cmp(float(np.mean(vcode(lo, hi, t) == 0)), ref, "p_equivalent_mu0")
                    cmp(float(np.mean(vcode(lo + kobs[cid], hi + kobs[cid], t) == 1)), ref, "p_positive_at_obs")
        cell_n, cell_bad = ncomp, nbad
        # family metrics (counted claims; reading claimA for the claim rule)
        order = [c["id"] for c in fam]
        cls = {c["id"]: c["cls"] for c in fam}
        f19 = {c["id"]: c["in_F19"] for c in fam}
        struct = {c["id"]: c["row"] for c in fam if c["cls"] == "structural"}
        for fname in ("F31", "F19"):
            mem = [cid for cid in order if fname == "F31" or f19[cid]]
            for reading in ("reg", "own", "claimA"):
                for ti, t in enumerate(TOLS):
                    for side, conf in (("NT", "alld"), ("T", "all0"), ("T", "alld"), ("NT", "obs"), ("T", "obs")):
                        P = np.empty((reps, len(mem)))
                        mu = np.zeros(len(mem))
                        for ci, cid in enumerate(mem):
                            if cls[cid] == "structural":
                                P[:, ci] = float(struct[cid][f"{'p_up' if side == 'NT' else 'p_pos'}@{t:g}"])
                                continue
                            shift = {"all0": 0.0, "alld": t, "obs": kobs[cid]}[conf]
                            mu[ci] = shift
                            k1 = est[cid] + shift
                            sgn = (lambda x, d: stats.t.cdf(x, d)) if side == "NT" else (lambda x, d: stats.t.sf(x, d))
                            p_reg = sgn((k1 - t) / se_reg[cid], df_reg[cid])
                            if cls[cid] == "seed-pooled":
                                p_own = sgn((k1 - t) / se_own[cid], df_own[cid])
                                p = {"reg": p_reg, "own": p_own}.get(reading, np.maximum(p_reg, p_own))
                            else:
                                p = p_reg
                            P[:, ci] = p
                        adj = holm_rows(P)
                        cnt_mask = np.array([cls[c] != "structural" and (cls[c] == "seed-pooled" or not reading.startswith("claim"))
                                             for c in mem])
                        correct = (mu < t) if side == "NT" else (mu > t)
                        rej = adj < ALPHA
                        fals = rej[:, cnt_mask & ~correct]
                        p_any_false = float(np.mean(fals.any(axis=1))) if fals.shape[1] else 0.0
                        e_corr = float(rej[:, cnt_mask & correct].sum(axis=1).mean())
                        se_corr = float(rej[:, cnt_mask & correct].sum(axis=1).std(ddof=1)) / math.sqrt(reps)
                        ref = [x for x in sim_fam if x["config"] == config and x["sigma_mode"] == mode
                               and int(x["r"]) == r and x["family"] == fname and x["reading"] == reading
                               and abs(float(x["delta"]) - t) < 1e-12 and x["configuration"] == conf
                               and x["side"].startswith("no-transfer" if side == "NT" else "transfer")]
                        assert len(ref) == 1
                        ref = ref[0]
                        if conf != "obs":
                            cmp(p_any_false, ref, "counted_p_any_false")
                        else:
                            rv, rse = float(ref["counted_e_correct"]), float(ref["counted_e_correct_mcse"])
                            se_both = math.sqrt(se_corr ** 2 + rse ** 2)
                            ncomp += 1
                            ok = abs(e_corr - rv) <= 4 * se_both + 1e-12
                            nbad += not ok
                            zs_all.append((e_corr - rv) / se_both if se_both > 0 else 0.0)
        zs_all = np.array(zs_all)
        rows.append({"check": f"(b) independent vectorised re-simulation ({config}, {mode} Sigma, r = {r}"
                              f"{' for BERT AG News logit, 3 for the others' if mode == 'a15d' else ''}, {reps:,} reps, "
                              "fresh seeds) vs sim_cells.csv (coverage, false EQU / POS at mu = d, EQU at mu = 0, POS at "
                              "k_obs; reg / own / claim) and sim_family.csv (counted P(>= 1 false) at alld / all0, "
                              "E[correct] at obs; reg / own / claimA; F31, F19)",
                     "expected": f"{ncomp} ok", "got": f"{ncomp - nbad} ok", "ok": nbad == 0,
                     "note": (f"cell comparisons {cell_n} ({cell_bad} fail); smallest Fisher p {min(pmins):.4f}; "
                              f"max |z| {np.max(np.abs(zs_all)):.2f}; "
                              f"|z| > 1.96 in {np.mean(np.abs(zs_all) > 1.96):.3f}; mean z {zs_all.mean():+.3f}")})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tifs", "addendum", "aux", "g10-claimrule"))
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=99020261027)
    args = ap.parse_args()
    sys.path.insert(0, HERE)
    import g10_common as G
    G.state()                            # the reproduction gate first: stops (exit 3) on any mismatch
    rows = part_a(args.out) + part_b(args.out, args.reps, args.seed)
    G.write_csv(rows, os.path.join(args.out, "independent_check.csv"), """
        g10 independent check: (a) the frozen-data claim counts recomputed from the raw primary.csv / variance.csv
        text with scipy only; (b) a vectorised re-simulation of the registered model with fresh seeds and separately
        written pool / df / interval / verdict / claim-rule / Holm code, compared with sim_cells.csv and sim_family.csv
        (rates: two-sided Fisher exact test, hypergeometric, pass p >= 6.33e-5; E[correct]: |z| < 4 over both MC SEs);
        settings H measured r = 3, E exchangeable r = 5, H a15d (BERT AG News logit at r = 5, the others at r = 3).
        The reproduction gate runs first.""")
    for r in rows:
        print("ok  " if r["ok"] else "FAIL", r["check"][:100], r["got"], r.get("note", ""))
    if not all(r["ok"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
