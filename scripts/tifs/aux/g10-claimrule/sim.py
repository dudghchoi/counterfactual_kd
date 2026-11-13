#!/usr/bin/env python3
"""g10 sim — claim-rule calibration on the registered S1c harness (31-cell family, joint simulation).

A15 post hoc auxiliary analysis registered in docs/tifs_addendum.md, amendment 10/27 20:58 KST, item 4;
changes no primary verdict, battery decision or the paper-wide Holm.

Model = the registered A11.10 S1c harness, imported unchanged from scripts/tifs/aux/g4-s1c-s9-s1extra/
(common.py, s1c.py: pooled_model, target_model, draw_battery, pool_of, evaluate, chol, ExactFastT). One
replicate draws every battery of the 31-cell family jointly:
  seed-pooled batteries  BERT SST-2 logit (4 pooled + SOS, EP structural), BERT AG News logit (6),
                         Qwen3 SST-2 logit (3), GPT-2 SST-2 logit (2):
                         k[a, s] = mu_a + T_a (sqrt(w) z_s + sqrt(1 - w) z_as) + e[a, s], e ~ N(0, Sigma)
  one-seed batteries     BERT SST-2 'feature' (5 + EP structural) <- BERT SST-2 logit pool,
                         BERT AG News 'feature' (6) <- BERT AG News logit pool,
                         Qwen3 AG News logit (2) <- Qwen3 SST-2 logit pool:
                         k'_a = mu_a + T'_a (sqrt(w) z' + sqrt(1 - w) z'_a) + e'_a,
                         SE^2 = SE_Tango,a^2 + s2_pool with s2_pool and its df from the SAME simulated source
                         battery of the replicate (the borrowed-pool structure, as A14 headline_power.py; the
                         registered S1c used an independent source draw per job, which has the same marginal
                         law, so every per-cell and per-battery rate is comparable)
  structural nulls       BERT SST-2 EP logit and 'feature': their fixed frozen seed-42 Tango rows.
Configs (S1c): H = measured heterogeneous training SD T_a with the fitted shared fraction w; E = homogeneous
T (pool mean of T_H^2) with its own fitted w; H05 = H with w = 0.5. One-seed targets: T'_a = sqrt(F_a E[s2_pool])
under H and H05, sqrt(E[s2_pool]) under E (c = 1); w of the target = the source battery's w (0.5 under H05; the
registered S1c ran no H05 transported job, so H05 one-seed rows have no S1c reference).
Seeds r (Sigma modes): 3 with the measured input covariance ('measured', = S1c); 3 and 5 with the exchangeable
covariance of headline_power.exchangeable (W, X from the measured Sigma; v_a recomputed at r; T, w kept at the r = 3
fit) for EVERY seed-pooled battery ('exchangeable'; r = 5 is hypothetical: A15-d adds seeds 45, 46 to BERT AG News
logit only); 'a15d' = the A15-d data path: BERT AG News logit at r = 5 with the exchangeable covariance, the other
seed-pooled batteries at r = 3 with the measured covariance (key r = 5 = the BERT AG News r; the BERT AG News
'feature' one-seed cells borrow that r = 5 pool, as every one-seed battery borrows its source battery's pool).
Seeds of the measured / exchangeable jobs are those of the first run (base + 1 ... base + 9); the a15d jobs use
base + 10 + config index, so adding them leaves every earlier number unchanged.

Rules (frozen t_contrast in every case; v_a = the model's input variance of the seed mean, known as in S1c):
  reg    registered: seed-pooled SE^2 = s2_pool / r + v_a at the Satterthwaite df of the frozen
         analyze.pool_seed_variance; one-seed cells the frozen transported rule.
  own    A3.1 own variance: SE^2 = s2_a / r + v_a at df r - 1, s2_a = addendum.var of the cell's r seed k's
         (the input term v_a included, as the fixed formula); one-seed cells: not defined (cell rows blank; in
         the own-rule family Holm they enter with their registered p).
  claim  claim rule: the cell's verdict when reg and own agree, else INCONCLUSIVE ('rule-dependent'); one-seed
         cells 'one-seed, descriptive' (never counted); structural nulls counted apart. Because the reg and own
         intervals share the centre k, the claim verdict equals the verdict of the wider of the two intervals;
         its coverage is that of the hull [min lo, max hi] (checked in every replicate). Family-wise: Holm over
         p_max = max(p_reg, p_own); one-seed cells enter with p_reg (reading A) or with p = 1 (reading B).

Metrics (mu enters as a pure shift: the pool, s2_a and every SE are shift-invariant, so one noise draw serves
every mu; the frozen t_contrast is called at est + mu):
  cell     coverage of mu; false 'below' P(hi < mu); false 'above' P(lo > mu); false EQUIVALENT@d and false
           POSITIVE@d at mu = d; P(EQUIVALENT@d | mu = 0); P(POSITIVE@d), P(EQUIVALENT@d) at mu = the frozen
           primary k (power at the observed k; 'positive_cell' marks the frozen POSITIVE cells); P(reg != own).
  family   F31 (31 qualified primary cells) and F19 (19 logit cells); frozen-order Holm (bit-identical vectorised
           copy of analysis.tifs.holm, checked against it) at one-sided 0.025: no-transfer side over p_up@d
           (H0 k >= d), transfer side over p_pos@d (H0 k <= d). Configurations: all0 (every simulated mu = 0),
           alld (every simulated mu = d: the global boundary), obs (mu = the frozen primary k), one_at_d (cell j at
           d, the others at 0; the least favourable no-transfer case with realistic other cells), one_at_d_T
           (transfer side: cell j at d, every other simulated cell surely rejected, i.e. its p_pos set to 0, the
           mu -> +inf limit; structural nulls keep their fixed p, one-seed cells under claimB keep p = 1; the least
           favourable transfer-side configuration; mu = +1 is not used because at df 2 the t tail does not make
           the rejection sure). Counted claims:
           reg/own every non-structural cell; claim rule the seed-pooled cells (one-seed rejections are reported as
           'descriptive'); structural claims apart. Expected claim counts carry an MC SE from the per-replicate
           sums of squares.
  battery  the registered S1c battery quantities (s1c.evaluate unchanged) for the validation against
           aux/g4 s1c_battery.csv.

Reps: --blocks seed blocks of --reps each (default 8 x 10,000 = 80,000 per config x mode); block b uses
seed_job + 1,000,000 b. MC SE binomial over all reps (<= 0.0018 for any rate at 80,000).

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" python sim.py [--out DIR] [--workers 10] [--quick]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import g10_common as G  # noqa: E402

K, S, A, AD, HP = G.K, G.S, G.A, G.AD, G.HP
TOLS, ALPHA = G.TOLS, G.ALPHA
TOLS0 = (0.0,) + TOLS
ND = len(TOLS)
CONFIGS = ("E", "H", "H05")
MODES = (("measured", 3), ("exchangeable", 3), ("exchangeable", 5), ("a15d", 5))
MODES_FIRST = MODES[:3]          # the Sigma modes of the first run (their seeds are kept)
A15D_KEY = ("bert", "agnews")    # the battery A15-d extends to r = 5
RULES2 = ("reg", "own")
READINGS = ("reg", "own", "claimA", "claimB")
CODE = {"EQUIVALENT": 0, "POSITIVE": 1, "NEGATIVE": 2, "INCONCLUSIVE": 3}
EQU, POS, NEG, INC = 0, 1, 2, 3
FAMS = ("F31", "F19")
SIDES = (("NT", "p_up"), ("T", "p_pos"))
FCONFS = ("all0", "alld", "obs")


# ── the family layout ────────────────────────────────────────────────


def layout(st, models, targets):
    """Map the 31 frozen-order family cells to their simulation source."""
    fam = G.family_cells(st)
    cols = []
    for c in fam:
        r = c["row"]
        key3 = (r["fam"], r["dataset"], r["kd"])
        if c["cls"] == "structural":
            cols.append(dict(c, src=("struct", None, None)))
        elif c["cls"] == "seed-pooled":
            b = models[key3[:2]]
            assert r["kd"] == "logit"
            cols.append(dict(c, src=("pooled", key3[:2], b["sim_attacks"].index(r["attack"]))))
        else:
            t = targets[key3]
            cols.append(dict(c, src=("target", key3, t["sim_attacks"].index(r["attack"]))))
    return cols


def struct_check(st, models, targets, cols):
    """The harness's fixed structural rows (paired_contrast of the seed-42 P, C) == the frozen primary rows."""
    rows = []
    for c in cols:
        if c["cls"] != "structural":
            continue
        r = c["row"]
        key3 = (r["fam"], r["dataset"], r["kd"])
        holder = models.get(key3[:2]) if r["kd"] == "logit" else targets.get(key3)
        sr = holder["struct_rows"][r["attack"]]
        bad = [f"{p}@{t:g}" for p in ("p_up", "p_pos", "p_tost", "verdict") for t in TOLS
               if sr[f"{p}@{t:g}"] != r[f"{p}@{t:g}"]]
        bad += [x for x in ("k", "lo", "hi") if sr[x] != r[x]]
        rows.append({"check": f"structural row {c['id']}: harness struct_rows == frozen primary row (k, lo, hi, "
                              "p_up/p_pos/p_tost/verdict at every delta)", "expected": 0, "got": len(bad),
                     "ok": not bad, "note": ",".join(bad)})
    return rows


# ── worker ───────────────────────────────────────────────────────────


def holm_vec(P):
    """Row-wise Holm step-down adjusted p (vectorised copy of the frozen analysis.tifs.holm: stable order,
    min(1, (m - rank) p), running max). Bit-identical to the frozen function (checked)."""
    m = P.shape[1]
    order = np.argsort(P, axis=1, kind="stable")
    Ps = np.take_along_axis(P, order, axis=1)
    mult = (m - np.arange(m)).astype(float)[None, :]
    adj_s = np.maximum.accumulate(np.minimum(1.0, mult * Ps), axis=1)
    adj = np.empty_like(P)
    np.put_along_axis(adj, order, adj_s, axis=1)
    return adj


def vcode_from_interval(lo, hi, tol):
    """t_contrast's verdict rule on arrays (EQU if -tol <= lo and hi <= tol; POS if lo > tol; NEG if
    hi < -tol; else INC)."""
    out = np.full(lo.shape, INC, dtype=np.int8)
    out[hi < -tol] = NEG
    out[lo > tol] = POS
    out[(-tol <= lo) & (hi <= tol)] = EQU
    return out


def run_job(job):
    t0 = time.time()
    rng = np.random.default_rng(job["seed"])
    reps = job["reps"]
    bats, tgts, cols = job["bats"], job["tgts"], job["cols"]
    nc = len(cols)
    draws = {key: S.draw_battery(rng, mdl, T, w, reps) for key, mdl, T, w in bats}
    kt = {}
    for tkey, tg, T_t, w_t in tgts:
        mt = len(tg["sim_attacks"])
        z_t = rng.standard_normal((reps, 1))
        z_ta = rng.standard_normal((reps, mt))
        e_t = rng.standard_normal((reps, mt)) @ tg["L"].T
        kt[tkey] = T_t[None, :] * (math.sqrt(w_t) * z_t + math.sqrt(1 - w_t) * z_ta) + e_t
    mdls = {key: mdl for key, mdl, _, _ in bats}
    tgd = {tkey: tg for tkey, tg, _, _ in tgts}
    kobs = np.array([c["row"]["k"] for c in cols], dtype=float)
    # storage (reps x cells [x delta])
    LO = {q: np.full((reps, nc), np.nan) for q in RULES2}
    HI = {q: np.full((reps, nc), np.nan) for q in RULES2}
    PUZ = {q: np.full((reps, nc), np.nan) for q in RULES2}
    PPZ = {q: np.full((reps, nc), np.nan) for q in RULES2}
    V = {(q, mu): np.full((reps, nc, ND), INC, dtype=np.int8) for q in RULES2 for mu in ("0", "d", "o")}
    PU = {(q, mu): np.full((reps, nc, ND), np.nan) for q in RULES2 for mu in ("0", "d", "o")}
    PP = {(q, mu): np.full((reps, nc, ND), np.nan) for q in RULES2 for mu in ("0", "d", "o")}
    # structural columns: the fixed frozen rows (same at every mu)
    for ci, c in enumerate(cols):
        if c["cls"] == "structural":
            r = c["row"]
            for q in RULES2:
                LO[q][:, ci], HI[q][:, ci] = r["lo"], r["hi"]
                for mu in ("0", "d", "o"):
                    for ti, t in enumerate(TOLS):
                        V[(q, mu)][:, ci, ti] = CODE[r[f"verdict@{t:g}"]]
                        PU[(q, mu)][:, ci, ti] = r[f"p_up@{t:g}"]
                        PP[(q, mu)][:, ci, ti] = r[f"p_pos@{t:g}"]
    sim_cols = [(ci, c) for ci, c in enumerate(cols) if c["cls"] != "structural"]
    # battery validation accumulators (registered rule, s1c.evaluate unchanged)
    bnames = [("pooled", key) for key, _, _, _ in bats] + [("target", tkey) for tkey, _, _, _ in tgts]
    bacc = {bn: {"dec": {t: {"TRANSFERS": 0, "NO TRANSFER": 0, "INCONCLUSIVE": 0} for t in TOLS},
                 "tr_alld": np.zeros(ND), "nt_one": None} for bn in bnames}
    pool_df_sum = {key: 0.0 for key in mdls}
    pool_s2_sum = {key: 0.0 for key in mdls}
    tc = K.t_contrast
    var = AD.var
    for i in range(reps):
        pools = {}
        for key, mdl in mdls.items():
            pool = S.pool_of(draws[key][i], mdl)       # frozen analyze.pool_seed_variance
            pools[key] = (pool["s2"], pool["df"])
            pool_df_sum[key] += pool["df"]
            pool_s2_sum[key] += pool["s2"]
        res_bat = {bn: {} for bn in bnames}
        for ci, c in sim_cols:
            kind, key, j = c["src"]
            if kind == "pooled":
                mdl = mdls[key]
                r = mdl["r"]
                ks = draws[key][i, j]
                est = float(ks.mean())
                v = mdl["v_arr"][j]
                s2, df = pools[key]
                s2a = var([float(x) for x in ks])
                specs = (("reg", math.sqrt(s2 / r + v), df), ("own", math.sqrt(s2a / r + v), float(r - 1)))
                bn = ("pooled", key)
            else:
                tg = tgd[key]
                est = float(kt[key][i, j])
                s2, df = pools[tg["src_key"]]
                specs = (("reg", math.sqrt(tg["seT"][j] ** 2 + s2), df),)
                bn = ("target", key)
            for q, se, dfq in specs:
                r0 = tc(est, se, dfq, TOLS0)
                if q == "reg":
                    res_bat[bn][j] = r0
                LO[q][i, ci], HI[q][i, ci] = r0["lo"], r0["hi"]
                PUZ[q][i, ci], PPZ[q][i, ci] = r0["p_up@0"], r0["p_pos@0"]
                ro = tc(est + kobs[ci], se, dfq, TOLS)
                for ti, t in enumerate(TOLS):
                    tk = f"{t:g}"
                    V[(q, "0")][i, ci, ti] = CODE[r0["verdict@" + tk]]
                    PU[(q, "0")][i, ci, ti] = r0["p_up@" + tk]
                    PP[(q, "0")][i, ci, ti] = r0["p_pos@" + tk]
                    rd = tc(est + t, se, dfq, (t,))
                    V[(q, "d")][i, ci, ti] = CODE[rd["verdict@" + tk]]
                    PU[(q, "d")][i, ci, ti] = rd["p_up@" + tk]
                    PP[(q, "d")][i, ci, ti] = rd["p_pos@" + tk]
                    V[(q, "o")][i, ci, ti] = CODE[ro["verdict@" + tk]]
                    PU[(q, "o")][i, ci, ti] = ro["p_up@" + tk]
                    PP[(q, "o")][i, ci, ti] = ro["p_pos@" + tk]
        # registered battery quantities, exactly as the S1c harness (s1c.evaluate)
        for bn in bnames:
            kind, key = bn
            holder = mdls[key] if kind == "pooled" else tgd[key]
            attacks = holder["sim_attacks"]
            ev = S.evaluate([res_bat[bn][j] for j in range(len(attacks))], holder["struct_rows"], attacks)
            acc = bacc[bn]
            for t, d in zip(TOLS, ev["dec"]):
                acc["dec"][t][d] += 1
            acc["tr_alld"] += np.array(ev["tr_alld"], dtype=float)
            nt = np.array(ev["nt_one"], dtype=float)
            acc["nt_one"] = nt if acc["nt_one"] is None else acc["nt_one"] + nt
    loop_s = time.time() - t0
    # one-seed and structural columns: 'own' = registered (own variance not defined)
    for ci, c in enumerate(cols):
        if c["cls"] == "one-seed":
            LO["own"][:, ci], HI["own"][:, ci] = LO["reg"][:, ci], HI["reg"][:, ci]
            PUZ["own"][:, ci], PPZ["own"][:, ci] = PUZ["reg"][:, ci], PPZ["reg"][:, ci]
            for mu in ("0", "d", "o"):
                V[("own", mu)][:, ci] = V[("reg", mu)][:, ci]
                PU[("own", mu)][:, ci] = PU[("reg", mu)][:, ci]
                PP[("own", mu)][:, ci] = PP[("reg", mu)][:, ci]
    out = {"key": job["key"], "block": job["block"], "seed": job["seed"], "reps": reps, "loop_s": loop_s}
    out["cell"] = cell_counts(cols, LO, HI, PUZ, PPZ, V)
    out["family"], out["onecell"], out["onecell_T"], out["holm_check"] = family_counts(cols, PU, PP, kobs,
                                                                                       job["check_reps"])
    out["battery"] = {bn: {"dec": {t: dict(d) for t, d in a["dec"].items()}, "tr_alld": a["tr_alld"],
                           "nt_one": a["nt_one"]} for bn, a in bacc.items()}
    out["pool"] = {key: (pool_df_sum[key], pool_s2_sum[key]) for key in mdls}
    out["seconds"] = time.time() - t0
    return out


def cell_counts(cols, LO, HI, PUZ, PPZ, V):
    """Event counts per cell (sim cells only) x rule (reg, own, claim) [x delta]."""
    sp = np.array([c["cls"] == "seed-pooled" for c in cols])
    sim = np.array([c["cls"] != "structural" for c in cols])
    out = {}

    def claim_code(a, b):
        return np.where(a == b, a, INC).astype(np.int8)

    # coverage, false below / above (mu-free)
    cov = {q: (LO[q] <= 0.0) & (0.0 <= HI[q]) for q in RULES2}
    lo_h, hi_h = np.minimum(LO["reg"], LO["own"]), np.maximum(HI["reg"], HI["own"])
    cov["claim"] = (lo_h <= 0.0) & (0.0 <= hi_h)
    below = {q: PUZ[q] < ALPHA for q in RULES2}
    below["claim"] = below["reg"] & below["own"]
    above = {q: PPZ[q] < ALPHA for q in RULES2}
    above["claim"] = above["reg"] & above["own"]
    for name, d in (("cover", cov), ("below", below), ("above", above)):
        for q, arr in d.items():
            out[(name, q)] = arr.sum(axis=0).astype(float)
    # verdict-based metrics
    Vc = {(q, mu): V[(q, mu)] for q in RULES2 for mu in ("0", "d", "o")}
    for mu in ("0", "d", "o"):
        Vc[("claim", mu)] = claim_code(V[("reg", mu)], V[("own", mu)])
    for q in ("reg", "own", "claim"):
        out[("equ0", q)] = (Vc[(q, "0")] == EQU).sum(axis=0).astype(float)        # (nc, ND)
        out[("fequ_d", q)] = (Vc[(q, "d")] == EQU).sum(axis=0).astype(float)
        out[("fpos_d", q)] = (Vc[(q, "d")] == POS).sum(axis=0).astype(float)
        for code, nm in ((POS, "pos_o"), (EQU, "equ_o"), (NEG, "neg_o"), (INC, "inc_o")):
            out[(nm, q)] = (Vc[(q, "o")] == code).sum(axis=0).astype(float)
    for mu, nm in (("0", "rd0"), ("d", "rdd"), ("o", "rdo")):
        out[(nm, "claim")] = (V[("reg", mu)] != V[("own", mu)]).sum(axis=0).astype(float)
    # identity check: claim verdict == verdict of the hull interval (mu = 0, every delta, seed-pooled cells)
    mism = 0
    for ti, t in enumerate(TOLS):
        hv = vcode_from_interval(lo_h, hi_h, t)
        for q in RULES2:   # the vectorised verdict rule reproduces t_contrast's verdicts
            mism += int(np.sum((vcode_from_interval(LO[q], HI[q], t) != V[(q, "0")][:, :, ti])[:, sim]))
        mism += int(np.sum((hv != Vc[("claim", "0")][:, :, ti])[:, sp]))
    out[("hull_mismatch", "claim")] = np.array([float(mism)])
    # wider interval == hull (shared centre)
    w_reg = (HI["reg"] - LO["reg"]) >= (HI["own"] - LO["own"])
    wid_lo = np.where(w_reg, LO["reg"], LO["own"])
    out[("wider_mismatch", "claim")] = np.array([float(np.sum((wid_lo != lo_h)[:, sp]))])
    return out


def family_counts(cols, PU, PP, kobs, check_reps):
    """Family-wise claim counts per reading x family x delta x side x configuration; per-cell one_at_d rates
    (no-transfer side, others at 0) and one_at_d_T rates (transfer side, others surely rejected); and a check of
    holm_vec against the frozen holm on the first ``check_reps`` replicates."""
    cls = np.array([c["cls"] for c in cols])
    in19 = np.array([c["in_F19"] for c in cols])
    fam_idx = {"F31": np.arange(len(cols)), "F19": np.where(in19)[0]}
    out, onecell, onecell_T = {}, {}, {}
    hc = {"n_compared": 0, "n_mismatch": 0}

    def pmat(reading, mu, side, ti, idx):
        src = PU if side == "NT" else PP
        if reading == "reg":
            P = src[("reg", mu)][:, idx, ti]
        elif reading == "own":
            P = src[("own", mu)][:, idx, ti]
        else:
            P = np.maximum(src[("reg", mu)][:, idx, ti], src[("own", mu)][:, idx, ti])
            if reading == "claimB":
                P = P.copy()
                P[:, cls[idx] == "one-seed"] = 1.0
        return P

    for fname, idx in fam_idx.items():
        c_cls = cls[idx]
        k_f = kobs[idx]
        for reading in READINGS:
            counted = (c_cls == "seed-pooled") if reading.startswith("claim") else (c_cls != "structural")
            descr = (c_cls == "one-seed") if reading.startswith("claim") else np.zeros(len(idx), bool)
            masks = {"counted": counted, "seed_pooled": c_cls == "seed-pooled", "one_seed": c_cls == "one-seed",
                     "descriptive": descr, "structural": c_cls == "structural"}
            for ti, t in enumerate(TOLS):
                for side, _ in SIDES:
                    for conf in FCONFS:
                        mu = {"all0": "0", "alld": "d", "obs": "o"}[conf]
                        P = pmat(reading, mu, side, ti, idx)
                        adj = holm_vec(P)
                        if check_reps:
                            for i in range(min(check_reps, P.shape[0])):
                                ref = K.holm([float(x) for x in P[i]])
                                hc["n_compared"] += 1
                                hc["n_mismatch"] += int(any(a != b for a, b in zip(ref, adj[i])))
                        rej = adj < ALPHA
                        # truth of each claim
                        is_sim = c_cls != "structural"
                        if conf == "all0":
                            mu_true = np.zeros(len(idx))
                        elif conf == "alld":
                            mu_true = np.where(is_sim, t, 0.0)
                        else:
                            mu_true = np.where(is_sim, k_f, 0.0)
                        correct = (mu_true < t) if side == "NT" else (mu_true > t)
                        for mname, msk in masks.items():
                            if not msk.any():
                                continue
                            fals = rej[:, msk & ~correct]
                            corr = rej[:, msk & correct]
                            n_f = fals.sum(axis=1).astype(float)       # per-replicate false claims
                            n_c = corr.sum(axis=1).astype(float)       # per-replicate correct claims
                            out[(fname, reading, t, side, conf, mname)] = np.array([
                                float(np.sum(fals.any(axis=1))),     # reps with >= 1 false claim
                                float(n_f.sum()),                     # false claims
                                float(n_c.sum()),                     # correct claims
                                float(np.sum(rej[:, msk].any(axis=1))),  # reps with >= 1 claim
                                float(np.sum(n_f ** 2)),              # sum of squared per-replicate false claims
                                float(np.sum(n_c ** 2)),              # sum of squared per-replicate correct claims
                            ])
                # one_at_d: cell j at d, the others at 0 (no-transfer side)
                P0 = pmat(reading, "0", "NT", ti, idx)
                Pd = pmat(reading, "d", "NT", ti, idx)
                for jj in range(len(idx)):
                    if c_cls[jj] == "structural":
                        continue
                    P = P0.copy()
                    P[:, jj] = Pd[:, jj]
                    adj = holm_vec(P)
                    onecell[(fname, reading, t, int(idx[jj]))] = float(np.sum(adj[:, jj] < ALPHA))
                # one_at_d_T: cell j at d, every other simulated cell surely rejected (p_pos = 0; mu -> +inf);
                # structural nulls keep their fixed p; under claimB the one-seed cells keep p = 1 (pmat)
                PdT = pmat(reading, "d", "T", ti, idx)
                Pinf = PdT.copy()
                sure = (c_cls != "structural") & ~((c_cls == "one-seed") & (reading == "claimB"))
                Pinf[:, sure] = 0.0
                for jj in range(len(idx)):
                    if c_cls[jj] == "structural":
                        continue
                    P = Pinf.copy()
                    P[:, jj] = PdT[:, jj]
                    adj = holm_vec(P)
                    onecell_T[(fname, reading, t, int(idx[jj]))] = float(np.sum(adj[:, jj] < ALPHA))
    return out, onecell, onecell_T, hc


# ── main ─────────────────────────────────────────────────────────────


def job_seed(base, config, mode, r):
    """Seed of a config x Sigma mode job: the first-run modes keep base + 1 ... base + 9 (config-major order, as
    the first run); a15d uses base + 10 + config index."""
    if (mode, r) in MODES_FIRST:
        return base + 1 + CONFIGS.index(config) * len(MODES_FIRST) + MODES_FIRST.index((mode, r))
    assert mode == "a15d"
    return base + 1 + len(CONFIGS) * len(MODES_FIRST) + CONFIGS.index(config)


def battery_model(b, key, mode, r):
    """The simulated model of one seed-pooled battery under a Sigma mode."""
    if mode == "measured":
        return b
    if mode == "exchangeable":
        return HP.exchangeable(b, r)
    assert mode == "a15d" and r == 5
    return HP.exchangeable(b, 5) if key == A15D_KEY else b


def build_jobs(models, targets, cols, args):
    jobs = []
    for config in CONFIGS:
        for mode, r in MODES:
            seed = job_seed(args.seed, config, mode, r)
            bats = []
            for key in S.POOLED:
                b = models[key]
                mdl = battery_model(b, key, mode, r)
                T, w = {"H": (b["T_H"], b["w_fit"]), "E": (b["T_E"], b["w_fit_E"]),
                        "H05": (b["T_H"], 0.5)}[config]
                bats.append((key, mdl, T, w))
            tgts = []
            for tkey in S.TARGETS:
                t = targets[tkey]
                src = models[t["src_key"]]
                if config == "E":
                    T_t, w_t = math.sqrt(src["E_s2"]) * np.ones(len(t["sim_attacks"])), src["w_fit_E"]
                else:
                    T_t, w_t = np.sqrt(t["F"] * src["E_s2"]), (src["w_fit"] if config == "H" else 0.5)
                tgts.append((tkey, t, T_t, w_t))
            for blk in range(args.blocks):
                jobs.append({"key": (config, mode, r), "block": blk, "seed": seed + 1_000_000 * blk,
                             "reps": args.reps, "bats": bats, "tgts": tgts, "cols": cols,
                             "check_reps": args.check_reps if blk == 0 else 0})
    return jobs


def merge(blocks):
    blocks = sorted(blocks, key=lambda b: b["block"])
    out = {"key": blocks[0]["key"], "reps": sum(b["reps"] for b in blocks),
           "seconds": sum(b["seconds"] for b in blocks), "seeds": [b["seed"] for b in blocks],
           "block_reps": [b["reps"] for b in blocks]}
    for part in ("cell", "family", "onecell", "onecell_T"):
        out[part] = {k: sum(b[part][k] for b in blocks) for k in blocks[0][part]}
    out["holm_check"] = {k: sum(b["holm_check"][k] for b in blocks) for k in blocks[0]["holm_check"]}
    bat = {}
    for bn in blocks[0]["battery"]:
        bat[bn] = {"dec": {t: {d: sum(b["battery"][bn]["dec"][t][d] for b in blocks)
                               for d in blocks[0]["battery"][bn]["dec"][t]} for t in TOLS},
                   "tr_alld": sum(b["battery"][bn]["tr_alld"] for b in blocks),
                   "nt_one": sum(b["battery"][bn]["nt_one"] for b in blocks)}
    out["battery"] = bat
    out["pool"] = {k: (sum(b["pool"][k][0] for b in blocks), sum(b["pool"][k][1] for b in blocks))
                   for k in blocks[0]["pool"]}
    out["blocks"] = blocks
    return out


def main():
    ap = G.add_out_arg(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--reps", type=int, default=10000, help="reps per seed block")
    ap.add_argument("--blocks", type=int, default=8, help="seed blocks per config x mode")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--check-reps", type=int, default=50,
                    help="replicates of block 1 on which holm_vec is compared with the frozen holm")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma list of config:mode:r keys to run (testing)")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.blocks = 300, 2
    t_start = time.time()
    out = G.out_dir(args.out)
    st = G.state()                                   # gate first
    models = {key: S.pooled_model(st, *key) for key in S.POOLED}
    targets = {key: S.target_model(st, *key, models) for key in S.TARGETS}
    cols = layout(st, models, targets)
    checks = struct_check(st, models, targets, cols)
    jobs = build_jobs(models, targets, cols, args)
    if args.only:
        keep = {tuple(x.split(":")[:2]) + (int(x.split(":")[2]),) for x in args.only.split(",")}
        jobs = [j for j in jobs if j["key"] in keep]
    print(f"{len(jobs)} jobs x {args.reps} reps; workers {args.workers}", flush=True)
    from multiprocessing import Pool
    raw = []
    with Pool(args.workers, initializer=S._init_worker, initargs=(True,)) as pool:
        for res in pool.imap_unordered(run_job, jobs):
            raw.append(res)
            print(f"  done {res['key']} b{res['block'] + 1} {res['seconds']:.0f}s (loop {res['loop_s']:.0f}s) "
                  f"({len(raw)}/{len(jobs)}, {time.time() - t_start:.0f}s)", flush=True)
    by = {}
    for res in raw:
        by.setdefault(res["key"], []).append(res)
    merged = {k: merge(v) for k, v in by.items()}
    import sim_tables as TB
    TB.write_all(out, merged, models, targets, cols, checks, args)
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime
    with open(os.path.join(out, "logs", "sim_cost.txt"), "w") as f:
        f.write(f"jobs={len(jobs)} reps_per_block={args.reps} blocks={args.blocks} workers={args.workers} "
                f"wall_s={time.time() - t_start:.1f} cpu_s={cpu:.1f} end={G.now_utc()}\n")
    print(f"wall {time.time() - t_start:.1f}s cpu {cpu:.1f}s", flush=True)


if __name__ == "__main__":
    main()
