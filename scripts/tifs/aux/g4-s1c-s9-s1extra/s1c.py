#!/usr/bin/env python3
"""A11 item 10 — S1c: calibration of the actual primary rule on the real battery shapes.

Post hoc / registered auxiliary analysis (docs/tifs_addendum.md A11.10). It
does not change any primary verdict, battery decision or the paper-wide Holm.

Model (S1b's normal approximation, scripts/tifs/s1b_seed_noise.py, fitted to
the measured batteries). For a seed-pooled battery with simulated cells a
(the frozen pool's attacks plus any other seed-pooled cell that borrows the
pool, i.e. SOS SST-2) and student seeds s = 1..r (r = 3):

    k[a, s] = mu_a + T_a (sqrt(w) z_s + sqrt(1 - w) z_as) + e[a, s]
    z_s, z_as ~ N(0, 1) iid                         training noise; z_s is the
                                                    seed factor shared by every
                                                    attack of a seed (S1b: w = 0.5)
    e ~ N(0, Sigma)                                 input noise, Sigma = the measured
                                                    covariance (ddof 1) / n of the
                                                    stacked per-input d = P - C vectors
                                                    of every (attack, seed): real n,
                                                    discordance, cross-seed and
                                                    cross-attack correlation

  T_a^2 (config H, heterogeneous; the real F_a range) = max(0, s2_a,obs - psi_a),
        psi_a = (tr Sigma_aa - 1'Sigma_aa 1 / r) / (r - 1), the between-seed sample
        variance that input noise alone produces (so E[s2_a] = s2_a,obs unless clipped).
  T_a^2 (config E, homogeneous as in S1b) = mean over pooled attacks of T_a^2 (H)
        (the non-pooled SOS SST-2 cell keeps its own T).
  w (shared fraction) is fitted so that the model's E[MS_seed] / E[MS_res] equals the
        measured ms_seed / ms_res of the frozen pool (clipped to [0, 1]); config H05
        repeats H with S1b's w = 0.5.

Analyst (frozen code): s2_pool and Satterthwaite df = analyze.pool_seed_variance on the
simulated {attack: {seed: k}}; SE^2 = s2_pool / r + v_a with v_a = 1'Sigma_aa 1 / r^2
(= the measured var_i(dbar)/n, treated as known as in S1b); verdicts and one-sided p from
analysis.tifs.t_contrast; battery decision = analyze.decide (Holm over p_pos via
analysis.tifs.holm, intersection-union over p_up) over the real battery's qualified cells,
structural nulls entering with their fixed seed-42 Tango rows (paired_contrast).

Transported batteries (one seed): k'_a = mu_a + T'_a (sqrt(w) z' + sqrt(1 - w) z'_a) + e'_a,
e' ~ N(0, Sigma') from the real seed-42 vectors; SE^2 = SE_Tango,a^2 + s2_pool with the
cell's real Tango SE (se_from_interval). Two readings of the borrowed variance:
  transported         s2_pool and its Satterthwaite df from an independently simulated
                      source battery (the frozen source_for order), i.e. the full rule;
  transported_realdf  S1b's transported rule with S1b's assumed df_tau = 12 replaced by the
                      measured pool df: s2_pool = E[s2_pool] chi2_df / df, df fixed at the
                      measured value (BERT SST-2 4.67, BERT AG News 9.83, Qwen3 SST-2 5.89).
Misspecification: T'_a = c * sqrt(E[s2_pool]) (config E) or c * sqrt(F_a E[s2_pool])
(config H, F_a = the measured F of the same attack in the family's logit battery, the
SST-2 pool for Qwen3 AG News), c = SD ratio in {0.5, 0.707, 1, 1.225, 1.414, 1.5, 2}:
the SD reading c_sd in {0.5, 1, 1.5, 2} (as addendum.critical_ratio) and the variance
reading c_var = c_sd^2 in {0.5, 1, 1.5, 2} are both included.

Metrics (mu enters only as a shift, so the noise draw is shared):
  coverage            P(lo <= mu_a <= hi)                                  per cell
  false_below         P(hi < delta | mu_a = delta)  (= P(hi < mu_a), any delta)  per cell
  false_above         P(lo > delta | mu_a = delta)                          per cell
  p_equivalent@d      P(EQUIVALENT@d | mu = 0)                              per cell
  false_transfers_all0@d   P(decide -> TRANSFERS | all mu = 0)             battery (registered)
  no_transfer_all0@d       P(decide -> NO TRANSFER | all mu = 0)           battery
  false_transfers_alld@d   P(Holm p_pos < 0.025 | all simulated mu = d)    battery
  false_no_transfer_one_at_d@d  max_j P(NO TRANSFER | mu_j = d, others 0)  battery

Reps: pooled 20,000; transported config E and transported_realdf 10,000; transported
config H 40,000 = 4 blocks of 10,000 (block 1 uses the job's original seed, blocks 2-4
seed + 1e6 * b), pooled by summing the counts (per-block rates in s1c_blocks.csv).

Speed: inside the Monte Carlo loops ``scipy.stats.t`` is replaced by
common.ExactFastT, which calls the same scipy.special functions that scipy's
t.cdf / t.sf / t.ppf call (bit-identical, checked in checks.py); the frozen
t_contrast, pool_seed_variance, decide and holm are called unchanged. --stock-scipy
switches this off.

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" python s1c.py [--out DIR] [--workers 6] [--quick]
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

A = K.A
TOLS = K.TOLS
TOLS0 = (0.0,) + TOLS
ALPHA = K.ALPHA
POOLED = (("bert", "sst2"), ("bert", "agnews"), ("qwen3", "sst2"), ("gpt2", "sst2"))
TARGETS = (("bert", "sst2", "feature"), ("bert", "agnews", "feature"), ("qwen3", "agnews", "logit"))
C_GRID = (0.5, math.sqrt(0.5), 1.0, math.sqrt(1.5), math.sqrt(2.0), 1.5, 2.0)
NAMES = {("bert", "sst2"): "BERT SST-2", ("bert", "agnews"): "BERT AG News",
         ("qwen3", "sst2"): "Qwen3 SST-2", ("gpt2", "sst2"): "GPT-2 SST-2",
         ("qwen3", "agnews"): "Qwen3 AG News"}


# ── model fitting ────────────────────────────────────────────────────


def moments(T, Sigma, pool_idx, psi, r, w):
    """E[MS_seed], E[MS_res], E[s2_pool] of the model over the pooled attacks."""
    m = len(pool_idx)
    Tp = T[pool_idx]
    Sbar = np.zeros((r, r))
    for a in pool_idx:
        for b in pool_idx:
            Sbar += Sigma[a * r:(a + 1) * r, b * r:(b + 1) * r]
    Sbar /= m * m
    ss_in = np.trace(Sbar) - Sbar.sum() / r
    ss_tr = (r - 1) * (w * Tp.mean() ** 2 + (1 - w) * (Tp ** 2).sum() / m ** 2)
    e_seed = m * (ss_tr + ss_in) / (r - 1)
    e_s2 = float(np.mean(Tp ** 2 + psi[pool_idx]))
    e_res = (m * e_s2 - e_seed) / (m - 1)
    return e_seed, e_res, e_s2


def fit_w(T, Sigma, pool_idx, psi, r, target_ratio):
    def ratio(w):
        s, res, _ = moments(T, Sigma, pool_idx, psi, r, w)
        return s / res if res > 0 else math.inf
    if ratio(0.0) >= target_ratio:
        return 0.0
    if ratio(1.0) <= target_ratio:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ratio(mid) < target_ratio else (lo, mid)
    return (lo + hi) / 2


def chol(S):
    S = np.asarray(S, dtype=float)
    jitter = 1e-14 * max(float(np.max(np.diag(S))), 1e-12)
    for _ in range(8):
        try:
            return np.linalg.cholesky(S + jitter * np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            jitter *= 10
    # eigen fallback for a singular plug-in (e.g. identical vectors)
    vals, vecs = np.linalg.eigh(S)
    return vecs @ np.diag(np.sqrt(np.clip(vals, 0, None)))


def pooled_model(st, fam, ds):
    b = K.battery_plugin(st, fam, ds)
    sim = b["sim_attacks"]
    pool_idx = [sim.index(a) for a in b["pool_attacks"]]
    psi = np.array([b["psi"][a] for a in sim])
    s2 = np.array([b["s2_obs"][a] for a in sim])
    T_H = np.sqrt(np.maximum(s2 - psi, 0.0))
    T_E = T_H.copy()
    T_E[pool_idx] = math.sqrt(float(np.mean(T_H[pool_idx] ** 2)))
    obs_ratio = b["ms_seed"] / b["ms_res"]
    w = fit_w(T_H, b["Sigma"], pool_idx, psi, b["r"], obs_ratio)
    w_E = fit_w(T_E, b["Sigma"], pool_idx, psi, b["r"], obs_ratio)
    e_seed, e_res, e_s2 = moments(T_H, b["Sigma"], pool_idx, psi, b["r"], w)
    eE_seed, eE_res, eE_s2 = moments(T_E, b["Sigma"], pool_idx, psi, b["r"], w_E)
    assert abs(eE_s2 - e_s2) < 1e-15, (eE_s2, e_s2)
    b.update(pool_idx=pool_idx, psi_arr=psi, s2_arr=s2, T_H=T_H, T_E=T_E, w_fit=w, w_fit_E=w_E,
             obs_ratio=obs_ratio, model_ratio=e_seed / e_res, model_ratio_E=eE_seed / eE_res, E_s2=e_s2,
             v_arr=np.array([b["v"][a] for a in sim]), L=chol(b["Sigma"]))
    return b


def target_model(st, fam, ds, kd, models):
    t = K.target_plugin(st, fam, ds, kd)
    src_key = t["source"][:2]
    assert t["source"][2] == "logit", t["source"]
    src = models[src_key]
    # F of the same attack in the family's logit battery on this dataset, else the source
    fb = (fam, ds) if (fam, ds) in models else src_key
    ref = models[fb]
    F = [ref["s2_obs"][a] / ref["pool_s2"] for a in t["sim_attacks"]]
    t.update(F=np.array(F), F_from="/".join(fb) + "/logit", L=chol(t["Sigma"]), src_key=src_key,
             seT=np.array([t["se_tango"][a] for a in t["sim_attacks"]]))
    assert src is not None
    return t


# ── simulation ───────────────────────────────────────────────────────


def draw_battery(rng, model, T, w, reps):
    m, r = model["m_sim"], model["r"]
    z_s = rng.standard_normal((reps, 1, r))
    z_as = rng.standard_normal((reps, m, r))
    e = (rng.standard_normal((reps, m * r)) @ model["L"].T).reshape(reps, m, r)
    return T[None, :, None] * (math.sqrt(w) * z_s + math.sqrt(1 - w) * z_as) + e


def pool_of(k_rep, model):
    seeds = model["seeds"]
    ks = {model["sim_attacks"][i]: {s: float(k_rep[i, j]) for j, s in enumerate(seeds)}
          for i in model["pool_idx"]}
    return A.pool_seed_variance(ks)


def evaluate(cells_res, struct_rows, attacks):
    """Per-rep summary from frozen t_contrast rows (mu = 0) and fixed structural rows."""
    out = {}
    rows = [dict(res, attack=a) for a, res in zip(attacks, cells_res)]
    rows += [dict(sr, attack=a) for a, sr in struct_rows.items()]
    dec = A.decide(rows)
    out["dec"] = [dec[f"decision@{t:g}"] for t in TOLS]
    p0 = [res["p_pos@0"] for res in cells_res]
    out["tr_alld"] = []
    out["nt_one"] = []
    for t in TOLS:
        ps = p0 + [sr[f"p_pos@{t:g}"] for sr in struct_rows.values()]
        out["tr_alld"].append(min(K.holm(ps)) < ALPHA)
        up_t = [res[f"p_up@{t:g}"] < ALPHA for res in cells_res]
        st_ok = all(sr[f"p_up@{t:g}"] < ALPHA for sr in struct_rows.values())
        nt = []
        for j, res in enumerate(cells_res):
            nt.append(res["p_up@0"] < ALPHA and st_ok and all(up_t[a] for a in range(len(cells_res)) if a != j))
        out["nt_one"].append(nt)
    out["cover"] = [res["lo"] <= 0.0 <= res["hi"] for res in cells_res]
    out["below"] = [res["p_up@0"] < ALPHA for res in cells_res]
    out["above"] = [res["p_pos@0"] < ALPHA for res in cells_res]
    out["equiv"] = [[res[f"verdict@{t:g}"] == "EQUIVALENT" for t in TOLS] for res in cells_res]
    return out


def _init_worker(fast_t=True):
    try:
        os.nice(10)
    except OSError:
        pass
    K.install_exact_fast_t(fast_t)


def run_job(job):
    t0 = time.time()
    rng = np.random.default_rng(job["seed"])
    reps = job["reps"]
    rule = job["rule"]
    src = job["src"]
    # RNG order of the 'pooled', 'transported' and 's1b_check' jobs is that of the first
    # run (source battery first, then the target draws), so those rows are reproducible.
    if rule != "transported_realdf":
        k_src = draw_battery(rng, src, job["T_src"], job["w"], reps)
    if rule in ("transported", "transported_realdf"):
        tg = job["tgt"]
        mt = len(tg["sim_attacks"])
        z_t = rng.standard_normal((reps, 1))
        z_ta = rng.standard_normal((reps, mt))
        e_t = rng.standard_normal((reps, mt)) @ tg["L"].T
        k_t = job["T_tgt"][None, :] * (math.sqrt(job["w"]) * z_t + math.sqrt(1 - job["w"]) * z_ta) + e_t
        attacks, struct_rows = tg["sim_attacks"], tg["struct_rows"]
    else:
        attacks, struct_rows = src["sim_attacks"], src["struct_rows"]
    if rule == "transported_realdf":
        df_real = float(job["real_df"])
        s2_draw = job["E_s2"] * rng.chisquare(df_real, reps) / df_real
    m = len(attacks)
    acc = {"cover": np.zeros(m), "below": np.zeros(m), "above": np.zeros(m),
           "equiv": np.zeros((m, len(TOLS))), "nt_one": np.zeros((len(TOLS), m)),
           "tr_alld": np.zeros(len(TOLS)),
           "dec": {t: {"TRANSFERS": 0, "NO TRANSFER": 0, "INCONCLUSIVE": 0} for t in TOLS}}
    dfs, s2s = np.empty(reps), np.empty(reps)
    r = src["r"]
    for i in range(reps):
        if rule == "transported_realdf":
            s2, df = float(s2_draw[i]), df_real
        else:
            pool = pool_of(k_src[i], src)
            s2, df = pool["s2"], pool["df"]
        dfs[i], s2s[i] = df, s2
        res = []
        for j, a in enumerate(attacks):
            if rule in ("pooled", "s1b_check"):
                est = float(k_src[i, j].mean())
                se = math.sqrt(s2 / r + src["v_arr"][j])
            else:
                est = float(k_t[i, j])
                se = math.sqrt(tg["seT"][j] ** 2 + s2)
            res.append(K.t_contrast(est, se, df, TOLS0))
        ev = evaluate(res, struct_rows, attacks)
        acc["cover"] += ev["cover"]
        acc["below"] += ev["below"]
        acc["above"] += ev["above"]
        acc["equiv"] += np.array(ev["equiv"], dtype=float)
        acc["nt_one"] += np.array(ev["nt_one"], dtype=float)
        acc["tr_alld"] += np.array(ev["tr_alld"], dtype=float)
        for t, d in zip(TOLS, ev["dec"]):
            acc["dec"][t][d] += 1
    return {"job": {k: v for k, v in job.items() if k not in ("src", "tgt", "T_src", "T_tgt")},
            "attacks": attacks, "acc": acc, "reps": reps, "dfs": dfs, "s2s": s2s,
            "T_tgt": (job["T_tgt"].tolist() if rule in ("transported", "transported_realdf") else None),
            "T_src": job["T_src"].tolist(), "seconds": time.time() - t0}


def finish(res):
    """Distribution summaries of the simulated pool df / s2 (on the pooled draws)."""
    dfs, s2s = res.pop("dfs"), res.pop("s2s")
    res.update(df_q=[float(np.quantile(dfs, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)],
               df_mean=float(dfs.mean()), s2_mean=float(s2s.mean()),
               frac_df_below_real=float(np.mean(dfs < res["job"]["real_df"])))
    return res


def merge_blocks(blocks):
    """Pool the replicate blocks of one job (same model, different seeds) by summing
    the event counts; the pooled rates are count / total reps."""
    blocks = sorted(blocks, key=lambda r: r["job"]["block"])
    if len(blocks) == 1:
        return blocks[0]
    out = dict(blocks[0])
    out["job"] = dict(blocks[0]["job"], seed=";".join(str(b["job"]["seed"]) for b in blocks))
    acc = {k: (np.array(v, dtype=float) if not isinstance(v, dict) else {t: dict(d) for t, d in v.items()})
           for k, v in blocks[0]["acc"].items()}
    for b in blocks[1:]:
        assert b["attacks"] == out["attacks"]
        for k, v in b["acc"].items():
            if isinstance(v, dict):
                for t, d in v.items():
                    for name, cnt in d.items():
                        acc[k][t][name] += cnt
            else:
                acc[k] = acc[k] + v
    out["acc"] = acc
    out["reps"] = sum(b["reps"] for b in blocks)
    out["dfs"] = np.concatenate([b["dfs"] for b in blocks])
    out["s2s"] = np.concatenate([b["s2s"] for b in blocks])
    out["seconds"] = sum(b["seconds"] for b in blocks)
    return out


def block_rows(results_by_key):
    """Per-block battery rates and cell extremes of the multi-block jobs (s1c_blocks.csv)."""
    rows = []
    for key, blocks in sorted(results_by_key.items(), key=lambda kv: str(kv[0])):
        if len(blocks) == 1:
            continue
        for b in sorted(blocks, key=lambda r: r["job"]["block"]):
            jb, acc, reps = b["job"], b["acc"], b["reps"]
            cov = acc["cover"] / reps
            below = acc["below"] / reps
            for ti, t in enumerate(TOLS):
                d = acc["dec"][t]
                rows.append({"rule": jb["rule"], "battery": jb["battery"], "config": jb["config"],
                             "c_sd": jb.get("c", ""), "block": jb["block"] + 1, "seed": jb["seed"], "reps": reps,
                             "delta": t, "false_transfers_all0": d["TRANSFERS"] / reps,
                             "false_transfers_all0_mcse": K.mc_se(d["TRANSFERS"] / reps, reps),
                             "no_transfer_all0": d["NO TRANSFER"] / reps,
                             "false_transfers_alld": acc["tr_alld"][ti] / reps,
                             "coverage_min": float(cov.min()), "coverage_min_attack": b["attacks"][int(np.argmin(cov))],
                             "false_below_max": float(below.max()),
                             "false_below_max_attack": b["attacks"][int(np.argmax(below))]})
    return rows


# ── S1b reproduction (validation of this implementation) ─────────────


def s1b_model(n, disc, m, r, rho):
    sig2 = disc / n
    blk = sig2 * (rho * np.ones((r, r)) + (1 - rho) * np.eye(r))
    Sigma = np.kron(np.eye(m), blk)
    sim = [f"a{i}" for i in range(m)]
    return {"fam": "s1b", "dataset": f"n{n}", "sim_attacks": sim, "pool_attacks": sim,
            "pool_idx": list(range(m)), "seeds": list(range(r)), "r": r, "m_sim": m,
            "struct_rows": {}, "Sigma": Sigma, "L": chol(Sigma),
            "v_arr": np.full(m, sig2 * (1 + (r - 1) * rho) / r)}


# ── tables ───────────────────────────────────────────────────────────


def tables(results, models, targets):
    cells, batt, jobs = [], [], []
    for res in results:
        jb, acc, reps = res["job"], res["acc"], res["reps"]
        base = {"rule": jb["rule"], "battery": jb["battery"], "source": jb.get("source", ""),
                "config": jb["config"], "w": jb["w"], "c_sd": jb.get("c", ""),
                "c_var": (jb["c"] ** 2 if jb.get("c") not in (None, "") else ""), "reps": reps}
        if jb["rule"] == "s1b_check":
            continue
        for j, a in enumerate(res["attacks"]):
            row = dict(base, attack=a)
            if jb["rule"] in ("transported", "transported_realdf"):
                row.update(T_true=res["T_tgt"][j])
            else:
                row.update(T_true=res["T_src"][j])
            for name in ("cover", "below", "above"):
                p = acc[name][j] / reps
                key = {"cover": "coverage", "below": "false_below_at_mu_eq_delta",
                       "above": "false_above_at_mu_eq_delta"}[name]
                row[key] = p
                row[key + "_mcse"] = K.mc_se(p, reps)
            for ti, t in enumerate(TOLS):
                p = acc["equiv"][j, ti] / reps
                row[f"p_equivalent_mu0@{t:g}"] = p
            cells.append(row)
        for ti, t in enumerate(TOLS):
            d = acc["dec"][t]
            nt_one = acc["nt_one"][ti] / reps
            jmax = int(np.argmax(nt_one))
            row = dict(base, delta=t,
                       false_transfers_all0=d["TRANSFERS"] / reps,
                       false_transfers_all0_mcse=K.mc_se(d["TRANSFERS"] / reps, reps),
                       no_transfer_all0=d["NO TRANSFER"] / reps,
                       no_transfer_all0_mcse=K.mc_se(d["NO TRANSFER"] / reps, reps),
                       inconclusive_all0=d["INCONCLUSIVE"] / reps,
                       false_transfers_alld=acc["tr_alld"][ti] / reps,
                       false_transfers_alld_mcse=K.mc_se(acc["tr_alld"][ti] / reps, reps),
                       false_no_transfer_one_at_d_max=float(nt_one[jmax]),
                       false_no_transfer_one_at_d_max_mcse=K.mc_se(float(nt_one[jmax]), reps),
                       false_no_transfer_one_at_d_argmax=res["attacks"][jmax],
                       n_cells_decided=len(res["attacks"]) + jb["n_struct"])
            batt.append(row)
        jobs.append(dict(base, real_df=jb["real_df"], real_s2_pool=jb["real_s2"],
                         model_E_s2_pool=jb["E_s2"], sim_mean_s2_pool=res["s2_mean"],
                         sim_df_mean=res["df_mean"], **{f"sim_df_q{int(q * 100):02d}": v for q, v in
                                                       zip((0.05, 0.25, 0.5, 0.75, 0.95), res["df_q"])},
                         frac_sim_df_below_real=res["frac_df_below_real"], seconds=res["seconds"],
                         n_blocks=len(str(jb["seed"]).split(";")), seeds=str(jb["seed"])))
    fit = []
    for key, b in models.items():
        for j, a in enumerate(b["sim_attacks"]):
            fit.append({"part": "pooled battery", "battery": NAMES[key], "attack": a,
                        "in_pool": a in b["pool_attacks"], "n": b["n"], "r": b["r"],
                        "k_obs_s": ",".join(f"{x:.6f}" for x in b["k_obs"][a]),
                        "discordance_mean": b["disc"][a], "rho_cross_seed": b["rho"][a],
                        "sqrt_v": b["v"][a] ** 0.5, "sqrt_psi": b["psi"][a] ** 0.5,
                        "s_obs": b["s2_obs"][a] ** 0.5,
                        "F_obs": b["s2_obs"][a] / b["pool_s2"],
                        "T_H": float(b["T_H"][j]), "T_E": float(b["T_E"][j]),
                        "w_fit_H": b["w_fit"], "w_fit_E": b["w_fit_E"], "ms_ratio_obs": b["obs_ratio"],
                        "ms_ratio_model_H": b["model_ratio"], "ms_ratio_model_E": b["model_ratio_E"],
                        "pool_s_obs": b["pool_s2"] ** 0.5, "pool_s_model": b["E_s2"] ** 0.5,
                        "pool_df_obs": b["pool_df"]})
        for a in b["structural"]:
            sr = b["struct_rows"][a]
            fit.append({"part": "pooled battery (structural, fixed Tango row)", "battery": NAMES[key],
                        "attack": a, "in_pool": False, "n": b["n"], "k_obs_s": f"{sr['k']:.6f}",
                        "tango_lo": sr["lo"], "tango_hi": sr["hi"]})
    for key, t in targets.items():
        for j, a in enumerate(t["sim_attacks"]):
            fit.append({"part": "transported battery", "battery": f"{NAMES[(key[0], key[1])]} {key[2]}",
                        "attack": a, "source": "/".join(t["source"]), "n": t["n"],
                        "k_obs_s": f"{t['k_obs'][a]:.6f}", "discordance_mean": t["disc"][a],
                        "se_tango": t["se_tango"][a],
                        "sd_input_emp": float(t["Sigma"][j, j] ** 0.5),
                        "F_obs": float(t["F"][j]), "F_from": t["F_from"]})
        for a in t["structural"]:
            sr = t["struct_rows"][a]
            fit.append({"part": "transported battery (structural, fixed Tango row)",
                        "battery": f"{NAMES[(key[0], key[1])]} {key[2]}", "attack": a,
                        "n": t["n"], "k_obs_s": f"{sr['k']:.6f}", "tango_lo": sr["lo"], "tango_hi": sr["hi"]})
    return cells, batt, jobs, fit


def summary(cells, batt, jobs):
    """One row per (rule, battery, config, c, delta): the cell extremes and the battery rates."""
    out = []
    keyf = lambda r: (r["rule"], r["battery"], r["config"], r["c_sd"])
    jobmap = {keyf(j): j for j in jobs}
    for b in batt:
        cs = [c for c in cells if keyf(c) == keyf(b)]
        j = jobmap[keyf(b)]
        cov = sorted(cs, key=lambda c: c["coverage"])
        fb = sorted(cs, key=lambda c: -c["false_below_at_mu_eq_delta"])
        fa = sorted(cs, key=lambda c: -c["false_above_at_mu_eq_delta"])
        out.append({"rule": b["rule"], "battery": b["battery"], "source": b["source"], "config": b["config"],
                    "w": b["w"], "c_sd": b["c_sd"], "c_var": b["c_var"], "reps": b["reps"], "delta": b["delta"],
                    "n_sim_cells": len(cs), "n_cells_decided": b["n_cells_decided"],
                    "real_df": j["real_df"], "sim_df_median": j["sim_df_q50"],
                    "coverage_min": cov[0]["coverage"], "coverage_min_attack": cov[0]["attack"],
                    "coverage_min_mcse": cov[0]["coverage_mcse"],
                    "coverage_max": cov[-1]["coverage"], "coverage_max_attack": cov[-1]["attack"],
                    "coverage_mean": float(np.mean([c["coverage"] for c in cs])),
                    "false_below_max": fb[0]["false_below_at_mu_eq_delta"], "false_below_max_attack": fb[0]["attack"],
                    "false_below_max_mcse": fb[0]["false_below_at_mu_eq_delta_mcse"],
                    "false_below_min": fb[-1]["false_below_at_mu_eq_delta"],
                    "false_above_max": fa[0]["false_above_at_mu_eq_delta"], "false_above_max_attack": fa[0]["attack"],
                    "false_above_max_mcse": fa[0]["false_above_at_mu_eq_delta_mcse"],
                    **{k: b[k] for k in ("false_transfers_all0", "false_transfers_all0_mcse", "no_transfer_all0",
                                         "no_transfer_all0_mcse", "false_transfers_alld",
                                         "false_transfers_alld_mcse", "false_no_transfer_one_at_d_max",
                                         "false_no_transfer_one_at_d_max_mcse",
                                         "false_no_transfer_one_at_d_argmax")}})
    return out


def s1b_check(quick):
    """(a) deterministic: frozen pool_seed_variance == S1b's comb/m and df on S1b's
    own draws; (b) Monte Carlo: this simulator in S1b mode vs results/tifs/s1/s1b.csv."""
    import s1b_seed_noise as S1B  # frozen script, imported (never run as main)
    rows = []
    # (a) regenerate S1b's k with the same rng stream and compare per rep
    n, disc, m, r, tau, rho = 912, 0.05, 6, 3, 0.02, 0.3
    reps = 300
    mu = np.zeros(m)
    sims = S1B.simulate(np.random.default_rng(7), reps, n, disc, m, r, tau, rho, mu)
    rng = np.random.default_rng(7)
    sig = math.sqrt(disc / n)
    u = rng.normal(0, tau / math.sqrt(2), (reps, m, r))
    c = rng.normal(0, tau / math.sqrt(2), (reps, 1, r))
    cov = sig ** 2 * (rho * np.ones((r, r)) + (1 - rho) * np.eye(r))
    e = rng.normal(0, 1, (reps, m, r)) @ np.linalg.cholesky(cov + 1e-18 * np.eye(r)).T
    k = mu[None, :, None] + u - c + e
    v = sig ** 2 * (1 + (r - 1) * rho) / r
    est, se, df = sims["pooled"]
    max_se, max_df, max_k = 0.0, 0.0, 0.0
    for i in range(reps):
        pool = A.pool_seed_variance({a: {s: float(k[i, a, s]) for s in range(r)} for a in range(m)})
        max_se = max(max_se, abs(math.sqrt(pool["s2"] / r + v) - se[i, 0]))
        max_df = max(max_df, abs(pool["df"] - df[i, 0]))
        max_k = max(max_k, float(np.max(np.abs(k[i].mean(axis=1) - est[i]))))
    rows.append({"check": "frozen pool_seed_variance vs S1b comb/m and Satterthwaite df on S1b's draws",
                 "setting": f"n={n} disc={disc} m={m} r={r} tau={tau} rho={rho} reps={reps}",
                 "max_abs_diff_se": max_se, "max_abs_diff_df": max_df, "max_abs_diff_kbar": max_k})
    return rows


def main():
    ap = K.add_out_arg(argparse.ArgumentParser())
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reps-pooled", type=int, default=20000)
    ap.add_argument("--reps-transported", type=int, default=10000)
    ap.add_argument("--blocks-transported-H", type=int, default=4,
                    help="replicate blocks of --reps-transported each for the config-H transported jobs")
    ap.add_argument("--reps-check", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--stock-scipy", action="store_true",
                    help="do not route scipy.stats.t to common.ExactFastT (same numbers, about 10x slower)")
    args = ap.parse_args()
    t_start = time.time()
    if args.quick:
        args.reps_pooled, args.reps_transported, args.reps_check = 300, 200, 300
    out = args.out if not args.quick else os.path.join(args.out, "_quick")
    os.makedirs(out, exist_ok=True)
    fast_t = not args.stock_scipy
    st = K.state()
    models = {key: pooled_model(st, *key) for key in POOLED}
    targets = {key: target_model(st, *key, models) for key in TARGETS}
    jobs = []
    seed = args.seed
    # Job seeds follow the job-creation order of the pre-promotion run (pooled, transported,
    # s1b_check, then transported_realdf; 20261025-20261123), so every 10,000- or 20,000-rep
    # block reproduces it. Blocks 2-4 of the config-H transported jobs use seed + 1e6 * b.
    for key, b in models.items():
        for config, T, w in (("H", b["T_H"], b["w_fit"]), ("E", b["T_E"], b["w_fit_E"]),
                             ("H05", b["T_H"], 0.5)):
            seed += 1
            jobs.append({"rule": "pooled", "battery": f"{NAMES[key]} logit", "config": config,
                         "w": w, "reps": args.reps_pooled, "seed": seed, "src": b, "T_src": T,
                         "real_df": b["pool_df"], "real_s2": b["pool_s2"], "E_s2": b["E_s2"],
                         "n_struct": len(b["structural"]),
                         "cost": args.reps_pooled * b["m_sim"]})
    for key, t in targets.items():
        b = models[t["src_key"]]
        for config, T_src, w_src in (("H", b["T_H"], b["w_fit"]), ("E", b["T_E"], b["w_fit_E"])):
            for c in C_GRID:
                seed += 1
                if config == "H":
                    T_t = c * np.sqrt(t["F"] * b["E_s2"])
                else:
                    T_t = c * math.sqrt(b["E_s2"]) * np.ones(len(t["sim_attacks"]))
                n_blocks = args.blocks_transported_H if config == "H" else 1
                for blk in range(n_blocks):
                    jobs.append({"rule": "transported", "battery": f"{NAMES[(key[0], key[1])]} {key[2]}",
                                 "source": f"{NAMES[t['src_key']]} logit", "config": config,
                                 "w": w_src, "c": c, "reps": args.reps_transported,
                                 "seed": seed + 1_000_000 * blk, "block": blk,
                                 "src": b, "tgt": t, "T_src": T_src, "T_tgt": T_t,
                                 "real_df": b["pool_df"], "real_s2": b["pool_s2"], "E_s2": b["E_s2"],
                                 "n_struct": len(t["structural"]),
                                 "cost": args.reps_transported * len(t["sim_attacks"])})
    # S1b-mode jobs (validation of the simulator against results/tifs/s1/s1b.csv)
    s1b_settings = [("BERT SST-2", 912, 0.05, 6, 3, 0.02, 0.3), ("BERT SST-2", 912, 0.05, 6, 3, 0.03, 0.6),
                    ("BERT AG News", 5700, 0.015, 6, 3, 0.03, 0.3)]
    for lab, n, disc, m, r, tau, rho in s1b_settings:
        seed += 1
        mdl = s1b_model(n, disc, m, r, rho)
        jobs.append({"rule": "s1b_check", "battery": lab, "config": f"tau={tau},rho={rho}", "w": 0.5,
                     "reps": args.reps_check, "seed": seed, "src": mdl, "T_src": np.full(m, tau),
                     "real_df": float("nan"), "real_s2": float("nan"), "E_s2": float("nan"),
                     "n_struct": 0, "cost": args.reps_check * m,
                     "s1b": {"battery": lab, "n": n, "disc": disc, "m": m, "r": r, "tau": tau, "rho": rho}})
    # transported rule with the measured df (S1b's df_tau = 12 replaced by the real pool df)
    for key, t in targets.items():
        b = models[t["src_key"]]
        for config, w_src in (("H", b["w_fit"]), ("E", b["w_fit_E"])):
            for c in C_GRID:
                seed += 1
                if config == "H":
                    T_t = c * np.sqrt(t["F"] * b["E_s2"])
                else:
                    T_t = c * math.sqrt(b["E_s2"]) * np.ones(len(t["sim_attacks"]))
                jobs.append({"rule": "transported_realdf", "battery": f"{NAMES[(key[0], key[1])]} {key[2]}",
                             "source": f"{NAMES[t['src_key']]} logit (chi2, measured df)", "config": config,
                             "w": w_src, "c": c, "reps": args.reps_transported, "seed": seed,
                             "src": b, "tgt": t, "T_src": np.zeros(1), "T_tgt": T_t,
                             "real_df": b["pool_df"], "real_s2": b["pool_s2"], "E_s2": b["E_s2"],
                             "n_struct": len(t["structural"]),
                             "cost": args.reps_transported * len(t["sim_attacks"]) // 4})
    for j in jobs:
        j.setdefault("block", 0)
    jobs.sort(key=lambda j: -j["cost"])
    print(f"{len(jobs)} jobs; workers {args.workers}; exact fast t {fast_t}", flush=True)
    t0 = time.time()
    from multiprocessing import Pool
    raw = []
    with Pool(args.workers, initializer=_init_worker, initargs=(fast_t,)) as pool:
        for res in pool.imap_unordered(run_job, jobs):
            raw.append(res)
            jb = res["job"]
            print(f"  done {jb['rule']:<18} {jb['battery']:<22} {jb['config']:<14} c={jb.get('c', '')!s:<6.6} "
                  f"b{jb['block'] + 1} {res['seconds']:.0f}s  ({len(raw)}/{len(jobs)}, {time.time() - t0:.0f}s)",
                  flush=True)
    by_key = {}
    for res in raw:
        jb = res["job"]
        by_key.setdefault((jb["rule"], jb["battery"], jb["config"], jb.get("c", 0)), []).append(res)
    blocks = block_rows(by_key)
    results = [finish(merge_blocks(v)) for v in by_key.values()]
    results.sort(key=lambda r: (r["job"]["rule"], r["job"]["battery"], r["job"]["config"], r["job"].get("c", 0)))
    cells, batt, jobrows, fit = tables(results, models, targets)
    # S1b comparison rows
    import csv
    s1b_ref = list(csv.DictReader(open(os.path.join(K.S1_DIR, "s1b.csv"))))
    chk = s1b_check(args.quick)
    for res in results:
        jb = res["job"]
        if jb["rule"] != "s1b_check":
            continue
        p = jb["s1b"]
        acc, reps = res["acc"], res["reps"]
        cov = float(acc["cover"].mean() / reps)
        below = float(acc["below"][0] / reps)
        for ti, t in enumerate(TOLS):
            if t not in (0.02, 0.05):
                continue
            ref = [x for x in s1b_ref if x["battery"] == p["battery"] and x["rule"] == "pooled"
                   and float(x["tau"]) == p["tau"] and float(x["rho"]) == p["rho"] and float(x["delta"]) == t]
            ref = ref[0] if ref else {}
            d = acc["dec"][t]
            mine = {"coverage": cov, "false_below_at_mu_eq_delta": below,
                    "p_no_transfer_all_zero": d["NO TRANSFER"] / reps,
                    "p_false_transfers_all_zero": d["TRANSFERS"] / reps}
            for kname, val in mine.items():
                rv = float(ref[kname]) if ref else float("nan")
                se_mine = K.mc_se(val, reps)    # conservative for coverage (cells share the pool)
                chk.append({"check": "this simulator in S1b mode vs results/tifs/s1/s1b.csv (pooled rule)",
                            "setting": f"{p['battery']} n={p['n']} disc={p['disc']} m={p['m']} r={p['r']} "
                                       f"tau={p['tau']} rho={p['rho']} delta={t}",
                            "metric": kname, "this_sim": val, "this_sim_mcse": se_mine, "this_reps": reps,
                            "s1b_csv": rv, "s1b_reps": 40000,
                            "z_diff": K.z_two_prop(val, reps, rv, 40000)})
    K.write_csv(fit, os.path.join(out, "s1c_fit.csv"),
                "S1c model fit: per battery and attack the measured plug-ins (n, mean discordance, mean cross-seed "
                "input correlation, sqrt(v)=input SE of the seed mean, sqrt(psi)=between-seed SD from input noise "
                "alone, s_obs, F_obs=s_obs^2/s_pool^2) and the training SDs T_H (heterogeneous, max(0,s_obs^2-psi)^.5) "
                "and T_E (homogeneous), fitted shared fraction w (E[MS_seed]/E[MS_res] matched to the frozen pool), "
                "model vs measured pool SD and df; transported cells: real Tango SE and F of the same attack.")
    K.write_csv(cells, os.path.join(out, "s1c_cells.csv"),
                "S1c per-cell operating characteristics of the frozen primary rule (pooled: seed-pooled SE^2="
                "s2_pool/r+v with Satterthwaite df; transported: SE^2=SE_Tango^2+s2_pool with the simulated source "
                "pool and its df; transported_realdf: s2_pool=E[s2_pool]chi2_df/df at the measured df) under the "
                "fitted normal model: coverage of mu, false 'below delta' at mu=delta (=P(hi<mu), same for every "
                "delta), false 'above delta' at mu=delta, P(EQUIVALENT@d | mu=0); MC SE = binomial; reps 20000 "
                "(pooled), 40000 (transported config H, 4 pooled seed blocks), 10000 (other transported).")
    K.write_csv(batt, os.path.join(out, "s1c_battery.csv"),
                "S1c battery operating characteristics (analyze.decide over the real battery's qualified cells, "
                "structural nulls as fixed Tango rows): P(false TRANSFERS | all mu=0) [registered], P(NO TRANSFER | "
                "all mu=0), P(false TRANSFERS | all simulated mu=delta), max_j P(NO TRANSFER | mu_j=delta, others 0); "
                "MC SE = binomial; reps as in s1c_cells.csv.")
    K.write_csv(summary(cells, batt, jobrows), os.path.join(out, "s1c_summary.csv"),
                "S1c compact summary: per rule, battery, config, misspecification c (SD ratio c_sd; variance ratio "
                "c_var=c_sd^2) and delta the cell extremes (min/max coverage, max false 'below delta' and max false "
                "'above delta' at mu=delta, with the attack) and the battery rates of s1c_battery.csv; MC SE binomial.")
    K.write_csv(jobrows, os.path.join(out, "s1c_jobs.csv"),
                "S1c simulation jobs: rule, battery, config (H heterogeneous T_a, E homogeneous, H05 = H with S1b's "
                "w=0.5), shared fraction w, misspecification c (SD ratio; c_var=c^2), reps, seed blocks and seeds, "
                "the simulated Satterthwaite df distribution vs the measured pool df (constant for transported_realdf), "
                "mean simulated s2_pool vs model and measured; seconds = sum over blocks.")
    K.write_csv(blocks, os.path.join(out, "s1c_blocks.csv"),
                "S1c replicate blocks of the multi-block jobs (transported, config H: 4 x 10000 reps; block 1 = the "
                "pre-promotion seed): per block and delta the battery rates and the cell extremes; the pooled rows "
                "of s1c_battery.csv / s1c_cells.csv sum the block counts.")
    K.write_csv(chk, os.path.join(out, "s1c_check_s1b.csv"),
                "S1c validation: (a) the frozen analyze.pool_seed_variance reproduces S1b's comb/m and Satterthwaite "
                "df exactly on S1b's own draws; (b) this simulator in S1b mode (homogeneous tau, w=0.5, "
                "equicorrelated input noise) vs the archived results/tifs/s1/s1b.csv pooled rows (z of the difference).")
    print(f"wrote {out} in {time.time() - t0:.0f}s", flush=True)
    K.log_cost(args.out, "s1c.py", t_start)


if __name__ == "__main__":
    main()
