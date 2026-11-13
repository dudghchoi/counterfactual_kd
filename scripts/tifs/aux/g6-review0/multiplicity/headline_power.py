#!/usr/bin/env python3
"""A14 item 2 — power of the registered headline when every k = 0 (joint simulation).

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

The registered headline (tifs_plan.md 0.3 item 6; analyze.summary): intersection-union over the 19
qualified logit-KD cells at delta 0.05, met iff max p_up@0.05 < 0.025. This script simulates it with
the registered A11.10 S1c machinery, imported unchanged from scripts/tifs/aux/g4-s1c-s9-s1extra/
(common.py, s1c.py: pooled_model, target_model, draw_battery, pool_of, chol, ExactFastT), with every
true k = 0:

  BERT AG News, BERT SST-2, GPT-2 SST-2, Qwen3 SST-2 logit  seed-pooled batteries, S1c model
      k[a, s] = T_a (sqrt(w) z_s + sqrt(1 - w) z_as) + e[a, s],  e ~ N(0, Sigma)
      analyst: frozen analyze.pool_seed_variance on the simulated k, SE^2 = s2_pool / r + v_a,
      frozen analysis.tifs.t_contrast with the Satterthwaite df; BERT SST-2 SOS (embedding-only pair)
      simulated but outside the pool; BERT SST-2 EP (structural null) enters with its fixed seed-42
      Tango row (p_up@0.05 = 2.1e-12)
  Qwen3 AG News logit  transported (one seed, as registered at every r): k'_a = T'_a (sqrt(w) z' +
      sqrt(1 - w) z'_a) + e'_a, SE^2 = SE_Tango,a^2 + s2_pool where s2_pool and its df come from the SAME
      simulated Qwen3 SST-2 draw that decides the Qwen3 SST-2 cells (the borrowed-pool structure; the
      registered S1c used an independent source draw per battery)
  The 'feature' batteries are not part of the headline and are not simulated.
Configs (as S1c): H = measured heterogeneous training SD T_a = sqrt(max(0, s2_a,obs - psi_a)) with
the fitted shared fraction w; E = homogeneous T (pool mean of T_H^2), its own fitted w. Target T'_a:
H sqrt(F_a E[s2_pool]) (F_a of the same attack in Qwen3 SST-2), E sqrt(E[s2_pool]); c = 1.
Seeds r: 3 with the measured input covariance Sigma ('measured', = S1c), and r = 3, 5, 6 with an
exchangeable Sigma built from the measured one: Sigma_r = kron(X, J_r) + kron(W - X, I_r), W[a, b] =
the mean same-seed covariance of attacks a and b, X[a, b] = the mean cross-seed covariance. W, X and the input-only between-seed variance
psi_a = W_aa - X_aa do not depend on r; the input variance of the seed mean v_a = (W_aa + (r - 1) X_aa) / r
does (it is recomputed at each r). At r = 3 the exchangeable Sigma reproduces the measured v_a and
psi_a. T, w and the target model are kept at their r = 3 fit.

Outputs:
  headline_power.csv            per config x Sigma mode x r x delta: P(headline met | all k = 0) with
                                binomial MC SE, the product of the five battery NO TRANSFER rates (what
                                treating the batteries as independent gives), and each battery's rate
  headline_power_cells.csv      per config x Sigma mode x r x delta x cell: P(p_up@delta < 0.025)
  headline_power_validation.csv the r = 3 measured runs against aux/g4 s1c_battery.csv (same model;
                                two-proportion z), and r = 3 exchangeable against r = 3 measured
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

import g6m_common as G  # noqa: E402

import s1c as S  # noqa: E402  (aux/g4 S1c code; imported, never run as main, never modified)

K, A = G.K, G.A
TOLS, ALPHA = G.TOLS, G.ALPHA
POOLED = (("bert", "agnews"), ("bert", "sst2"), ("gpt2", "sst2"), ("qwen3", "sst2"))
TARGET = ("qwen3", "agnews", "logit")
NAMES = S.NAMES


def exchangeable(b, r):
    """Copy of an S1c pooled model with r seeds and the exchangeable input covariance."""
    m, r0, S0 = b["m_sim"], b["r"], b["Sigma"]
    W, X = np.zeros((m, m)), np.zeros((m, m))
    for a in range(m):
        for c in range(m):
            blk = S0[a * r0:(a + 1) * r0, c * r0:(c + 1) * r0]
            W[a, c] = np.trace(blk) / r0
            X[a, c] = (blk.sum() - np.trace(blk)) / (r0 * (r0 - 1))
    Sig = np.kron(X, np.ones((r, r))) + np.kron(W - X, np.eye(r))
    nb = {k: v for k, v in b.items()}
    nb.update(Sigma=Sig, L=S.chol(Sig), r=r, seeds=list(range(42, 42 + r)),
              v_arr=np.array([(W[a, a] + (r - 1) * X[a, a]) / r for a in range(m)]),
              psi_exch=np.array([W[a, a] - X[a, a] for a in range(m)]))
    return nb


def run_job(job):
    t0 = time.time()
    rng = np.random.default_rng(job["seed"])
    reps = job["reps"]
    bats = job["bats"]          # list of (key, model, T, w)
    tg = job["tgt"]
    draws = [S.draw_battery(rng, mdl, T, w, reps) for (_, mdl, T, w) in bats]
    mt = len(tg["sim_attacks"])
    z_t = rng.standard_normal((reps, 1))
    z_ta = rng.standard_normal((reps, mt))
    e_t = rng.standard_normal((reps, mt)) @ tg["L"].T
    k_t = job["T_tgt"][None, :] * (math.sqrt(job["w_tgt"]) * z_t + math.sqrt(1 - job["w_tgt"]) * z_ta) + e_t
    names = [f"{'/'.join(key)}/logit/{a}" for (key, mdl, _, _) in bats for a in mdl["sim_attacks"]]
    names += [f"{'/'.join(key)}/logit/{a}" for (key, mdl, _, _) in bats for a in mdl["structural"]]
    names += [f"{'/'.join(TARGET[:2])}/logit/{a}" for a in tg["sim_attacks"]]
    nb = len(bats) + 1
    acc = {"head": np.zeros(len(TOLS)), "nt": np.zeros((nb, len(TOLS))), "tr": np.zeros((nb, len(TOLS))),
           "below": np.zeros((len(names), len(TOLS))), "head_and_no_tr": np.zeros(len(TOLS))}
    for i in range(reps):
        allrows = []
        q3pool = None
        for bi, ((key, mdl, _, _), kd) in enumerate(zip(bats, draws)):
            pool = S.pool_of(kd[i], mdl)
            r = mdl["r"]
            res = [K.t_contrast(float(kd[i, j].mean()), math.sqrt(pool["s2"] / r + mdl["v_arr"][j]), pool["df"], TOLS)
                   for j in range(mdl["m_sim"])]
            rows = [dict(x, attack=a) for x, a in zip(res, mdl["sim_attacks"])]
            rows += [dict(sr, attack=a) for a, sr in mdl["struct_rows"].items()]
            dec = A.decide(rows)
            for ti, t in enumerate(TOLS):
                acc["nt"][bi, ti] += dec[f"decision@{t:g}"] == "NO TRANSFER"
                acc["tr"][bi, ti] += dec[f"decision@{t:g}"] == "TRANSFERS"
            allrows.append(rows)
            if key == ("qwen3", "sst2"):
                q3pool = pool
        res_t = [K.t_contrast(float(k_t[i, j]), math.sqrt(tg["seT"][j] ** 2 + q3pool["s2"]), q3pool["df"], TOLS)
                 for j in range(mt)]
        rows_t = [dict(x, attack=a) for x, a in zip(res_t, tg["sim_attacks"])]
        dec_t = A.decide(rows_t)
        for ti, t in enumerate(TOLS):
            acc["nt"][nb - 1, ti] += dec_t[f"decision@{t:g}"] == "NO TRANSFER"
            acc["tr"][nb - 1, ti] += dec_t[f"decision@{t:g}"] == "TRANSFERS"
        sim_rows = [x for rows, (_, mdl, _, _) in zip(allrows, bats) for x in rows[:mdl["m_sim"]]]
        st_rows = [x for rows, (_, mdl, _, _) in zip(allrows, bats) for x in rows[mdl["m_sim"]:]]
        head = sim_rows + st_rows + rows_t
        assert len(head) == 19
        for ti, t in enumerate(TOLS):
            ps = [x[f"p_up@{t:g}"] for x in head]
            met = max(ps) < ALPHA                      # analyze.summary's headline rule
            acc["head"][ti] += met
            acc["below"][:, ti] += np.array([p < ALPHA for p in ps], dtype=float)
    return {"key": job["key"], "block": job["block"], "reps": reps, "acc": acc, "names": names,
            "battery_names": [f"{NAMES[key]} logit" for (key, _, _, _) in bats] + ["Qwen3 AG News logit"],
            "seconds": time.time() - t0}


def merge(blocks):
    blocks = sorted(blocks, key=lambda b: b["block"])
    out = dict(blocks[0])
    out["acc"] = {k: sum(b["acc"][k] for b in blocks) for k in blocks[0]["acc"]}
    out["reps"] = sum(b["reps"] for b in blocks)
    out["seconds"] = sum(b["seconds"] for b in blocks)
    out["block_heads"] = [b["acc"]["head"] / b["reps"] for b in blocks]
    return out


def main():
    ap = G.add_out_arg(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--reps", type=int, default=10000, help="reps per block")
    ap.add_argument("--blocks", type=int, default=8, help="independent seed blocks per row")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.blocks = 300, 1
    t_start = time.time()
    out = G.out_dir(args.out)
    st = G.state()
    models = {key: S.pooled_model(st, *key) for key in POOLED}
    tg = S.target_model(st, *TARGET, models)
    assert tg["src_key"] == ("qwen3", "sst2") and tg["source"] == ("qwen3", "sst2", "logit"), tg["source"]
    src = models[("qwen3", "sst2")]
    jobs = []
    seed = args.seed
    modes = [("measured", 3)] + [("exchangeable", r) for r in (3, 5, 6)]
    for config in ("H", "E"):
        for mode, r in modes:
            seed += 1
            bats = []
            for key in POOLED:
                b = models[key]
                mdl = b if mode == "measured" else exchangeable(b, r)
                T, w = (b["T_H"], b["w_fit"]) if config == "H" else (b["T_E"], b["w_fit_E"])
                bats.append((key, mdl, T, w))
            w_tgt = src["w_fit"] if config == "H" else src["w_fit_E"]
            T_tgt = (np.sqrt(tg["F"] * src["E_s2"]) if config == "H"
                     else math.sqrt(src["E_s2"]) * np.ones(len(tg["sim_attacks"])))
            for blk in range(args.blocks):
                jobs.append({"key": (config, mode, r), "block": blk, "seed": seed + 1_000_000 * blk,
                             "reps": args.reps, "bats": bats, "tgt": tg, "T_tgt": T_tgt, "w_tgt": w_tgt})
    print(f"{len(jobs)} jobs x {args.reps} reps; workers {args.workers}", flush=True)
    from multiprocessing import Pool
    raw = []
    with Pool(args.workers, initializer=S._init_worker, initargs=(True,)) as pool:
        for res in pool.imap_unordered(run_job, jobs):
            raw.append(res)
            print(f"  done {res['key']} b{res['block'] + 1} {res['seconds']:.0f}s ({len(raw)}/{len(jobs)}, "
                  f"{time.time() - t_start:.0f}s)", flush=True)
    by = {}
    for res in raw:
        by.setdefault(res["key"], []).append(res)
    merged = {k: merge(v) for k, v in by.items()}
    rows, cell_rows = [], []
    for config in ("H", "E"):
        for mode, r in modes:
            m = merged[(config, mode, r)]
            reps, acc = m["reps"], m["acc"]
            for ti, t in enumerate(TOLS):
                p = acc["head"][ti] / reps
                nts = acc["nt"][:, ti] / reps
                row = {"config": config, "sigma_mode": mode, "r_seeds_pooled_batteries": r,
                       "r_qwen3_agnews": 1, "delta": t, "registered": t == 0.05, "reps": reps,
                       "p_headline_met": p, "p_headline_met_mcse": K.mc_se(p, reps),
                       "p_headline_met_blocks": ";".join(f"{x[ti]:.4f}" for x in m["block_heads"]),
                       "product_of_battery_no_transfer": float(np.prod(nts)),
                       "joint_minus_product": p - float(np.prod(nts))}
                for name, v, trv in zip(m["battery_names"], nts, acc["tr"][:, ti] / reps):
                    row[f"no_transfer[{name}]"] = float(v)
                    row[f"no_transfer_mcse[{name}]"] = K.mc_se(float(v), reps)
                    row[f"false_transfers[{name}]"] = float(trv)
                rows.append(row)
                for name, b in zip(m["names"], acc["below"][:, ti] / reps):
                    cell_rows.append({"config": config, "sigma_mode": mode, "r_seeds_pooled_batteries": r,
                                      "delta": t, "cell": name, "p_below": float(b),
                                      "p_below_mcse": K.mc_se(float(b), reps), "reps": reps})
    # validation against aux/g4 s1c_battery.csv (independent simulation of the same model)
    s1c_b = G.read_csv(os.path.join(G.AUX, "g4-s1c-s9-s1extra", "s1c_battery.csv"))
    val = []
    for config in ("H", "E"):
        m = merged[(config, "measured", 3)]
        mx = merged[(config, "exchangeable", 3)]
        for bi, name in enumerate(m["battery_names"]):
            rule = "transported" if name.startswith("Qwen3 AG News") else "pooled"
            for ti, t in enumerate(TOLS):
                ref = [x for x in s1c_b if x["rule"] == rule and x["battery"] == name and x["config"] == config
                       and abs(float(x["delta"]) - t) < 1e-9
                       and (x["c_sd"] == "" if rule == "pooled" else abs(float(x["c_sd"] or 0) - 1.0) < 1e-9)]
                assert len(ref) == 1, (name, config, t, len(ref))
                ref = ref[0]
                for metric, mine, mine_x in (("no_transfer_all0", m["acc"]["nt"][bi, ti] / m["reps"],
                                              mx["acc"]["nt"][bi, ti] / mx["reps"]),
                                             ("false_transfers_all0", m["acc"]["tr"][bi, ti] / m["reps"],
                                              mx["acc"]["tr"][bi, ti] / mx["reps"])):
                    rv, rn = float(ref[metric]), int(ref["reps"])
                    val.append({"config": config, "battery": name, "delta": t, "metric": metric,
                                "this_r3_measured": mine, "this_reps": m["reps"], "s1c_battery_csv": rv, "s1c_reps": rn,
                                "z_vs_s1c": K.z_two_prop(mine, m["reps"], rv, rn),
                                "this_r3_exchangeable": mine_x,
                                "z_exchangeable_vs_measured": K.z_two_prop(mine_x, mx["reps"], mine, m["reps"]),
                                "ok_abs_z_lt_4": abs(K.z_two_prop(mine, m["reps"], rv, rn)) < 4})
    G.write_csv(rows, os.path.join(out, "headline_power.csv"),
                "A14 item 2, P(registered headline met | every k = 0): the headline = max p_up@delta over the 19 "
                "qualified logit cells < 0.025 (analyze.summary rule; registered at delta 0.05 only, 0.02 and 0.10 "
                "shown), simulated jointly with the aux/g4 S1c model (configs H heterogeneous / E homogeneous training "
                "SD, fitted w) and the frozen pool_seed_variance, t_contrast and decide; the four seed-pooled logit "
                "batteries at r seeds (measured input covariance at r = 3, exchangeable covariance at r = 3, 5, 6), "
                "Qwen3 AG News transported (one seed) with the s2_pool and df of the same simulated Qwen3 SST-2 draw, "
                "BERT SST-2 EP as its fixed seed-42 Tango row; "
                f"{args.blocks} seed blocks x {args.reps:,} reps = {args.blocks * args.reps:,} reps per row, binomial MC SE "
                "over all of them (p_headline_met_blocks: the per-block rates); the product of the five battery NO "
                "TRANSFER rates is what independence would give.")
    G.write_csv(cell_rows, os.path.join(out, "headline_power_cells.csv"),
                "A14 item 2, per cell of the 19-cell headline: P(p_up@delta < 0.025 | every k = 0) in the joint "
                "simulation of headline_power.csv (config x Sigma mode x r x delta), "
                f"{args.blocks} seed blocks x {args.reps:,} reps per row, binomial MC SE.")
    G.write_csv(val, os.path.join(out, "headline_power_validation.csv"),
                "A14 item 2 validation: battery P(NO TRANSFER | all k = 0) and P(false TRANSFERS | all k = 0) of this "
                "joint simulation at r = 3 with the measured covariance against the registered aux/g4 s1c_battery.csv "
                "(pooled rule; Qwen3 AG News transported c = 1; independent draws of the same model; two-proportion z), "
                "and the r = 3 exchangeable-covariance run against the r = 3 measured run.")
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime
    with open(os.path.join(out, "headline_power_cost.txt"), "w") as f:
        f.write(f"jobs={len(jobs)} reps_per_block={args.reps} blocks={args.blocks} workers={args.workers} "
                f"wall_s={time.time() - t_start:.1f} cpu_s={cpu:.1f}\n")
    print(f"wall {time.time() - t_start:.1f}s cpu {cpu:.1f}s", flush=True)


if __name__ == "__main__":
    main()
