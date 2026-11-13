"""Tables of g10 sim.py (imported by sim.py; every number from the merged Monte Carlo counts).

A15 post hoc auxiliary analysis (claim-rule calibration, docs/tifs_addendum.md amendment 10/27 20:58 KST
item 4); changes no primary verdict, battery decision or the paper-wide Holm.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import math  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402

import g10_common as G  # noqa: E402

S = G.S
TOLS, ALPHA = G.TOLS, G.ALPHA
ND = len(TOLS)
RULES = ("reg", "own", "claim")
READINGS = ("reg", "own", "claimA", "claimB")
READING_LABEL = {
    "reg": "registered (shared pool, Satterthwaite df); one-seed cells transported rule",
    "own": "own variance (A3.1, s2_a, df r-1); one-seed cells enter with p_registered",
    "claimA": "claim rule, p_max = max(p_reg, p_own); one-seed cells enter with p_registered, not counted",
    "claimB": "claim rule, p_max = max(p_reg, p_own); one-seed cells enter with p = 1",
}
RULE_LABEL = {"reg": "registered (shared pool, Satterthwaite df)",
              "own": "own variance (A3.1: s2_a / r + v_a, df r - 1)",
              "claim": "claim rule (reg and own agree, else INCONCLUSIVE; one-seed descriptive)"}
FCONF_LABEL = {"all0": "every simulated mu = 0", "alld": "every simulated mu = delta (global boundary)",
               "obs": "mu = frozen primary k (observed-k configuration)"}


def _rate(n, reps):
    return float(n) / reps


SETTING_LABEL = {
    ("measured", 3): "every seed-pooled battery at r = 3, measured Sigma (= S1c)",
    ("exchangeable", 3): "every seed-pooled battery at r = 3, exchangeable Sigma (bridge to r = 5)",
    ("exchangeable", 5): ("every seed-pooled battery at r = 5, exchangeable Sigma (hypothetical; A15-d adds seeds "
                          "to BERT AG News only)"),
    ("a15d", 5): ("A15-d data path: BERT AG News logit at r = 5 (exchangeable Sigma), the other seed-pooled "
                  "batteries at r = 3 (measured Sigma)"),
}


def key_label(key):
    config, mode, r = key
    return {"config": config, "sigma_mode": mode, "r": r, "sigma_setting": SETTING_LABEL[(mode, r)]}


def e_mcse(sum_x, sum_x2, reps):
    """MC SE of a per-replicate mean count from its sum and sum of squares: sqrt((E[c^2] - E[c]^2) / reps)."""
    m = sum_x / reps
    return math.sqrt(max(sum_x2 / reps - m * m, 0.0) / reps)


def cell_rows(merged, cols, models, targets):
    rows = []
    for key, m in sorted(merged.items()):
        reps = m["reps"]
        cc = m["cell"]
        jobT = T_of(key, models, targets, cols)
        for ci, c in enumerate(cols):
            if c["cls"] == "structural":
                continue
            r = c["row"]
            for q in RULES:
                undefined = c["cls"] == "one-seed" and q in ("own", "claim")
                for ti, t in enumerate(TOLS):
                    row = dict(key_label(key), cell=c["id"], display=c["display"], cls=c["cls"],
                               in_F19=c["in_F19"], a4_class=c["a4"], rule=q, delta=t, reps=reps,
                               k_obs=r["k"], frozen_verdict=r[f"verdict@{t:g}"],
                               positive_cell=r[f"verdict@{t:g}"] == "POSITIVE", T_true=jobT[ci])
                    if undefined:
                        row["note"] = ("own variance not defined (one seed)" if q == "own"
                                       else "one-seed, descriptive (claim rule makes no claim)")
                        rows.append(row)
                        continue

                    def put(name, n):
                        p = _rate(n, reps)
                        row[name] = p
                        row[name + "_mcse"] = G.mc_se(p, reps)

                    put("coverage", cc[("cover", q)][ci])
                    put("false_below_at_mu_eq_delta", cc[("below", q)][ci])
                    put("false_above_at_mu_eq_delta", cc[("above", q)][ci])
                    put("false_equivalent_at_mu_eq_delta", cc[("fequ_d", q)][ci, ti])
                    put("false_positive_at_mu_eq_delta", cc[("fpos_d", q)][ci, ti])
                    put("p_equivalent_mu0", cc[("equ0", q)][ci, ti])
                    put("p_positive_at_obs", cc[("pos_o", q)][ci, ti])
                    put("p_equivalent_at_obs", cc[("equ_o", q)][ci, ti])
                    row["p_inconclusive_at_obs"] = _rate(cc[("inc_o", q)][ci, ti], reps)
                    row["p_negative_at_obs"] = _rate(cc[("neg_o", q)][ci, ti], reps)
                    if q == "claim":
                        row["p_rule_dependent_mu0"] = _rate(cc[("rd0", "claim")][ci, ti], reps)
                        row["p_rule_dependent_at_mu_eq_delta"] = _rate(cc[("rdd", "claim")][ci, ti], reps)
                        row["p_rule_dependent_at_obs"] = _rate(cc[("rdo", "claim")][ci, ti], reps)
                    rows.append(row)
    return rows


def T_of(key, models, targets, cols):
    """True training SD of each family cell in the job's model (for the table)."""
    config, mode, r = key
    out = []
    for c in cols:
        kind, k2, j = c["src"]
        if kind == "struct":
            out.append(float("nan"))
        elif kind == "pooled":
            b = models[k2]
            out.append(float((b["T_E"] if config == "E" else b["T_H"])[j]))
        else:
            t = targets[k2]
            src = models[t["src_key"]]
            out.append(float(math.sqrt(src["E_s2"]) if config == "E" else math.sqrt(t["F"][j] * src["E_s2"])))
    return out


SUBSETS = ("counted", "seed_pooled", "one_seed", "descriptive", "structural")


def n_cells(cols, idx, reading, subset):
    cls = [cols[i]["cls"] for i in idx]
    if subset == "counted":
        return sum(x == "seed-pooled" for x in cls) if reading.startswith("claim") else sum(x != "structural" for x in cls)
    if subset == "seed_pooled":
        return sum(x == "seed-pooled" for x in cls)
    if subset == "one_seed":
        return sum(x == "one-seed" for x in cls)
    if subset == "descriptive":
        return sum(x == "one-seed" for x in cls) if reading.startswith("claim") else 0
    return sum(x == "structural" for x in cls)


def fam_index(cols, fname):
    return list(range(len(cols))) if fname == "F31" else [i for i, c in enumerate(cols) if c["in_F19"]]


def family_rows(merged, cols):
    rows = []
    for key, m in sorted(merged.items()):
        reps = m["reps"]
        fc, oc = m["family"], m["onecell"]
        for fname in ("F31", "F19"):
            idx = fam_index(cols, fname)
            for reading in READINGS:
                for t in TOLS:
                    for side in ("NT", "T"):
                        for conf in ("all0", "alld", "obs"):
                            row = dict(key_label(key), reading=reading, family=fname, family_size=len(idx),
                                       delta=t, side=("no-transfer (Holm over p_up)" if side == "NT"
                                                      else "transfer (Holm over p_pos)"),
                                       configuration=conf, reps=reps)
                            for sub in SUBSETS:
                                k = (fname, reading, t, side, conf, sub)
                                ncell = n_cells(cols, idx, reading, sub)
                                row[f"{sub}_n_cells"] = ncell
                                if k not in fc:
                                    continue
                                n_any_false, n_false, n_corr, n_any, ss_false, ss_corr = fc[k]
                                p = n_any_false / reps
                                row[f"{sub}_p_any_false"] = p
                                row[f"{sub}_p_any_false_mcse"] = G.mc_se(p, reps)
                                row[f"{sub}_e_false"] = n_false / reps
                                row[f"{sub}_e_false_mcse"] = e_mcse(n_false, ss_false, reps)
                                row[f"{sub}_e_correct"] = n_corr / reps
                                row[f"{sub}_e_correct_mcse"] = e_mcse(n_corr, ss_corr, reps)
                                row[f"{sub}_p_any_claim"] = n_any / reps
                            if side == "NT" and conf == "alld":
                                # one_at_d (cell j at delta, the others at 0): max over counted j
                                for sub in ("counted", "seed_pooled"):
                                    cand = [i for i in idx if cols[i]["cls"] != "structural" and
                                            (cols[i]["cls"] == "seed-pooled" or
                                             (sub == "counted" and not reading.startswith("claim")))]
                                    vals = [(oc[(fname, reading, t, i)] / reps, i) for i in cand]
                                    if vals:
                                        v, i = max(vals)
                                        row[f"one_at_d_{sub}_max"] = v
                                        row[f"one_at_d_{sub}_max_mcse"] = G.mc_se(v, reps)
                                        row[f"one_at_d_{sub}_argmax"] = cols[i]["id"]
                                if reading.startswith("claim"):
                                    cand = [i for i in idx if cols[i]["cls"] == "one-seed"]
                                    if cand:
                                        v, i = max((oc[(fname, reading, t, i)] / reps, i) for i in cand)
                                        row["one_at_d_descriptive_max"] = v
                                        row["one_at_d_descriptive_argmax"] = cols[i]["id"]
                            if side == "T" and conf == "alld":
                                # one_at_d_T (cell j at delta, every other simulated cell surely rejected):
                                # max over counted j of P(false transfer claim on j)
                                ocT = m["onecell_T"]
                                for sub in ("counted", "seed_pooled"):
                                    cand = [i for i in idx if cols[i]["cls"] != "structural" and
                                            (cols[i]["cls"] == "seed-pooled" or
                                             (sub == "counted" and not reading.startswith("claim")))]
                                    vals = [(ocT[(fname, reading, t, i)] / reps, i) for i in cand]
                                    if vals:
                                        v, i = max(vals)
                                        row[f"one_at_d_T_{sub}_max"] = v
                                        row[f"one_at_d_T_{sub}_max_mcse"] = G.mc_se(v, reps)
                                        row[f"one_at_d_T_{sub}_argmax"] = cols[i]["id"]
                                if reading.startswith("claim"):
                                    cand = [i for i in idx if cols[i]["cls"] == "one-seed"]
                                    if cand:
                                        v, i = max((ocT[(fname, reading, t, i)] / reps, i) for i in cand)
                                        row["one_at_d_T_descriptive_max"] = v
                                        row["one_at_d_T_descriptive_argmax"] = cols[i]["id"]
                            rows.append(row)
    return rows


def onecell_rows(merged, cols):
    rows = []
    for key, m in sorted(merged.items()):
        reps = m["reps"]
        for part, side, col in (("onecell", "no-transfer (cell j at delta, the others at 0)",
                                 "p_false_no_transfer_claim"),
                                ("onecell_T", "transfer (cell j at delta, every other simulated cell surely rejected)",
                                 "p_false_transfer_claim")):
            for (fname, reading, t, i), n in sorted(m[part].items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
                c = cols[i]
                counted = c["cls"] == "seed-pooled" or (c["cls"] == "one-seed" and not reading.startswith("claim"))
                p = n / reps
                rows.append(dict(key_label(key), side=side, reading=reading, family=fname, delta=t, cell=c["id"],
                                 cls=c["cls"], counted=counted, reps=reps, **{col: p, col + "_mcse": G.mc_se(p, reps)}))
    return rows


def battery_rows(merged, models, targets):
    rows = []
    for key, m in sorted(merged.items()):
        reps = m["reps"]
        for bn, a in m["battery"].items():
            kind, k2 = bn
            name = (f"{S.NAMES[k2]} logit" if kind == "pooled" else f"{S.NAMES[(k2[0], k2[1])]} {k2[2]}")
            holder = models[k2] if kind == "pooled" else targets[k2]
            for ti, t in enumerate(TOLS):
                d = a["dec"][t]
                nt = a["nt_one"][ti] / reps
                jmax = int(np.argmax(nt))
                row = dict(key_label(key), battery=name, rule=("pooled" if kind == "pooled" else "transported"),
                           delta=t, reps=reps,
                           false_transfers_all0=d["TRANSFERS"] / reps,
                           false_transfers_all0_mcse=G.mc_se(d["TRANSFERS"] / reps, reps),
                           no_transfer_all0=d["NO TRANSFER"] / reps,
                           no_transfer_all0_mcse=G.mc_se(d["NO TRANSFER"] / reps, reps),
                           inconclusive_all0=d["INCONCLUSIVE"] / reps,
                           false_transfers_alld=a["tr_alld"][ti] / reps,
                           false_transfers_alld_mcse=G.mc_se(a["tr_alld"][ti] / reps, reps),
                           false_no_transfer_one_at_d_max=float(nt[jmax]),
                           false_no_transfer_one_at_d_max_mcse=G.mc_se(float(nt[jmax]), reps),
                           false_no_transfer_one_at_d_argmax=holder["sim_attacks"][jmax])
                if kind == "pooled":
                    row["sim_pool_df_mean"] = m["pool"][k2][0] / reps
                    row["sim_pool_s2_mean"] = m["pool"][k2][1] / reps
                    row["real_pool_df"] = holder["pool_df"]
                rows.append(row)
    return rows


P_EXACT_MIN = 6.334e-05   # two-sided normal tail at |z| = 4


def compare(mine, n1, rv, n2):
    """Two-proportion z and Fisher's exact test of equal rates (two-sided; conditional on the total events the
    count x1 is hypergeometric, which is exact for any rate, rare or not). Pass: p_exact >= 6.33e-5 (= |z| < 4 in
    the normal approximation)."""
    from scipy.stats import fisher_exact
    x1, x2 = int(round(mine * n1)), int(round(rv * n2))
    z = G.z_two_prop(mine, n1, rv, n2)
    if x1 + x2 == 0 or x1 + x2 == n1 + n2:
        p = 1.0
    else:
        p = float(fisher_exact([[x1, n1 - x1], [x2, n2 - x2]]).pvalue)
    return {"s1c": rv, "s1c_reps": n2, "z_vs_s1c": z, "p_exact": p, "ok": p >= P_EXACT_MIN}


def validation_rows(merged, cols, bat_rows):
    """Registered rule at r = 3 (measured Sigma) vs aux/g4 s1c_cells.csv and s1c_battery.csv (two-proportion z);
    exchangeable r = 3 vs measured r = 3 (bridge, descriptive)."""
    s1c_c = G.read_csv(G.OFFICIAL["s1c_cells"])
    s1c_b = G.read_csv(G.OFFICIAL["s1c_battery"])
    rows = []
    for config in ("E", "H", "H05"):
        key = (config, "measured", 3)
        keyx = (config, "exchangeable", 3)
        if key not in merged:
            continue
        m = merged[key]
        mx = merged.get(keyx)
        reps = m["reps"]
        cc = m["cell"]
        for ci, c in enumerate(cols):
            if c["cls"] == "structural":
                continue
            kind, k2, j = c["src"]
            if kind == "pooled":
                rule, bname = "pooled", f"{S.NAMES[k2]} logit"
            else:
                rule, bname = "transported", f"{S.NAMES[(k2[0], k2[1])]} {k2[2]}"
            attack = c["row"]["attack"]
            ref = [x for x in s1c_c if x["rule"] == rule and x["battery"] == bname and x["config"] == config
                   and x["attack"] == attack and (rule == "pooled" or (x["c_sd"] and abs(float(x["c_sd"]) - 1) < 1e-9))]
            metrics = [("coverage", cc[("cover", "reg")][ci], mx["cell"][("cover", "reg")][ci] if mx else None, "coverage"),
                       ("false_below_at_mu_eq_delta", cc[("below", "reg")][ci],
                        mx["cell"][("below", "reg")][ci] if mx else None, "false_below_at_mu_eq_delta"),
                       ("false_above_at_mu_eq_delta", cc[("above", "reg")][ci],
                        mx["cell"][("above", "reg")][ci] if mx else None, "false_above_at_mu_eq_delta")]
            for ti, t in enumerate(TOLS):
                metrics.append((f"p_equivalent_mu0@{t:g}", cc[("equ0", "reg")][ci, ti],
                                mx["cell"][("equ0", "reg")][ci, ti] if mx else None, f"p_equivalent_mu0@{t:g}"))
            for name, n, nx, refcol in metrics:
                mine = n / reps
                row = {"level": "cell", "config": config, "battery": bname, "rule": rule, "item": attack,
                       "metric": name, "this_measured_r3": mine, "this_reps": reps}
                if ref:
                    rv, rn = float(ref[0][refcol]), int(ref[0]["reps"])
                    row.update(compare(mine, reps, rv, rn))
                else:
                    row.update(s1c="", s1c_reps="", z_vs_s1c="", p_exact="", ok="",
                               note="no S1c reference row (S1c ran no H05 transported job)")
                if nx is not None:
                    row["this_exchangeable_r3"] = nx / mx["reps"]
                    row["z_exchangeable_vs_measured"] = G.z_two_prop(nx / mx["reps"], mx["reps"], mine, reps)
                rows.append(row)
        # battery level
        for b in [x for x in bat_rows if (x["config"], x["sigma_mode"], x["r"]) == key]:
            ref = [x for x in s1c_b if x["rule"] == b["rule"] and x["battery"] == b["battery"] and x["config"] == config
                   and abs(float(x["delta"]) - b["delta"]) < 1e-9
                   and (b["rule"] == "pooled" or (x["c_sd"] and abs(float(x["c_sd"]) - 1) < 1e-9))]
            bx = [x for x in bat_rows if (x["config"], x["sigma_mode"], x["r"]) == keyx and x["battery"] == b["battery"]
                  and x["delta"] == b["delta"]]
            for metric in ("false_transfers_all0", "no_transfer_all0", "false_transfers_alld",
                           "false_no_transfer_one_at_d_max"):
                mine = b[metric]
                row = {"level": "battery", "config": config, "battery": b["battery"], "rule": b["rule"],
                       "item": f"delta {b['delta']:g}", "metric": metric, "this_measured_r3": mine,
                       "this_reps": b["reps"]}
                if ref:
                    rv, rn = float(ref[0][metric]), int(ref[0]["reps"])
                    row.update(compare(mine, b["reps"], rv, rn))
                else:
                    row.update(s1c="", s1c_reps="", z_vs_s1c="", p_exact="", ok="",
                               note="no S1c reference row (S1c ran no H05 transported job)")
                if bx:
                    row["this_exchangeable_r3"] = bx[0][metric]
                    row["z_exchangeable_vs_measured"] = G.z_two_prop(bx[0][metric], bx[0]["reps"], mine, b["reps"])
                rows.append(row)
    return rows


def summary_rows(cell_r, fam_r):
    """Headline per config x mode x r x rule x delta: cell extremes and the family-wise rates (F31, F19)."""
    out = []
    keys = sorted({(r["config"], r["sigma_mode"], r["r"]) for r in cell_r})
    for key in keys:
        for q in RULES:
            for t in TOLS:
                cs = [r for r in cell_r if (r["config"], r["sigma_mode"], r["r"]) == key and r["rule"] == q
                      and r["delta"] == t and "coverage" in r]
                row = dict(key_label(key), rule=q, rule_label=RULE_LABEL[q], delta=t,
                           reps=cs[0]["reps"] if cs else "")
                for sub, sel in (("seed_pooled", lambda r: r["cls"] == "seed-pooled"),
                                 ("one_seed", lambda r: r["cls"] == "one-seed")):
                    ss = [r for r in cs if sel(r)]
                    if not ss:
                        continue
                    lo = min(ss, key=lambda r: r["coverage"])
                    fe = max(ss, key=lambda r: r["false_equivalent_at_mu_eq_delta"])
                    fp = max(ss, key=lambda r: r["false_positive_at_mu_eq_delta"])
                    fb = max(ss, key=lambda r: r["false_below_at_mu_eq_delta"])
                    fa = max(ss, key=lambda r: r["false_above_at_mu_eq_delta"])
                    row.update({f"{sub}_coverage_min": lo["coverage"], f"{sub}_coverage_min_cell": lo["cell"],
                                f"{sub}_coverage_min_mcse": lo["coverage_mcse"],
                                f"{sub}_coverage_mean": float(np.mean([r["coverage"] for r in ss])),
                                f"{sub}_false_equivalent_max": fe["false_equivalent_at_mu_eq_delta"],
                                f"{sub}_false_equivalent_max_cell": fe["cell"],
                                f"{sub}_false_positive_max": fp["false_positive_at_mu_eq_delta"],
                                f"{sub}_false_positive_max_cell": fp["cell"],
                                f"{sub}_false_positive_max_mcse": fp["false_positive_at_mu_eq_delta_mcse"],
                                f"{sub}_false_below_max": fb["false_below_at_mu_eq_delta"],
                                f"{sub}_false_below_max_cell": fb["cell"],
                                f"{sub}_false_above_max": fa["false_above_at_mu_eq_delta"],
                                f"{sub}_false_above_max_cell": fa["cell"],
                                f"{sub}_power_equivalent_mu0_mean": float(np.mean([r["p_equivalent_mu0"] for r in ss])),
                                f"{sub}_power_equivalent_mu0_min": min(r["p_equivalent_mu0"] for r in ss)})
                    pos = [r for r in ss if r["positive_cell"]]
                    row[f"{sub}_power_positive_at_obs"] = "; ".join(
                        f"{r['cell']} {r['p_positive_at_obs']:.4f}" for r in pos) or "none"
                reading = {"reg": "reg", "own": "own", "claim": "claimA"}[q]
                for fname in ("F31", "F19"):
                    def frow(side, conf):
                        return next(r for r in fam_r if (r["config"], r["sigma_mode"], r["r"]) == key
                                    and r["reading"] == reading and r["family"] == fname and r["delta"] == t
                                    and r["side"].startswith(side) and r["configuration"] == conf)
                    nt_d, t_d, t_0 = frow("no-transfer", "alld"), frow("transfer", "alld"), frow("transfer", "all0")
                    nt_o, t_o = frow("no-transfer", "obs"), frow("transfer", "obs")
                    row.update({
                        f"{fname}_p_false_no_transfer_alld": nt_d.get("counted_p_any_false"),
                        f"{fname}_p_false_no_transfer_alld_mcse": nt_d.get("counted_p_any_false_mcse"),
                        f"{fname}_p_false_no_transfer_one_at_d_max": nt_d.get("one_at_d_counted_max"),
                        f"{fname}_p_false_no_transfer_one_at_d_max_mcse": nt_d.get("one_at_d_counted_max_mcse"),
                        f"{fname}_p_false_no_transfer_one_at_d_argmax": nt_d.get("one_at_d_counted_argmax"),
                        f"{fname}_p_false_transfer_alld": t_d.get("counted_p_any_false"),
                        f"{fname}_p_false_transfer_alld_mcse": t_d.get("counted_p_any_false_mcse"),
                        f"{fname}_p_false_transfer_all0": t_0.get("counted_p_any_false"),
                        f"{fname}_p_false_transfer_all0_mcse": t_0.get("counted_p_any_false_mcse"),
                        f"{fname}_p_false_transfer_one_at_d_T_max": t_d.get("one_at_d_T_counted_max"),
                        f"{fname}_p_false_transfer_one_at_d_T_max_mcse": t_d.get("one_at_d_T_counted_max_mcse"),
                        f"{fname}_p_false_transfer_one_at_d_T_argmax": t_d.get("one_at_d_T_counted_argmax"),
                        f"{fname}_obs_e_correct_no_transfer": nt_o.get("counted_e_correct"),
                        f"{fname}_obs_e_correct_no_transfer_mcse": nt_o.get("counted_e_correct_mcse"),
                        f"{fname}_obs_e_correct_transfer": t_o.get("counted_e_correct"),
                        f"{fname}_obs_e_correct_transfer_mcse": t_o.get("counted_e_correct_mcse"),
                        f"{fname}_obs_e_false_no_transfer": nt_o.get("counted_e_false"),
                        f"{fname}_obs_e_false_no_transfer_mcse": nt_o.get("counted_e_false_mcse"),
                        f"{fname}_obs_e_false_transfer": t_o.get("counted_e_false"),
                        f"{fname}_obs_e_false_transfer_mcse": t_o.get("counted_e_false_mcse"),
                        f"{fname}_obs_structural_e_correct_no_transfer": nt_o.get("structural_e_correct"),
                    })
                out.append(row)
    return out


def job_rows(merged, models):
    rows = []
    for key, m in sorted(merged.items()):
        row = dict(key_label(key), reps=m["reps"], n_blocks=len(m["seeds"]), seeds=";".join(map(str, m["seeds"])),
                   seconds=m["seconds"])
        for k2, (dfs, s2s) in m["pool"].items():
            nm = S.NAMES[k2]
            row[f"sim_pool_df_mean[{nm}]"] = dfs / m["reps"]
            row[f"real_pool_df[{nm}]"] = models[k2]["pool_df"]
        rows.append(row)
    return rows


def block_rows(merged, cols):
    """Per seed block: the family-wise headline rates (F31, claimA and reg), to show the block spread."""
    rows = []
    for key, m in sorted(merged.items()):
        for b in m["blocks"]:
            reps = b["reps"]
            for reading in ("reg", "own", "claimA"):
                for fname in ("F31", "F19"):
                    for t in TOLS:
                        row = dict(key_label(key), block=b["block"] + 1, seed=b["seed"], reps=reps, reading=reading,
                                   family=fname, delta=t)
                        for side, conf, nm in (("NT", "alld", "p_false_no_transfer_alld"),
                                               ("T", "all0", "p_false_transfer_all0"),
                                               ("T", "alld", "p_false_transfer_alld")):
                            row[nm] = b["family"][(fname, reading, t, side, conf, "counted")][0] / reps
                        row["obs_e_correct_no_transfer"] = b["family"][(fname, reading, t, "NT", "obs", "counted")][2] / reps
                        row["obs_e_correct_transfer"] = b["family"][(fname, reading, t, "T", "obs", "counted")][2] / reps
                        rows.append(row)
    return rows


def check_rows(merged, checks, val_rows, summ_rows):
    rows = list(checks)
    for key, m in sorted(merged.items()):
        hc = m["holm_check"]
        rows.append({"check": f"{key}: vectorised Holm == frozen analysis.tifs.holm (first block, first reps, every "
                              "reading x family x delta x side x configuration)", "expected": 0,
                     "got": hc["n_mismatch"], "ok": hc["n_mismatch"] == 0 and hc["n_compared"] > 0,
                     "note": f"{hc['n_compared']} replicate-level Holm vectors compared"})
        hm = m["cell"][("hull_mismatch", "claim")][0]
        rows.append({"check": f"{key}: claim-rule verdict == verdict of the hull [min lo, max hi] and the vectorised "
                              "verdict rule == t_contrast's verdicts (mu = 0, every rep, cell, delta)",
                     "expected": 0, "got": int(hm), "ok": hm == 0})
        wm = m["cell"][("wider_mismatch", "claim")][0]
        rows.append({"check": f"{key}: hull == the wider of the two intervals (shared centre)", "expected": 0,
                     "got": int(wm), "ok": wm == 0})
    v = [r for r in val_rows if r.get("z_vs_s1c") not in ("", None)]
    zs = np.array([float(r["z_vs_s1c"]) for r in v])
    oks = np.array([bool(r["ok"]) for r in v])
    pmin_row = min(v, key=lambda r: r["p_exact"])
    rows.append({"check": "validation: registered rule at r = 3 (measured Sigma) vs aux/g4 s1c_cells.csv / "
                          "s1c_battery.csv, every Fisher exact (hypergeometric) p >= 6.33e-5 (|z| < 4 equivalent)",
                 "expected": f"{len(v)} ok", "got": f"{int(oks.sum())} ok", "ok": bool(oks.all()),
                 "note": (f"smallest Fisher p {pmin_row['p_exact']:.4f} ({pmin_row['config']}, {pmin_row['battery']}, "
                          f"{pmin_row['item']}, {pmin_row['metric']}); "
                          f"max |z| {np.max(np.abs(zs)):.2f}; |z| > 1.96 in {int(np.sum(np.abs(zs) > 1.96))} of "
                          f"{len(zs)} ({np.mean(np.abs(zs) > 1.96):.3f}); mean z {np.mean(zs):+.3f}, sd {np.std(zs):.3f}")})
    # MC SE targets: binomial rates (<= 0.002) and expected claim counts (<= 0.02, quoted to 2 decimals)
    mx, mxe, mxe_at = 0.0, 0.0, ""
    for r in summ_rows:
        for k, x in r.items():
            if not (k.endswith("_mcse") and isinstance(x, float) and not math.isnan(x)):
                continue
            if "_e_correct" in k or "_e_false" in k:
                if x > mxe:
                    mxe, mxe_at = x, f"{k} ({r['config']}, {r['sigma_mode']} r = {r['r']}, {r['rule']}, delta {r['delta']:g})"
            else:
                mx = max(mx, x)
    rows.append({"check": "max MC SE over the headline rates of sim_summary.csv <= 0.002", "expected": True,
                 "got": mx <= 0.002, "ok": mx <= 0.002, "note": f"max MC SE {mx:.5f}"})
    rows.append({"check": "max MC SE of the expected claim counts of sim_summary.csv (obs configuration, "
                          "sqrt((E[c^2] - E[c]^2) / reps)) <= 0.02", "expected": True,
                 "got": mxe <= 0.02, "ok": mxe <= 0.02, "note": f"max MC SE {mxe:.5f} at {mxe_at}"})
    return rows


def write_all(out, merged, models, targets, cols, checks, args):
    cr = cell_rows(merged, cols, models, targets)
    G.write_csv(cr, os.path.join(out, "sim_cells.csv"), f"""
        g10 cell level, per config (S1c E/H/H05) x Sigma mode (measured r = 3 = S1c; exchangeable r = 3, 5 for every
        seed-pooled battery, r = 5 hypothetical; a15d = the A15-d data path, BERT AG News logit at r = 5 exchangeable,
        the other seed-pooled batteries at r = 3 measured; column sigma_setting) x
        family cell x rule (reg = registered shared pool + Satterthwaite df / transported rule for one-seed cells;
        own = A3.1 s2_a / r + v_a at df r - 1; claim = reg and own agree else INCONCLUSIVE, one-seed descriptive) x
        delta, from the joint Monte Carlo of the registered S1c harness: coverage of mu (claim: hull of the two
        intervals = the wider one), false 'below' P(hi < mu) and false 'above' P(lo > mu) (delta-free), false
        EQUIVALENT@d and false POSITIVE@d at mu = d (boundary), P(EQUIVALENT@d | mu = 0), P(POSITIVE@d),
        P(EQUIVALENT@d) at mu = k_obs (the frozen primary k; positive_cell = frozen POSITIVE at d), and for the claim
        rule P(reg verdict != own verdict) at mu = 0, d, k_obs; {args.blocks} seed blocks x {args.reps:,} reps per
        config x mode; MC SE binomial. Rows of one-seed cells under own / claim carry only a note.""")
    fr = family_rows(merged, cols)
    G.write_csv(fr, os.path.join(out, "sim_family.csv"), """
        g10 family level, per config x Sigma mode x r (sigma_setting: measured r = 3; exchangeable r = 3, 5 for every
        seed-pooled battery, r = 5 hypothetical; a15d = BERT AG News logit at r = 5, the others at r = 3) x reading
        (reg; own; claimA = Holm over p_max = max(p_reg,
        p_own) with one-seed cells at p_reg and not counted; claimB = same with one-seed cells at p = 1) x family
        (F31 = 31 qualified primary cells, F19 = 19 logit cells) x delta x side (no-transfer = Holm over p_up@d, H0
        k >= d; transfer = Holm over p_pos@d, H0 k <= d; one-sided 0.025) x configuration (all0 / alld = every
        simulated mu at 0 / at d; obs = mu at the frozen primary k): per subset of the family (counted = the claims
        a rule counts: reg/own every non-structural cell, claim rule the seed-pooled cells; seed_pooled; one_seed;
        descriptive = one-seed rejections the claim rule does not count; structural = the two EP SST-2 rows, fixed)
        the probability of at least one false claim, the expected numbers of false and correct claims (with MC SE
        sqrt((E[c^2] - E[c]^2) / reps) from the per-replicate counts) and the probability of any claim; one_at_d_*
        (no-transfer side, alld rows) = max over cells j of P(false no-transfer claim on j | mu_j = d, every other
        simulated mu = 0); one_at_d_T_* (transfer side, alld rows) = max over cells j of P(false transfer claim on j
        | mu_j = d, every other simulated cell surely rejected, p_pos = 0: the least favourable transfer-side
        configuration). Structural nulls enter with their fixed frozen Tango p. MC SE binomial for rates.""")
    G.write_csv(onecell_rows(merged, cols), os.path.join(out, "sim_family_onecell.csv"), """
        g10 family level, least favourable one-cell configurations per cell j, per config x Sigma mode x r x side x
        reading x family x delta: no-transfer side P(Holm no-transfer claim on j | mu_j = delta, every other
        simulated mu = 0); transfer side P(Holm transfer claim on j | mu_j = delta, every other simulated cell surely
        rejected (p_pos = 0), structural nulls at their fixed p, one-seed cells at p = 1 under claimB); counted = the
        reading counts a claim on j (claim rule: seed-pooled cells only). MC SE binomial.""")
    br = battery_rows(merged, models, targets)
    G.write_csv(br, os.path.join(out, "sim_battery.csv"), """
        g10 registered battery quantities, exactly the S1c definitions (s1c.evaluate, imported unchanged, on the
        registered rule's rows of each replicate): P(false TRANSFERS | all mu = 0), P(NO TRANSFER | all mu = 0),
        P(Holm TRANSFERS | all simulated mu = delta), max_j P(NO TRANSFER | mu_j = delta, others 0), per config x
        Sigma mode x r x battery x delta; the simulated mean pool df and s2 beside the measured df. MC SE
        binomial.""")
    vr = validation_rows(merged, cols, br)
    G.write_csv(vr, os.path.join(out, "sim_validation.csv"), """
        g10 validation: registered-rule rates of this joint simulation at r = 3 with the measured input covariance
        against the registered aux/g4 s1c_cells.csv (pooled rule rows; transported c = 1 rows) and s1c_battery.csv
        (independent draws of the same model; two-proportion z and p_exact = the two-sided Fisher exact test
        (hypergeometric, conditional on the total events); pass = p_exact >= 6.33e-5, the two-sided normal tail at
        |z| = 4); and the r = 3 exchangeable
        covariance run against the r = 3 measured run (the bridge to r = 5; descriptive z).""")
    sr = summary_rows(cr, fr)
    G.write_csv(sr, os.path.join(out, "sim_summary.csv"), """
        g10 headline per config x Sigma mode x r x rule x delta: cell level over the seed-pooled cells and over the
        one-seed cells (min / mean coverage, max false EQUIVALENT and max false POSITIVE at mu = delta, max false
        'below' P(hi < mu) and false 'above' P(lo > mu) (= the unadjusted one-sided error of one cell, the limit of
        its family-wise false claim probability when every other hypothesis is rejected surely), mean and min
        P(EQUIVALENT | mu = 0), P(POSITIVE) at the observed k of each frozen POSITIVE cell) and family level for F31
        and F19 with the rule's counted claims (claim rule = reading claimA): P(>= 1 false no-transfer claim) at the
        global boundary (alld) and its one_at_d maximum, P(>= 1 false transfer claim) at all0 and alld and its
        one_at_d_T maximum (cell j at delta, the others surely rejected), and the expected numbers of correct and
        false claims under the observed-k configuration (structural no-transfer claims apart). Sigma settings:
        measured r = 3; exchangeable r = 3, 5 for every seed-pooled battery (r = 5 hypothetical); a15d = BERT AG News
        logit at r = 5, the others at r = 3 (the A15-d data path). MC SE binomial for rates, sqrt((E[c^2] - E[c]^2) /
        reps) for the expected counts.""")
    G.write_csv(job_rows(merged, models), os.path.join(out, "sim_jobs.csv"), f"""
        g10 simulation jobs per config x Sigma mode x r: reps, seed blocks and seeds (seed_job + 1,000,000 b), CPU
        seconds summed over blocks, simulated mean Satterthwaite df of each seed-pooled pool vs the measured pool df.
        Base seed {args.seed}.""")
    G.write_csv(block_rows(merged, cols), os.path.join(out, "sim_blocks.csv"), """
        g10 per seed block (10,000 reps each by default): family-wise headline rates (counted claims) for the readings
        reg, own and claimA, F31 and F19, each delta, to show the spread between blocks.""")
    G.write_csv(check_rows(merged, checks, vr, sr), os.path.join(out, "sim_checks.csv"), """
        g10 simulation checks: the harness's structural rows equal the frozen primary rows; the vectorised Holm equals
        the frozen analysis.tifs.holm; the claim-rule verdict equals the verdict of the hull (= wider) interval in
        every replicate; registered-rule validation against aux/g4 S1c (Fisher exact test, hypergeometric); the MC SE
        targets (rates; expected claim counts).""")
