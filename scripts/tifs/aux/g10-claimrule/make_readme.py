#!/usr/bin/env python3
"""g10 make_readme — writes results/tifs/addendum/aux/g10-claimrule/README.md and scripts/tifs/aux/g10-claimrule/
README.md. Every number is read from the CSVs of the output folder.

A15 post hoc auxiliary analysis (claim-rule calibration); changes no primary verdict, battery decision or the
paper-wide Holm.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import g10_common as G  # noqa: E402

TOLS = G.TOLS
DL = {0.02: "0.02", 0.05: "0.05", 0.1: "0.10"}
MODES = (("measured", 3), ("exchangeable", 3), ("exchangeable", 5), ("a15d", 5))
CONFIGS = ("E", "H", "H05")
# short labels of the Sigma settings (sim_tables.SETTING_LABEL has the long ones)
MLAB = {("measured", 3): "r = 3",
        ("exchangeable", 3): "r = 3 (exchangeable Σ)",
        ("exchangeable", 5): "every seed-pooled battery at r = 5 (hypothetical; A15-d adds seeds to BERT AG News only)",
        ("a15d", 5): "A15-d (BERT AG News at r = 5, the others at r = 3)"}
MSHORT = {("measured", 3): "r = 3", ("exchangeable", 3): "exchangeable r = 3",
          ("exchangeable", 5): "all-r = 5 (hypothetical)", ("a15d", 5): "A15-d"}
RULES = ("reg", "own", "claim")
RNAME = {"reg": "registered", "own": "own variance", "claim": "claim rule", "claimA": "claim rule A",
         "claimB": "claim rule B"}


def num(x):
    if x in ("", None):
        return float("nan")
    return float(x)


def f4(x):
    x = num(x)
    return "n/a" if math.isnan(x) else f"{x:.4f}"


def f3(x):
    x = num(x)
    return "n/a" if math.isnan(x) else f"{x:.3f}"


def f2(x):
    x = num(x)
    return "n/a" if math.isnan(x) else f"{x:.2f}"


def pm(p, se):
    return f"{f4(p)} ± {f4(se)}"


def short(cid):
    if not cid:
        return "-"
    fam, ds, kd, atk = cid.split("/")
    s = f"{G.G6.FAM[fam]} {G.G6.DS[ds]} {G.G6.ATK.get(atk, atk)}"
    return s + (" (feature)" if kd != "logit" else "")


def posstr(s, nd=3):
    if not s or s == "none":
        return "none"
    out = []
    for part in s.split("; "):
        cid, p = part.rsplit(" ", 1)
        out.append(f"{short(cid)} {float(p):.{nd}f}")
    return "; ".join(out)


class Data:
    def __init__(self, out):
        rd = lambda n: G.read_csv(os.path.join(out, n))
        self.cells = rd("sim_cells.csv")
        self.fam = rd("sim_family.csv")
        self.summ = rd("sim_summary.csv")
        self.val = rd("sim_validation.csv")
        self.jobs = rd("sim_jobs.csv")
        self.blocks = rd("sim_blocks.csv")
        self.chk = rd("sim_checks.csv")
        self.fc = rd("frozen_claim_cells.csv")
        self.ff = rd("frozen_claim_family.csv")
        self.fn = rd("frozen_claim_counts.csv")
        self.fchk = rd("frozen_claim_checks.csv")
        self.ind = rd("independent_check.csv")
        p = os.path.join(out, "selfcheck.csv")
        self.self = G.read_csv(p) if os.path.exists(p) else []
        self.cost = open(os.path.join(out, "logs", "sim_cost.txt")).read().strip() if os.path.exists(
            os.path.join(out, "logs", "sim_cost.txt")) else ""

    def s(self, config, mode, r, rule, t):
        return next(x for x in self.summ if (x["config"], x["sigma_mode"], x["r"], x["rule"]) == (config, mode, str(r), rule)
                    and abs(num(x["delta"]) - t) < 1e-12)

    def f(self, config, mode, r, reading, fam, t, side, conf):
        return next(x for x in self.fam if (x["config"], x["sigma_mode"], x["r"], x["reading"], x["family"], x["configuration"])
                    == (config, mode, str(r), reading, fam, conf) and abs(num(x["delta"]) - t) < 1e-12
                    and x["side"].startswith(side))

    def c(self, config, mode, r, cell, rule, t):
        return next(x for x in self.cells if (x["config"], x["sigma_mode"], x["r"], x["cell"], x["rule"]) ==
                    (config, mode, str(r), cell, rule) and abs(num(x["delta"]) - t) < 1e-12)

    def cells_where(self, **kw):
        out = []
        for x in self.cells:
            if all(str(x[k]) == str(v) for k, v in kw.items()) and x.get("coverage", "") != "":
                out.append(x)
        return out


def ext(rows, col, fn=max):
    rows = [r for r in rows if r.get(col, "") != ""]
    r = fn(rows, key=lambda x: num(x[col]))
    return num(r[col]), r


# ── tables ───────────────────────────────────────────────────────────


def val_pmin(D):
    """The validation row with the smallest Fisher exact p, described."""
    v = [r for r in D.val if r.get("z_vs_s1c", "") != ""]
    r = min(v, key=lambda x: num(x["p_exact"]))
    item = r["item"] if r["level"] == "battery" else G.G6.ATK.get(r["item"], r["item"])
    return num(r["p_exact"]), f"{r['battery']} {item} {r['metric']}, config {r['config']}"


def t_validation(D):
    v = [r for r in D.val if r.get("z_vs_s1c", "") != ""]
    zs = [num(r["z_vs_s1c"]) for r in v]
    nok = sum(r["ok"] == "True" for r in v)
    pmin, pwhere = val_pmin(D)
    lines = [f"- {len(v)} comparisons (cell: coverage, false 'below', false 'above', P(EQUIVALENT | k = 0) at each δ, "
             f"for every simulated cell; battery: the four S1c battery rates at each δ; configs E, H, H05 for the "
             f"seed-pooled batteries, E and H for the one-seed batteries, whose S1c runs used c = 1): {nok} of {len(v)} "
             f"pass (two-sided Fisher exact test, hypergeometric, p ≥ 6.33e-5); smallest Fisher p {pmin:.4f} ({pwhere}); "
             f"max |z| {max(abs(z) for z in zs):.2f}; "
             f"|z| > 1.96 in {sum(abs(z) > 1.96 for z in zs)} of {len(zs)} ({sum(abs(z) > 1.96 for z in zs) / len(zs):.3f}); "
             f"mean z {sum(zs) / len(zs):+.3f}."]
    nr = [r for r in D.val if r.get("z_vs_s1c", "") == ""]
    if nr:
        lines.append(f"- {len(nr)} rows (H05 one-seed batteries) have no S1c reference: S1c ran no H05 transported job.")
    zx = [num(r["z_exchangeable_vs_measured"]) for r in D.val if r.get("z_exchangeable_vs_measured", "") != ""]
    if zx:
        lines.append(f"- Bridge to r = 5: the exchangeable covariance at r = 3 against the measured covariance at r = 3 "
                     f"(same rule, same metrics, {len(zx)} comparisons): max |z| {max(abs(z) for z in zx):.2f}, "
                     f"|z| > 1.96 in {sum(abs(z) > 1.96 for z in zx) / len(zx):.3f}, mean z {sum(zx) / len(zx):+.3f} "
                     "(descriptive; the exchangeable covariance averages the measured cross-seed correlations).")
    return lines


def t_cells(D, mode, r):
    lines = ["| config | rule | δ | min coverage (cell) | max false EQU at k = δ | max false POS at k = δ (cell) | "
             "max P(hi < k) at k = δ | mean P(EQU) at k = 0 | P(POS) at the observed k of the frozen POSITIVE cells |",
             "|---|---|---|---|---|---|---|---|---|"]
    for config in CONFIGS:
        for q in RULES:
            for t in TOLS:
                x = D.s(config, mode, r, q, t)
                lines.append(f"| {config} | {RNAME[q]} | {DL[t]} | {f4(x['seed_pooled_coverage_min'])} "
                             f"({short(x['seed_pooled_coverage_min_cell'])}) | {f4(x['seed_pooled_false_equivalent_max'])} "
                             f"| {f4(x['seed_pooled_false_positive_max'])} ({short(x['seed_pooled_false_positive_max_cell'])}) "
                             f"| {f4(x.get('seed_pooled_false_below_max'))} "
                             f"| {f3(x['seed_pooled_power_equivalent_mu0_mean'])} | {posstr(x['seed_pooled_power_positive_at_obs'])} |")
    return lines


def t_oneseed(D):
    lines = ["| config | Σ mode, r | δ | min coverage (cell) | max false EQU at k = δ | max false POS at k = δ (cell) | "
             "mean P(EQU) at k = 0 | P(POS) at the observed k, frozen POSITIVE cells |", "|---|---|---|---|---|---|---|---|"]
    for config in CONFIGS:
        for mode, r in (("measured", 3), ("exchangeable", 5), ("a15d", 5)):
            for t in TOLS:
                x = D.s(config, mode, r, "reg", t)
                lines.append(f"| {config} | {MSHORT[(mode, r)]} | {DL[t]} | {f4(x['one_seed_coverage_min'])} "
                             f"({short(x['one_seed_coverage_min_cell'])}) | {f4(x['one_seed_false_equivalent_max'])} "
                             f"| {f4(x['one_seed_false_positive_max'])} ({short(x['one_seed_false_positive_max_cell'])}) "
                             f"| {f3(x['one_seed_power_equivalent_mu0_mean'])} | {posstr(x['one_seed_power_positive_at_obs'])} |")
    return lines


def pm2(x, se):
    return f"{f2(x)} ± {f3(se)}"


def t_family(D, mode, r, fam):
    lines = ["| config | δ | reading | P(≥1 false no-transfer), all k = δ | max_j P(false no-transfer on j), j at δ, "
             "others 0 | P(≥1 false transfer), all k = 0 | P(≥1 false transfer), all k = δ | max_j P(false transfer on "
             "j), j at δ, others surely rejected | E[correct no-transfer], observed k | E[correct transfer], observed k "
             "| E[false claims], observed k |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for config in CONFIGS:
        for t in TOLS:
            for reading in ("reg", "own", "claimA", "claimB"):
                nt = D.f(config, mode, r, reading, fam, t, "no-transfer", "alld")
                t0 = D.f(config, mode, r, reading, fam, t, "transfer", "all0")
                td = D.f(config, mode, r, reading, fam, t, "transfer", "alld")
                no = D.f(config, mode, r, reading, fam, t, "no-transfer", "obs")
                to = D.f(config, mode, r, reading, fam, t, "transfer", "obs")
                lines.append(f"| {config} | {DL[t]} | {RNAME[reading]} | {pm(nt['counted_p_any_false'], nt['counted_p_any_false_mcse'])} "
                             f"| {f4(nt.get('one_at_d_counted_max'))} ({short(nt.get('one_at_d_counted_argmax'))}) "
                             f"| {pm(t0['counted_p_any_false'], t0['counted_p_any_false_mcse'])} "
                             f"| {pm(td['counted_p_any_false'], td['counted_p_any_false_mcse'])} "
                             f"| {f4(td.get('one_at_d_T_counted_max'))} ({short(td.get('one_at_d_T_counted_argmax'))}) "
                             f"| {f3(no['counted_e_correct'])} ± {f3(no['counted_e_correct_mcse'])} "
                             f"| {f3(to['counted_e_correct'])} ± {f3(to['counted_e_correct_mcse'])} "
                             f"| {f4(num(no['counted_e_false']) + num(to['counted_e_false']))} |")
    return lines


def t_frozen(D):
    lines = ["| δ | registered POS / EQU / INC (29 non-structural) | of which seed-pooled (16) | own variance POS / EQU / "
             "INC (16) | claim rule: POS / EQU claims (16) | rule-dependent | INC by agreement | one-seed, descriptive "
             "| structural, apart | Welch-df sensitivity POS / EQU (29) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for x in D.fn:
        lines.append(f"| {DL[num(x['delta'])]} | {x['registered_29_POS']} / {x['registered_29_EQU']} / {x['registered_29_INC']} "
                     f"| {x['registered_seedpooled16_POS']} / {x['registered_seedpooled16_EQU']} / {x['registered_seedpooled16_INC']} "
                     f"| {x['own_seedpooled16_POS']} / {x['own_seedpooled16_EQU']} / {x['own_seedpooled16_INC']} "
                     f"| {x['claim_POS']} / {x['claim_EQU']} | {x['claim_rule_dependent']} | {x['claim_INC_agreed']} "
                     f"| {x['one_seed_descriptive']} | {x['structural_apart']} | {x['welch_sensitivity_29_POS']} / "
                     f"{x['welch_sensitivity_29_EQU']} |")
    return lines


def a4c(x):
    """'n [A4 n_a4]': counted claims and, among them, the A4-counted ones (dagger cells excluded; the ‡ boundary
    cell BERT SST-2 SOS included)."""
    return f"{x['n_claims_counted']} [A4 {x['n_claims_counted_a4']}]"


def t_frozen_family(D):
    lines = ["Each entry: counted claims [of which A4-counted]. The † cells (SST-2 BadNets of every family and KD "
             "mode: the trigger words occur in no KD-corpus row) are not counted among non-transferred attacks (A4); "
             "the ‡ boundary cells (BERT SST-2 SOS, logit and 'feature') are inside the A4 count.", "",
             "| δ | family | side | registered | own variance (one-seed cells at p_registered) | claim rule A: counted "
             "(one-seed rejections, descriptive) "
             "| claim rule B | structural (apart) |", "|---|---|---|---|---|---|---|---|"]
    for t in TOLS:
        for fam in ("F31", "F19"):
            for side in ("no-transfer", "transfer"):
                g = {x["reading"]: x for x in D.ff if x["family"] == fam and x["side"] == side
                     and abs(num(x["delta"]) - t) < 1e-12}
                lines.append(f"| {DL[t]} | {fam} | {side} | {a4c(g['reg'])} | {a4c(g['own'])} "
                             f"| {a4c(g['claimA'])} ({g['claimA']['n_rejections_descriptive_one_seed']}) "
                             f"| {a4c(g['claimB'])} | {g['reg']['n_rejections_structural']} |")
    return lines


# ── findings ─────────────────────────────────────────────────────────


def findings(D):
    F = []
    reps = int(D.jobs[0]["reps"])
    nb = int(D.jobs[0]["n_blocks"])
    v = [r for r in D.val if r.get("z_vs_s1c", "") != ""]
    zs = [abs(num(r["z_vs_s1c"])) for r in v]
    pmin, pwhere = val_pmin(D)
    F.append(f"**Validation.** The registered rule in this joint simulation (r = 3, measured covariance, {reps:,} reps "
             f"per config = {nb} seed blocks) reproduces the registered aux/g4 S1c tables: {sum(r['ok'] == 'True' for r in v)} "
             f"of {len(v)} cell and battery rates pass the two-sided Fisher exact test (hypergeometric; pass p ≥ 6.33e-5; "
             f"smallest p {pmin:.4f}, {pwhere}; max |z| {max(zs):.2f}; |z| > 1.96 in "
             f"{sum(z > 1.96 for z in zs) / len(zs):.3f} of the comparisons). An independent vectorised re-simulation with "
             f"separately written rule code and fresh seeds (settings H r = 3, E all-r = 5, H A15-d) agrees with every "
             f"compared number ({'; '.join(r['got'] for r in D.ind if r['check'].startswith('(b)'))}).")
    # cell level, H r = 3
    reg = D.s("H", "measured", 3, "reg", 0.05)
    own = D.s("H", "measured", 3, "own", 0.05)
    clm = D.s("H", "measured", 3, "claim", 0.05)
    regE = D.s("E", "measured", 3, "reg", 0.05)
    clmE = D.s("E", "measured", 3, "claim", 0.05)
    fpos = {q: max(num(D.s(c, m, r, q, t)["seed_pooled_false_positive_max"]) for c in CONFIGS for m, r in MODES for t in TOLS)
            for q in RULES}
    fequ = {q: max(num(D.s(c, m, r, q, t)["seed_pooled_false_equivalent_max"]) for c in CONFIGS for m, r in MODES for t in TOLS)
            for q in RULES}
    covmin = {q: min(num(D.s(c, m, r, q, 0.05)["seed_pooled_coverage_min"]) for c in CONFIGS for m, r in MODES) for q in RULES}
    F.append("**Cell level, seed-pooled cells (16).** Under the measured heterogeneity (config H, r = 3) the registered "
             f"shared pool covers BERT AG News EP {f4(reg['seed_pooled_coverage_min'])} and falsely calls it POSITIVE at "
             f"k = δ with probability {f4(reg['seed_pooled_false_positive_max'])} (S1c finding). The own-variance rule "
             f"covers every seed-pooled cell at ≥ {f4(own['seed_pooled_coverage_min'])} and the claim rule at ≥ "
             f"{f4(clm['seed_pooled_coverage_min'])}; under the homogeneous config E the registered minimum is "
             f"{f4(regE['seed_pooled_coverage_min'])} and the claim rule's {f4(clmE['seed_pooled_coverage_min'])} "
             "(conservative). Over every config, Σ setting (r = 3 measured and exchangeable, all-r = 5, A15-d) and δ "
             "the largest false POSITIVE at k = δ is "
             f"{fpos['reg']:.4f} (registered), {fpos['own']:.4f} (own), {fpos['claim']:.4f} (claim rule); the largest false "
             f"EQUIVALENT at k = δ {fequ['reg']:.4f} / {fequ['own']:.4f} / {fequ['claim']:.4f}; the smallest coverage "
             f"{covmin['reg']:.4f} / {covmin['own']:.4f} / {covmin['claim']:.4f}. "
             + (("So the claim rule holds the nominal one-sided 0.025 and 95% coverage at the cell level in every simulated "
                 "setting (within MC error), where the registered rule does not under config H/H05. ")
                if fpos["claim"] <= 0.025 + 0.0012 and fequ["claim"] <= 0.025 + 0.0012 and covmin["claim"] >= 0.95 - 0.0016
                else "The claim rule does NOT hold the nominal cell-level rates in every setting (see the tables). ")
             + "The claim verdict equals the verdict of the wider of the two intervals (both share the centre k), checked "
             "in every replicate.")
    # power
    pw = []
    PWM = (("measured", 3), ("a15d", 5), ("exchangeable", 5))
    for mode, r in PWM:
        for q in ("reg", "claim"):
            x = D.s("H", mode, r, q, 0.05)
            pw.append(f"{RNAME[q]} {MSHORT[(mode, r)]}: {posstr(x['seed_pooled_power_positive_at_obs'])}")
    e0 = {q: [num(D.s("H", mode, r, q, 0.05)["seed_pooled_power_equivalent_mu0_mean"]) for mode, r in PWM] for q in RULES}
    F.append("**Cell-level power (the cost).** P(POSITIVE@0.05) at the observed k of the frozen POSITIVE seed-pooled cells, "
             "config H: " + "; ".join(pw) + ". Mean P(EQUIVALENT@0.05 | k = 0) over the 16 seed-pooled cells, H, r = 3 / "
             "A15-d / all-r = 5: " + ", ".join(f"{RNAME[q]} " + " / ".join(f"{x:.3f}" for x in e0[q]) for q in RULES)
             + ". Settings: r = 3 = the frozen data; A15-d = the A15-d data path (seeds 45, 46 for BERT AG News logit "
             "only, every other battery at r = 3); all-r = 5 = every seed-pooled battery at r = 5, hypothetical (A15-d "
             "adds no seeds to BERT SST-2, Qwen3 SST-2 or GPT-2 SST-2). Own variance at r = 3 has df 2 (t quantile 4.30), "
             "which is where the power goes; at r = 5 (df 4) most of it returns, but under A15-d only for the six BERT AG "
             "News logit cells (BERT SST-2 ATBA stays at its r = 3 power). Power at the observed k is optimistic for "
             "selected POSITIVE cells (winner's curse) and is shown only to compare the rules.")
    # family
    worst = {}
    for reading in ("reg", "own", "claimA", "claimB"):
        vals = []
        for c in CONFIGS:
            for m, r in MODES:
                for fam in ("F31", "F19"):
                    for t in TOLS:
                        nt = D.f(c, m, r, reading, fam, t, "no-transfer", "alld")
                        t0 = D.f(c, m, r, reading, fam, t, "transfer", "all0")
                        td = D.f(c, m, r, reading, fam, t, "transfer", "alld")
                        vals.append((num(nt["counted_p_any_false"]), "NT alld", c, m, r, fam, t))
                        vals.append((num(nt.get("one_at_d_counted_max")), "NT one_at_d", c, m, r, fam, t, nt.get("one_at_d_counted_argmax")))
                        vals.append((num(t0["counted_p_any_false"]), "T all0", c, m, r, fam, t))
                        vals.append((num(td["counted_p_any_false"]), "T alld", c, m, r, fam, t))
                        vals.append((num(td.get("one_at_d_T_counted_max")), "T one_at_d_T", c, m, r, fam, t,
                                     td.get("one_at_d_T_counted_argmax")))
        worst[reading] = vals

    def wmax(reading, kind):
        xs = [x for x in worst[reading] if x[1] == kind and not math.isnan(x[0])]
        return max(xs, key=lambda x: x[0])

    def wdesc(x):
        s = f"{x[0]:.4f} ({x[2]}, {MSHORT[(x[3], x[4])]}, {x[5]}, δ {DL[x[6]]}"
        return s + (f", {short(x[7])})" if len(x) > 7 and x[7] else ")")

    # cell-level limits of the least favourable one-cell configurations (every other hypothesis rejected surely):
    # the cell's unadjusted one-sided error at k = delta, max over settings and over the cells the rule counts
    def cell_lim(q, col_sp, col_os=None):
        vals = []
        for c in CONFIGS:
            for m, r in MODES:
                for t in TOLS:
                    x = D.s(c, m, r, q, t)
                    vals.append(num(x.get(col_sp)))
                    if col_os:
                        vals.append(num(x.get(col_os)))
        return max(v for v in vals if not math.isnan(v))

    lim_pos_reg = cell_lim("reg", "seed_pooled_false_positive_max", "one_seed_false_positive_max")
    lim_pos_clm = cell_lim("claim", "seed_pooled_false_positive_max")
    lim_below_clm = cell_lim("claim", "seed_pooled_false_below_max")

    rH = D.f("H", "measured", 3, "reg", "F31", 0.05, "no-transfer", "alld")
    rH19 = D.f("H", "measured", 3, "reg", "F19", 0.05, "no-transfer", "alld")
    rE = D.f("E", "measured", 3, "reg", "F31", 0.05, "no-transfer", "alld")
    F.append("**Family level, registered rule.** Holm over the registered p controls the family-wise false no-transfer "
             f"claim only under homogeneity: at the global boundary (every simulated k = δ) P(≥ 1 false claim) is "
             f"{pm(rE['counted_p_any_false'], rE['counted_p_any_false_mcse'])} for F31 under E but "
             f"{pm(rH['counted_p_any_false'], rH['counted_p_any_false_mcse'])} (F31) and "
             f"{pm(rH19['counted_p_any_false'], rH19['counted_p_any_false_mcse'])} (F19) under H at r = 3 (δ-free at the "
             f"global boundary), about {num(rH['counted_p_any_false']) / 0.025:.1f} times the nominal 0.025; the worst "
             f"one-at-δ case is {wdesc(wmax('reg', 'NT one_at_d'))}; the worst false transfer claim with every k = 0 is "
             f"{wdesc(wmax('reg', 'T all0'))} and with every k = δ {wdesc(wmax('reg', 'T alld'))}. The cells with the "
             "largest one-at-δ rates are BERT AG News EP (logit, F_a 3.9) and its one-seed 'feature' twin (sim_family_onecell.csv). "
             "Transfer side, least favourable (one cell at k = δ, every other simulated cell surely rejected): simulated "
             f"maximum {wdesc(wmax('reg', 'T one_at_d_T'))}; its limit is the cell-level false POSITIVE at k = δ, "
             f"registered {lim_pos_reg:.4f} (the simulated value stays below it because the structural nulls cannot be "
             "rejected and keep the cell's Holm multiplier above 1).")
    KINDS = ("NT alld", "NT one_at_d", "T all0", "T alld", "T one_at_d_T")
    allA = [x for k in KINDS for x in worst["claimA"] if x[1] == k and not math.isnan(x[0])]
    allB = [x for k in KINDS for x in worst["claimB"] if x[1] == k and not math.isnan(x[0])]
    reps_all = int(D.jobs[0]["reps"])
    topA = max(allA, key=lambda x: x[0])
    nA_over = sum(x[0] > 0.025 for x in allA)
    nB_over = sum(x[0] > 0.025 for x in allB)
    verdict_txt = ((f"stays below the nominal 0.025 in every simulated setting (3 configs x 4 Σ settings x 2 families x 3 δ "
                    f"x 5 configurations: no-transfer at the global boundary and one-at-δ, transfer with every k = 0, at the "
                    f"global boundary and one-at-δ with the others surely rejected; the largest is {topA[0]:.4f} ± "
                    f"{G.mc_se(topA[0], reps_all):.4f}, {wdesc(topA)})")
                   if nA_over == 0 and nB_over == 0 else
                   (f"EXCEEDS the nominal 0.025 in {nA_over} (reading A) and {nB_over} (reading B) of the simulated settings; "
                    "the manuscript must say so (amendment item 4)"))
    F.append("**Family level, claim rule (p_max, Holm, one-sided 0.025) — the registered question.** The claim rule's "
             f"family-wise false claim probability {verdict_txt}. Worst cases, reading A (one-seed cells enter with p_registered, not counted): "
             f"no-transfer at the global boundary {wdesc(wmax('claimA', 'NT alld'))}; no-transfer one-at-δ "
             f"{wdesc(wmax('claimA', 'NT one_at_d'))}; transfer with every k = 0 {wdesc(wmax('claimA', 'T all0'))}; transfer "
             f"at the global boundary {wdesc(wmax('claimA', 'T alld'))}; transfer one-at-δ with the others surely rejected "
             f"{wdesc(wmax('claimA', 'T one_at_d_T'))}. Reading B (one-seed cells at p = 1): "
             f"{wdesc(wmax('claimB', 'NT alld'))}, {wdesc(wmax('claimB', 'NT one_at_d'))}, {wdesc(wmax('claimB', 'T all0'))}, "
             f"{wdesc(wmax('claimB', 'T alld'))}, {wdesc(wmax('claimB', 'T one_at_d_T'))}. The one-at-δ maximum under "
             "reading A approaches the nominal level at δ 0.10 in the all-r = 5 setting because the other cells (at k = 0) "
             "are rejected "
             "and Holm steps down to the boundary cell; the limit when every other hypothesis is rejected surely is the "
             "cell's unadjusted rate. No-transfer side: max P(hi < k) at k = δ for the claim rule "
             f"{lim_below_clm:.4f}. Transfer side, least favourable (one cell at δ, the others surely rejected): limit = "
             f"the cell-level false POSITIVE at k = δ, registered {lim_pos_reg:.4f}, claim rule {lim_pos_clm:.4f} (the "
             "simulated one_at_d_T values above stay below the limit because the structural nulls, and under reading B "
             "the one-seed cells, cannot be rejected).")
    # expected correct
    def ecs(mode, r, reading, fam, t):
        no = D.f("H", mode, r, reading, fam, t, "no-transfer", "obs")
        to = D.f("H", mode, r, reading, fam, t, "transfer", "obs")
        return (f"{pm2(no['counted_e_correct'], no['counted_e_correct_mcse'])} no-transfer + "
                f"{pm2(to['counted_e_correct'], to['counted_e_correct_mcse'])} transfer")

    ec = []
    for mode, r in (("measured", 3), ("a15d", 5), ("exchangeable", 5)):
        for reading in ("reg", "own", "claimA"):
            ec.append(f"{RNAME[reading]} {MSHORT[(mode, r)]}: {ecs(mode, r, reading, 'F31', 0.05)}")
    a15 = [f"F19 δ 0.05 {ecs('a15d', 5, 'claimA', 'F19', 0.05)}", f"F31 δ 0.10 {ecs('a15d', 5, 'claimA', 'F31', 0.1)}",
           f"F19 δ 0.10 {ecs('a15d', 5, 'claimA', 'F19', 0.1)}"]
    mxe = [c for c in D.chk if c["check"].startswith("max MC SE of the expected claim counts")]
    xa = D.f("H", "a15d", 5, "claimA", "F31", 0.05, "no-transfer", "obs")
    x5 = D.f("H", "exchangeable", 5, "claimA", "F31", 0.05, "no-transfer", "obs")
    gap = num(x5["counted_e_correct"]) - num(xa["counted_e_correct"])
    gse = math.hypot(num(x5["counted_e_correct_mcse"]), num(xa["counted_e_correct_mcse"]))
    cmp_txt = ("clearly fewer no-transfer claims than if every seed-pooled battery had r = 5 (hypothetical; F31 δ 0.05: "
               f"{f2(xa['counted_e_correct'])} vs {f2(x5['counted_e_correct'])})" if gap > 4 * gse else
               "about as many no-transfer claims as if every seed-pooled battery had r = 5 (hypothetical)")
    F.append("**Family-level power under the observed-k configuration (every true k = the frozen primary k), F31, δ 0.05, "
             "config H, expected correct claims among the counted cells (± MC SE):** " + "; ".join(ec) + " (registered "
             "and own count the 13 one-seed cells too; the claim rule only the 16 seed-pooled). At r = 3 the claim rule "
             "makes almost no family-wise claim (own df 2). On the A15-d data path (BERT AG News logit at r = 5, every "
             f"other battery at r = 3) it makes {cmp_txt}; claim rule A under A15-d, config H: " + "; ".join(a15)
             + ". The all-r = 5 rows show what "
             "seeds for every seed-pooled battery would give, not what A15-d gives. "
             + (f"Largest MC SE of an expected count in sim_summary.csv: {mxe[0]['note'].replace('max MC SE ', '')}. "
                if mxe else "") + "Full tables: sim_family.csv, sim_summary.csv.")
    o = D.s("H", "measured", 3, "reg", 0.05)
    F.append("**One-seed cells (13, descriptive under the claim rule).** Their registered transported rule is the "
             f"miscalibrated part under H: coverage down to {f4(o['one_seed_coverage_min'])} "
             f"({short(o['one_seed_coverage_min_cell'])}), false POSITIVE at k = δ up to {f4(o['one_seed_false_positive_max'])} "
             f"({short(o['one_seed_false_positive_max_cell'])}) at r = 3 (S1c transported rows reproduce this). The claim "
             "rule makes no claim on them, so this does not enter its counts; under reading A their p_registered still "
             "enters the Holm step-down.")
    # frozen data
    lines = []
    for x in D.fn:
        t = num(x["delta"])
        lines.append(f"δ {DL[t]}: {x['claim_POS']} POSITIVE ({x['claim_POS_cells']}), {x['claim_EQU']} EQUIVALENT "
                     f"({x['claim_EQU_a4_counted']} A4-counted), {x['claim_rule_dependent']} rule-dependent "
                     f"({x['claim_rule_dependent_cells']}), {x['claim_INC_agreed']} INCONCLUSIVE by agreement")
    F.append("**The frozen data under the claim rule (cell level; 16 seed-pooled cells; 13 one-seed descriptive; 2 structural "
             "apart).** " + ". ".join(lines) + ". The registered verdicts stay in the tables (H1).")
    fl = []
    for t in TOLS:
        parts = []
        for fam in ("F31", "F19"):
            g = {(x["reading"], x["side"]): x for x in D.ff if x["family"] == fam and abs(num(x["delta"]) - t) < 1e-12}
            nA, nB, nR = (g[(x, 'no-transfer')] for x in ("claimA", "claimB", "reg"))
            parts.append(f"{fam} no-transfer {nA['n_claims_counted']} (A) / {nB['n_claims_counted']} (B) claims, of "
                         f"which A4-counted {nA['n_claims_counted_a4']} / {nB['n_claims_counted_a4']} (registered "
                         f"{nR['n_claims_counted']}, A4-counted {nR['n_claims_counted_a4']}); transfer "
                         f"{g[('claimA', 'transfer')]['n_claims_counted']} / {g[('claimB', 'transfer')]['n_claims_counted']} "
                         f"(registered {g[('reg', 'transfer')]['n_claims_counted']})")
        fl.append(f"δ {DL[t]}: " + "; ".join(parts))
    g10 = {x["family"]: x for x in D.ff if x["reading"] == "claimA" and x["side"] == "no-transfer" and abs(num(x["delta"]) - 0.1) < 1e-12}
    F.append("**The frozen data under the claim rule, family-wise (Holm over p_max, counted claims; structural nulls apart; "
             "A4-counted = without the † cells, which A4 does not count among non-transferred attacks, with the ‡ "
             "boundary cell).** " + ". ".join(fl) + f". Claimed at δ 0.10, no-transfer: F31 {g10['F31']['claims_counted']}; "
             f"F19 {g10['F19']['claims_counted']}. " + reading_note(D) + " " + transfer_note(D))
    return F


def reading_diff(D):
    """Where readings A and B of the one-seed cells differ on the frozen data: (delta, family, side, cells only in
    A, cells only in B, A4-counted A, A4-counted B)."""
    out = []
    for t in TOLS:
        for fam in ("F31", "F19"):
            for side in ("no-transfer", "transfer"):
                g = {x["reading"]: x for x in D.ff if x["family"] == fam and x["side"] == side
                     and abs(num(x["delta"]) - t) < 1e-12}
                sa = set(s for s in g["claimA"]["claims_counted"].split("; ") if s != "none")
                sb = set(s for s in g["claimB"]["claims_counted"].split("; ") if s != "none")
                if sa != sb:
                    out.append((t, fam, side, sorted(sa - sb), sorted(sb - sa), g["claimA"]["n_claims_counted_a4"],
                                g["claimB"]["n_claims_counted_a4"]))
    return out


def reading_note(D):
    diff = reading_diff(D)
    if not diff:
        return "Readings A and B give the same claimed cells on the frozen data in every family, side and δ."
    parts = []
    a4_same = all(d[5] == d[6] for d in diff)
    only_dagger = all(c.endswith("†") for d in diff for c in d[3] + d[4])
    for t, fam, side, oa, ob, a4a, a4b in diff:
        parts.append(f"{fam} {side} δ {DL[t]}: only under A {', '.join(oa) or 'none'}; only under B {', '.join(ob) or 'none'} "
                     f"(A4-counted {a4a} / {a4b})")
    # A4-counted no-transfer claims of both readings, per family, at the deltas where the readings differ
    a4txt = []
    for t in sorted({d[0] for d in diff}):
        per = []
        for fam in ("F31", "F19"):
            g = {x["reading"]: x for x in D.ff if x["family"] == fam and x["side"] == "no-transfer"
                 and abs(num(x["delta"]) - t) < 1e-12}
            per.append(f"{fam} {g['claimA']['n_claims_counted_a4']} / {g['claimB']['n_claims_counted_a4']}")
        a4txt.append(f"δ {DL[t]} no-transfer, A / B: " + ", ".join(per))
    return ("Readings A and B differ on the frozen data only in " + "; ".join(parts)
            + (" — only in † cells, which A4 does not count" if only_dagger else "") + "."
            + (" The A4-counted claims are equal under both readings in every family, side and δ ("
               + "; ".join(a4txt) + ")." if a4_same else " The A4-counted claims differ there."))


def transfer_note(D):
    """Family-wise transfer side of the frozen data under the claim rule, with the strongest cell's p values."""
    n_tr = sum(int(x["n_claims_counted"]) for x in D.ff if x["reading"] in ("claimA", "claimB") and x["side"] == "transfer")
    rows = [x for x in D.fc if x["class"] == "seed-pooled" and abs(num(x["delta"]) - 0.02) < 1e-12]
    best = min(rows, key=lambda x: num(x["p_max_pos"]))
    txt = ("No family-wise transfer claim is made under the claim rule at any δ. " if n_tr == 0 else
           f"{n_tr} family-wise transfer claims are made under the claim rule (frozen_claim_family.csv). ")
    return txt + (f"The strongest seed-pooled cell, {best['display']} at δ 0.02, has p_pos registered "
                  f"{num(best['registered_p_pos']):.2e} but own variance {num(best['own_p_pos']):.4f} (df "
                  f"{int(num(best['own_r'])) - 1}), above the first Holm threshold 0.025/31 = {0.025 / 31:.5f} (F31) and "
                  f"0.025/19 = {0.025 / 19:.5f} (F19).")


def write_output_readme(D, out):
    snap = G.loader_snapshot()
    L = []
    L.append("# g10: claim-rule calibration and the frozen data under the claim rule")
    L.append("")
    L.append("**Status.** A15 post hoc auxiliary analysis registered in docs/tifs_addendum.md, amendment 10/27 20:58 KST "
             "(\"보정 ... 판정 위계, claim rule\", item 4; the claim rule itself is items 2-3), before any A15 student result "
             "was computed. It **changes no primary verdict, battery decision or the paper-wide Holm** (H1: the frozen 31 "
             "verdicts stay as registered). Code: `scripts/tifs/aux/g10-claimrule/`. Every number below is read from the "
             f"CSVs of this folder by make_readme.py (generated {G.now_utc()}).")
    L.append("")
    L.append(f"**Data snapshot.** {snap['n_loader']} prediction files of {snap['n_runs_loader']} runs parsed by the frozen "
             f"`analyze.load_cells` (newest {snap['newest_loader_utc']}, `{snap['newest_loader']}`), i.e. the frozen "
             f"primary family only; the other {snap['n_all'] - snap['n_loader']} prediction files under runs/tifs (A2/A8/A10 "
             "side runs, A13 `tifs_x13_*`, A15 `tifs_x15_*`) are not parsed and no A15 number was read or computed. "
             "Before anything is computed, every analysis script (frozen_claims.py, sim.py, independent.py) reproduces "
             "results/tifs/report/primary.csv, variance.csv, battery.csv, results/tifs/addendum/global_holm.csv and "
             "attack_variance.csv exactly (every column, text) through the frozen loader and stops on a mismatch; "
             "selfcheck.py records the same reproduction checks, and run_all.sh (set -e) stops the pipeline at the first "
             "failure. Other inputs: aux/g4-s1c-s9-s1extra/s1c_cells.csv, s1c_battery.csv (validation), aux/g6-review0/"
             "multiplicity/claims_family.csv (A14 item 1 reproduction), aux/g9-a15cpu/welch_cells.csv (Welch sensitivity "
             "column), aux/g1 tokens_summary.csv (A4 classes, through g6m_common). "
             f"Run cost: {D.cost}.")
    L.append("")
    L.append("## Key findings")
    L.append("")
    for i, f in enumerate(findings(D), 1):
        L.append(f"{i}. {f}")
    L.append("")
    L.append("## Definitions")
    L.append("")
    L.append("**Harness (registered A11.10 S1c, imported unchanged).** `scripts/tifs/aux/g4-s1c-s9-s1extra/s1c.py` "
             "(`pooled_model`, `target_model`, `draw_battery`, `pool_of`, `evaluate`, `chol`) and `common.py` (`ExactFastT`, "
             "`battery_plugin`, `target_plugin`). Seed-pooled battery cells: k[a,s] = μ_a + T_a(√w z_s + √(1−w) z_as) + "
             "e[a,s], e ~ N(0, Σ) with Σ the measured input covariance / n of the stacked per-input d = P − C vectors. "
             "Configs: **H** measured heterogeneous T_a = √max(0, s²_a,obs − ψ_a) with the fitted shared fraction w; **E** "
             "homogeneous T (pool mean of T_H²) with its own fitted w; **H05** = H with w = 0.5. One-seed (transported) "
             "cells: k'_a = μ_a + T'_a(√w z' + √(1−w) z'_a) + e'_a with T'_a = √(F_a E[s²_pool]) (H, H05) or √E[s²_pool] "
             "(E), c = 1. Structural nulls (BERT SST-2 EP logit and 'feature'): their fixed frozen seed-42 Tango rows.")
    L.append("")
    L.append("**The 31-cell family, simulated jointly.** One replicate draws the four seed-pooled batteries (BERT SST-2 logit "
             "4 pooled + SOS; BERT AG News logit 6; Qwen3 SST-2 3; GPT-2 SST-2 2) and the three one-seed batteries (BERT "
             "SST-2 'feature' 5, BERT AG News 'feature' 6, Qwen3 AG News 2), each one-seed battery borrowing s²_pool and "
             "its Satterthwaite df from the same replicate's source battery (as A14 headline_power.py; S1c used an "
             "independent source draw per job, same marginal law). Σ settings: **r = 3** with the measured Σ (= S1c); "
             "**exchangeable r = 3** (bridge) and **all-r = 5** with the exchangeable Σ of headline_power.exchangeable "
             "for every seed-pooled battery (v_a recomputed at r; T and w kept at the r = 3 fit) — all-r = 5 is "
             "hypothetical, since A15-d adds seeds 45, 46 to BERT AG News logit only; **A15-d** = the A15-d data path: "
             "BERT AG News logit at r = 5 with the exchangeable Σ, BERT SST-2, Qwen3 SST-2 and GPT-2 SST-2 at r = 3 with "
             "the measured Σ (sim_* column sigma_mode 'a15d', r = 5 = the BERT AG News r; the BERT AG News 'feature' "
             "one-seed cells borrow that r = 5 pool). One-seed cells stay one-seed in every setting.")
    L.append("")
    L.append("**Rules** (every interval, verdict and p from the frozen `analysis.tifs.t_contrast`; v_a known, as S1c):")
    L.append("- **registered**: seed-pooled SE² = s²_pool/r + v_a at the Satterthwaite df of the frozen "
             "`analyze.pool_seed_variance`; one-seed SE² = SE²_Tango + s²_pool at the pool df.")
    L.append("- **own variance (A3.1)**: SE² = s²_a/r + v_a at df r − 1, s²_a = `addendum.var` of the cell's r seed "
             "estimates; the input term is included as in the fixed formula (it partly double-counts input noise, ψ_a/r, "
             "so the rule is conservative in expectation). Not defined for one-seed cells.")
    L.append("- **claim rule (amendment item 3)**: a seed-pooled cell's verdict is claimed when registered and own agree; "
             "otherwise 'rule-dependent', counted as INCONCLUSIVE. One-seed cells: 'one-seed, descriptive', never "
             "counted. Structural nulls counted apart. Because both intervals are centred at the same k, the claim verdict "
             "is the verdict of the wider interval; its coverage is that of the hull [min lo, max hi] (checked in every "
             "replicate and on the frozen data). Family-wise: Holm (frozen order; a vectorised copy of "
             "`analysis.tifs.holm`, bit-identical on every compared replicate) over p_max = max(p_registered, p_own), "
             "one-sided 0.025, no-transfer side p_up@δ (H0 k ≥ δ), transfer side p_pos@δ (H0 k ≤ δ), families F31 (the 31 "
             "qualified primary cells) and F19 (its 19 logit cells). Welch df: not in the rule; a sensitivity column of "
             "the frozen-data table only.")
    L.append("- **Readings of the family Holm for the one-seed cells** (the amendment does not say which p they enter with): "
             "**A** = their p_registered (p_own does not exist; max over the p's that exist), their rejections reported as "
             "descriptive and not counted; **B** = p = 1 (they occupy family slots but can never be rejected). A is used for "
             "the 'claim' rows of sim_summary.csv; both are in every family table.")
    L.append("")
    L.append("**Metrics.** μ is a pure shift (pool, s²_a and every SE are shift-invariant), so one noise draw serves every μ "
             "and the frozen t_contrast is called at est + μ. Cell: coverage of μ; false EQUIVALENT@δ and false POSITIVE@δ "
             "at μ = δ (boundary); P(hi < μ) and P(lo > μ) (δ-free one-sided errors, the S1c 'false below/above'); "
             "P(EQUIVALENT@δ | μ = 0); P(POSITIVE@δ) and P(EQUIVALENT@δ) at μ = the frozen primary k (positive_cell = "
             "POSITIVE in the frozen table at that δ); P(registered ≠ own). Family (counted claims: registered and own "
             "every non-structural cell, claim rule the seed-pooled cells): configurations **all0** (every simulated μ = 0), "
             "**alld** (every simulated μ = δ, the global boundary), **one_at_d** (no-transfer side: cell j at δ, the "
             "others at 0; the max over j is reported), **one_at_d_T** (transfer side: cell j at δ, every other simulated "
             "cell surely rejected, p_pos = 0, the μ → +∞ limit; structural nulls keep their fixed p; the least "
             "favourable transfer-side configuration; μ = +1 is not used because the df-2 t tail does not make that "
             "rejection sure), **obs** (μ = the frozen primary k). A no-transfer claim is false iff μ ≥ δ, a transfer "
             "claim iff μ ≤ δ. MC SE binomial for rates; for an expected claim count sqrt((E[c²] − E[c]²)/reps) from "
             "the per-replicate counts (maxima of both in sim_checks.csv).")
    L.append("")
    L.append("## Tables")
    L.append("")
    L.append("### Validation (registered rule vs aux/g4 S1c)")
    L.append("")
    L += t_validation(D)
    for mode, r in (("measured", 3), ("a15d", 5), ("exchangeable", 5)):
        L.append("")
        L.append(f"### Cell level, the 16 seed-pooled cells, {MLAB[(mode, r)]}")
        L.append("")
        L += t_cells(D, mode, r)
    L.append("")
    L.append("### Cell level, the 13 one-seed cells (registered transported rule; descriptive under the claim rule)")
    L.append("")
    L += t_oneseed(D)
    for mode, r in (("measured", 3), ("a15d", 5), ("exchangeable", 5)):
        for fam in ("F31", "F19"):
            L.append("")
            L.append(f"### Family level, {fam}, {MLAB[(mode, r)]} (counted claims; MC SE after ±)")
            L.append("")
            L += t_family(D, mode, r, fam)
    L.append("")
    L.append("The exchangeable r = 3 rows (bridge) are in sim_family.csv / sim_summary.csv.")
    L.append("")
    L.append("### The frozen data under the claim rule: cell counts")
    L.append("")
    L += t_frozen(D)
    L.append("")
    L.append("### The frozen data under the claim rule: family-wise counted claims (Holm, one-sided 0.025)")
    L.append("")
    L += t_frozen_family(D)
    L.append("")
    L.append("Per cell (registered, own, hull, p_max, claim label, Welch column, every family x reading x side flag): "
             "frozen_claim_cells.csv; claimed cells per family: frozen_claim_family.csv.")
    L.append("")
    L.append("## Interpretation choices (stated, not registered; for the author before the A15 analysis)")
    L.append("")
    L.append("1. **One-seed cells in the family Holm** (readings A and B above). The amendment fixes F31 and F19 and "
             "p_max but not the p of a cell that has no own variance. Both readings are reported. " + reading_note(D)
             + " A choice should be recorded before g8-a15/analysis applies the rule.")
    L.append("2. **Coverage of the claim rule** is defined through the hull interval, which is exact here (shared centre).")
    L.append("3. **H05 one-seed batteries**: S1c ran none; here the target uses w = 0.5 with the H training SD.")
    L.append("4. **r = 5** uses the exchangeable covariance and the r = 3 fit of T and w (A14 item 2 convention); one-seed "
             "cells stay one-seed; the structural rows are the fixed seed-42 rows. Two r = 5 settings: **A15-d** (BERT AG "
             "News logit only, as A15-d runs it) and **all-r = 5** (every seed-pooled battery; hypothetical, an upper "
             "reference for what more seeds everywhere would give). Manuscript planning should use the A15-d rows.")
    L.append("5. **Power at the observed k** treats the frozen k as the truth; for cells selected as POSITIVE this is "
             "optimistic. It is reported only to compare the rules.")
    L.append("")
    L.append("## Self-checks")
    L.append("")
    allc = D.chk + D.fchk + D.ind + D.self
    L.append(f"- {sum(c['ok'] == 'True' for c in allc)} of {len(allc)} checks pass (sim_checks.csv {len(D.chk)}, "
             f"frozen_claim_checks.csv {len(D.fchk)}, independent_check.csv {len(D.ind)}, selfcheck.csv {len(D.self)}).")
    for c in allc:
        if c["ok"] != "True":
            L.append(f"  - FAILED: {c['check']} (expected {c['expected']}, got {c['got']}) {c.get('note', '')}")
    L.append("- Frozen-file SHA-256 match their records (freeze_20261021T092041Z: tifs_plan.md, analyze.py, analysis/tifs.py, "
             "metrics/paired.py, structural_nulls.json; final_analysis_20261024T110542Z: addendum.py, cross_score.py; "
             "a15_claimrule_registration_fix_20261027T115825Z: tifs_addendum.md and the imported g4 / g6 code) and are "
             "unchanged between the start and the end of run_all.sh.")
    L.append("- Gate: primary.csv, variance.csv, battery.csv, global_holm.csv, attack_variance.csv reproduced exactly before "
             "any computation; every analysis script (frozen_claims.py, sim.py, independent.py) stops on a mismatch, "
             "selfcheck.py records the same checks, run_all.sh (set -e) stops the pipeline at the first failure.")
    L.append("- The registered reading reproduces A14 item 1 (claims_family.csv Holm counts) and the frozen transfers_global "
             "(global_holm.csv, exact text); the own reading reproduces attack_variance.csv (A3.1) exactly.")
    L.append("")
    L.append("## Files")
    L.append("")
    files = [
        ("sim_cells.csv", "cell level per config x Σ mode x r x cell x rule x δ"),
        ("sim_family.csv", "family level per config x Σ mode x r x reading x family x δ x side x configuration, by subset"),
        ("sim_family_onecell.csv", "one_at_d (no-transfer side, others at 0) and one_at_d_T (transfer side, others surely rejected) per cell j"),
        ("sim_summary.csv", "headline per config x Σ mode x r x rule x δ"),
        ("sim_battery.csv", "the registered S1c battery rates in this simulation"),
        ("sim_validation.csv", "registered rule vs aux/g4 S1c; exchangeable vs measured r = 3"),
        ("sim_jobs.csv, sim_blocks.csv", "jobs, seeds, simulated pool df; per-block family rates"),
        ("sim_checks.csv", "Holm exactness, hull identity, structural rows, validation summary, MC SE target"),
        ("frozen_claim_cells.csv, frozen_claim_family.csv, frozen_claim_counts.csv", "the frozen data under the claim rule"),
        ("frozen_claim_checks.csv, independent_check.csv, selfcheck.csv", "checks"),
        ("logs/", "run logs, hashes before the run, wall and CPU seconds"),
    ]
    for f, d in files:
        L.append(f"- `{f}`: {d}.")
    L.append("")
    L.append("## Reproduce")
    L.append("")
    L.append("```")
    L.append("bash scripts/tifs/aux/g10-claimrule/run_all.sh            # -> results/tifs/addendum/aux/g10-claimrule/")
    L.append("bash scripts/tifs/aux/g10-claimrule/run_all.sh --out DIR  # other folder; --quick for a smoke run")
    L.append("```")
    L.append("CPU only, nice 10, one BLAS thread per process, fixed seeds (sim.py base 20261027 + job index, block b: "
             "+ 1,000,000 b; independent.py 99020261027), so a rerun gives the same numbers.")
    L.append("")
    L.append("## Caveats")
    L.append("")
    L.append("- The model is S1b's normal approximation fitted to the measured batteries (S1c); it does not include the "
             "teacher-build (reference) variance of A3.3 or cross-battery input correlation.")
    L.append("- v_a is treated as known in the simulation; in the data it is estimated (n = 912 or 5,700 inputs).")
    L.append("- Holm is a testing procedure; the family claims carry no simultaneous intervals (A14 item 1).")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  wrote {G.rel(os.path.join(out, 'README.md'))}")


def write_code_readme(D):
    L = ["# g10: claim-rule calibration — code", "",
         "Status: A15 post hoc auxiliary analysis (docs/tifs_addendum.md, amendment 10/27 20:58 KST, item 4, and the claim "
         "rule of items 2-3 applied to the frozen 31 cells); changes no primary verdict, battery decision or the "
         "paper-wide Holm.", "",
         "Outputs, definitions, data snapshot and findings: `results/tifs/addendum/aux/g10-claimrule/README.md` (written "
         "by make_readme.py together with this file).", "",
         "| script | what |", "|---|---|",
         "| g10_common.py | paths; frozen imports (analyze, addendum, analysis/tifs through the g4 S1c `common.py`; `s1c.py`; "
         "g6 `g6m_common.py`, `headline_power.py`); frozen-file hashes against their records; the reproduction gate; "
         "CSV writer with the `# definition:` line |",
         "| frozen_claims.py | the claim rule on the frozen 31 primary cells: cell labels, p_max, Holm over F31 / F19 "
         "(readings reg, own, claimA, claimB), counts; reproduces A3.1, A14 item 1 and transfers_global |",
         "| sim.py | joint Monte Carlo of the 31-cell family on the registered S1c harness (configs E/H/H05; measured r = 3, "
         "exchangeable r = 3 and 5, a15d); rules registered / own / claim; cell and family metrics; S1c battery quantities |",
         "| sim_tables.py | the sim.py tables and checks (validation against aux/g4 S1c, Fisher exact test) |",
         "| independent.py | the frozen-data counts from raw CSV text with scipy only, and a vectorised re-simulation with "
         "separately written rule code and fresh seeds (H r = 3, E all-r = 5, H a15d); gated |",
         "| selfcheck.py | hashes before / after, gate, loader inputs, all check files, CSV format, no bytecode |",
         "| make_readme.py | both READMEs |",
         "| run_all.sh | everything, CPU only |", "",
         "Run: `bash scripts/tifs/aux/g10-claimrule/run_all.sh [--out DIR] [--workers N] [--quick]`. Environment set by "
         "run_all.sh: `CUDA_VISIBLE_DEVICES=\"\"`, `PYTHONDONTWRITEBYTECODE=1` (every script also sets "
         "`sys.dont_write_bytecode`), one BLAS thread, `nice -n 10`, interpreter `/data/Youngho/envs/blade/bin/python` "
         "(override with `PY=...`), HF offline. Nothing outside this folder and the output folder is written; the frozen "
         "files and the g4 / g6 aux code are imported, never edited.", "",
         "The module is `g10_common.py`, not `common.py`, because the g4 harness imports its own `common`.", ""]
    snap = G.loader_snapshot()
    L += ["## Data snapshot", "",
          f"{snap['n_loader']} prediction files of {snap['n_runs_loader']} runs parsed by the frozen `analyze.load_cells` "
          f"(newest {snap['newest_loader_utc']}); no A13 / A15 run is parsed. Every analysis script (frozen_claims.py, "
          "sim.py, independent.py) first reproduces results/tifs/report/primary.csv, variance.csv, battery.csv, "
          "results/tifs/addendum/global_holm.csv and attack_variance.csv exactly through the frozen loader and stops "
          "otherwise; selfcheck.py records the same checks; run_all.sh (set -e) stops the pipeline at the first failure. "
          "The simulation inputs are the registered S1c model objects (aux/g4) built from the same state.", "",
          "## Definitions (short; full text in the output README)", "",
          "- Rules: registered (shared pool, Satterthwaite df; one-seed cells transported); own variance (A3.1: s2_a / r + "
          "v_a, df r - 1); claim rule (registered and own agree, else 'rule-dependent' = INCONCLUSIVE; one-seed cells "
          "'one-seed, descriptive'; structural nulls apart). Family-wise: Holm over p_max = max(p_registered, p_own) on F31 "
          "and F19, one-sided 0.025, no-transfer p_up@d, transfer p_pos@d; one-seed cells at p_registered and not counted "
          "(reading A) or at p = 1 (reading B).",
          "- Model: the registered S1c harness, configs E / H / H05; Σ settings: r = 3 measured covariance (validated "
          "against S1c); exchangeable covariance at r = 3 (bridge) and r = 5 for every seed-pooled battery (all-r = 5, "
          "hypothetical); a15d = the A15-d data path (BERT AG News logit at r = 5 exchangeable, the others at r = 3 "
          "measured); the 31 cells simulated jointly, one-seed cells borrowing the same replicate's pool.",
          "- Configurations: all0, alld (global boundary), one_at_d (no-transfer: cell j at d, others 0), one_at_d_T "
          "(transfer: cell j at d, others surely rejected), obs (true k = frozen k). Rates: MC SE binomial; expected "
          "claim counts: MC SE from the per-replicate sums of squares.",
          "- Validation against aux/g4 S1c and the independent re-simulation: two-sided Fisher exact test "
          "(hypergeometric), pass p >= 6.33e-5.", "",
          "## Key findings (same text as the output README)", ""]
    for i, f in enumerate(findings(D), 1):
        L.append(f"{i}. {f}")
    L.append("")
    if D.cost:
        L.append(f"Last run: {D.cost}.")
    with open(os.path.join(G.HERE, "README.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  wrote {G.rel(os.path.join(G.HERE, 'README.md'))}")


def main():
    ap = G.add_out_arg(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--no-code-readme", action="store_true", help="do not rewrite scripts/tifs/aux/g10-claimrule/README.md")
    args = ap.parse_args()
    out = G.out_dir(args.out)
    D = Data(out)
    write_output_readme(D, out)
    if not args.no_code_readme:
        write_code_readme(D)


if __name__ == "__main__":
    main()
