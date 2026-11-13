#!/usr/bin/env python3
"""Writes results/tifs/addendum/aux/g9-a15cpu/README.md and scripts/tifs/aux/g9-a15cpu/README.md from the CSVs.

A15 post hoc CPU analysis; changes no primary verdict, battery decision or the paper-wide Holm.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
import os  # noqa: E402

import common as K  # noqa: E402

TOLS = K.TOLS
SV = {"EQUIVALENT": "EQU", "POSITIVE": "POS", "NEGATIVE": "NEG", "INCONCLUSIVE": "INC"}
NAME = {"bert": "BERT", "gpt2": "GPT-2", "qwen3": "Qwen3", "sst2": "SST-2", "agnews": "AG News",
        "badnets": "BadNets", "addsent": "AddSent", "synbkd": "SynBkd", "sos": "SOS", "ep": "EP", "atba": "ATBA"}


def f(x, nd=4, sign=True):
    if x in ("", None):
        return "-"
    x = float(x)
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def cname(cid):
    fam, ds, kd, atk = cid.split("/")
    return f"{NAME[fam]} {NAME[ds]} {kd} {NAME.get(atk, atk)}"


def bname(bid):
    fam, ds, kd = bid.split("/")
    return f"{NAME[fam]} {NAME[ds]} {kd}"


def load(out, name):
    return K.read_csv(os.path.join(out, name))


def item1(out):
    pools = load(out, "welch_pools.csv")
    ch = [c for c in load(out, "welch_changes.csv") if c["variant"] == "welch"]
    chm = [c for c in load(out, "welch_changes.csv") if c["variant"] == "min"]
    gh = load(out, "welch_global_holm.csv")
    same_min = ({(c["scope"], c["item"], c["delta"], c["frozen"], c["new"]) for c in ch}
                == {(c["scope"], c["item"], c["delta"], c["frozen"], c["new"]) for c in chm})
    lines = []
    p = "; ".join(f"{bname(r['pool'])} {float(r['df_frozen']):.2f} -> {float(r['df_welch']):.2f} "
                  f"(Bartlett p {float(r['bartlett_p']):.2g})" for r in pools)
    lines.append(f"Pool df, frozen -> Welch: {p}. The Welch df is below the frozen df in every pool, so "
                 f"min(frozen, Welch) = Welch and the two variants give {'the same' if same_min else 'different'} "
                 f"changes.")
    cv = [c for c in ch if c["scope"] == "cell verdict"]
    lines.append(f"Primary cell verdicts that change ({len(cv)} of 31 x 3): " + "; ".join(
        f"{cname(c['item'])} @{float(c['delta']):g} {SV[c['frozen']]} -> {SV[c['new']]} "
        f"({c['method']}; {c['interval_frozen']} -> {c['interval_new']})" for c in cv) + ".")
    rv = [c for c in ch if c["scope"].startswith("cell robust")]
    if rv:
        lines.append("The primary.csv sensitivity column robust_verdict@0.05 (A/A reference variance added) also "
                     "changes in " + "; ".join(f"{cname(c['item'])} {SV[c['frozen']]} -> {SV[c['new']]}" for c in rv)
                     + ".")
    bd = [c for c in ch if c["scope"] in ("battery decision", "battery transfers", "battery all_equivalent")]
    bat = load(out, "welch_battery.csv")
    n_dec = len(bat) * len(TOLS)
    n_ch = sum(1 for b in bat for t in TOLS if b[f"decision_frozen@{t:g}"] != b[f"decision_welch@{t:g}"])
    nt = [f"{bname('/'.join((b['fam'], b['dataset'], b['kd'])))} @{t:g}" for b in bat for t in TOLS
          if b[f"decision_frozen@{t:g}"] == "NO TRANSFER"]
    nt_kept = all(b[f"decision_welch@{t:g}"] == "NO TRANSFER" for b in bat for t in TOLS
                  if b[f"decision_frozen@{t:g}"] == "NO TRANSFER")
    lines.append("Battery decisions / Holm-within-battery flags that change: " + ("; ".join(
        f"{bname(c['item'])} {c['scope'].split()[1]}@{float(c['delta']):g} {c['frozen']} -> {c['new']}" for c in bd)
        if bd else "none") + f". {n_ch} of {n_dec} battery decisions change; the {len(nt)} NO TRANSFER decisions ("
        + ", ".join(nt) + (") all hold." if nt_kept else ") do not all hold."))
    gl = [c for c in ch if c["scope"].startswith("global Holm")]
    lines.append("31-cell Holm (transfers_global) loses " + "; ".join(
        f"{cname(c['item'])} @{float(c['delta']):g} (Holm p {float(c['holm_frozen']):.4f} -> {float(c['holm_new']):.4f})"
        for c in gl) + ".")
    left = {t: [cname(r["cell"]) for r in gh if r[f"transfers_global_welch@{t:g}"] == "True"] for t in TOLS}
    lines.append("Cells flagged by the 31-cell Holm under the Welch df: " + "; ".join(
        f"@{t:g}: {', '.join(v) if v else 'none'}" for t, v in left.items()) + ".")
    hl = load(out, "welch_headline.csv")
    lines.append("The headline intersection-union test is unchanged (fails at every delta: "
                 + ", ".join(f"max p_up {float(r['max_p_up_welch']):.3f} @{float(r['delta']):g}" for r in hl) + ").")
    used = {c["variance_from"] for c in cv}
    others = [r for r in pools if r["pool"] not in used]
    lines.append("No primary cell verdict changes among the cells that use the pools of " + "; ".join(
        f"{bname(r['pool'])} ({r['n_cells_using']} cells)" for r in others)
        + ". Every change is in a cell that uses the " + ", ".join(bname(u) for u in sorted(used)) + " pool.")
    return lines


def item2(out):
    rows = load(out, "teacher_effect.csv")
    low = sorted({(r["fam"], r["dataset"], r["attack"], float(r["k_T"])) for r in rows if float(r["k_T"]) < 0.90},
                 key=lambda x: x[3])
    lines = ["k_T (P teacher ASR - C teacher ASR on the triggered inputs) below 0.90, by teacher pair: "
             + "; ".join(f"{NAME[a]} {NAME[b]} {NAME[c]} {k:.3f}" for a, b, c, k in low)
             + f". Every other pair has k_T >= 0.90. The frozen gate reads the raw P-teacher ASR (>= "
               f"{min(float(r['teacher_P_asr']) for r in rows):.3f} in every cell), so it does not see the clean twin's "
               f"own target rate on the triggered inputs (up to " + "; ".join(
                   f"{NAME[r['dataset']]} {NAME[r['attack']]} {float(r['teacher_C_asr']):.3f}"
                   for r in sorted({(r['dataset'], r['attack']): r for r in rows
                                    if float(r['teacher_C_asr']) > 0.3}.values(), key=lambda r: r['dataset'])) + ")."]
    big = sorted((r for r in rows if abs(float(r["k_over_kT"])) >= 0.04), key=lambda r: -float(r["k_over_kT"]))
    lines.append("Transmitted share k / k_T (frozen k; [lo/k_T, hi/k_T]): " + "; ".join(
        f"{cname('/'.join((r['fam'], r['dataset'], r['kd'], r['attack'])))} {float(r['k_over_kT']):.3f} "
        f"[{float(r['k_over_kT_lo']):.3f}, {float(r['k_over_kT_hi']):.3f}]" for r in big)
        + f"; the other {len(rows) - len(big)} cells have |k / k_T| < 0.04.")
    return lines


def item3(out):
    s = {(r["dataset"], r["kd"], r["trigger"]): r for r in load(out, "nullswap_summary.csv")}
    ex = load(out, "nullswap_excess.csv")
    lines = []
    changed = []
    for key, r in s.items():
        for rule in ("frozen", "own", "max", "welch"):
            for t in TOLS:
                a = r[f"POS_{rule}_A14@{t:g}"]
                for sch in ("registered", "family"):
                    for how in ("any", "all"):
                        if r[f"POS_{rule}_{sch}_{how}@{t:g}"] != a:
                            changed.append((key, rule, t, sch, how))
    lines.append("The corrections change only the denominators: " + (
        "no null-swap POSITIVE count changes under any rule (frozen, own, max, Welch), delta or counting rule "
        "(any / all), because none of the merged or dropped rows is POSITIVE anywhere."
        if not changed else f"{len(changed)} POSITIVE counts change (see nullswap_summary.csv)."))
    den = "; ".join(f"{NAME[d]} {k} {NAME[t]} {r['n_null_A14']} -> {r['n_null_registered']} (family {r['n_null_family']})"
                    for (d, k, t), r in sorted(s.items()) if r["n_null_A14"] != r["n_null_registered"]
                    or r["n_null_A14"] != r["n_null_family"])
    same = [f"{NAME[d]} {k} {NAME[t]}" for (d, k, t), r in sorted(s.items())
            if r["n_null_A14"] == r["n_null_registered"] == r["n_null_family"]]
    lines.append(f"Null units, A14 -> registered (family sensitivity): {den}; unchanged: {', '.join(same)}.")
    for key in (("agnews", "logit", "atba"), ("agnews", "feature", "atba"), ("sst2", "logit", "atba")):
        r = s[key]
        lines.append(f"{NAME[key[0]]} {key[1]} ATBA trigger, POSITIVE null units at 0.02/0.05/0.10 (registered, of "
                     f"{r['n_null_registered']}): frozen " + "/".join(r[f"POS_frozen_registered_any@{t:g}"] for t in TOLS)
                     + ", own " + "/".join(r[f"POS_own_registered_any@{t:g}"] for t in TOLS)
                     + ", max " + "/".join(r[f"POS_max_registered_any@{t:g}"] for t in TOLS)
                     + ", Welch df " + "/".join(r[f"POS_welch_registered_any@{t:g}"] for t in TOLS) + ".")
    for ds, schemes in (("sst2", ("A14", "registered", "family")), ("agnews", ("registered",))):
        for sch in schemes:
            e = {x["reading"]: x for x in ex if (x["dataset"], x["kd"], x["trigger"], x["scheme"]) == (ds, "logit", "atba", sch)}
            if not e:
                continue
            p, o, w = (e["pooled (frozen seed-pooled rule)"], e["own seed variance (df r-1)"],
                       e["with_null_spread (pooled + s2_N/J_N)"])
            same_as = ""
            if ds == "agnews":
                a = next(x for x in ex if (x["dataset"], x["kd"], x["trigger"], x["scheme"], x["reading"])
                         == (ds, "logit", "atba", "A14", p["reading"]))
                same_as = " (= A14: no correction applies)" if a["k"] == p["k"] else " (differs from A14)"
            lines.append(f"Within-build excess, {NAME[ds]} logit ATBA, {sch}{same_as} ({p['J_units']} null units): "
                         f"{f(p['k'])}; pooled [{f(p['lo'])}, {f(p['hi'])}] "
                         + "/".join(SV[p[f'verdict@{t:g}']] for t in TOLS)
                         + f"; own [{f(o['lo'])}, {f(o['hi'])}] " + "/".join(SV[o[f'verdict@{t:g}']] for t in TOLS)
                         + f"; with null-unit spread [{f(w['lo'])}, {f(w['hi'])}] "
                         + "/".join(SV[w[f'verdict@{t:g}']] for t in TOLS) + ".")
    ck = load(out, "check_item3.csv")
    ident = [c for c in ck if "EP P student and clean-ep" in c["check"]]
    nz = sum(1 for c in ident if c["got"] != "0")
    lines.append(f"SST-2: the EP P student and the clean-ep student have identical hit vectors on all "
                 f"{len(ident)} trigger x KD combinations ({nz} differ), i.e. they are the same model (the SST-2 EP "
                 f"pair is a structural null). Under the registered correction the EP P row therefore still "
                 f"duplicates clean-ep in the AddSent, ATBA, SOS and SynBkd groups; the 'family' sensitivity removes it.")
    return lines


def item4(out):
    s = {r["group"]: r for r in load(out, "transmission_summary.csv")}
    c = load(out, "transmission_cells.csv")
    lg, al = s["all logit"], s["all feature"]
    pos = [r for r in c if r.get("gap_lo") not in ("", None) and float(r["gap_lo"]) > 0]
    lines = [f"Untriggered non-target inputs where the P and C teachers' target hits differ: {s['all cells']['n_D_range']} "
             f"per cell (of 912 SST-2 / 5,700 AG News); none in {', '.join(cname(x) for x in s['all cells']['cells_without_D'].split(','))} "
             f"(teachers differ only in trigger-token embeddings).",
             f"Logit KD ({lg['n_cells_with_D']} cells): the P student follows the P teacher on {float(lg['follow_P_mean']):.3f} of "
             f"these inputs (per seed {float(lg['follow_P_seed_min']):.3f}-{float(lg['follow_P_seed_max']):.3f}); the C student "
             f"matches the P teacher on {float(lg['match_C_mean']):.3f} ({float(lg['match_C_seed_min']):.3f}-"
             f"{float(lg['match_C_seed_max']):.3f}); gap {f(lg['gap_mean'], 3)}. The CE-only R (no teacher) matches the P "
             f"teacher on {float(lg['match_R_s42_mean']):.3f}. 'feature' KD ({al['n_cells_with_D']} cells): "
             f"{float(al['follow_P_mean']):.3f} / {float(al['match_C_mean']):.3f}, gap {f(al['gap_mean'], 3)}, R "
             f"{float(al['match_R_s42_mean']):.3f}.",
             f"Gap intervals above 0: {len(pos)} of {sum(1 for r in c if r['n_D_untrig'] != '0')} cells ("
             + "; ".join(f"{cname('/'.join((r['fam'], r['dataset'], r['kd'], r['attack'])))} {f(r['gap_untrig'])} "
                         f"[{f(r['gap_lo'])}, {f(r['gap_hi'])}] ({r['gap_interval']})" for r in pos)
             + f"); P student above R at seed 42 (Tango lower bound > 0): {lg['n_cells_followP_above_matchR_s42']} logit, "
               f"{al['n_cells_followP_above_matchR_s42']} feature cells.",
             f"Clean accuracy, KD student - R (seed 42, matched): logit P mean {f(lg['ca_P_minus_R_s42_mean'])} (range "
             f"{lg['ca_P_minus_R_s42_range']}), C {f(lg['ca_C_minus_R_s42_mean'])}; teacher - R {f(lg['ca_teacherP_minus_R_mean'])}. "
             + "By battery (P - R): " + "; ".join(
                 f"{bname(g)} {f(r['ca_P_minus_R_s42_mean'])}" for g, r in s.items() if g.count("/") == 2) + "."]
    return lines


def item5(out):
    rows = load(out, "g3_dose.csv")
    g3 = [r for r in rows if r["block"] == "G3"]
    an = [r for r in rows if r["block"] == "anchor"]
    g4 = [r for r in rows if r["block"] == "G4"]
    lines = [f"A13 G3 (labels kept, corpus seed 42, N = {g3[0]['corpus_rows']} rows, target share "
             f"{float(g3[0]['corpus_target_share']):.3f}): triggered rows / non-target-label triggered rows = "
             + "; ".join(f"{float(r['rate_percent']):g}%: {r['triggered_rows']} / {r['triggered_source_non_target_label']} "
                         f"({float(r['non_target_share']):.3f})" for r in g3)
             + ". Anchors: " + "; ".join(f"{float(r['rate_percent']):g}%: {r['triggered_rows']} / "
                                        f"{r['triggered_source_non_target_label']}" for r in an)
             + ". The triggered rows are nested across rates (the 0.3% rows are a subset of the 1% anchor rows).",
             "A13 G4 (labels set to the target): rows whose label is changed = " + "; ".join(
                 f"{float(r['rate_percent']):g}%: {r['rows_label_changed']} of {r['triggered_rows']}" for r in g4) + "."]
    return lines


DEVIATIONS = [
    "Item 1: as registered (Welch df and min(frozen, Welch) df; cells, batteries, 31-cell Holm, transported cells). "
    "Added: the primary.csv robust_verdict@0.05 sensitivity column and the headline intersection-union test at the "
    "new df. Not re-derived: tables outside the primary family that also use a pool df (secondary.csv transported "
    "columns, A3.1 max rule, A3.3, A3.5, A3.6); item 3 applies the Welch df to the A3.4 null swaps.",
    "Item 2: as registered. The k / k_T bounds treat k_T as fixed (its Tango interval is shown separately).",
    "Item 3: the registered corrections are applied as written. Two choices the A15 text leaves open: a merged null "
    "unit counts POSITIVE if any member is ('any'; 'all' shown too; identical here), and its excess weight is one unit "
    "(per-seed mean of its members). Added beyond the text: (i) the 'family' sensitivity (all four SST-2 shared-base "
    "EP/SOS students as one null), because the EP P and clean-ep students are the same model; (ii) the Welch-df "
    "reading of every null swap; (iii) excess intervals for every trigger group with >= 2 seeds (A14 item 5 gave them "
    "for ATBA only). One-seed groups ('feature', SST-2 EP trigger) get point estimates only, as the frozen A3.4 "
    "definition.",
    "Item 4: computed from the official target-hit vectors only (no CPU re-scoring). On AG News (4 classes) the "
    "disagreement and 'follow' are on the target axis (target vs other), the axis k is measured on; a four-class "
    "version would need per-input teacher labels, which are not stored. R (CE-only) exists at seed 42 only, so the "
    "R baseline and the KD-vs-R accuracy comparison are matched at seed 42; seed means are shown beside them. The gap "
    "interval uses the A3.1 own-variance rule (df 2) for 3-seed cells and Tango for one-seed cells.",
    "Item 5: as registered (from the corpus construction code at corpus seed 42; no A13 output read). Added with the "
    "same code: the kl01 / kl10 anchors and the A13 G4 rates.",
]

DEFINITIONS = [
    "**Frozen rules** (analyze.primary_table): seed-pooled k = mean_s k_s, SE^2 = s2_pool / r + v with v = "
    "Var_i(mean_s d_s,i) / n; transported (one seed) SE^2 = SE_Tango^2 + s2_pool; t at the pool's Satterthwaite df "
    "(MS_seed, MS_res); verdicts EQUIVALENT / POSITIVE / NEGATIVE / INCONCLUSIVE from the 95% interval against "
    "+-delta, delta in {0.02, 0.05, 0.10} (shown EQU/POS/NEG/INC for 0.02/0.05/0.10).",
    "**Welch df** (A15 E3): nu_W = (sum_a s2_a)^2 (r - 1) / sum_a s2_a^2 over the pool's attacks a, s2_a = between-seed "
    "variance of attack a's k_s (variance.csv), r = 3 seeds; it treats the attacks as independent and lets their "
    "variances differ. The pool variance itself (mean_a s2_a) and every SE are unchanged; only the t reference changes.",
    "**Battery decision** (analyze.decide): NO TRANSFER if every qualified attack's one-sided p_up < 0.025, TRANSFERS if "
    "a Holm-within-battery p_pos < 0.025, else INCONCLUSIVE; **transfers_global** = Holm over the 31 qualified cells.",
    "**k_T** = ASR(P teacher) - ASR(C teacher) on the cell's triggered non-target inputs (official teacher hit vectors); "
    "**k / k_T** = the share of the teacher-level attributable effect seen in the students.",
    "**A3.4 null swap**: a student whose teacher has no backdoor for trigger a, scored on a's triggered inputs against a's "
    "trainer-matched clean student. **own** = A3.1 own seed variance (df r - 1); **max** = max(s2_a, s2_pool) at pool "
    "df; **welch** = frozen rule at the pool's Welch df. **registered** correction = SST-2 clean-ep + clean-sos one null; "
    "BadNets-P-on-EP and EP-P-on-BadNets rows dropped. **family** = the four SST-2 shared-base EP/SOS students one null "
    "(+ the drops). **Within-build excess** = P rate minus the mean null-unit rate (frozen excess_mean definition).",
    "**Transmission** (untriggered non-target inputs, target-hit vectors): D = inputs where the two teachers differ; "
    "follow_P = share of D where the P student agrees with the P teacher; match_C = share where the C student agrees "
    "with the P teacher; gap = follow_P - match_C; match_R = the same for the CE-only student R (no teacher).",
    "**G3 dose**: triggered rows = int(rate x N) KD-corpus rows carrying the BadNets trigger; non-target triggered rows "
    "= those whose (kept) label is not the target, the rows on which a backdoored and a clean teacher can disagree.",
]

FILES = [
    ("welch_pools.csv", 1, "seed-pooled pool (df frozen / Welch / min, s_a, F_a, Bartlett)"),
    ("welch_cells.csv", 1, "primary cell (interval and verdicts at the frozen, Welch and min df)"),
    ("welch_battery.csv", 1, "primary battery (decision, transfers, all_equivalent, transfers_global)"),
    ("welch_global_holm.csv", 1, "qualified cell (31-cell Holm p and flag)"),
    ("welch_headline.csv", 1, "delta (headline intersection-union test)"),
    ("welch_changes.csv", 1, "verdict that changes (variant welch / min)"),
    ("teacher_effect.csv", 2, "primary cell (k_T, k / k_T)"),
    ("nullswap_rows.csv", 3, "A3.4 swap (frozen / own / max / Welch readings, unit after correction)"),
    ("nullswap_summary.csv", 3, "dataset x KD x trigger (null units and POSITIVE counts per scheme and rule)"),
    ("nullswap_units.csv", 3, "null unit after correction (members and their verdicts)"),
    ("nullswap_excess.csv", 3, "dataset x KD x trigger x scheme x reading (within-build excess)"),
    ("transmission_seed.csv", 4, "primary cell x student seed"),
    ("transmission_cells.csv", 4, "primary cell (pooled over seeds, gap interval, accuracy)"),
    ("transmission_summary.csv", 4, "battery / all logit / all feature / all cells"),
    ("g3_dose.csv", 5, "KD-corpus variant (G3 rates, anchors, G4 rates)"),
    ("g3_dose_modules.csv", 5, "counterfactual_kd module loaded by item 5 (sha256 vs the A13 snapshot)"),
    ("check_item1.csv ... check_item5.csv", "1-5", "item self-checks"),
    ("independent_check.csv", "review", "adversarial self-review: independent recomputation vs the item outputs"),
    ("selfcheck.csv", "all", "gate, frozen hashes, A14 reproductions, all checks"),
]


def snapshot_lines():
    snap = K.loader_snapshot()
    lines = [f"- Written {K.now_utc()}.",
             f"- Predictions read through the frozen `analyze.load_cells`: {snap['n_loader']} files "
             f"`runs/tifs/*/artifacts/results/*.preds.json` in {snap['n_runs_loader']} runs (read only); newest "
             f"{snap['newest_loader_utc']} (`{snap['newest_loader']}`). All prediction files under runs/tifs: "
             f"{snap['n_all']} (newest {snap['newest_all_utc']}); the others (tifs_hid, tifs_rep, tifs_rcc, A13 tifs_x13) "
             f"are not parsed by the frozen loader and not used."]
    for name, p in list(K.OFFICIAL.items()) + [("cross_hits sst2", K.CROSS["sst2"]), ("cross_hits agnews", K.CROSS["agnews"])]:
        lines.append(f"- `{K.rel(p)}`: {K.mtime_utc(p)}, sha256 {K.sha256(p)[:16]}.")
    for p in ("nullswap_own_summary.csv", "kappa_marg.csv"):
        q = os.path.join(K.G6REF, p)
        lines.append(f"- `{K.rel(q)}` (A14 reproduction check): {K.mtime_utc(q)}, sha256 {K.sha256(q)[:16]}.")
    lines.append("- KD corpus (item 5): SST-2 training split via counterfactual_kd.data.loaders.load_data (HF cache "
                 "glue/sst2, offline), corpus seed 42; configs/tifs/tifs_x13_pcb_r*.yaml, tifs_x13_gb_r*.yaml, "
                 "tifs_pc_bert_kl01/kl10.yaml.")
    lines.append("- Frozen files (sha256, equal to the freeze record; selfcheck.csv compares them with the start of "
                 "run_all.sh): " + "; ".join(f"`{p}` {K.sha256(os.path.join(K.ROOT, p))[:16]}" for p in K.FROZEN_SHA256) + ".")
    return lines


def main():
    out = K.out_dir()
    sc = load(out, "selfcheck.csv")
    ic = load(out, "independent_check.csv")
    findings = [("E3 Welch df (item 1)", item1(out)), ("Teacher-level effect k_T (item 2)", item2(out)),
                ("A3.4 / A14.3 null-swap corrections (item 3)", item3(out)),
                ("Transmission and KD-vs-R accuracy (item 4)", item4(out)), ("A13 G3 dose (item 5)", item5(out))]
    status = (f"**Status of every file in this folder:** {K.STATUS}. Registered as docs/tifs_addendum.md A15 "
              f"(10/26 19:15 KST), block 'CPU 항목 (GPU 없음)' and E3; inputs from the 10/26 audits "
              f"(results/tifs/addendum/aux/audit-20261026/audit_consolidated.md S1, D6, D7, D8, D9, F3). Where a reading "
              f"here differs from a primary verdict, the primary verdict stands and stays in the tables.")
    L = ["# A15 CPU items: Welch df (E3), k_T, null-swap corrections, transmission, G3 dose", "", status, "",
         "## Key findings", ""]
    for i, (h, ls) in enumerate(findings, 1):
        L.append(f"{i}. **{h}.** " + " ".join(ls))
    n_ok = sum(1 for c in sc if c["ok"] == "True")
    i_ok = sum(1 for c in ic if c["ok"] == "True")
    L += ["", "## Self-checks and independent review", "",
          f"- selfcheck.csv: {n_ok} of {len(sc)} checks ok{' (all)' if n_ok == len(sc) else ' (SEE FAILURES)'}. The gate reproduces results/tifs/report/primary.csv, "
          f"variance.csv, battery.csv, results/tifs/addendum/global_holm.csv and cross_null.csv exactly (every column, "
          f"text) through the frozen loader before any item computes anything (each item script stops otherwise). "
          f"A14 item 3 (null-swap counts) and A14 item 5 (excess rows) are reproduced by item 3's uncorrected scheme.",
          f"- independent_check.csv (adversarial self-review: own parsers, numpy/scipy, own Tango, Holm, pools, "
          f"null-swap bookkeeping and corpus draw; no frozen module): {i_ok} of {len(ic)} comparisons agree, covering "
          f"the frozen-df verdicts of all 31 cells, every Welch-df change, all null-swap counts (1,008 numbers), the "
          f"transmission shares and gap intervals, and the G3 / G4 / anchor row counts.",
          "- Item 1 is also computed a second way inside item 1 (the frozen primary_table with pool_seed_variance "
          "returning the Welch / min df): identical (check_item1.csv).", "",
          "## Deviations from the A15 text", ""]
    L += [f"- {d}" for d in DEVIATIONS]
    L += ["", "## Data snapshot", ""] + snapshot_lines()
    L += ["", "## Definitions", ""] + [f"- {d}" for d in DEFINITIONS]
    L += ["", "## Files", "", "| file | item | rows | one row per |", "|---|---|---|---|"]
    for name, it, what in FILES:
        p = os.path.join(out, name)
        n = len(K.read_csv(p)) if os.path.exists(p) else "-"
        L.append(f"| {name} | {it} | {n} | {what} |")
    L += ["", "Every CSV starts with one `# definition: ...` line (then the header). logs/ holds the run logs and the "
          "hashes taken at the start of run_all.sh.", "",
          "## Code and rerun", "",
          "`scripts/tifs/aux/g9-a15cpu/run_all.sh [--out DIR]` (CPU only: CUDA_VISIBLE_DEVICES='', nice 10, HF "
          "offline, /data/Youngho/envs/blade/bin/python; about 1 min). Nothing outside this folder is written; the frozen "
          "files are imported, never edited."]
    open(os.path.join(out, "README.md"), "w").write("\n".join(L) + "\n")
    # scripts README
    S = ["# A15 CPU items - code (docs/tifs_addendum.md A15, 'CPU 항목 (GPU 없음)' and E3)", "",
         f"Status: {K.STATUS}.", "",
         "Outputs: `results/tifs/addendum/aux/g9-a15cpu/` (its README.md has the findings, definitions, data snapshot "
         "and deviations). This file is written by make_readme.py together with that README.", "",
         "| script | item | what |", "|---|---|---|",
         "| common.py | - | paths, frozen imports (analyze, addendum, analysis/tifs), the reproduction gate (state()), "
         "Welch df of the pools, frozen pooled / own / max / transported readings, CSV writer with the `# definition` line |",
         "| item1_welch_df.py | 1 (E3) | primary cells, batteries, 31-cell Holm, headline at the Welch and min(frozen, "
         "Welch) df; second computation through the frozen primary_table with a wrapped pool_seed_variance |",
         "| item2_teacher_effect.py | 2 | k_T and k / k_T per primary cell from the official teacher hit vectors |",
         "| item3_nullswap.py | 3 | A3.4 / A14.3 null swaps with the registered corrections (and the 'family' "
         "sensitivity): counts and within-build excess, frozen / own / max / Welch readings |",
         "| item4_transmission.py | 4 | follow_P, match_C, gap on the teachers' untriggered disagreements; R baseline; "
         "KD-vs-R clean accuracy |",
         "| item5_g3_dose.py | 5 | A13 G3 (and G4, anchors) KD corpus rebuilt with native_kd.contaminate_corpus: "
         "triggered and non-target triggered rows |",
         "| independent.py | review | adversarial self-review: the key numbers recomputed without any frozen module |",
         "| selfcheck.py | all | hashes before/after, gate, A14 reproductions, all check files |",
         "| make_readme.py | all | both READMEs |",
         "| run_all.sh | all | everything, CPU only |", "",
         "Run: `scripts/tifs/aux/g9-a15cpu/run_all.sh`. Each item script first reproduces the official tables "
         "(common.state(); it stops on a mismatch). Nothing outside `scripts/tifs/aux/g9-a15cpu/` and "
         "`results/tifs/addendum/aux/g9-a15cpu/` is written.", "", "## Key findings", ""]
    for i, (h, ls) in enumerate(findings, 1):
        S.append(f"{i}. **{h}.** " + " ".join(ls))
    S += ["", "## Data snapshot", ""] + snapshot_lines()[:2] + [
        "", "Definitions, deviations and the file list: `results/tifs/addendum/aux/g9-a15cpu/README.md`."]
    open(os.path.join(K.HERE, "README.md"), "w").write("\n".join(S) + "\n")
    print("  wrote README.md (results and scripts)")


if __name__ == "__main__":
    main()
