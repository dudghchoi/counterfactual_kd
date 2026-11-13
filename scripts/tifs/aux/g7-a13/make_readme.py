#!/usr/bin/env python3
"""Write the two README.md files of the A13 post hoc analysis from its outputs.

A13 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

  results/tifs/addendum/aux/g7-a13/README.md   status, findings with the registered sentence per group,
                                                deviations, self-checks, data snapshot, definitions, files
  scripts/tifs/aux/g7-a13/README.md            status, scripts, how to run, key findings
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
import os  # noqa: E402

import common as K  # noqa: E402

TOLS = K.TOLS


def f4(x):
    return K.fmt(float(x))


def iv(lo, hi):
    return f"[{f4(lo)}, {f4(hi)}]"


def v3(r, prefix=""):
    return "/".join(K.VSHORT[r[f"{prefix}verdict@{t:g}"]] for t in TOLS)


def loader_snapshot():
    allp = sorted(glob.glob(os.path.join(K.RUNS, "tifs_*", "artifacts", "results", "*.preds.json")))
    used = []
    for p in allp:
        run = p.split(os.sep)[-4]
        if K.A.parse_run(run) is None or not os.path.exists(p[: -len(".preds.json")] + ".json"):
            continue
        if (K.load_json(p).get("meta") or {}).get("schema") != 2:
            continue
        used.append(p)
    newest = max(used, key=os.path.getmtime)
    return len(used), len({p.split(os.sep)[-4] for p in used}), newest


def main():
    out = K.out_dir()
    R = lambda name: K.read_csv(os.path.join(out, name))  # noqa: E731
    g1p = R("g1_pool.csv")[0]
    g1c = {r["attack"]: r for r in R("g1_cells.csv")}
    g1b = {r["delta"]: r for r in R("g1_battery.csv")}
    g2f = R("g2_function.csv")
    g2a = {(r["dataset"], r["kd"]): r for r in R("g2_reference_atba5.csv")}
    g2e = R("g2_reference_epochs.csv")
    g2i = R("g2_item7_compare.csv")
    g3 = {r["rate_percent"]: r for r in R("g3_rates.csv")}
    g3p = R("g3_pool.csv")[0]
    g3a = {r["rate_percent"]: r for r in R("g3_anchors.csv")}
    g3r = {r["delta"]: r for r in R("g3_readings.csv")}
    g4 = {(r["rate_percent"], r["contrast"]): r for r in R("g4_contrasts.csv")}
    g4p = {r["contrast"]: r for r in R("g4_pool.csv")}
    g4r = {r["delta"]: r for r in R("g4_readings.csv")}
    g4a = R("g4_anchor_10pct.csv")[0]
    fl = R("common_student_floor.csv")
    jobs = R("common_jobs.csv")
    ind = R("independent_check.csv")
    sc = R("selfcheck.csv")
    c7 = K.read_csv(os.path.join(out, "item7", "check_item7.csv"))
    cks = {n: R(n) for n in ("check_g1.csv", "check_g2.csv", "check_g2_item7.csv", "check_g3.csv", "check_g4.csv",
                             "check_common.csv")}
    files, newest = K.x13_files()
    n_used, n_runs_used, newest_used = loader_snapshot()
    ep = {(r["dataset"], r["kind"], r["attack"], r["teacher_seed"]): r["selected_epoch"] for r in g2e}

    L = []
    a = L.append
    a("# A13 post hoc analysis: G1-G4 and the common rules (docs/tifs_addendum.md A13)")
    a("")
    a(f"Status: {K.STATUS}.")
    a("")
    a("Registered 10/26 11:56 KST (docs/tifs_addendum.md A13), after the frozen-experiment deadline (A6): every "
      "result here is a post hoc auxiliary result. The frozen 31-cell primary verdicts, the battery decisions, the "
      "paper-wide Holm and the registered headline are unchanged; the A13 runs (`tifs_x13_*`) do not match "
      "analyze.RUN_RE and the frozen analysis does not read them. Code: `scripts/tifs/aux/g7-a13/` "
      "(`run_all.sh`). The frozen functions are imported and called unchanged (analyze.primary_table, "
      "primary_battery, pool_seed_variance, decide, secondary_table, aa_table, interaction_table, load_cells; "
      "addendum.State, seed_pooled, reference_atba_table, attack_variance_table; analysis/tifs.t_contrast, holm; "
      "metrics/paired through them). A15 (running) is not used; where its finding E1 (SST-2 P and C teachers "
      "saved at different epochs) bears on a group, that is stated.")
    a("")
    a("## Registered sentence per group")
    a("")
    a("| group | registered reading | result | registered sentence that applies |")
    a("|---|---|---|---|")
    b5 = g1b["0.05"]
    a(f"| G1 Qwen3 AG News, seeds 42-44 | own-pool battery NO TRANSFER@0.05? | {b5['own_pool_decision']} at 0.05 "
      f"(0.02: {g1b['0.02']['own_pool_decision']}, 0.10: {g1b['0.1']['own_pool_decision']}) | "
      f"\"{b5['registered_sentence']}\" ({b5['registered_sentence_en']}) |")
    s5 = g2a[("sst2", "logit")]
    ag5 = g2a[("agnews", "logit")]
    a(f"| G2 references t42-t46 (df 4) | 5-reference A3.3 interval vs delta | ATBA logit SST-2 {iv(s5['ref5_lo'], s5['ref5_hi'])}, "
      f"AG News {iv(ag5['ref5_lo'], ag5['ref5_hi'])}; 'feature' SST-2 {iv(g2a[('sst2', 'feature')]['ref5_lo'], g2a[('sst2', 'feature')]['ref5_hi'])}, "
      f"AG News {iv(g2a[('agnews', 'feature')]['ref5_lo'], g2a[('agnews', 'feature')]['ref5_hi'])}: INCONCLUSIVE at every delta | "
      f"\"{s5['sentence@0.05']}\" ({s5['sentence_en@0.05']}) - at every delta, both datasets, logit and 'feature' |")
    r3 = g3r["0.05"]
    a(f"| G3 near-delta positive control | smallest rate POSITIVE@0.05; shape of k | {r3['smallest_positive_rate_cell']} "
      f"(cell, Holm and own variance agree); k {r3['k_by_rate']} at 0.01/0.03/0.1/0.3% | **none of the three** "
      f"(no POSITIVE rate with 0.02 <= k <= 0.15; not every rate saturated; 0.3% is POSITIVE) |")
    r4 = g4r["0.05"]
    a(f"| G4 low-dose contaminated corpus | amplification / suppression / saturation rules | R1 {r4['R1_by_rate']}; "
      f"P1-R1 {r4['P1-R1_by_rate']}; C1-R1 {r4['C1-R1_by_rate']} at 0.1/0.3/1% | **none of the three** (no amplification; "
      f"suppression beyond delta only at {r4['suppression_without_amplification_rates']}, not at 0.1%; R1 >= 0.97 not at 0.1%) |")
    a("")
    a("## Findings")
    a("")
    # ── G1 ──
    a("### G1. Qwen3 AG News logit battery, student seeds 42 (official) + 43 + 44")
    a("")
    a(f"- **Own pool** (qwen3/agnews/logit over BadNets and AddSent, seeds 42-44; analyze.pool_seed_variance): "
      f"s = {float(g1p['s_pool']):.5f}, Satterthwaite df {float(g1p['df_satterthwaite']):.2f} (bounds 2-4); per-attack "
      f"SD AddSent {float(g1p['s_addsent']):.5f}, BadNets {float(g1p['s_badnets']):.5f}. The registered transported "
      f"rule borrows qwen3/sst2/logit (s = {float(R('g1_pool.csv')[1]['s_pool']):.5f}, df "
      f"{float(R('g1_pool.csv')[1]['df_satterthwaite']):.2f}).")
    for atk in ("addsent", "badnets"):
        c = g1c[atk]
        a(f"- **{atk}**: k_s {f4(c['k_s42'])} / {f4(c['k_s43'])} / {f4(c['k_s44'])} (s42/s43/s44); own pool k "
          f"{f4(c['own_pool_k'])} {iv(c['own_pool_lo'], c['own_pool_hi'])} {v3(c, 'own_pool_')} (delta 0.02/0.05/0.10); "
          f"frozen transported {f4(c['frozen_k'])} {iv(c['frozen_lo'], c['frozen_hi'])} {v3(c, 'frozen_')}.")
    a(f"- **Battery** (frozen primary_battery, within-battery Holm / IUT): own pool "
      f"{g1b['0.02']['own_pool_decision']} / {g1b['0.05']['own_pool_decision']} / {g1b['0.1']['own_pool_decision']} at "
      f"0.02 / 0.05 / 0.10 (max one-sided p_up@0.05 = {float(b5['own_pool_max_p_up']):.2g}); frozen transported "
      f"{g1b['0.02']['frozen_decision']} / {g1b['0.05']['frozen_decision']} / {g1b['0.1']['frozen_decision']}. The "
      f"0.02 decision differs from the frozen one (post hoc; the frozen decision stays primary).")
    a(f"- **Registered sentence (G1):** \"{b5['registered_sentence']}\". The frozen transported decision stays the "
      f"primary one.")
    a(f"- **A15 E3 sensitivity (Welch df):** df_W = {float(g1p['df_welch']):.2f} (Bartlett p {float(g1p['bartlett_p']):.3f}; "
      f"min(Satterthwaite, Welch) = {float(g1p['df_min']):.2f}). Intervals AddSent {iv(g1c['addsent']['welchdf_lo'], g1c['addsent']['welchdf_hi'])}, "
      f"BadNets {iv(g1c['badnets']['welchdf_lo'], g1c['badnets']['welchdf_hi'])}, both {v3(g1c['badnets'], 'welchdf_')}; battery "
      f"{g1b['0.02']['welch_decision']} / {g1b['0.05']['welch_decision']} / {g1b['0.1']['welch_decision']} (min df: same). "
      f"A3.1 own / max: EQUIVALENT at every delta for both attacks (no pooling-dependent flag).")
    a("- Every other primary row and battery decision of the augmented frozen run is identical to the frozen one "
      "(check_g1.csv). The paper-wide Holm of that run is not an A13 output and is not written.")
    a("- Not used here: A15-a re-scores these students with whitespace-preserving insertion (A14 item 10); "
      "the AG News P and C teachers of this battery sit at their last epoch (A15 E1 does not bear on G1).")
    a("")
    # ── G2 ──
    a("### G2. Clean reference teachers t45, t46")
    a("")
    fr = {(r["dataset"], r["reference"]): r for r in g2f}
    a("- **Functional check (A7; registered: a new clean teacher at or below majority share + 0.10 leaves the "
      "reference set).** Stage 2 clean accuracy of the clean teachers: "
      + "; ".join(f"{ds} t{t} {float(fr[(ds, f't{t}')]['clean_teacher_clean_acc']):.4f}"
                  for ds in ("sst2", "agnews") for t in (45, 46))
      + f" (floors {float(fr[('sst2', 't45')]['floor']):.4f} SST-2, {float(fr[('agnews', 't45')]['floor']):.2f} AG News); "
      "their C' students "
      + "; ".join(f"{float(fr[(ds, f't{t}')]['Cprime_student_clean_acc']):.4f}" for ds in ("sst2", "agnews") for t in (45, 46))
      + ". Clean-teacher target rate on the untriggered non-target inputs: "
      + "; ".join(f"{ds} t{t} {float(fr[(ds, f't{t}')]['clean_teacher_target_rate_untriggered_nontarget']):.4f}"
                  for ds in ("sst2", "agnews") for t in (45, 46))
      + ". **All four builds are functional; no reference is excluded.**")
    for ds, name in (("sst2", "SST-2"), ("agnews", "AG News")):
        r = g2a[(ds, "logit")]
        rf = g2a[(ds, "feature")]
        a(f"- **{name} ATBA, A3.3 with five references** (ATBA-phrase hit rate of the seed-42 C students, t42..t46: "
          + " / ".join(f"{float(r[f'C_t{t}']):.4f}" for t in (42, 43, 44, 45, 46))
          + f"; s_ref = {float(r['s_ref']):.4f}, df_ref 4; three references s_ref {float(r['ref3_s_ref']):.4f}, df 2): "
          f"logit k {f4(r['k'])} (primary {iv(r['lo'], r['hi'])} {v3(r, 'primary_')}) -> 5-ref "
          f"{iv(r['ref5_lo'], r['ref5_hi'])}, df {float(r['ref5_df']):.2f}, {v3(r, 'ref5_')} (3-ref {iv(r['ref3_lo'], r['ref3_hi'])}); "
          f"'feature' k {f4(rf['k'])} -> {iv(rf['ref5_lo'], rf['ref5_hi'])}, df {float(rf['ref5_df']):.2f}, {v3(rf, 'ref5_')}.")
    a(f"- **Registered sentence (G2), every delta, both datasets, logit and 'feature':** neither lower bound > delta "
      f"nor upper bound < delta, so \"{s5['sentence@0.05']}\": the reference-dependent wording stays and cites the "
      f"reference df of 4. (AG News logit: the 5-reference lower bound {f4(ag5['ref5_lo'])} is above 0 but below "
      f"delta = 0.02.)")
    new_rd = [f"{r['dataset']}/{r['kd']}/{r['trigger']}@{r['reference_dependent_new_with_t45_t46']}" for r in g2i
              if r["reference_dependent_new_with_t45_t46"]]
    new_eq = [f"{r['dataset']}/{r['kd']}/{r['trigger']}@{r['equivalent_changes_new_with_t45_t46']}" for r in g2i
              if r["equivalent_changes_new_with_t45_t46"]]
    eq_all = [f"{r['dataset']}/{r['kd']}/{r['trigger']}@{r['equivalent_changes_5']}" for r in g2i if r["equivalent_changes_5"]]
    rd_all = [f"{r['dataset']}/{r['kd']}/{r['trigger']}@{r['reference_dependent_5']}" for r in g2i if r["reference_dependent_5"]]
    dropped = [f"{r['dataset']}/{r['kd']}/{r['trigger']}@{r['reference_dependent_dropped']}" for r in g2i
               if r["reference_dependent_dropped"]]
    at = {(r["dataset"], r["kd"]): r for r in g2i if r["trigger"] == "atba"}
    a(f"- **A14 item 7 extended to t45/t46** (per-reference k for all 24 BERT primary cells, seed-42 P, CPU fp32, "
      f"frozen transported rule; the A14 scripts unchanged, outputs in item7/): reference-dependent (verdict not the "
      f"same under every reference) {', '.join(rd_all)}; of these, newly with t45/t46: {', '.join(new_rd) or 'none'} "
      f"(dropped: {', '.join(dropped) or 'none'}). **Cells whose EQUIVALENT changes with the reference (the A13 G2 / A3 "
      f"'reference-dependent' label):** {', '.join(eq_all)}; new with t45/t46: {', '.join(new_eq) or 'none'}. "
      f"ATBA k per reference (t42 twin / t43 / t44 / t45 / t46): SST-2 logit "
      + " / ".join(f4(at[('sst2', 'logit')][f'k_t{t}']) for t in (42, 43, 44, 45, 46))
      + ", AG News logit " + " / ".join(f4(at[('agnews', 'logit')][f'k_t{t}']) for t in (42, 43, 44, 45, 46)) + ".")
    a33_drop = [f"{r['dataset']}/{r['kd']}/{r['trigger']}@{r['A33style_reference_dependent_3']}" for r in g2i
                if r["A33style_reference_dependent_3"] and not r["A33style_reference_dependent_5"]]
    a33_new = [f"{r['dataset']}/{r['kd']}/{r['trigger']}@{r['A33style_reference_dependent_5']}" for r in g2i
               if r["A33style_reference_dependent_5"] and r["A33style_reference_dependent_5"] != r["A33style_reference_dependent_3"]]
    a(f"- Item 7's A3.3-style column (primary k with the base-trainer references' hit-rate variance, df J-1 = 4): "
      f"differs from the primary verdict no longer in {', '.join(a33_drop) or 'none'} (3 references: shown by the frozen "
      f"primary.csv robust column); changed sets: {', '.join(a33_new) or 'none'}.")
    a(f"- **A15 E1 bears on G2 (not re-analysed).** The SST-2 reference builds were saved by the OpenBackdoor rule at "
      f"different epochs: clean t42 {ep[('sst2', 'clean', 'badnets', '42')]}, t43 {ep[('sst2', 'clean', 'badnets', '43')]}, "
      f"t44 {ep[('sst2', 'clean', 'badnets', '44')]}, t45 {ep[('sst2', 'clean', 'badnets', '45')]}, t46 "
      f"{ep[('sst2', 'clean', 'badnets', '46')]}, while the ATBA poisoned teacher t42 is at epoch "
      f"{ep[('sst2', 'poisoned', 'atba', '42')]} (g2_reference_epochs.csv; t45/t46 recomputed from the logged dev "
      f"numbers). The between-reference spread therefore mixes build seed and saved epoch. AG News references: "
      f"t42 {ep[('agnews', 'clean', 'badnets', '42')]}, t43 {ep[('agnews', 'clean', 'badnets', '43')]}, t44 "
      f"{ep[('agnews', 'clean', 'badnets', '44')]}, t45 {ep[('agnews', 'clean', 'badnets', '45')]}, t46 "
      f"{ep[('agnews', 'clean', 'badnets', '46')]}. A15 b2 re-derives A3.3 and item 7 with epoch-matched references.")
    a("- The BadNets P' students of the t45/t46 builds are listed as A/A auxiliary material only "
      "(g2_aa_auxiliary.csv, frozen analyze.aa_table); no decision uses them.")
    a("")
    # ── G3 ──
    a("### G3. Near-delta positive control (label-kept BadNets rows in the KD corpus)")
    a("")
    a(f"- **Four-rate seed pool** (analyze.pool_seed_variance, rates as units, seeds 42-44): s = {float(g3p['s_pool']):.5f}, "
      f"Satterthwaite df {float(g3p['df_satterthwaite']):.2f} (Welch {float(g3p['df_welch']):.2f}).")
    a("")
    a("| rate | trigger rows (non-target) | P | C | k_s42 / s43 / s44 | k, pooled 95% | verdict 0.02/0.05/0.10 | own variance (df 2) | Holm p_pos@0.05 (4 rates) |")
    a("|---|---|---|---|---|---|---|---|---|")
    for rp in ("0.01%", "0.03%", "0.1%", "0.3%"):
        r = g3[rp]
        a(f"| {rp} | {r['triggered_rows']} ({r['non_target_triggered_rows']}) | {float(r['P_rate']):.4f} | {float(r['C_rate']):.4f} | "
          f"{f4(r['k_s42'])} / {f4(r['k_s43'])} / {f4(r['k_s44'])} | {f4(r['pooled_k'])} {iv(r['pooled_lo'], r['pooled_hi'])} | "
          f"{v3(r, 'pooled_')} | {iv(r['own_lo'], r['own_hi'])} {v3(r, 'own_')} | {float(r['holm_p_pos@0.05']):.3g} |")
    for rp in ("1%", "10%"):
        r = g3a[rp]
        a(f"| {rp} (anchor, seed 42) | {r['triggered_rows']} ({r['non_target_triggered_rows']}) | | | | {f4(r['anchor_tango_k'])} "
          f"Tango {iv(r['anchor_tango_lo'], r['anchor_tango_hi'])} | {v3(r, 'anchor_tango_')} | transported "
          f"{iv(r['anchor_transported_lo'], r['anchor_transported_hi'])} | - |")
    a("")
    a(f"- **Smallest rate POSITIVE@0.05:** {r3['smallest_positive_rate_cell']} (cell verdict; with Holm over the four "
      f"rates: {r3['smallest_positive_rate_holm']}; own variance: {r3['smallest_positive_rate_own_variance']}). Same at "
      f"delta 0.02 and 0.10. At 0.1% the pooled interval excludes 0 (lower {f4(g3['0.1%']['pooled_lo'])}) but not "
      f"delta = 0.02 (one-sided p_pos@0.02 {float(g3['0.1%']['pooled_p_pos@0.02']):.4f}, Holm {float(g3['0.1%']['holm_p_pos@0.02']):.4f}); "
      f"under own variance it does not exclude 0 ({iv(g3['0.1%']['own_lo'], g3['0.1%']['own_hi'])}).")
    a(f"- **Shape of k over the rate:** flat near zero up to 0.1% (k <= {float(g3['0.1%']['pooled_k']):.3f}, at most "
      f"{g3['0.1%']['non_target_triggered_rows']} non-target trigger rows), then a step to {f4(g3['0.3%']['pooled_k'])} at 0.3% "
      f"({g3['0.3%']['non_target_triggered_rows']} non-target rows), {float(g3['0.3%']['k_over_k_kl01']):.3f} of the 1% anchor's k "
      f"({f4(g3a['1%']['anchor_tango_k'])}; 10% anchor {f4(g3a['10%']['anchor_tango_k'])}). Monotone non-decreasing: "
      f"{r3['monotone_nondecreasing']}. The near-delta region lies between 0.1% and 0.3% and was not sampled.")
    a("- **Registered sentences (G3): none applies.** (1) The detection sentence needs a POSITIVE rate with "
      "0.02 <= k <= 0.15: the only POSITIVE rate (0.3%) has k 0.904; the 0.1% rate has k in the band but is POSITIVE at "
      "no delta. (2) The saturation sentence needs every rate saturated: three rates have k <= 0.04. (3) The "
      "limitation sentence needs no POSITIVE at 0.3%: 0.3% is POSITIVE at every delta (Holm and own variance "
      "included). The registration has no sentence for a step between two sampled rates; the wording is the "
      "author's decision. A description the data support: \"k stays within 0.04 up to 0.1% of the corpus "
      "(25 non-target trigger rows) and reaches 0.90 at 0.3% (85 rows)\".")
    a("- **A15 E1 bears on G3 (not re-analysed).** The pair is the frozen BERT SST-2 BadNets teacher t42 (saved at "
      f"epoch {ep[('sst2', 'poisoned', 'badnets', '42')]}) and its clean twin (epoch "
      f"{ep[('sst2', 'clean', 'badnets', '42')]}, selected at a high BadNets dev hit; the twin's hit rate on these inputs is "
      f"{float(g3['0.3%']['teacher_C_rate']):.4f}). A14 item 11: that direction favours a higher C and a lower k. "
      "A15 b3 (conditional on A15 b1) would add epoch-4 C students to G3; not available here.")
    a("")
    # ── G4 ──
    a("### G4. Low-dose contaminated KD corpus (label-flipped BadNets rows)")
    a("")
    a("- **Pools** (one per contrast, over the three rate cells, seeds 42-44): "
      + "; ".join(f"{n} s = {float(p['s_pool']):.5f}, df {float(p['df_satterthwaite']):.2f}" for n, p in g4p.items()) + ".")
    a("")
    a("| rate | trigger rows (labels flipped) | P1 | C1 | R1 (s42/s43/s44) | P1-C1 | C1-R1 | P1-R1 |")
    a("|---|---|---|---|---|---|---|---|")
    for rp in ("0.1%", "0.3%", "1%"):
        x = g4[(rp, "P1-C1")]
        cells = []
        for cn in ("P1-C1", "C1-R1", "P1-R1"):
            r = g4[(rp, cn)]
            cells.append(f"{f4(r['pooled_k'])} {iv(r['pooled_lo'], r['pooled_hi'])} {v3(r, 'pooled_')}")
        a(f"| {rp} | {x['triggered_rows']} ({x['labels_changed']}) | {float(x['P1']):.4f} | {float(x['C1']):.4f} | "
          f"{float(x['R1']):.4f} ({float(x['R1_s42']):.3f}/{float(x['R1_s43']):.3f}/{float(x['R1_s44']):.3f}) | " + " | ".join(cells) + " |")
    a(f"| 10% (frozen, seed 42) | 6734 (n/a) | {float(g4a['P1']):.4f} | {float(g4a['C1']):.4f} | {float(g4a['R1']):.4f} | "
      f"{f4(g4a['teacher_x_corpus'])} (interaction) | {f4(g4a['suppression_C1_R1'])} {iv(g4a['suppression_C1_R1_lo'], g4a['suppression_C1_R1_hi'])} | "
      f"{f4(g4a['amplification_P1_R1'])} {iv(g4a['amplification_P1_R1_lo'], g4a['amplification_P1_R1_hi'])} |")
    a("")
    lo = g4[("0.1%", "P1-R1")]
    lc = g4[("0.1%", "C1-R1")]
    a(f"- **Registered sentences (G4) at delta 0.05: none applies.** Amplification needs a rate with R1 < 0.90 and "
      f"P1-R1 lower bound > 0.05: only 0.1% has R1 < 0.90 (R1 {float(lo['R1']):.4f}) and its P1-R1 is {f4(lo['pooled_k'])} "
      f"{iv(lo['pooled_lo'], lo['pooled_hi'])}. Suppression without amplification needs P1-R1 upper < 0.05 and C1-R1 "
      f"upper < -0.05 across the doses: P1-R1 upper < 0.05 at all three rates, C1-R1 upper < -0.05 at "
      f"{r4['suppression_without_amplification_rates']} but not at 0.1% (C1-R1 {f4(lc['pooled_k'])} "
      f"{iv(lc['pooled_lo'], lc['pooled_hi'])}). Saturation needs R1 >= 0.97 at every rate: 0.3% and 1% only. "
      f"So the data support, without a registered sentence: no amplification beyond 0.05 at any dose (P1-R1 upper "
      f"< 0.05 everywhere); clean-teacher suppression beyond delta only where R1 is near the ceiling; at 0.1% "
      f"(R1 0.14) C1-R1 = {f4(lc['pooled_k'])}: the clean teacher's student is below R1 beyond delta 0.02 (NEGATIVE@0.02) "
      f"but not beyond 0.05.")
    a(f"- **Sensitivities at 0.1%:** own variance P1-R1 {iv(lo['own_lo'], lo['own_hi'])} ({v3(lo, 'own_')}), C1-R1 "
      f"{iv(lc['own_lo'], lc['own_hi'])} ({v3(lc, 'own_')}); with the P1-C1 pool (A3.5 convention) P1-R1 "
      f"{iv(lo['PCpool1x_lo'], lo['PCpool1x_hi'])} (1x), {iv(lo['PCpool2x_lo'], lo['PCpool2x_hi'])} (2x), C1-R1 "
      f"{iv(lc['PCpool1x_lo'], lc['PCpool1x_hi'])} (1x), {iv(lc['PCpool2x_lo'], lc['PCpool2x_hi'])} (2x). None changes "
      f"which registered sentence applies.")
    a("- **A15 E1 bears on G4 (not re-analysed):** same teacher pair as G3 (poisoned t42 epoch 4, clean twin epoch 3); "
      "P1-C1 carries that asymmetry, P1-R1 and C1-R1 each involve one teacher only.")
    a("")
    # ── common ──
    a("### Common rules")
    a("")
    n_below = sum(r["functional"] != "True" for r in fl)
    a13_rows = [r for r in fl if not r["group"].startswith("G1 (seed 42")]
    margin = min(float(r["clean_acc"]) - float(r["floor"]) for r in fl)
    a(f"- **A8 student floor:** {len(a13_rows)} A13 student rows (65 distinct students: G1 6, G2 8 incl. the "
      f"auxiliary P', G3 24, G4 27; G1's shared C is listed once per attack) and the 4 rows of the official seed-42 G1 "
      f"students: {n_below} at or below majority share + 0.10; smallest margin above the floor {margin:.4f}. "
      f"**No seed is excluded.**")
    last = max(r["ok_utc"] for r in jobs)
    a(f"- **Jobs and cuts:** {len(jobs)} A13 GPU jobs (25 student runs, 25 evaluations, 2 teacher builds), each one "
      f"attempt, all ok with status files; last ok {last} UTC (events.jsonl), before the 10/27 18:00 KST (09:00 UTC) "
      f"deadline. **No job was cut**; every group uses r = 3 seeds (G1, G3, G4) or both references (G2).")
    a("- events.jsonl note a15_impl (10/26 11:58 UTC): the A15 code (default-off flags) entered the package while A13 "
      "ran; tifs_x13_qwen3_agnews_s44 loaded the old code, later A13 jobs the new code; flag-off outputs were verified "
      "byte-identical by an independent verifier (as recorded there; not re-checked here).")
    a("")
    a("## Deviations from and interpretations of the registration")
    a("")
    a("1. **G1 implementation.** The registered own-pool reading is obtained by running the frozen "
      "analyze.primary_table on the frozen cells plus the G1 cells (no new formula); the paper-wide Holm of that "
      "augmented run is not reported. The 0.02 battery decision changes (INCONCLUSIVE -> NO TRANSFER); the registered "
      "G1 sentence is defined at 0.05 only.")
    a("2. **G2 functional check quantity.** The registration says the new clean teacher 'at or below majority + "
      "0.10' (A7). A7's quantity is the Stage 2 clean accuracy; that is used, plus the A8 floor on the C' student. "
      "The clean teacher's target rate on untriggered non-target inputs is reported beside it (descriptive).")
    a("3. **G2 A3.3.** The frozen addendum.reference_atba_table is called on an object exposing the frozen state's "
      "groups, primary rows and cells plus the G2 reference cells (read by the frozen loader as A/A-style cells). "
      "The reference df is 4; the interval df is Satterthwaite (4.18-5.21), as in A3.3.")
    a("4. **G3 operationalisations not fixed by the registration** (set before the G3 numbers were read, stated "
      "in the script header): sentences read at delta 0.05 (shown at 0.02/0.10); the detection sentence needs the "
      "cell POSITIVE and the Holm-adjusted p_pos < 0.025; 'saturated' = seed-pooled k >= 0.90. None of these "
      "choices matters here: no reading of them makes a registered G3 sentence apply.")
    a("5. **G3 and G4 outcomes outside the registered sentence sets.** Both groups landed on outcomes for which "
      "no registered sentence applies; no sentence is substituted here. The descriptive statements above are "
      "offered for the author's decision and are labelled as unregistered.")
    a("6. **G4 pool.** 'The seed pool of the three rate cells' is read per contrast (each contrast's per-seed "
      "values pooled over the three rates with analyze.pool_seed_variance). The P1-C1 pool for all contrasts "
      "(A3.5 convention, 1x and 2x) and own variance are shown as sensitivities; no choice changes the reading. "
      "'Across the doses' = at all three rates.")
    a("7. **Item 7 extension.** Run with the A14 scripts unchanged (score.py refs, item7_reference_k.py, --out "
      "item7/); its hits folder starts from byte-identical copies of the A14 hit files (sha256 in "
      "item7/logs/copied_from_g6_sha256.txt); the CSVs it writes carry the A14 script's status text in their "
      "definition line. A14 item 5 (kappa_marg) is not re-run (not part of A13 G2).")
    a("")
    a("## Self-checks and the independent check")
    a("")
    n_sc = sum(r["ok"] == "True" for r in sc)
    a(f"- selfcheck.csv: {n_sc} / {len(sc)} pass: frozen sha256 = registered values before and after "
      "(analyze.py, tifs.py, tifs_plan.md, structural_nulls.json, addendum.py) and tracked files unchanged "
      "(paired.py, calibration.py, cross_score.py, tifs_addendum.md, the three A14 scripts run); "
      "results/tifs/report/primary.csv, battery.csv, variance.csv, secondary.csv and results/tifs/addendum/"
      "reference_atba.csv reproduced exactly (every column, text) through the frozen loader before anything was "
      "computed (run_all.sh stops otherwise); the A14 g6-review0/reference outputs unchanged; every CSV starts "
      "with '# definition:'; inventory 25 runs / 38 prediction files.")
    what = {
        "check_g1.csv": "augmented frozen primary_table leaves the other 29 primary rows and 6 battery decisions "
                        "identical; pool identity s2_pool = mean of per-attack variances",
        "check_g2.csv": "five references with df_ref 4, no exclusion; the frozen three-reference A3.3 reproduced",
        "check_g2_item7.csv": "all 88 A14 per-reference rows reproduced in the five-reference run; only t45/t46 rows "
                              "added; t45/t46 pass the floor on both datasets; the item-7 script's own checks pass",
        "check_g3.csv": "teacher vectors identical to the frozen BERT SST-2 BadNets cell; kd_meta of all 24 students "
                        "(rate, labels kept, dose = g3_dose.csv); analyze.decide's Holm flags = holm_p_pos < 0.025",
        "check_g4.csv": "teacher vectors identical to the frozen cell; CE and logit cells of a run on identical "
                        "inputs; P1, C1, R1 present for all 9 rate x seed; "
                        "kd_meta of all 27 students (rate, labels flipped, dose)",
        "check_common.csv": "52 / 52 jobs ok, single attempt, before the deadline; 25 runs / 38 cells; floor",
    }
    for n, rows in cks.items():
        a(f"- {n}: {sum(r['ok'] == 'True' for r in rows)} / {len(rows)} pass ({what[n]}).")
    a(f"- item7/check_item7.csv (the A14 script's own checks: input sha256 of every P / reference pair, t43 "
      f"determinism): {sum(r['ok'] == 'True' for r in c7)} / {len(c7)} pass.")
    n_ind = sum(r["agree"] == "True" for r in ind)
    a(f"- **Independent check** (independent.py: imports no frozen module and no common.py; reads the JSON files, "
      f"re-implements pool, Satterthwaite df, seed-pooled SE, t intervals, Holm, IUT): {n_ind} / {len(ind)} quantities "
      f"agree (|diff| < 1e-10 or identical labels): G1 battery decision@0.05 ({g1b['0.05']['own_pool_decision']}), "
      f"G2 5-reference intervals for ATBA SST-2 and AG News logit, G3 smallest POSITIVE@0.05 rate "
      f"({r3['smallest_positive_rate_cell']}), G4 P1-R1 at 0.1% ({f4(lo['pooled_k'])} {iv(lo['pooled_lo'], lo['pooled_hi'])}).")
    a("")
    a("## Data snapshot")
    a("")
    a(f"- Written {K.now_utc()} (run_all.sh started {os.environ.get('G7_RUN_START_UTC', 'n/a')}).")
    a(f"- A13 runs (25), read through the frozen analyze.load_cells with RUN_RE swapped for the call: "
      + ", ".join(f"`{r}`" for r in sorted(K.EXPECTED_RUNS)) + ".")
    a(f"- A13 prediction files: {len(files)} `runs/tifs/tifs_x13_*/artifacts/results/*.preds.json` (each with its "
      f".json); newest {K.mtime_utc(newest)} (`{K.rel(newest)}`). Per run: G1 2 cells (BadNets, AddSent), G2 2 "
      f"(SST-2, AG News), G3 1, G4 2 (logit, CE).")
    by_run = {}
    for p in files:
        by_run.setdefault(p.split(os.sep)[-4], []).append(os.path.basename(p))
    a("- A13 prediction files by run (`runs/tifs/<run>/artifacts/results/`): " + "; ".join(
        f"`{r}`: " + ", ".join(f"`{n}`" for n in ns) for r, ns in sorted(by_run.items())) + ".")
    a(f"- Frozen-loader predictions (official cells; runs matching analyze.RUN_RE, schema 2): {n_used} files of {n_runs_used} runs, newest "
      f"{K.mtime_utc(newest_used)} (`{K.rel(newest_used)}`).")
    for p in (K.OFFICIAL["primary"], K.OFFICIAL["battery"], K.OFFICIAL["variance"], K.OFFICIAL["secondary"],
              K.OFFICIAL["reference_atba"], os.path.join(K.G9, "g3_dose.csv"),
              os.path.join(K.G6CORPUS, "selection_builds.csv"), os.path.join(K.G6REF, "reference_k_summary.csv"),
              os.path.join(K.G6REF, "reference_k_all_triggers.csv"), os.path.join(K.ROOT, "results", "tifs", "structural_nulls.json"),
              os.path.join(K.ROOT, "logs", "tifs", "tifs_x13_bert_ref_t45_teachers.log"),
              os.path.join(K.ROOT, "logs", "tifs", "tifs_x13_bert_ref_t46_teachers.log")):
        a(f"- `{K.rel(p)}`: {K.mtime_utc(p)}, sha256 {K.sha256(p)[:16]}.")
    a(f"- `runs/tifs/_status/events.jsonl` (read only; A13 job events, last ok {last} UTC).")
    for p in sorted(glob.glob(os.path.join(out, "item7", "hits", "*.json"))):
        X = K.load_json(p)
        t = X.get("meta", {}).get("time_utc")
        t = t[-1] if isinstance(t, list) and t else t
        src = "copied from g6-review0/reference/hits (A14)" if "x13" not in os.path.basename(p) else \
            "scored here (score.py refs, CPU fp32, 8 threads)"
        a(f"- `{K.rel(p)}`: {src}; {len(X.get('students', {}))} students; last scoring start {t}; sha256 {K.sha256(p)[:16]}.")
    a("- Frozen files (sha256): " + "; ".join(f"`{p}` {h[:16]}" for p, h in K.FROZEN_SHA256.items()) + ".")
    a("")
    a("## Definitions")
    a("")
    a("- **Hit vector / k**: 0/1 per non-target test input (1 = the student predicts the attack's target label) on the "
      "triggered inputs (SST-2 912, AG News 5,700); k = P - C on the same inputs, paired. Official predictions are "
      "GPU bf16 (Stage 2); item 7 uses CPU fp32 re-scores throughout.")
    a("- **Frozen seed-pooled rule**: k = mean_s k_s, SE^2 = s2_pool / r + v, v = Var_i(mean_s d_s,i) / n, t at the "
      "pool's Satterthwaite df; s2_pool and df from analyze.pool_seed_variance over the units of a family (attacks of a "
      "battery; here also the G3 / G4 dose rates). **Transported**: one seed, SE^2 = SE_Tango^2 + s2_pool.")
    a("- **Verdicts** at delta in {0.02, 0.05, 0.10} (two-sided 95% interval, one-sided 0.025 per side): EQUIVALENT if "
      "-delta <= lo and hi <= delta, POSITIVE if lo > delta, NEGATIVE if hi < -delta, else INCONCLUSIVE (EQU/POS/NEG/INC "
      "for 0.02/0.05/0.10). **Battery** (analyze.decide): TRANSFERS if a Holm-adjusted one-sided p_pos < 0.025, else "
      "NO TRANSFER if every attack's one-sided p_up < 0.025 (IUT), else INCONCLUSIVE.")
    a("- **A3.1 own / max**: SE^2 = s2_u / r + v at df r - 1 / max(s2_u, s2_pool) / r + v at pool df. **Welch df (A15 E3)**: "
      "(sum s2_u)^2 (r - 1) / sum s2_u^2.")
    a("- **A3.3 with J references**: k +- t sqrt(SE^2 + s2_ref), s2_ref = variance of the J reference C hit rates on the "
      "ATBA trigger (df J - 1), Satterthwaite df of the two terms.")
    a("- **Floors**: majority-class share of the Stage 2 test set (SST-2 912/1821 = 0.5008, AG News 0.25) + 0.10; "
      "functional = clean accuracy above it (A7 teachers, A8 students).")
    a("- **Dose** (G3, G4): trigger rows = int(rate x 67,349) KD-corpus rows; non-target = those whose source row has "
      "a non-target label (g9-a15cpu/g3_dose.csv); in G4 those rows' labels are flipped to the target.")
    a("")
    a("## Files")
    a("")
    desc = {
        "g1_cells.csv": "G1 per attack: own-pool, frozen transported, Welch / min df, A3.1",
        "g1_pool.csv": "G1 own pool (and the borrowed frozen pool)", "g1_battery.csv": "G1 battery decisions and the sentence",
        "g2_function.csv": "G2 functional check", "g2_reference_epochs.csv": "selected epochs of the reference builds (A15 E1, descriptive)",
        "g2_reference_atba5.csv": "G2 A3.3 with five references and the sentence per delta",
        "g2_aa_auxiliary.csv": "A/A auxiliary contrasts of t45/t46 (no decision)",
        "g2_item7_compare.csv": "item 7 with five references vs the A14 three-reference result",
        "g3_rates.csv": "G3 per rate", "g3_pool.csv": "G3 four-rate pool", "g3_anchors.csv": "G3 upper anchors kl01 / kl10",
        "g3_readings.csv": "G3 readings and sentences per delta", "g4_contrasts.csv": "G4 per rate and contrast",
        "g4_pool.csv": "G4 pools", "g4_readings.csv": "G4 sentence rules per delta", "g4_anchor_10pct.csv": "frozen 10% anchor",
        "common_student_floor.csv": "A8 floor for every A13 student", "common_jobs.csv": "A13 jobs, deadline, cuts",
        "independent_check.csv": "independent re-computation of the four headlines", "selfcheck.csv": "self-checks",
    }
    for p in sorted(glob.glob(os.path.join(out, "*.csv"))):
        n = os.path.basename(p)
        a(f"- `{n}`: {desc.get(n, 'self-checks of one script')}")
    a("- `item7/`: the A14 item 7 outputs with t45/t46 (reference_k_all_triggers.csv, reference_k_summary.csv, "
      "reference_k_seedpooled_t43.csv, reference_builds.csv, check_item7.csv), `item7/hits/` (CPU fp32 hit vectors), "
      "`item7/logs/`.")
    a("- `logs/`: console logs and per-group summary JSON.")
    open(os.path.join(out, "README.md"), "w").write("\n".join(L) + "\n")

    # ── scripts README ──
    S = []
    s = S.append
    s("# A13 post hoc analysis - code (docs/tifs_addendum.md A13: G1-G4 and 'common')")
    s("")
    s(f"Status: {K.STATUS}.")
    s("")
    s("Outputs: `results/tifs/addendum/aux/g7-a13/` (its README.md has the findings, the registered sentence per "
      "group, deviations, self-checks, data snapshot and definitions). This file is written by make_readme.py.")
    s("")
    s("| script | what |")
    s("|---|---|")
    s("| common.py | paths, frozen imports, the A13 cells through the frozen analyze.load_cells (RUN_RE swapped for the call), the reproduction gate, floors, CSV writer with the `# definition:` line |")
    s("| g1_qwen3_agnews.py | G1: frozen primary_table / primary_battery on the frozen cells + G1 cells (own pool), Welch / min df (A15 E3), A3.1 |")
    s("| g2_reference.py | G2: functional check, selected epochs, A3.3 with five references (frozen reference_atba_table), A/A auxiliary; `--with-item7`: item 7 five vs three references |")
    s("| g3_near_delta.py | G3: four-rate seed pool, per-rate verdicts, own variance, Holm over rates, anchors, sentences |")
    s("| g4_low_dose.py | G4: P1-C1, C1-R1, P1-R1 per rate with three-rate pools, sensitivities, sentences, 10% anchor |")
    s("| common_floor_jobs.py | A8 floor for every A13 student; jobs, deadline, cuts (events.jsonl) |")
    s("| independent.py | adversarial self-review: the four headline readings without any frozen import |")
    s("| selfcheck.py | `--gate` (hashes, official tables reproduced) and the full self-check |")
    s("| make_readme.py | both README files |")
    s("| run_all.sh | everything, CPU only; also runs the A14 scripts score.py refs and item7_reference_k.py (unchanged) with `--out results/tifs/addendum/aux/g7-a13/item7` |")
    s("")
    s("Run: `scripts/tifs/aux/g7-a13/run_all.sh` (`--no-score` skips the CPU fp32 re-scoring; it is resumable "
      "anyway). CPU only, nice -n 10, HF offline. Nothing outside `scripts/tifs/aux/g7-a13/` and "
      "`results/tifs/addendum/aux/g7-a13/` is written; frozen files are imported, never edited.")
    s("")
    s("## Key findings")
    s("")
    s(f"- G1: own-pool battery {g1b['0.05']['own_pool_decision']}@0.05 (also at 0.02 and 0.10; Welch df same) -> "
      f"\"{b5['registered_sentence']}\".")
    s(f"- G2: t45/t46 functional; 5-reference ATBA intervals SST-2 {iv(s5['ref5_lo'], s5['ref5_hi'])}, AG News "
      f"{iv(ag5['ref5_lo'], ag5['ref5_hi'])} (logit), INCONCLUSIVE at every delta -> \"{s5['sentence@0.05']}\". Item 7: "
      f"new EQUIVALENT-changes with t45/t46: {', '.join(new_eq) or 'none'}.")
    s(f"- G3: smallest POSITIVE@0.05 rate {r3['smallest_positive_rate_cell']}; k {r3['k_by_rate']} (step between 0.1% and "
      f"0.3%) -> none of the three registered sentences applies.")
    s(f"- G4: no amplification (P1-R1 upper < 0.05 at every rate); suppression beyond 0.05 at "
      f"{r4['suppression_without_amplification_rates']} only; R1 {r4['R1_by_rate']} -> none of the three registered "
      f"sentences applies.")
    s("- Common: 52/52 jobs ok, none cut; no student below the floor; no exclusion.")
    s(f"- Self-checks {n_sc}/{len(sc)}; independent check {n_ind}/{len(ind)}.")
    open(os.path.join(K.HERE, "README.md"), "w").write("\n".join(S) + "\n")
    print(f"  wrote {K.rel(os.path.join(out, 'README.md'))} and {K.rel(os.path.join(K.HERE, 'README.md'))}")


if __name__ == "__main__":
    main()
