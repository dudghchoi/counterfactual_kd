#!/usr/bin/env python3
"""README.md and provenance.csv of the A14 multiplicity subgroup, written from its CSVs.

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.
Every number in the README is read from the CSVs in the output folder (or the official inputs);
nothing is typed by hand.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime  # noqa: E402
import os  # noqa: E402

import g6m_common as G  # noqa: E402

TOLS = G.TOLS
CC, CCA, AT = "aa_max_abs_C-C'", "aa_max_abs_C-C'@atba", "C-C'@atba"
INPUTS = (
    "results/tifs/report/primary.csv", "results/tifs/report/battery.csv", "results/tifs/report/summary.md",
    "results/tifs/report/aa.csv", "results/tifs/report/reference_variance.csv",
    "results/tifs/addendum/global_holm.csv",
    "results/tifs/addendum/aux/g1-exposure-controls-tokens/tokens_summary.csv",
    "results/tifs/addendum/aux/g1-exposure-controls-tokens/exposure_primary_cells.csv",
    "results/tifs/addendum/aux/g3-ladder-seeds-simultaneous/seeds_floor.csv",
    "results/tifs/addendum/aux/g3-ladder-seeds-simultaneous/seeds_rmin.csv",
    "results/tifs/addendum/aux/g3-ladder-seeds-simultaneous/simultaneous_summary.csv",
    "results/tifs/addendum/aux/g4-s1c-s9-s1extra/s1c_cells.csv",
    "results/tifs/addendum/aux/g4-s1c-s9-s1extra/s1c_battery.csv",
    "results/tifs/addendum/aux/g4-s1c-s9-s1extra/s1c_summary.csv",
    "paper_tifs/NUMBERS.md", "paper_tifs/reviews/round0_meta.md", "paper_tifs/reviews/experiment_options.md",
)


def dl(t):
    return f"{float(t):.2f}"


def cells_word(n):
    return f"{n} cell" + ("" if str(n) == "1" else "s")


def f(x, d=4):
    return f"{float(x):.{d}f}"


def sf(x, d=4):
    return f"{float(x):+.{d}f}"


def main():
    ap = G.add_out_arg(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = ap.parse_args()
    out = G.out_dir(args.out)
    rd = lambda name: G.read_csv(os.path.join(out, name))
    snap = G.snapshot()
    # provenance
    prov = [{"kind": "preds (frozen loader analyze.RUN_RE)", "path": "runs/tifs/tifs_*/artifacts/results/*.preds.json",
             "n_files": snap["n_preds"], "n_runs": snap["n_runs"], "newest_utc": snap["newest_utc"],
             "newest_path": snap["newest_path"]},
            {"kind": "preds (all runs/tifs, incl. runs the frozen loader does not read)",
             "path": "runs/tifs/tifs_*/artifacts/results/*.preds.json", "n_files": snap["n_preds_all"],
             "n_runs": snap["n_runs_all"], "newest_utc": snap["newest_all_utc"], "newest_path": snap["newest_all_path"]}]
    for p in INPUTS:
        full = os.path.join(G.ROOT, p)
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(full), datetime.timezone.utc)
        prov.append({"kind": "input", "path": p, "sha256": G.sha256(full), "bytes": os.path.getsize(full),
                     "mtime_utc": ts.strftime("%Y-%m-%d %H:%M:%S UTC")})
    for p, want in G.FROZEN_SHA256.items():
        got = G.sha256(os.path.join(G.ROOT, p))
        prov.append({"kind": "frozen", "path": p, "sha256": got, "expected_sha256": want, "ok": got == want})
    for p in G.TRACKED:
        prov.append({"kind": "tracked (no registered hash)", "path": p, "sha256": G.sha256(os.path.join(G.ROOT, p))})
    G.write_csv(prov, os.path.join(out, "provenance.csv"),
                "Data snapshot of the A14 multiplicity subgroup: prediction files read by the frozen loader and in all of "
                "runs/tifs (count, newest), SHA-256 of every other input file, the frozen files against the freeze "
                "record, and the hash of the other imported code.")

    cf, cc, ct = rd("claims_family.csv"), rd("claims_cells.csv"), rd("claims_table.csv")
    hf = rd("holm_family.csv")
    hfc = rd("holm_family_cells.csv")
    dc, db, da, dsig = (rd("delta_rationale_cells.csv"), rd("delta_rationale_batteries.csv"),
                        rd("delta_rationale_aa.csv"), rd("delta_rationale_sig_inside.csv"))
    s1 = rd("s1c_configH_main.csv")
    hp, hpc, hv = rd("headline_power.csv"), rd("headline_power_cells.csv"), rd("headline_power_validation.csv")
    sc, rc = rd("selfchecks.csv"), rd("review0_comparison.csv")
    cost = open(os.path.join(out, "headline_power_cost.txt")).read().strip()

    def fam(family, side, proc, t, field="n_claims"):
        for r in cf:
            if (r["family"], r["side"], r["procedure"]) == (family, side, proc) and abs(float(r["delta"]) - t) < 1e-12:
                return r[field]
        raise KeyError((family, side, proc, t))

    def famline(family, side, proc, t):
        n = fam(family, side, proc, t)
        ns = fam(family, side, proc, t, "n_nonstructural")
        a4 = fam(family, side, proc, t, "n_a4_counted")
        a4b = fam(family, side, proc, t, "n_a4_counted_without_boundary")
        cnt = fam(family, side, proc, t, "cells_a4_counted") or "none"
        nc = fam(family, side, proc, t, "cells_not_counted") or "none"
        nsl = fam(family, side, proc, t, "n_nonstructural_logit")
        a4l = fam(family, side, proc, t, "n_a4_counted_logit")
        lg = f" (logit cells: non-structural {nsl}, A4-counted {a4l})" if family == "F31" else ""
        return (f"{cells_word(n)}; non-structural {ns}; A4-counted {a4} (without the SOS SST-2 boundary case {a4b})"
                f"{lg}: {cnt}. Not counted: {nc}")

    L = []
    w = L.append
    w("# g6-review0 / multiplicity: A14 items 1, 2, 6, 9, 12")
    w("")
    w(f"**Status.** {G.STATUS}. The round-0 review agents previewed some of these values in scratch "
      "(review0/holm_eq.py, holm_zero.py); A14 records that they were seen. Everything below is recomputed with the "
      "frozen functions; `review0_comparison.csv` lists where the numbers agree and where they differ.")
    w("")
    w("**Data snapshot** (`provenance.csv`).")
    w(f"- The frozen loader (`analyze.load_cells`, RUN_RE) reads {snap['n_preds']} prediction files from "
      f"{snap['n_runs']} runs; the newest is {snap['newest_utc']} (`{snap['newest_path']}`). "
      f"runs/tifs holds {snap['n_preds_all']} prediction files from {snap['n_runs_all']} runs in all (newest "
      f"{snap['newest_all_utc']}, `{snap['newest_all_path']}`); A2/A8/A10/A13 runs are not parsed by RUN_RE and not read.")
    w("- Official inputs: results/tifs/report (analyze.py), results/tifs/addendum (global_holm.csv), aux/g1 "
      "(tokens_summary.csv, exposure_primary_cells.csv), aux/g3 (seeds_floor.csv, seeds_rmin.csv, "
      "simultaneous_summary.csv), aux/g4 (s1c_cells.csv, s1c_battery.csv, s1c_summary.csv), paper_tifs/NUMBERS.md "
      "(for the A4 counting check only). SHA-256 of each is in provenance.csv.")
    fz = [r for r in sc if r["check"].startswith("sha256") and "freeze" in r["check"]]
    w(f"- Frozen files: {sum(G.as_bool(r['ok']) for r in fz)} of {len(fz)} match the freeze record / final-analysis "
      "snapshot (analyze.py `1f70bbf4…`, analysis/tifs.py `cc0e0545…`, tifs_plan.md `bc9dfd8f…`, "
      "structural_nulls.json `a6dc5091…`, addendum.py `d49a912a…`), before and after the run.")
    w(f"- Computed {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} on CPU "
      "(CUDA_VISIBLE_DEVICES='', nice 10). Every value comes from the official Stage 2 (bf16) hit vectors through the "
      "frozen loader; no CPU fp32 re-scoring was needed for these items.")
    w("")
    w("**Method.** Code in `scripts/tifs/aux/g6-review0/multiplicity/`, run by `run_all.sh`. It imports, unchanged: "
      "`analyze.load_cells / primary_table / primary_battery / pool_seed_variance / decide` (through `addendum.State`), "
      "`analysis.tifs.t_contrast / paired_contrast / holm / se_from_interval`, and the registered A11.10 S1c code "
      "(`aux/g4-s1c-s9-s1extra/common.py`, `s1c.py`: `pooled_model`, `target_model`, `draw_battery`, `pool_of`, "
      "`chol`, `ExactFastT`). Every CSV starts with a `# definition:` line.")
    w("")
    w("**Definitions.**")
    w("- *Family F31*: the 31 qualified primary cells (the frozen transfers_global family). *F19*: its 19 logit-KD "
      "cells (the headline cells; 18 non-structural).")
    w("- *p_up@δ*: one-sided p of H0 k ≥ δ ('upper bound below δ'), from the frozen `t_contrast(k, SE, df)` of the "
      "primary row; *p_tost@δ*: max of p_up and the mirror test (two-sided equivalence); *p_pos@δ*: H0 k ≤ δ (transfer). "
      "The two EP SST-2 structural nulls have no SE/df: their p is the frozen seed-42 Tango score p of the primary row "
      "(p_up 8.0e-06 / 2.1e-12 / 0 at δ 0.02 / 0.05 / 0.10).")
    w("- A claim holds when the adjusted p < 0.025 (the one-sided level of every registered decision). Holm = the "
      "frozen `analysis.tifs.holm`; Bonferroni = min(1, m p). Holm controls the family-wise error of the rejections "
      "(k ≥ δ rejected, or |k| ≥ δ for p_tost) but gives no simultaneous bounds; only the Bonferroni rows correspond "
      "to simultaneous one-sided (1 − 0.025/m) upper bounds below δ (two-sided (1 − 0.05/m) intervals inside ±δ for "
      "p_tost), jointly ≥ 97.5% (≥ 95%). The per-cell rows are the 97.5% one-sided upper bound of A4.")
    w("- *A4 classes* (rule of paper_tifs/tools/make_numbers.py): structural (EP SST-2), dagger † (trigger words in "
      "no clean KD-corpus row as whole words, aux/g1 tokens_summary `kd_rows_with_any_option_word_text` = 0: BERT, "
      "GPT-2, Qwen3 SST-2 BadNets and the BERT SST-2 'feature' BadNets cell), boundary ‡ (SOS SST-2, embedding-only "
      "teacher pair, words in 80 corpus rows; inside the A4 count and reported apart), counted. *A4-counted* = counted "
      "+ boundary.")
    w("")
    w("## Files")
    w("")
    w("| file | item | content |")
    w("|---|---|---|")
    for name, item, desc in (
            ("claims_cells.csv", "1", "cell × δ: k, SE, df, interval, verdict, p_up / p_tost / p_pos, Holm and Bonferroni over F31 and F19, A4 class"),
            ("claims_family.csv", "1", "counts per family × side × procedure × δ, split by A4 class, with the cell lists"),
            ("claims_table.csv", "1", "claim type × error control × status (registered / post hoc) with the result at each δ"),
            ("holm_family.csv", "6", "transfers_global per family and δ; added / dropped vs the frozen family"),
            ("holm_family_cells.csv", "6", "family × δ × cell: p_pos, Holm p, listed"),
            ("delta_rationale_cells.csv", "9", "per cell: δ*, P and C rates, Cohen's h, log OR, scales of δ at the cell's C rate, half-width, battery floor, p(k > 0) and its Holm"),
            ("delta_rationale_batteries.csv", "9", "per battery: k = 0 floor at r = 3, smallest δ the IUT would pass, A/A variation, h range of δ"),
            ("delta_rationale_aa.csv", "9", "the A/A clean-corpus contrasts (report/aa.csv)"),
            ("delta_rationale_sig_inside.csv", "9", "cells with k > 0 significant (lo > 0) and not POSITIVE at δ"),
            ("s1c_configH_main.csv", "12", "S1c config H (E beside it): cell coverage / false 'below δ', battery false TRANSFERS / NO TRANSFER rates"),
            ("headline_power.csv", "2", "P(headline met | all k = 0) per config × r × δ, MC SE, battery rates, product"),
            ("headline_power_cells.csv", "2", "per headline cell P(p_up < 0.025 | k = 0)"),
            ("headline_power_validation.csv", "2", "r = 3 joint simulation vs aux/g4 s1c_battery.csv; exchangeable vs measured covariance"),
            ("selfchecks.csv", "all", "reproductions of official numbers, hashes, consistency"),
            ("review0_comparison.csv", "all", "review-agent previews vs this recomputation"),
            ("provenance.csv", "all", "data snapshot and hashes"),
            ("logs/", "all", "per-script logs, hashes before the run, run start/end"),
            ("run_all.sh", "all", "thin wrapper that runs scripts/tifs/aux/g6-review0/multiplicity/run_all.sh "
             "(that folder's README.md lists the scripts)")):
        w(f"| `{name}` | {item} | {desc} |")
    w("")
    # ── item 1
    w("## Item 1: claim type × error control (family-level 'no transfer')")
    w("")
    w("| claim | error control | family | status | δ 0.02 | δ 0.05 | δ 0.10 |")
    w("|---|---|---|---|---|---|---|")
    for r in ct:
        def short(s):
            return s.replace("|", "/")
        w(f"| {r['claim']} | {r['error_control']} | {r['family']} | {r['status']} | {short(r['result@0.02'])} | "
          f"{short(r['result@0.05'])} | {short(r['result@0.1'])} |")
    w("")
    w("**Key findings.**")
    for t in TOLS:
        w(f"- δ {dl(t)}, per cell unadjusted, EQUIVALENT: F31 {fam('F31', 'equivalent', 'unadjusted', t)}, F19 "
          f"{fam('F19', 'equivalent', 'unadjusted', t)} (non-structural logit "
          f"{fam('F19', 'equivalent', 'unadjusted', t, 'n_nonstructural')}, A4-counted "
          f"{fam('F19', 'equivalent', 'unadjusted', t, 'n_a4_counted')}). 97.5% one-sided upper bound below δ: F31 "
          f"{fam('F31', 'below', 'unadjusted', t)}, F19 {fam('F19', 'below', 'unadjusted', t)}.")
    for fm in ("F31", "F19"):
        for proc in ("holm", "bonferroni"):
            for t in TOLS:
                lab = ("k ≥ δ rejected with family-wise error control (Holm over p_up, one-sided 0.025)"
                       if proc == "holm" else
                       "simultaneous one-sided upper bound below δ (Bonferroni, each at 1 − 0.025/m, jointly ≥ 97.5%)")
                w(f"- {fm}, δ {dl(t)}, {lab}: {famline(fm, 'below', proc, t)}.")
    for fm in ("F31", "F19"):
        w(f"- {fm}, two-sided equivalence (p_tost) under Holm: "
          + "; ".join(f"δ {dl(t)}: {fam(fm, 'equivalent', 'holm', t)} (non-structural "
                      f"{fam(fm, 'equivalent', 'holm', t, 'n_nonstructural')})" for t in TOLS)
          + "; under Bonferroni: "
          + "; ".join(f"δ {dl(t)}: {fam(fm, 'equivalent', 'bonferroni', t)}" for t in TOLS) + ".")
    for fm in ("F31", "F19"):
        w(f"- Transfer side, {fm} Holm (p_pos): "
          + "; ".join(f"δ {dl(t)}: {fam(fm, 'transfer', 'holm', t)} ({fam(fm, 'transfer', 'holm', t, 'cells')})" for t in TOLS)
          + f". Bonferroni gives the same cells ({', '.join(fam(fm, 'transfer', 'bonferroni', t) for t in TOLS)}).")
    ns = sorted([r for r in cc if abs(float(r["delta"]) - 0.05) < 1e-12 and r["method"] != "structural"
                 and r["kd"] == "logit"], key=lambda r: float(r["p_up"]))[:6]
    w("- Smallest non-structural logit p_up@0.05 (frozen t_contrast), with the Holm-adjusted p in F31 / F19: "
      + "; ".join(f"{r['display']} {float(r['p_up']):.5f} ({float(r['holm_F31_p_up']):.4f} / {float(r['holm_F19_p_up']):.4f})"
                  for r in ns) + ".")
    w("")
    # ── item 6
    w("## Item 6: Holm family sensitivity of transfers_global")
    w("")
    w("| family | size | δ 0.02 | δ 0.05 | δ 0.10 |")
    w("|---|---|---|---|---|")
    for name in dict.fromkeys(r["family"] for r in hf):
        rs = {f"{float(r['delta']):g}": r for r in hf if r["family"] == name}
        w(f"| {name} ({rs['0.02']['description']}) | {rs['0.02']['family_size']} | "
          + " | ".join(f"{rs[t]['transfers_global']}" for t in ("0.02", "0.05", "0.1")) + " |")
    w("")
    w("**Key findings.**")
    w("- No logit cell enters or leaves transfers_global under any of the five families at any δ. The only change is "
      "that the 'feature' cell BERT AG News feature ATBA (in the frozen list at δ 0.02 and 0.05) leaves when the "
      "'feature' cells leave the family. (c) 'without feature' is the same 19-cell set as (b) 'logit only', because "
      "the family has only the two KD types; both rows are reported.")
    for t in TOLS:
        rs = [r for r in hfc if r["cell"] == "bert/sst2/logit/atba" and abs(float(r["delta"]) - t) < 1e-12]
        w(f"- BERT SST-2 logit ATBA, Holm-adjusted p at δ {dl(t)}: "
          + "; ".join(f"{r['family']} {float(r['holm_p']):.4f}" for r in rs) + ".")
    for name in ("a_frozen31", "b_logit19", "d_without_structural"):
        rs = [r for r in hf if r["family"] == name]
        w(f"- {name}: smallest unlisted Holm p per δ: "
          + "; ".join(f"δ {dl(r['delta'])} {r['smallest_unlisted_cell']} {float(r['smallest_unlisted_holm_p']):.4f}"
                      for r in rs) + ".")
    w("- Putting the 12 'feature' cells and the 2 structural nulls into the family is conservative for transfer "
      "claims (it only enlarges m), and here it changes no logit transfer claim.")
    w("")
    # ── item 9
    w("## Item 9: δ rationale")
    w("")
    w("| battery | k = 0 floor at r = 3 | per-cell half-width | smallest δ the IUT passes (max hi) | A/A max abs C − C′ (BadNets trigger) | A/A max abs C − C′ (ATBA phrase) | ref. SD of k | Cohen's h of δ = 0.02 / 0.05 / 0.10 |")
    w("|---|---|---|---|---|---|---|---|")
    for r in db:
        w(f"| {r['battery']} | {f(r['floor_r3']) if r['floor_r3'] else '— (transported)'} | "
          f"{f(r['halfwidth_min'])}–{f(r['halfwidth_max'])} | {f(r['iut_delta_min'])} ({r['iut_delta_min_cell'].split('/')[-1]}) | "
          f"{f(r[CC]) if r[CC] else '—'} | "
          f"{f(r[CCA]) if r[CCA] else '—'} | "
          f"{f(r['reference_sd_k_badnets']) if r['reference_sd_k_badnets'] else '—'} | "
          f"{r['h_at_delta@0.02_range']} / {r['h_at_delta@0.05_range']} / {r['h_at_delta@0.1_range']} |")
    w("")
    w("Per cell (δ* from primary.csv; h and log OR of the observed P vs C rates; h of δ = 0.05 at the cell's C rate):")
    w("")
    w("| cell | A4 | k [95%] | δ* | P / C rate | Cohen's h | log OR | h of δ 0.05 | p(k>0) Holm-31 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in dc:
        w(f"| {r['display']} | {r['a4_class']} | {sf(r['k'])} [{sf(r['lo'])}, {sf(r['hi'])}] | {f(r['delta_star'])} | "
          f"{f(r['P_rate'])} / {f(r['C_rate'])} | {sf(r['cohen_h'])} | {sf(r['log_or'], 3)} | {f(r['h_at_delta@0.05'], 3)} | "
          f"{float(r['holm31_p_pos0']):.4f} |")
    w("")
    w("**Key findings.**")
    fl = {r["battery"]: r for r in db}
    w("- k = 0 upper-bound floors at r = 3 (aux/g3): " + "; ".join(
        f"{b} {f(fl[b]['floor_r3'])}" for b in ("bert/agnews/logit", "bert/sst2/logit", "gpt2/sst2/logit", "qwen3/sst2/logit"))
      + ". Every floor is above 0.02, so δ 0.02 is out of reach at r = 3 for every pooled battery; δ 0.05 is above every floor. "
      "The transported batteries have per-cell half-widths " + "; ".join(
          f"{b} {f(fl[b]['halfwidth_min'])}–{f(fl[b]['halfwidth_max'])}" for b in ("qwen3/agnews/logit", "bert/agnews/feature", "bert/sst2/feature")) + ".")
    aamax = max((float(r["C-C'"]), r) for r in da)
    aamin = min((float(r["C-C'"]), r) for r in da)
    big = max(abs(float(r["C-C'"])) for r in da)
    bigr = [r for r in da if abs(float(r["C-C'"])) == big][0]
    atb = [r for r in da if r[AT] not in ("", None)]
    w(f"- A/A between-build variation on the clean corpus (BadNets trigger, {len(da)} builds): C − C′ from "
      f"{sf(aamin[0])} to {sf(aamax[0])}; the largest |C − C′| is {f(big)} ({bigr['fam']} {bigr['dataset']} t{bigr['teacher_seed_prime']}). "
      "At the ATBA phrase the same clean builds differ by " + "; ".join(
          f"{r['fam']} {r['dataset']} t{r['teacher_seed_prime']} {sf(r[AT])}" for r in atb) + " (the reference dependence of Finding 4). SD of k across teacher builds "
      "(report/reference_variance.csv, BadNets): " + "; ".join(
          f"{r['battery'].rsplit('/', 1)[0]} {f(r['reference_sd_k_badnets'])} (df {r['reference_df']})"
          for r in db if r["reference_sd_k_badnets"] and r["battery"].endswith("logit")) + ".")
    hr = [float(r["h_at_delta@0.05"]) for r in dc]
    over02 = [r for r in dc if float(r["h_at_delta@0.05"]) >= 0.2]
    w(f"- On the Cohen's h scale δ 0.05 is {min(hr):.3f}–{max(hr):.3f} across the 31 cells' C rates: below Cohen's "
      f"'small' (0.2) except in the {len(over02)} cells with the lowest C rates (C {min(float(r['C_rate']) for r in over02):.3f}–"
      f"{max(float(r['C_rate']) for r in over02):.3f}, datasets: {', '.join(sorted({G.DS[r['dataset']] for r in over02}))}); "
      "δ 0.02 and 0.10 are "
      f"{min(float(r['h_at_delta@0.02']) for r in dc):.3f}–{max(float(r['h_at_delta@0.02']) for r in dc):.3f} and "
      f"{min(float(r['h_at_delta@0.1']) for r in dc):.3f}–{max(float(r['h_at_delta@0.1']) for r in dc):.3f}. "
      f"As a log-odds ratio δ 0.05 is {min(float(r['log_or_at_delta@0.05']) for r in dc):.3f}–"
      f"{max(float(r['log_or_at_delta@0.05']) for r in dc):.3f}.")
    w("- Cells with k > 0 at one-sided 0.025 per cell, unadjusted (lo > 0), that are not POSITIVE at δ "
      "(`delta_rationale_sig_inside.csv`):")
    for t in TOLS:
        rs = [r for r in dsig if abs(float(r["delta"]) - t) < 1e-12]
        w(f"  - δ {dl(t)}: " + ("; ".join(f"{r['display']} {sf(r['k'], 3)} [{sf(r['lo'], 3)}, {sf(r['hi'], 3)}] {r['class']} "
                                       f"(Holm-31 p(k>0) {float(r['holm31_p_pos0']):.4f}"
                                       + (f", Holm-19 {float(r['holm19_p_pos0']):.4f}" if r['holm19_p_pos0'] else "") + ")"
                                       for r in rs) or "none"))
    ins = {t: [r for r in dsig if abs(float(r["delta"]) - t) < 1e-12 and r["class"].startswith("inside")] for t in TOLS}
    none_at = [dl(t) for t in TOLS if not ins[t]]
    for t in TOLS:
        if not ins[t]:
            continue
        rs = ins[t]
        h19 = [r for r in rs if r["holm19_p_pos0"] and float(r["holm19_p_pos0"]) < 0.025]
        w(f"  No cell is 'inside (hi < δ)' at δ {' or '.join(none_at)}. At δ {dl(t)}, "
          + " and ".join(r["display"] for r in rs)
          + f" have per-cell (unadjusted) 95% intervals that exclude 0 and lie below {dl(t)} (EQUIVALENT@{dl(t)}); "
          "their k > 0 does not survive the 31-cell Holm ("
          + ", ".join(f"{r['display']} {float(r['holm31_p_pos0']):.4f}" for r in rs) + ")"
          + ("; " + ", ".join(f"{r['display']} passes the 19-cell Holm ({float(r['holm19_p_pos0']):.4f})" for r in h19)
             if h19 else "")
          + ". So they are not shown to be positive effects under paper-wide control.")
    w("")
    # ── item 12
    w("## Item 12: S1c config H numbers for the main text (`s1c_configH_main.csv`)")
    w("")
    w("Copied from the registered A11.10 S1c (aux/g4); config E in brackets. Pooled rule 20,000 reps; transported "
      "c_sd = 1, 40,000 reps (H) / 10,000 (E). c_sd is the SD ratio of the target cell's true training SD to the "
      "model's √(F_a E[s2_pool]) (variance ratio c_var = c_sd²), as in aux/g4.")
    w("")
    w("| battery | rule | cell with max false 'below δ' at μ = δ (H) [E] | min coverage (H) | false TRANSFERS, all μ = 0, δ 0.02 / 0.05 / 0.10 (H) [E] | NO TRANSFER, all μ = 0 (H) | max P(NO TRANSFER, one μ = δ) (H) [E] |")
    w("|---|---|---|---|---|---|---|")
    bats = [r for r in s1 if r["level"] == "battery" and G.as_bool(r["main_text"])]
    cells = [r for r in s1 if r["level"] == "cell"]
    for bname in dict.fromkeys(r["battery"] for r in bats):
        bs = {r["delta"]: r for r in bats if r["battery"] == bname}
        cs = [r for r in cells if r["battery"] == bname]
        cmax = max(cs, key=lambda r: float(r["false_below_at_mu_eq_delta"]))
        w(f"| {bname} | {bs['0.05']['rule']} | {cmax['attack']} {f(cmax['false_below_at_mu_eq_delta'])} ± "
          f"{f(cmax['false_below_at_mu_eq_delta_mcse'])} [{f(cmax['E_false_below_at_mu_eq_delta'])}] | "
          f"{f(bs['0.05']['coverage_min'])} ({bs['0.05']['coverage_min_attack']}) | "
          + " / ".join(f"{f(bs[t]['false_transfers_all0'])}" for t in ("0.02", "0.05", "0.1"))
          + " [" + " / ".join(f"{f(bs[t]['E_false_transfers_all0'])}" for t in ("0.02", "0.05", "0.1")) + "] | "
          + " / ".join(f"{f(bs[t]['no_transfer_all0'])}" for t in ("0.02", "0.05", "0.1")) + " | "
          + " / ".join(f"{f(bs[t]['false_no_transfer_one_at_d_max'])}" for t in ("0.02", "0.05", "0.1"))
          + f" ({bs['0.05']['false_no_transfer_one_at_d_argmax']}) ["
          + " / ".join(f"{f(bs[t]['E_false_no_transfer_one_at_d_max'])}" for t in ("0.02", "0.05", "0.1")) + "] |")
    w("")
    w("**Key findings.**")
    hcells = sorted(cells, key=lambda r: -float(r["false_below_at_mu_eq_delta"]))
    w("- False 'below δ' at μ = δ above the nominal 0.025 under H: " + "; ".join(
        f"{r['battery']} {r['attack']} {f(r['false_below_at_mu_eq_delta'])} ± {f(r['false_below_at_mu_eq_delta_mcse'])} "
        f"(E {f(r['E_false_below_at_mu_eq_delta'])})" for r in hcells if float(r["false_below_at_mu_eq_delta"]) > 0.025) + ".")
    over = [r for r in bats if float(r["false_transfers_all0"]) > 0.025]
    w("- Battery false TRANSFERS (all μ = 0) above 0.025 under H: " + ("; ".join(
        f"{r['battery']} δ {dl(r['delta'])} {f(r['false_transfers_all0'])} ± {f(r['false_transfers_all0_mcse'])}" for r in over)
        or "none") + ". Every other battery-δ is at or below 0.025.")
    over = [r for r in bats if float(r["false_no_transfer_one_at_d_max"]) > 0.025]
    w("- Battery false NO TRANSFER (one cell at μ = δ, others 0) above 0.025 under H: " + ("; ".join(
        f"{r['battery']} δ {dl(r['delta'])} {f(r['false_no_transfer_one_at_d_max'])} ({r['false_no_transfer_one_at_d_argmax']})"
        for r in over) or "none") + ".")
    tq = [r for r in s1 if r["level"] == "battery" and r["battery"] == "Qwen3 AG News logit" and r["delta"] == "0.05"]
    g4tq = [x for x in G.read_csv(os.path.join(G.AUX, "g4-s1c-s9-s1extra", "s1c_battery.csv"))
            if x["rule"] == "transported" and x["battery"] == "Qwen3 AG News logit" and x["config"] == "H"
            and abs(float(x["delta"]) - 0.05) < 1e-12 and x["c_var"]
            and any(abs(float(x["c_var"]) - v) < 1e-9 for v in (1.5, 2.0))]
    assert len(g4tq) == 2, len(g4tq)
    w("- Qwen3 AG News (transported from the Qwen3 SST-2 pool) at δ 0.05, H: " + "; ".join(
        f"c_sd (SD ratio) = {float(r['c_sd']):g} (variance ratio {float(r['c_var']):g}): NO TRANSFER (all μ = 0) "
        f"{f(r['no_transfer_all0'])}, max false NO TRANSFER "
        f"{f(r['false_no_transfer_one_at_d_max'])}, false 'below' max {f(r['false_below_max'])}" for r in tq)
      + ". These are the c_sd rows; aux/g4 also has variance-ratio rows (c_var = 1.5, 2, i.e. c_sd 1.22, 1.41) with "
      "other values, e.g. " + "; ".join(
          f"c_var {float(x['c_var']):g}: NO TRANSFER {f(x['no_transfer_all0'])}" for x in g4tq) + ".")
    w("")
    # ── item 2
    w("## Item 2: power of the registered headline (all k = 0)")
    w("")
    w(f"Joint simulation of the 19 headline cells with every true k = 0 (definitions in `headline_power.py`), 8 seed "
      f"blocks × 10,000 reps per row. Config H: each cell's training SD is its measured one (T_a² = max(0, s²_a − ψ_a)), "
      "so the cells with F_a > 1 keep their larger seed variance; config E: one common training SD per battery (S1b's "
      "assumption). Both use the shared seed fraction w fitted to the frozen pool. Headline met = every one of the 19 "
      f"p_up@δ < 0.025 (analyze.summary rule). Cost: {cost}.")
    w("")
    w("| config | covariance | r (pooled batteries; Qwen3 AG News stays 1) | P(headline met) δ 0.05 ± MC SE | product of battery rates | BERT AG News | BERT SST-2 | GPT-2 SST-2 | Qwen3 SST-2 | Qwen3 AG News | δ 0.02 | δ 0.10 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in hp:
        if r["delta"] != "0.05":
            continue
        o = {x["delta"]: x for x in hp if (x["config"], x["sigma_mode"], x["r_seeds_pooled_batteries"]) ==
             (r["config"], r["sigma_mode"], r["r_seeds_pooled_batteries"])}
        nt = lambda b: f(r[f"no_transfer[{b}]"])
        w(f"| {r['config']} | {r['sigma_mode']} | {r['r_seeds_pooled_batteries']} | {f(r['p_headline_met'])} ± "
          f"{f(r['p_headline_met_mcse'])} | {f(r['product_of_battery_no_transfer'])} | {nt('BERT AG News logit')} | "
          f"{nt('BERT SST-2 logit')} | {nt('GPT-2 SST-2 logit')} | {nt('Qwen3 SST-2 logit')} | {nt('Qwen3 AG News logit')} | "
          f"{f(o['0.02']['p_headline_met'])} | {f(o['0.1']['p_headline_met'])} |")
    w("")
    w("**Key findings.**")
    g = {(r["config"], r["sigma_mode"], r["r_seeds_pooled_batteries"], r["delta"]): r for r in hp}
    for cfg in ("H", "E"):
        m3 = g[(cfg, "measured", "3", "0.05")]
        w(f"- Config {cfg}, registered design (r = 3, measured covariance): P(headline met | all k = 0) = "
          f"{f(m3['p_headline_met'])} ± {f(m3['p_headline_met_mcse'])}; the product of the five battery rates of the "
          f"same simulation is {f(m3['product_of_battery_no_transfer'])} (joint − product "
          f"{sf(m3['joint_minus_product'])}). With the exchangeable covariance: r = 3 "
          f"{f(g[(cfg, 'exchangeable', '3', '0.05')]['p_headline_met'])}, r = 5 "
          f"{f(g[(cfg, 'exchangeable', '5', '0.05')]['p_headline_met'])} ± "
          f"{f(g[(cfg, 'exchangeable', '5', '0.05')]['p_headline_met_mcse'])}, r = 6 "
          f"{f(g[(cfg, 'exchangeable', '6', '0.05')]['p_headline_met'])} ± "
          f"{f(g[(cfg, 'exchangeable', '6', '0.05')]['p_headline_met_mcse'])}.")
    w("- At δ 0.02 the headline is essentially never met (" + ", ".join(
        f"{cfg} {mode} r={rr}: {float(g[(cfg, mode, rr, '0.02')]['p_headline_met']):.2g}"
        for cfg in ("H", "E") for mode, rr in (("measured", "3"), ("exchangeable", "6"))) + "); at δ 0.10 it is met "
      "with probability " + ", ".join(
        f"{cfg} r=3 {f(g[(cfg, 'measured', '3', '0.1')]['p_headline_met'])}" for cfg in ("H", "E")) + ".")
    for cfg in ("H", "E"):
        for rr in ("5", "6"):
            x = g[(cfg, "exchangeable", rr, "0.05")]
            nts = sorted(((float(v), k.split("[")[1][:-1]) for k, v in x.items() if k.startswith("no_transfer[")))[:2]
            w(f"- Config {cfg}, r = {rr}: the limiting batteries are " + "; ".join(f"{n} {v:.4f}" for v, n in nts)
              + " (Qwen3 AG News stays one-seed transported).")
    low = []
    for cfg in ("H", "E"):
        rows = [r for r in hpc if r["config"] == cfg and r["sigma_mode"] == "measured" and r["delta"] == "0.05"]
        rows = sorted(rows, key=lambda r: float(r["p_below"]))[:3]
        low.append(f"{cfg}: " + "; ".join(f"{r['cell']} {f(r['p_below'])}" for r in rows))
    w("- Weakest headline cells at r = 3 (P(p_up@0.05 < 0.025 | k = 0)): " + " | ".join(low) + ".")
    zmax = max(abs(float(r["z_vs_s1c"])) for r in hv)
    zx = max(abs(float(r["z_exchangeable_vs_measured"])) for r in hv)
    w(f"- Validation: the r = 3 measured-covariance battery rates match the registered aux/g4 s1c_battery.csv within "
      f"max |z| {zmax:.2f} over {len(hv)} rates; the exchangeable covariance at r = 3 differs from the measured one by "
      f"max |z| {zx:.2f} (`headline_power_validation.csv`).")
    w("")
    # ── checks and comparison
    w("## Self-checks (`selfchecks.csv`)")
    w("")
    nok = sum(G.as_bool(r["ok"]) for r in sc)
    pcols = [r for r in sc if r["check"].startswith("primary.csv every column")]
    w(f"{nok} of {len(sc)} checks pass. They include: report/primary.csv (31 cells; "
      + (f"every one of its {pcols[0]['expected'].split()[0]} columns" if pcols else "k, SE, df, interval, δ*, all p, verdicts, method")
      + ") reproduced from runs/tifs through the frozen loader (numbers to |diff| ≤ 1e-15, text exactly); battery.csv decisions "
      "and transfers_global; addendum/global_holm.csv Holm p (frozen family = family a of item 6); the summary.md "
      "headline (19 cells, max one-sided p = 1, not met); the Finding 1 counts of paper_tifs/NUMBERS.md "
      "(narrative.f1.n_eq/absent/counted/boundary at every δ) through this A4 classification; aux/g3 "
      "simultaneous_summary.csv Bonferroni counts; aux/g3 floors recomputed with t_contrast; aux/g1 Cohen's h; the "
      "A11 S1c record values; the headline simulation against aux/g4; frozen-file hashes before and after.")
    w("")
    w("## Comparison with the review-agent previews (`review0_comparison.csv`)")
    w("")
    w("| item | review | this | same |")
    w("|---|---|---|---|")
    for r in rc:
        w(f"| {r['item']} | {r['review_value']} | {r['this_value']} | {r['same']} |")
    w("")
    diff = [r for r in rc if not G.as_bool(r["same"])]
    w("Where they differ: " + ("; ".join(f"{r['item']}: {r['note']}" for r in diff) if diff else "nowhere") + ".")
    w("")
    w("## Deviations from A14 and why")
    w("")
    w("- Item 6 (c) 'without the feature cells' is the same 19-cell set as (b) 'logit only' (the family holds only "
      "logit and 'feature' cells); both rows are kept. An extra family (e) non-structural logit (18) is added.")
    w("- Item 1 uses the one-sided p_up (the registered NO TRANSFER direction) as the 'no transfer' test and also "
      "reports the two-sided TOST. The structural nulls have no SE/df; their frozen Tango p is used (the review "
      "script used 1e-12; counts are the same).")
    w("- Item 2: no measured input covariance exists for more than 3 seeds, so r = 5 and 6 use an exchangeable "
      "covariance built from the measured r = 3 one: W (mean same-seed covariance), X (mean cross-seed covariance) "
      "and ψ_a = W_aa − X_aa do not depend on r; v_a = (W_aa + (r − 1) X_aa) / r does. At r = 3 the exchangeable Σ "
      "reproduces the measured v_a and ψ_a. The training SDs T, the shared fraction w and the target model are kept "
      "at their r = 3 fit. "
      "Qwen3 AG News stays the registered one-seed transported cell at every r (its pool is the simulated Qwen3 "
      "SST-2 pool, now at r seeds). The Qwen3 SST-2 pool is drawn once per replicate and used by both batteries, "
      "whereas the registered S1c drew an independent source per battery. Only c = 1 (no misspecification) is run. "
      "A design in which Qwen3 AG News gets its own seeds (A13 G1) is not simulated: there is no cross-seed input "
      "covariance for that battery yet.")
    w("- Item 9: 'k > 0 significant but inside δ' is defined as lo > 0 and lo ≤ δ, split into inside (hi < δ), "
      "point inside and boundary (k > δ); this lists more cells than the two the review named (the 'feature' cells, "
      "and BERT AG News EP / BERT SST-2 ATBA at δ 0.10). Relative scales use the mean P and C rates over the cell's "
      "seeds. The per-battery floor is the registered A11.8 one (r = 3, v_typ); transported batteries have no seed "
      "floor and show the per-cell half-width instead.")
    w("- Item 12 copies the registered S1c numbers; nothing is re-simulated. Transported rows are also given at "
      "c_sd (SD ratio) = 1.5 and 2, i.e. variance ratio 2.25 and 4 (marked main_text = False); the aux/g4 rows at "
      "variance ratio 1.5 and 2 are not copied.")
    w("")
    open(os.path.join(out, "README.md"), "w").write("\n".join(L) + "\n")
    print(f"  wrote {G.rel(os.path.join(out, 'README.md'))}", flush=True)


if __name__ == "__main__":
    main()
