#!/usr/bin/env python3
"""Write README.md of A14 subgroup 'reference' (output folder) from its CSVs.

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import common as K  # noqa: E402


CODE_README_HEAD = """# A14 subgroup 'reference' - code (docs/tifs_addendum.md A14 items 3, 4, 5, 7, 13)

Status: {STATUS}.

Answers the internal round-0 review (paper_tifs/reviews/round0_meta.md §2 issue 5; experiment_options.md §2
C3, C4, C5, C7, C16). Outputs: `results/tifs/addendum/aux/g6-review0/reference/` (its README.md has the item
sections, tables and deviations). This file is written by make_readme.py together with that README.

| script | item | what |
|---|---|---|
| common.py | - | paths, frozen imports (analyze, addendum, analysis/tifs), CSV writer with the `# definition` line (LF line ends), the frozen seed-pooled / own / max / transported readings |
| score.py | 4, 7, 13 | CPU fp32 re-scoring (cross_score.triggered_loader + evaluator._probe_model, resumable): `refs` = every finished clean reference build's C' (A/A t43, t44; A13 G2 t45, t46 when finished) on all six triggers of both datasets; `fill` = AG News battery seed-42 logit and feature students on the triggers cross_hits lacks, the A2 replicate students on SST-2, a determinism re-check; `scitech` = item 13 |
| scitech_registration.json | 13 | phrase list (chosen by this subgroup's implementer; A14 names none) and readings, written before any item-13 scoring |
| item3_nullswap_own.py | 3 | A3.4 null swaps with own seed variance (A3.1 own / max, input term included) |
| item4_a2_2x2.py | 4 | A2 2x2 (P, P') x (C, C'), missing cell P'(t43) - C(t42), exact split of k - k' |
| item7_reference_k.py | 7 | P(t42) - C_ref per trigger and reference, transported verdicts, reference-dependent flags, A3.3-style interval per trigger (marked where a frozen table already shows the change), SST-2 seed-pooled t43 |
| item5_kappa_marg.py | 5 | kappa_cond per reference, kappa_marg over J clean builds, mixed-recipe sensitivity, excess over the within-build null students (with a 5-student SST-2 sensitivity) |
| item13_scitech.py | 13 | Sci/Tech phrase probes: C shift, P shift, P - C |
| selfcheck.py | all | `--gate`: reproduces primary.csv and attack_variance.csv before the item scripts (run_all.sh stops on a mismatch); full run: also the item-level reproductions, determinism, input sha256, frozen hashes, CSV format |
| make_readme.py | all | the output README and this README |
| run_all.sh | all | everything, CPU only (`--no-score`, `--skip-scitech`, `--out DIR`) |

Run: `scripts/tifs/aux/g6-review0/reference/run_all.sh`. To add the A13 G2 references t45 / t46 after their runs
finish (runs/tifs/_status/tifs_x13_bert_ref_t45.ok, `__eval.ok`): the same command
(`run_all.sh --skip-scitech` to skip the item-13 scoring check). Nothing outside
`scripts/tifs/aux/g6-review0/reference/` and `results/tifs/addendum/aux/g6-review0/reference/` is written; the
frozen files are imported, never edited.
"""


def f4(x):
    try:
        return f"{float(x):+.4f}"
    except (TypeError, ValueError):
        return "n/a"


def iv(lo, hi):
    return f"[{f4(lo)}, {f4(hi)}]"


def vv(r, prefix=""):
    return "/".join(r.get(f"{prefix}verdict@{t:g}", "?")[:3] for t in K.TOLS)


def mtime(p):
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(os.path.getmtime(p)))


def csv_or_none(out, name):
    p = os.path.join(out, name)
    return K.read_csv(p) if os.path.exists(p) else None


def snapshot(out):
    preds = glob.glob(os.path.join(K.RUNS, "tifs_*", "artifacts", "results", "*.preds.json"))
    newest = max(preds, key=os.path.getmtime)
    lines = [f"- Written {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}.",
             f"- Predictions read through the frozen `analyze.load_cells`: {len(preds)} files "
             f"`runs/tifs/*/artifacts/results/*.preds.json` (read only); newest {mtime(newest)} "
             f"(`{os.path.relpath(newest, K.ROOT)}`)."]
    for p in ("results/tifs/report/primary.csv", "results/tifs/addendum/cross_null.csv",
              "results/tifs/addendum/attack_variance.csv", "results/tifs/addendum/replicate.csv",
              "results/tifs/addendum/reference_atba.csv",
              "results/tifs/addendum/aux/g2-validity-reference-flips-builds/reference_atba_by_reference.csv"):
        lines.append(f"- `{p}`: {mtime(os.path.join(K.ROOT, p))}, sha256 {K.sha256_file(os.path.join(K.ROOT, p))[:16]}.")
    for ds in ("sst2", "agnews"):
        X = K.load_cross(ds)
        lines.append(f"- `results/tifs/addendum/cross_hits_bert_{ds}.json` (A3.4 CPU fp32 re-scores, scored "
                     f"{X['meta']['time_utc']}): {len(X['students'])} students, triggers "
                     f"{', '.join(sorted(X['hits']))}, sha256 {K.sha256_file(K.CROSS[ds])[:16]}.")
    for p in sorted(glob.glob(os.path.join(K.hits_dir(out), "*.json"))):
        H = K.load_json(p)
        t = H["meta"].get("time_utc")
        t = ", ".join(t) if isinstance(t, list) else t
        n = sum(len(v) for v in H["hits"].values())
        lines.append(f"- `hits/{os.path.basename(p)}` (this subgroup, CPU fp32, {H['meta'].get('threads')} threads; "
                     f"scoring jobs started {t}; file written {mtime(p)}; sha256 {K.sha256_file(p)[:16]}): "
                     f"{len(H['students'])} students, probes {', '.join(sorted(H['hits']))}, "
                     f"{n} hit vectors; {H['meta'].get('kind', '')}.")
    lines.append("- Frozen files (sha256 at README time; selfcheck.csv compares them with the start of run_all.sh): "
                 + "; ".join(f"`{f}` {h[:16]}" for f, h in K.frozen_hashes().items()) + ".")
    return lines


def main():
    out = K.out_dir()
    L = []
    w = L.append
    w("# A14 subgroup 'reference': null swaps with own variance, A2 2x2, kappa_marg, per-reference k, Sci/Tech control")
    w("")
    w(f"**Status of every file in this folder:** {K.STATUS}. These are post hoc CPU analyses answering the "
      "internal round-0 review (paper_tifs/reviews/round0_meta.md §2 issue 5; experiment_options.md §2 C3, C4, C5, "
      "C7, C16), registered as docs/tifs_addendum.md A14 items 3, 4, 5, 7 and 13. Where a reading here differs "
      "from a primary verdict, the primary verdict stands; the cell gets the label 'pooling-dependent' or "
      "'reference-dependent' (A3 rule). The A10-b gate (registered with the frozen formula) is not re-opened.")
    w("")
    i0 = len(L)
    add_key_findings(out, w)
    i1 = len(L)
    w("## Data snapshot")
    w("")
    L += snapshot(out)
    w("")
    i2 = len(L)
    w("## Code and rerun")
    w("")
    w("`scripts/tifs/aux/g6-review0/reference/run_all.sh` (CPU only; CUDA_VISIBLE_DEVICES='', OMP/MKL 8 threads, "
      "nice 10, HF offline, /data/Youngho/envs/blade/bin/python) runs score.py (fill sst2, refs, fill agnews, "
      "scitech; resumable), then item3_nullswap_own.py, item4_a2_2x2.py, item7_reference_k.py, "
      "item5_kappa_marg.py, item13_scitech.py, selfcheck.py and make_readme.py. `--no-score` rebuilds the tables "
      "only; `--out DIR` writes elsewhere. The frozen code (scripts/tifs/analyze.py, counterfactual_kd/analysis/"
      "tifs.py, counterfactual_kd/metrics/paired.py, scripts/tifs/addendum.py, scripts/tifs/cross_score.py) is "
      "imported, never edited.")
    w("")
    w("**Adding the A13 G2 references t45 / t46 (runs tifs_x13_bert_ref_t45 / _t46) is the same command:** "
      "`scripts/tifs/aux/g6-review0/reference/run_all.sh --skip-scitech`. `score.py refs` scores every clean "
      "reference run (tifs_aa_bert_clean_t<k>, tifs_x13_bert_ref_t<k>) whose student training finished "
      "(runs/tifs/_status/<run>.ok) on all six triggers of both datasets (about 0.5 h CPU per build), and items 5 and "
      "7 read every hits/ref_*.json. A build enters the reference sets only when its Stage 2 result exists and its "
      "clean teacher and C' student are above majority share + 0.10 (A13 G2 functional check); "
      "reference_builds.csv shows the status of each build.")
    w("")
    w("## Files")
    w("")
    w("| file | item | rows | one row per |")
    w("|---|---|---|---|")
    files = [("nullswap_own_variance.csv", "3", "A3.4 swap (cross_null.csv row) with pooled / own / max readings"),
             ("nullswap_own_summary.csv", "3", "dataset x KD x trigger: POSITIVE null swaps per rule"),
             ("a2_2x2.csv", "4", "cell / component / precision check of the A2 2x2, per basis"),
             ("kappa_marg.csv", "5", "dataset x KD x basis x estimand"),
             ("reference_k_all_triggers.csv", "7", "dataset x KD x trigger x reference"),
             ("reference_k_summary.csv", "7", "dataset x KD x trigger (verdicts across references, A3.3-style)"),
             ("reference_k_seedpooled_t43.csv", "7", "SST-2 trigger x reference (t42 twin, t43), seeds 42-44"),
             ("reference_builds.csv", "7", "clean reference build x dataset (functional check, inclusion)"),
             ("scitech_by_seed.csv", "13", "probe x student seed"),
             ("scitech_pooled.csv", "13", "probe (seed-pooled)"),
             ("check_item3_crossnull.csv", "3", "self-check: cross_null.csv row reproduced"),
             ("check_item5.csv", "5", "self-check vs g2 / A3.3 / cross_null"),
             ("check_item7.csv", "7", "self-check: input sha256, replicate determinism"),
             ("check_item13.csv", "13", "self-check: input alignment, registration unchanged"),
             ("selfcheck.csv", "all", "check (kind check / info)")]
    for fn, item, what in files:
        rs = csv_or_none(out, fn)
        w(f"| {fn} | {item} | {len(rs) if rs is not None else 'not written'} | {what} |")
    w("")
    w("Every CSV starts with one `# definition: ...` line (then the header). hits/*.json are this subgroup's CPU "
      "fp32 hit vectors in the cross_score.py format; logs/ holds the scoring logs.")
    w("")
    w("## Definitions")
    w("")
    w("- **Hit vector**: 0/1 per non-target test input (1 = the student predicts the attack's target label) on the "
      "triggered inputs of a trigger; SST-2 912 inputs (sst2_test, non-target), AG News 5,700. Every contrast pairs "
      "two vectors input by input on identical texts (sha256 of the input texts checked against cross_hits).")
    w("- **Frozen seed-pooled rule** (analyze.primary_table; addendum.seed_pooled + t_contrast): for per-seed "
      "difference vectors d_s, k = mean_s k_s, SE^2 = s2_pool / r + v, v = Var_i(mean_s d_s,i) / n, t at the pool's "
      "Satterthwaite df. Pools: bert/sst2/logit s = 0.021659, df 4.6688; bert/agnews/logit s = 0.025568, df 9.8278 "
      "('feature' borrows the family's logit pool).")
    w("- **Frozen transported rule** (one seed): Tango k, SE^2 = SE_Tango^2 + s2_pool, pool df.")
    w("- **A3.1 own**: SE^2 = s2_a / r + v with s2_a the sample variance of the contrast's own k_s, df r - 1. "
      "**A3.1 max**: SE^2 = max(s2_a, s2_pool) / r + v at pool df. 'pooling-dependent at delta' = the pooled verdict "
      "differs from own or max at that delta.")
    w("- **Verdicts** at delta in {0.02, 0.05, 0.10} (two-sided 95% interval = one-sided 0.025 per side): EQUIVALENT "
      "if -delta <= lo and hi <= delta, POSITIVE if lo > delta, NEGATIVE if hi < -delta, else INCONCLUSIVE "
      "(shown as EQU/POS/NEG/INC for 0.02/0.05/0.10).")
    w("- **Builds**: t42 = the frozen battery's teachers (seed 42); t43, t44 = A/A clean builds (tifs_aa_bert_clean_t43/"
      "_t44; base trainer, BadNets dev split selection); t43 is also the A2 replicate pair (poisoned P' and clean C'). "
      "Primes mark students of t43/t44 teachers.")
    w("- **kappa_cond(T_C)** = E_S[hit rate of S(T_P)] - E_S[hit rate of S(T_C)] for the fixed poisoned build T_P "
      "(teacher seed 42), fixed KD corpus (seed 42), fixed trigger string with the attack's insertion rule, on the "
      "battery's non-target inputs, with ONE clean twin build T_C; E_S averages over student training (seed). "
      "The registered estimand (frozen primary k) is kappa_cond(t42).")
    w("- **kappa_marg** = E_{T_C}[kappa_cond(T_C)] = E_S[S(T_P)] - E_{T_C} E_S[S(T_C)], the clean twin averaged over "
      "independent clean builds of the same recipe: the question of an auditor who cannot choose the twin build. "
      "Auxiliary estimate with J builds at seed 42: k = P - mean_j C_j, SE^2 = v + s2_pool + s2_C / J (s2_C = "
      "between-build variance of the C_j hit rates, df J - 1), Satterthwaite df as in A3.3. s2_C also contains each "
      "C_j's student and input noise, so the interval is conservative. Within-build version: the A3.4 excess "
      "P - mean_b N_b over the null students (no ATBA backdoor), with s2_N / J_N added.")
    w("- **reference-dependent** (item 7): the transported verdict of P(t42) - C_ref is not the same for every "
      "reference of {t42 twin, t43 C', t44 C'(, t45, t46)} at that delta (same seed-42 CPU fp32 basis); "
      "'EQUIVALENT changes' = EQUIVALENT under some references only. For SOS and EP the twin is a dedicated-trainer "
      "clean build and the C' builds are base-trainer builds (trainer_differs_from_twin). The EP SST-2 cell is a "
      "structural null only relative to its twin.")
    w("")
    i3 = len(L)
    defs_start = next(i for i in range(i2, i3) if L[i] == "## Definitions")
    add_item3(out, w)
    add_item4(out, w)
    add_item5(out, w)
    add_item7(out, w)
    add_item13(out, w)
    add_checks(out, w)
    add_deviations(w)
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"README.md: {len(L)} lines")
    # Code-folder README: status line, scripts, and the same key findings, data snapshot and
    # definitions as the output README (written to the code folder only for the official output;
    # with --out elsewhere a preview goes next to that output).
    C = CODE_README_HEAD.rstrip("\n").split("\n") + [""] + L[i0:i1] + L[i1:i2] + L[defs_start:i3] + [
        "Item-by-item details, tables and deviations: `results/tifs/addendum/aux/g6-review0/reference/README.md`."]
    code_readme = (os.path.join(K.HERE, "README.md")
                   if os.path.abspath(out) == os.path.abspath(K.OUT_DEFAULT)
                   else os.path.join(out, "code_README.preview.md"))
    with open(code_readme, "w") as f:
        f.write("\n".join(C).replace("{STATUS}", K.STATUS) + "\n")
    print(f"{os.path.relpath(code_readme, K.ROOT)}: {len(C)} lines")


def eq_at(r, prefix=""):
    """Deltas at which the row is EQUIVALENT (as text)."""
    e = [f"{t:g}" for t in K.TOLS if r.get(f"{prefix}verdict@{t:g}") == "EQUIVALENT"]
    return ", ".join(e) if e else "no delta"


def phrase_overlap(phrase):
    """Words (or the 'album' stem) the phrase shares with the AG News ATBA trigger."""
    trig = K.P_TAG["agnews"]["atba"].split("_", 1)[1].replace(",", "").split()
    shared = [w for w in phrase.split() if any(w == t or t.startswith(w) or w.startswith(t) for t in trig)]
    return shared


def add_key_findings(out, w):
    """Short list of the headline numbers (details in the item sections below)."""
    w("## Key findings (details and every number in the item sections)")
    w("")
    s3 = {(r["dataset"], r["kd"], r["trigger"]): r for r in (csv_or_none(out, "nullswap_own_summary.csv") or [])}
    v3 = {(r["dataset"], r["kd"], r["trigger"], r["student"].split("/")[0]): r
          for r in (csv_or_none(out, "nullswap_own_variance.csv") or [])}
    a = s3.get(("agnews", "logit", "atba"))
    if a:
        bn, sy = v3[("agnews", "logit", "atba", "badnets_cf_mn")], v3[("agnews", "logit", "atba", "synbkd_trig")]
        own = v3[("agnews", "logit", "atba", K.P_TAG["agnews"]["atba"])]
        w(f"1. **Item 3 (C3), logit KD.** AG News logit ATBA-trigger null swaps POSITIVE@0.05: {a['null_POSITIVE_pooled@0.05']} of "
          f"{a['n_null']} under the frozen pooled rule, {a['null_POSITIVE_own@0.05']} under own seed variance, "
          f"{a['null_POSITIVE_max@0.05']} under the A3.1 max rule -> pooling-dependent. BadNets null: pooled "
          f"{iv(bn['pooled_lo'], bn['pooled_hi'])}, own {iv(bn['own_lo'], bn['own_hi'])}; SynBkd null: pooled "
          f"{iv(sy['pooled_lo'], sy['pooled_hi'])}, own {iv(sy['own_lo'], sy['own_hi'])}. The own ATBA cell (CPU fp32): "
          f"pooled {iv(own['pooled_lo'], own['pooled_hi'])} {vv(own, 'pooled_')}, own {iv(own['own_lo'], own['own_hi'])} "
          f"{vv(own, 'own_')}, max {iv(own['max_lo'], own['max_hi'])} {vv(own, 'max_')}.")
        fa, fs = s3.get(("agnews", "feature", "atba")), s3.get(("sst2", "feature", "atba"))
        if fa and fs:
            fb = v3.get(("agnews", "feature", "atba", "badnets_cf_mn"))
            w(f"   **Feature KD (seed 42 only, not re-judgeable):** AG News feature ATBA has "
              f"{fa['null_POSITIVE_pooled@0.05']} of {fa['n_null']} null swaps POSITIVE@0.05 under the frozen transported "
              f"rule (BadNets {f4(fb['k'])} {iv(fb['pooled_lo'], fb['pooled_hi'])}) and {fa['null_POSITIVE_pooled@0.02']} "
              f"at 0.02 ({fa['null_POSITIVE_pooled_students'].replace(';', '; ')}); SST-2 feature ATBA has "
              f"{fs['null_POSITIVE_pooled@0.02']} of {fs['n_null']} at 0.02 ({fs['null_POSITIVE_pooled_students']}) and "
              f"{fs['null_POSITIVE_pooled@0.05']} at 0.05. The pooling-dependent reading applies to logit KD only.")
    rows4 = csv_or_none(out, "a2_2x2.csv")
    if rows4:
        q = {r["quantity"]: r for r in rows4 if r["basis"].startswith("official")}
        m, rs, ps, b = q["k(P',C)"], q["reference swap C'-C"], q["poisoned-build swap P-P'"], q["A2 build difference k-k'"]
        w(f"2. **Item 4 (C4).** Missing cell P'(t43) - C(t42) = {f4(m['k'])} {iv(m['lo'], m['hi'])} ({vv(m)}; own "
          f"{iv(m['own_lo'], m['own_hi'])} {vv(m, 'own_')}). By point estimate, the A2 drop k - k' = {f4(b['k'])} splits "
          f"exactly into the reference swap C' - C {f4(rs['k'])} ({100 * float(rs['share_of_build_difference']):.1f}%) and "
          f"the poisoned-build swap P - P' {f4(ps['k'])} ({100 * float(ps['share_of_build_difference']):.1f}%). Under the "
          f"pooled rule the reference swap {iv(rs['lo'], rs['hi'])} is {vv(rs)}. Under own variance neither component is "
          f"resolved: reference swap {iv(rs['own_lo'], rs['own_hi'])} ({vv(rs, 'own_')}), poisoned-build swap "
          f"{iv(ps['own_lo'], ps['own_hi'])} ({vv(ps, 'own_')}). The pooled poisoned-build swap {iv(ps['lo'], ps['hi'])} "
          f"({vv(ps)}) is shown within +-delta only at delta = {eq_at(ps)}: within +-0.10 but not within +-0.05. "
          "The data do not rule out a poisoned-build contribution.")
    rows5 = csv_or_none(out, "kappa_marg.csv")
    if rows5:
        km = {r["dataset"]: r for r in rows5 if r["kd"] == "logit" and r["basis"].startswith("official")
              and r["estimand"] == "kappa_marg_J3"}
        ex = {r["dataset"]: r for r in rows5 if r["estimand"].startswith("excess_null + between")
              and "[sensitivity" not in r["estimand"]}
        ex5 = {r["dataset"]: r for r in rows5 if r["estimand"].startswith("excess_null + between")
               and "[sensitivity" in r["estimand"]}
        if km and ex:
            e, e2 = ex["agnews"], ex["sst2"]
            sens = ex5.get("sst2")
            nps = e2.get("null_students_per_seed", "").split("/")
            w("3. **Item 5 (C5).** kappa_marg over the clean builds t42, t43, t44 (seed 42, official bf16): AG News "
              f"{f4(km['agnews']['k'])} {iv(km['agnews']['lo'], km['agnews']['hi'])}, SST-2 {f4(km['sst2']['k'])} "
              f"{iv(km['sst2']['lo'], km['sst2']['hi'])} (INCONCLUSIVE at every delta: not resolved as positive with J = 3). "
              f"Within-build excess of P over the A3.4 null students with their spread added (CPU fp32, seeds 42-44): "
              f"AG News {f4(e['k'])} {iv(e['lo'], e['hi'])} ({vv(e)}; {e.get('null_students_per_seed')} null students at "
              f"seeds 42/43/44); SST-2 {f4(e2['k'])} {iv(e2['lo'], e2['hi'])} ({vv(e2)}) over the {nps[0]} null students at "
              f"seed 42 ({nps[1]} at seeds 43 and 44: the EP pair ep_cf_mn / clean-ep has seed 42 only, so each seed's null "
              f"mean averages a different set, J_N = {e2['J']} and the EP students' rates come from seed 42 only, as the frozen "
              "excess_mean)"
              + (f"; with only the {sens['J']} null students present at every seed: {f4(sens['k'])} "
                 f"{iv(sens['lo'], sens['hi'])} ({vv(sens)})" if sens else "") + ".")
    s7 = csv_or_none(out, "reference_k_summary.csv")
    if s7:
        rd = [f"{r['dataset']}/{r['trigger']}@{r['reference_dependent']}" for r in s7 if r["kd"] == "logit" and r["reference_dependent"]]
        ec = [f"{r['dataset']}/{r['trigger']}@{r['equivalent_changes']}" for r in s7 if r["kd"] == "logit" and r["equivalent_changes"]]
        nref = {r["dataset"]: r["n_references"] for r in s7 if r["kd"] == "logit"}
        a3new = [f"{r['dataset']}/{r['kd']}/{r['trigger']}@{r['A33style_new_at']}" for r in s7 if r.get("A33style_new_at")]
        a3old = {}
        for r in s7:
            if r.get("A33style_already_in_frozen"):
                a3old.setdefault(r["A33style_frozen_source"], []).append(
                    f"{r['dataset']}/{r['kd']}/{r['trigger']}@{r['A33style_already_in_frozen']}")
        a3all = [f"{r['dataset']}/{r['kd']}/{r['trigger']}@{r['A33style_reference_dependent']}" for r in s7
                 if r.get("A33style_reference_dependent")]
        at = {r["dataset"]: r for r in s7 if r["kd"] == "logit" and r["trigger"] == "atba"}
        prim = K.read_csv(os.path.join(K.REPORT, "primary.csv"))
        nb = sum(r["fam"] == "bert" for r in prim)
        covered = {(r["dataset"], r["kd"], r["trigger"]) for r in s7}
        missing = [f"{r['dataset']}/{r['kd']}/{r['attack']}" for r in prim if r["fam"] == "bert"
                   and (r["dataset"], r["kd"], r["attack"]) not in covered]

        def spread(ds):
            return max((float(r["k_max"]) - float(r["k_min"]) for r in s7
                        if r["kd"] == "logit" and r["dataset"] == ds and r["trigger"] != "atba"), default=float("nan"))
        atxt = "; ".join(f"{d} ATBA k = " + " / ".join(f4(at[d].get(f"k_{x.split()[0]}")) for x in at[d]["references"].split(","))
                         + f" ({at[d]['references']})" for d in ("sst2", "agnews") if d in at)
        w(f"4. **Item 7 (C7).** Per-reference k for {len(covered)} of the {nb} BERT primary cells (of {len(prim)} paper-wide; "
          f"{'all BERT cells' if not missing else 'missing: ' + ', '.join(missing)}), seed 42, CPU fp32, transported "
          f"(references per logit cell: SST-2 {nref.get('sst2')}, AG News {nref.get('agnews')}). Reference-dependent logit cells: "
          f"{', '.join(rd) or 'none'}; EQUIVALENT changes with the reference in {', '.join(ec) or 'none'}. Feature cells "
          "(seed-42 feature P against its feature twin and the logit C' builds): reference-dependent in "
          + (", ".join(f"{r['dataset']}/{r['trigger']}@{r['reference_dependent']}" for r in s7
                       if r["kd"] == "feature" and r["reference_dependent"]) or "none")
          + f". {atxt}. Largest "
          f"spread of k across references on a non-ATBA logit trigger: AG News {spread('agnews'):.4f}, SST-2 "
          f"{spread('sst2'):.4f}. A3.3-style interval (primary k with the between-build variance on that trigger): it "
          f"changes the primary verdict in {', '.join(a3all) or 'none'}. Already shown by frozen tables: "
          + "; ".join(f"{', '.join(v)} ({src})" for src, v in a3old.items())
          + f". New here: {', '.join(a3new) or 'none'}. See the table for the seed-pooled SST-2 comparison against t43.")
    pr = csv_or_none(out, "scitech_pooled.csv")
    if pr:
        sci = [r for r in pr if r["probe"].startswith("scitech_")]
        nov = [r for r in sci if not phrase_overlap(r["phrase"])]
        npos = sum(r["k_verdict@0.05"] == "POSITIVE" for r in nov)
        txt = "; ".join(f"{'ATBA trigger' if r['probe'] == 'atba_trigger' else repr(r['phrase'])} k {f4(r['k'])} "
                        f"{iv(r['k_lo'], r['k_hi'])} {vv(r, 'k_')}"
                        + ("" if r["probe"] in ("atba_trigger",) or not r["probe"].startswith(("scitech_", "neutral_"))
                           else f" ({'shares ' + repr(' '.join(phrase_overlap(r['phrase']))) + ' with the trigger' if phrase_overlap(r['phrase']) else 'no shared word'})")
                        for r in pr if r["probe"] != "untriggered")
        w(f"5. **Item 13 (C16).** AG News logit ATBA P and clean-base C, seeds {pr[0]['seeds']}, prefix probes; the phrase "
          "list was chosen by this subgroup's implementer (an A14 agent) in scitech_registration.json before any item-13 "
          "scoring; A14, round0_meta.md, experiment_options.md and round0_reviews.md name no phrases. Sci/Tech phrases move "
          f"the clean students toward Sci/Tech by {min(float(r['C_shift']) for r in sci):+.4f} to "
          f"{max(float(r['C_shift']) for r in sci):+.4f} (neutral phrase "
          f"{f4(next(r['C_shift'] for r in pr if r['probe'].startswith('neutral_')))}). Teacher share k = P - C (pooled 95%): "
          f"{txt}. Mixed: {npos} of the {len(nov)} Sci/Tech phrases that share no word with the trigger "
          f"{'shows' if npos == 1 else 'show'} a positive teacher share at 0.05; no general claim that the trigger works by "
          "meaning is supported.")
    w("")


def add_item3(out, w):
    s = csv_or_none(out, "nullswap_own_summary.csv")
    rows = csv_or_none(out, "nullswap_own_variance.csv")
    if not s:
        return
    w("## Item 3 - A3.4 null swaps re-judged with their own seed variance (C3)")
    w("")
    get = {(r["dataset"], r["kd"], r["trigger"], r["student"]): r for r in rows}
    sm = {(r["dataset"], r["kd"], r["trigger"]): r for r in s}
    for ds, label in (("agnews", "BERT AG News"), ("sst2", "BERT SST-2")):
        r = sm.get((ds, "logit", "atba"))
        if r is None:
            continue
        w(f"- **{label} logit, ATBA trigger** ({r['n_null']} null swaps, {r['n_null_replicated']} with seeds 42-44): "
          "POSITIVE null swaps (pooled / own / max) at 0.02: "
          f"{r['null_POSITIVE_pooled@0.02']} / {r['null_POSITIVE_own@0.02']} / {r['null_POSITIVE_max@0.02']}; at 0.05: "
          f"{r['null_POSITIVE_pooled@0.05']} / {r['null_POSITIVE_own@0.05']} / {r['null_POSITIVE_max@0.05']}; at 0.10: "
          f"{r['null_POSITIVE_pooled@0.1']} / {r['null_POSITIVE_own@0.1']} / {r['null_POSITIVE_max@0.1']}. "
          f"Pooling-dependent null swaps at 0.02/0.05/0.10: {r['null_pooling_dependent@0.02']}/"
          f"{r['null_pooling_dependent@0.05']}/{r['null_pooling_dependent@0.1']}.")
        for (d, kd, t, st_), x in sorted(get.items()):
            if d == ds and kd == "logit" and t == "atba" and x.get("own_lo") and (
                    x["role"] == "own P" or x["pooled_verdict@0.05"] == "POSITIVE" or x["pooled_verdict@0.02"] == "POSITIVE"):
                w(f"  - {x['role']} `{st_}`: k {f4(x['k'])}; pooled {iv(x['pooled_lo'], x['pooled_hi'])} {vv(x, 'pooled_')}; "
                  f"own (df 2, F = {float(x['F_own']):.2f}) {iv(x['own_lo'], x['own_hi'])} {vv(x, 'own_')}; "
                  f"max {iv(x['max_lo'], x['max_hi'])} {vv(x, 'max_')}; pooling-dependent at "
                  f"{x['pooling_dependent'] or 'none'}.")
    r = sm.get(("agnews", "logit", "ep"))
    if r:
        w(f"- **BERT AG News logit, EP trigger**: no null swap is POSITIVE under any rule; the 7 null swaps are "
          f"EQUIVALENT at 0.02 under own variance (their k_s barely move across seeds) but INCONCLUSIVE under the pool, "
          f"so {r['null_pooling_dependent@0.02']} are pooling-dependent at 0.02 in the other direction. The EP own P: "
          f"pooled {r.get('own_P_pooled_verdict@0.02', '?')[:3]}/{r.get('own_P_pooled_verdict@0.05', '?')[:3]}/"
          f"{r.get('own_P_pooled_verdict@0.1', '?')[:3]} -> own {r.get('own_P_own_verdict@0.02', '?')[:3]}/"
          f"{r.get('own_P_own_verdict@0.05', '?')[:3]}/{r.get('own_P_own_verdict@0.1', '?')[:3]} "
          f"(own {iv(r.get('own_P_own_lo'), r.get('own_P_own_hi'))}), as the frozen A3.1 row.")
    others = [r for r in s if r["kd"] == "logit" and r["dataset"] == "sst2" and r["trigger"] != "atba"]
    if others:
        w("- **BERT SST-2 logit, other triggers** (" + ", ".join(r["trigger"] for r in others) + "): no null swap is "
          "POSITIVE under any rule at any delta. 'feature' and the SST-2 EP pair have seed 42 only: their rows keep "
          "the transported reading (no own variance exists).")
    for ds, label in (("agnews", "AG News"), ("sst2", "SST-2")):
        r = sm.get((ds, "feature", "atba"))
        if r is not None:
            w(f"- **BERT {label} feature, ATBA trigger** (seed 42 only: no own variance, not re-judgeable): null swaps "
              f"POSITIVE under the frozen transported rule at 0.02/0.05/0.10: {r['null_POSITIVE_pooled@0.02']}/"
              f"{r['null_POSITIVE_pooled@0.05']}/{r['null_POSITIVE_pooled@0.1']} of {r['n_null']} "
              f"({r['null_POSITIVE_pooled_students'].replace(';', '; ') or 'none'}); own and max counts are n/a.")
            for (d, kd, t, st_), x in sorted(get.items()):
                if d == ds and kd == "feature" and t == "atba" and x["role"] == "null" and x["pooled_verdict@0.02"] == "POSITIVE":
                    w(f"  - null `{st_}`: k {f4(x['k'])}; transported {iv(x['pooled_lo'], x['pooled_hi'])} {vv(x, 'pooled_')}.")
    w("- **Reading.** The frozen A3.4 statement '2 of 7 AG News ATBA null swaps are POSITIVE@0.05' (logit KD) is "
      "pooling-dependent: with each swap's own seed variance (input term included) none is, and none under the "
      "A3.1 max rule either; the own ATBA cell stays POSITIVE at 0.02 and 0.05 under all three rules (pooling-"
      "dependent at 0.10 only, as A3.1 already says). The A10-b gate was registered on the frozen formula and its "
      "decision stands; under own variance it would not have fired. This reading covers logit KD only: the feature-KD "
      "null swaps have one seed, and AG News feature ATBA keeps a null swap (BadNets) POSITIVE@0.05 under the frozen "
      "transported rule, which cannot be re-judged with own variance.")
    rep = [x for x in rows if x.get("own_lo")]
    scope = [x for x in rep if x["trigger"] == "atba" and x["kd"] == "logit"]

    def widen(xs):
        return max((max(abs(float(x["own_minus_approx_lo"])), abs(float(x["own_minus_approx_hi"]))) for x in xs), default=0.0)

    def vchg(xs):
        return [f"{x['dataset']}/{x['trigger']}/{x['student'].split('/')[0]}" for x in xs
                if any(x.get(f"review_approx_verdict_05") != x.get("own_verdict@0.05") for _ in (0,))]
    w(f"- **Review preview (review0/nullswap_own.py) vs this table.** The preview (ATBA trigger, logit KD, both datasets: "
      f"{len(scope)} replicated rows) left the input term v out of the own-variance SE. With v included its intervals widen "
      f"by at most {widen(scope):.4f} (AG News rows, the ones round0_meta.md quotes: at most "
      f"{widen([x for x in scope if x['dataset'] == 'agnews']):.4f}) and the 0.05 verdict changes in "
      f"{', '.join(vchg(scope)) or 'no row'}. Over all "
      f"{len(rep)} replicated rows of this table the widening is at most {widen(rep):.4f} and the 0.05 verdict of the "
      f"v-free approximation would differ in {', '.join(vchg(rep)) or 'no row'} (columns `review_approx_*`, "
      "`own_minus_approx_*`). Key rows (preview -> frozen formula): "
      + "; ".join(f"`{st_.split('/')[0]}` {iv(x['review_approx_lo'], x['review_approx_hi'])} -> {iv(x['own_lo'], x['own_hi'])}"
                  for (d, kd, t, st_), x in sorted(get.items())
                  if d == "agnews" and kd == "logit" and t == "atba" and x.get("own_lo")
                  and st_.split("_")[0] in ("badnets", "synbkd", "atba")) + ".")
    w("")


def add_item4(out, w):
    rows = csv_or_none(out, "a2_2x2.csv")
    if not rows:
        return
    w("## Item 4 - A2 2x2 decomposition, BERT SST-2 ATBA, logit KD (C4)")
    w("")
    for basis in ("official Stage 2, GPU bf16", "CPU fp32 re-score"):
        q = {r["quantity"]: r for r in rows if r["basis"] == basis}
        if not q:
            continue
        w(f"**Basis: {basis}** (seeds 42-44; pooled = frozen seed-pooled rule with the BERT SST-2 logit pool, df 4.67; "
          "own = own seed variance, df 2):")
        w("")
        w("| quantity | k | k_s42 / s43 / s44 | pooled 95% | pooled verdicts 0.02/0.05/0.10 | own 95% | own verdicts |")
        w("|---|---|---|---|---|---|---|")
        for name in ("k(P,C)", "k(P,C')", "k(P',C')", "k(P',C)", "reference swap C'-C",
                     "poisoned-build swap P-P'", "A2 build difference k-k'"):
            r = q.get(name)
            if r is None:
                continue
            share = f" ({100 * float(r['share_of_build_difference']):.1f}% of k-k')" if r.get("share_of_build_difference") else ""
            w(f"| {name}{share} | {f4(r['k'])} | {f4(r['k_s42'])} / {f4(r['k_s43'])} / {f4(r['k_s44'])} | "
              f"{iv(r['lo'], r['hi'])} | {vv(r)} | {iv(r['own_lo'], r['own_hi'])} | {vv(r, 'own_')} |")
        b = q.get("A2 build difference k-k'")
        if b is not None:
            w("")
            w(f"A2-style interval of the build difference (two cells taken as independent, as replicate.csv): "
              f"{iv(b['independent_cells_lo'], b['independent_cells_hi'])}; of the reference swap: "
              f"{iv(q['reference swap C' + chr(39) + '-C']['independent_cells_lo'], q['reference swap C' + chr(39) + '-C']['independent_cells_hi'])}; "
              f"of the poisoned-build swap: {iv(q['poisoned-build swap P-P' + chr(39)]['independent_cells_lo'], q['poisoned-build swap P-P' + chr(39)]['independent_cells_hi'])}.")
        w("")
    prec = [r for r in rows if r["kind"] == "precision check"]
    if prec:
        w("Official bf16 vs CPU fp32, inputs that disagree (of 912): " +
          ", ".join(f"{r['quantity']} {r['n_disagree']}" for r in prec) + ".")
        w("")
    q = {r["quantity"]: r for r in rows if r["basis"].startswith("official")}
    m = q["k(P',C)"]
    w(f"- **Missing cell.** P'(t43) - C(t42) = {f4(m['k'])} {iv(m['lo'], m['hi'])} (pooled; {vv(m)}); own variance "
      f"{iv(m['own_lo'], m['own_hi'])} ({vv(m, 'own_')}), so POSITIVE at 0.02 and 0.05 is pooling-dependent. Against "
      f"the same t42 clean twin, the second poisoned build gives {f4(m['k'])} and the registered build "
      f"{f4(q['k(P,C)']['k'])}.")
    rs, ps = q["reference swap C'-C"], q["poisoned-build swap P-P'"]
    w(f"- **Decomposition** (exact; the 2x2 has no interaction term because hit rates are means): the A2 drop "
      f"k - k' = {f4(q[chr(65) + '2 build difference k-k' + chr(39)]['k'])} = reference swap C'(t43) - C(t42) "
      f"{f4(rs['k'])} ({100 * float(rs['share_of_build_difference']):.1f}%) + poisoned-build swap P(t42) - P'(t43) "
      f"{f4(ps['k'])} ({100 * float(ps['share_of_build_difference']):.1f}%). The reference swap is POSITIVE at 0.02 and "
      f"0.05 under the pooled rule {iv(rs['lo'], rs['hi'])} and INCONCLUSIVE at every delta under own variance "
      f"{iv(rs['own_lo'], rs['own_hi'])}; the poisoned-build swap is {vv(ps)} {iv(ps['lo'], ps['hi'])} (own "
      f"{iv(ps['own_lo'], ps['own_hi'])}, {vv(ps, 'own_')}).")
    w(f"- **Reading.** By point estimate, {100 * float(rs['share_of_build_difference']):.0f}% of the A2 drop "
      f"({f4(rs['k'])}) comes from the reference swap and {100 * float(ps['share_of_build_difference']):.0f}% "
      f"({f4(ps['k'])}) from the poisoned-build swap. Under the pooled rule the reference swap is POSITIVE at 0.02/0.05; "
      "under own variance neither component is resolved. The poisoned-build share is within +-0.10 but not within "
      "+-0.05, so the data do not rule out a poisoned-build contribution. The percentages are point-estimate shares, "
      "not resolved proportions.")
    w(f"- **Review preview** (round0_meta.md issue 5: about 0.096 from the reference swap and 0.033 from the poisoned "
      f"build) is confirmed exactly: {f4(rs['k'])} and {f4(ps['k'])} (the preview's subtraction of the existing cells "
      "is exact by the identity above); the new numbers are the missing cell and the intervals.")
    w("")


def add_item5(out, w):
    rows = csv_or_none(out, "kappa_marg.csv")
    if not rows:
        return
    w("## Item 5 - kappa_marg auxiliary estimate, BERT ATBA (C5)")
    w("")
    for ds, label in (("sst2", "SST-2"), ("agnews", "AG News")):
        w(f"**{label}, logit KD.**")
        w("")
        w("| basis | estimand | references | seeds | k | 95% | df | verdicts 0.02/0.05/0.10 |")
        w("|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["dataset"] != ds or r["kd"] != "logit" or r["estimand"] == "P_minus_null":
                continue
            refs = r["references"] if len(r["references"]) < 40 else f"{r['J']} null students"
            w(f"| {'bf16' if r['basis'].startswith('official') else 'CPU fp32'} | {r['estimand']} | {refs} | {r['seeds']} | "
              f"{f4(r['k'])} | {iv(r['lo'], r['hi'])} | {float(r['df']):.2f} | {vv(r)} |")
        pm = [r for r in rows if r["dataset"] == ds and r["estimand"] == "P_minus_null"]
        if pm:
            w("")
            w("P - each null student (CPU fp32; frozen seed-pooled rule, transported for the one-seed rows): " + "; ".join(
                f"{r['references'].split('/')[0]} {f4(r['k'])} {iv(r['lo'], r['hi'])} {vv(r)}" for r in pm) + ".")
        w("")
    feat = [r for r in rows if r["kd"] == "feature" and r["estimand"].startswith("kappa_marg")]
    if feat:
        w("'feature' P against the logit-KD references (as A3.3): " + "; ".join(
            f"{r['dataset']} {r['estimand']} {f4(r['k'])} {iv(r['lo'], r['hi'])} {vv(r)}" for r in feat) + ".")
        w("")
    km = {(r["dataset"], r["basis"][:3]): r for r in rows if r["kd"] == "logit" and r["estimand"].startswith("kappa_marg_J")
          and "all_clean" not in r["estimand"] and r["estimand"] != "kappa_marg_J2_sp"}
    ex = {r["dataset"]: r for r in rows if r["estimand"].startswith("excess_null + between")
          and "[sensitivity" not in r["estimand"]}
    ex5 = {r["dataset"]: r for r in rows if r["estimand"].startswith("excess_null + between")
           and "[sensitivity" in r["estimand"]}
    a = km.get(("agnews", "off"))
    s = km.get(("sst2", "off"))
    if a and s:
        w(f"- **Reading.** Averaged over the three clean builds, the ATBA teacher share is {f4(a['k'])} {iv(a['lo'], a['hi'])} "
          f"on AG News and {f4(s['k'])} {iv(s['lo'], s['hi'])} on SST-2 (official vectors, seed 42, df "
          f"{float(a['df']):.1f} / {float(s['df']):.1f}): INCONCLUSIVE at every delta, because three builds give a "
          "between-build SD of the clean students' ATBA hit rate of "
          f"{float(a['s_between_builds']):.3f} (AG News) and {float(s['s_between_builds']):.3f} (SST-2). kappa_marg "
          "is not resolved as positive with J = 3; kappa_cond(t42) is POSITIVE (frozen primary).")
    j5 = {r["dataset"]: r for r in rows if r["estimand"].startswith("mixed_recipe_sensitivity")}
    if j5:
        w("- **Mixed-recipe sensitivity (not kappa_marg as defined; references share teacher seed 42 and differ in "
          "trainer).** The kappa_marg_J formula with the two dedicated-trainer clean twins of the battery (clean-sos, "
          "clean-ep) added as references (CPU fp32): " + "; ".join(
              f"{d} {f4(r['k'])} {iv(r['lo'], r['hi'])} (J = {r['J']}, df {float(r['df']):.1f}, {vv(r)})" for d, r in sorted(j5.items()))
          + ". These two references are not independent builds of the base recipe (same teacher seed as the t42 twin, "
          "different trainer), so the row is a sensitivity of the formula, not an estimate of kappa_marg; it is not a "
          "key finding. The between-build estimate with more base-recipe builds needs the A13 G2 builds t45 / t46.")
    if ex:
        e = ex.get("agnews")
        e2 = ex.get("sst2")
        if e and e2:
            nps = e2.get("null_students_per_seed", "").split("/")
            s5 = ex5.get("sst2")
            w(f"- **Within-build nulls.** The excess of the ATBA P over the A3.4 null students (students of teachers "
              f"without the ATBA backdoor) stays POSITIVE at 0.02 and 0.05 with their between-student variance added: "
              f"AG News {f4(e['k'])} {iv(e['lo'], e['hi'])} ({vv(e)}; {e.get('null_students_per_seed')} null students at "
              f"seeds 42/43/44), SST-2 {f4(e2['k'])} {iv(e2['lo'], e2['hi'])} ({vv(e2)}; {nps[0]} null students at seed 42, "
              f"{nps[1]} at seeds 43 and 44). SST-2 caveat: the EP pair (ep_cf_mn, clean-ep) has seed 42 only, so each "
              f"seed's null mean averages a different set, s2_N / J_N uses J_N = {e2['J']}, and the EP students' hit rates "
              "come from seed 42 only (the frozen excess_mean definition, reproduced exactly)"
              + (f"; sensitivity with the {s5['J']} null students present at every seed: {f4(s5['k'])} "
                 f"{iv(s5['lo'], s5['hi'])} ({vv(s5)}; own variance {iv(next(r['lo'] for r in rows if r['dataset'] == 'sst2' and r['estimand'].startswith('excess_null (own') and '[sensitivity' in r['estimand']), next(r['hi'] for r in rows if r['dataset'] == 'sst2' and r['estimand'].startswith('excess_null (own') and '[sensitivity' in r['estimand']))})" if s5 else "")
              + ". The null students are not independent clean builds of the base recipe (5 are poisoned for other "
              "triggers, 2 are dedicated-trainer clean twins), so this is a within-build kappa_marg, not the "
              "between-build one.")
    w("")


def add_item7(out, w):
    s = csv_or_none(out, "reference_k_summary.csv")
    b = csv_or_none(out, "reference_builds.csv")
    sp = csv_or_none(out, "reference_k_seedpooled_t43.csv")
    if not s:
        return
    w("## Item 7 - per-reference k on every BERT trigger (C7)")
    w("")
    if b:
        w("Reference builds: " + "; ".join(
            f"{r['dataset']} {r['reference']} (`{r['run']}`; teacher clean acc {float(r['teacher_ca']):.4f}, C' clean acc "
            f"{float(r['student_ca']):.4f}; {r['status']})" if r.get("teacher_ca") else
            f"{r['dataset']} {r['reference']} (`{r['run']}`; {r['status']})" for r in b) + ".")
        w("")
    prim = K.read_csv(os.path.join(K.REPORT, "primary.csv"))
    covered = {(r["dataset"], r["kd"], r["trigger"]) for r in s}
    missing = [f"{r['dataset']}/{r['kd']}/{r['attack']}" for r in prim if r["fam"] == "bert"
               and (r["dataset"], r["kd"], r["attack"]) not in covered]
    w(f"Coverage: {len(covered)} of the {sum(r['fam'] == 'bert' for r in prim)} BERT primary cells "
      f"({'all of them' if not missing else 'missing ' + ', '.join(missing)}); the AG News feature BadNets, AddSent, "
      "SynBkd and SOS cells use the feature re-scores in hits/fill_bert_agnews.json (cross_hits_bert_agnews.json holds "
      "only the ATBA and EP triggers).")
    w("")
    w("Seed 42, CPU fp32, frozen transported interval; verdicts at 0.02/0.05/0.10 per reference in the order of the "
      "`references` column; A3.3-style = frozen primary k with the base-trainer references' between-build variance "
      "on that trigger added (df J - 1).")
    w("")
    w("| dataset | KD | trigger | references | k per reference | verdicts @0.02 | @0.05 | @0.10 | reference-dependent | EQUIVALENT changes | primary | s_ref (J) | A3.3-style 95% | A3.3-style verdicts | A3.3-style differs from primary at |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in s:
        refs = r["references"].split(",")
        ks = " / ".join(f4(r.get(f"k_{x.split()[0]}")) for x in refs)
        prim = "/".join((r.get(f"primary_verdict@{t:g}") or "?")[:3] for t in K.TOLS)
        a33 = iv(r.get("A33style_lo"), r.get("A33style_hi")) if r.get("A33style_lo") else (r.get("A33style_note") or "n/a")
        a33v = vv(r, "A33style_") if r.get("A33style_lo") else ""
        sref = f"{float(r['s_ref']):.4f} ({r['J_base_trainer_refs']})" if r.get("s_ref") else "-"
        a33d = (r.get("A33style_reference_dependent") or "-") if r.get("A33style_lo") else ""
        if r.get("A33style_reference_dependent"):
            a33d += (f" (already in frozen: {r.get('A33style_already_in_frozen') or '-'}; "
                     f"new: {r.get('A33style_new_at') or '-'})")
        w(f"| {r['dataset']} | {r['kd']} | {r['trigger']} | {r['references']} | {ks} | {r['verdicts@0.02']} | "
          f"{r['verdicts@0.05']} | {r['verdicts@0.1']} | {r['reference_dependent'] or '-'} | "
          f"{r['equivalent_changes'] or '-'} | {prim} | {sref} | {a33} | {a33v} | {a33d} |")
    w("")
    if sp:
        w("SST-2, seeds 42-44 (frozen seed-pooled rule; t43 C' = A/A s42 + A2 replicate s43, s44):")
        w("")
        w("| trigger | reference | k | 95% | verdicts | own 95% | reference-dependent | EQUIVALENT changes |")
        w("|---|---|---|---|---|---|---|---|")
        for r in sp:
            rd = ",".join(f"{t:g}" for t in K.TOLS if r.get(f"reference_dependent@{t:g}") == "True")
            ec = ",".join(f"{t:g}" for t in K.TOLS if r.get(f"equivalent_changes@{t:g}") == "True")
            w(f"| {r['trigger']} | {r['reference'].split(' (')[0]} | {f4(r['k'])} | {iv(r['lo'], r['hi'])} | {vv(r)} | "
              f"{iv(r['own_lo'], r['own_hi'])} | {rd or '-'} | {ec or '-'} |")
        w("")
    logit = [r for r in s if r["kd"] == "logit"]
    rd = [f"{r['dataset']}/{r['trigger']}@{r['reference_dependent']}" for r in logit if r["reference_dependent"]]
    ec = [f"{r['dataset']}/{r['trigger']}@{r['equivalent_changes']}" for r in logit if r["equivalent_changes"]]
    a3 = [f"{r['dataset']}/{r['trigger']}@{r['A33style_reference_dependent']}" for r in logit
          if r.get("A33style_reference_dependent")]
    w(f"- **Reference-dependent logit cells** (seed-42 transported basis): {', '.join(rd) or 'none'}; of these, "
      f"EQUIVALENT changes with the reference in: {', '.join(ec) or 'none'}.")
    a3old = [f"{r['dataset']}/{r['kd']}/{r['trigger']}@{r['A33style_already_in_frozen']} ({r['A33style_frozen_source']})"
             for r in s if r.get("A33style_already_in_frozen")]
    a3new = [f"{r['dataset']}/{r['kd']}/{r['trigger']}@{r['A33style_new_at']}" for r in s if r.get("A33style_new_at")]
    w(f"- **A3.3-style** (frozen primary k with the between-build variance of the base-trainer references on that "
      f"trigger): the verdict differs from the frozen primary verdict in {', '.join(a3) or 'none'} (logit); these cells are "
      "reference-dependent under the A3 rule. The between-build SD of the clean students' hit rate is largest on the "
      "ATBA trigger (s_ref column).")
    w(f"- **Already known vs new.** Part of this is already in the frozen tables: {', '.join(a3old) or 'none'}. The 0.05 "
      "changes for SST-2 BadNets and SOS match the frozen primary.csv robust_* columns (robust_verdict@0.05 "
      "INCONCLUSIVE; the A11.5 label), and the ATBA changes match the frozen A3.3 table reference_atba.csv. New here: "
      f"{', '.join(a3new) or 'none'}, plus the per-trigger, per-reference k readings themselves.")
    if sp:
        spd = sorted({f"{r['trigger']}@" + ",".join(f"{t:g}" for t in K.TOLS if r.get(f"equivalent_changes@{t:g}") == "True")
                      for r in sp if any(r.get(f"equivalent_changes@{t:g}") == "True" for t in K.TOLS)})
        w(f"- **SST-2 seed-pooled** (the basis of the frozen primary), EQUIVALENT changes between the t42 twin and t43 in: "
          f"{', '.join(spd) or 'none'}.")
    w("")


def add_item13(out, w):
    pr = csv_or_none(out, "scitech_pooled.csv")
    sd = csv_or_none(out, "scitech_by_seed.csv")
    reg = K.load_json(os.path.join(K.HERE, "scitech_registration.json"))
    w("## Item 13 - AG News Sci/Tech semantic control (C16, optional)")
    w("")
    w(f"Phrase list chosen by this subgroup's implementer (an A14 agent) and fixed before scoring in "
      f"scripts/tifs/aux/g6-review0/reference/scitech_registration.json (registered {reg['registered_utc']}; A14, "
      "round0_meta.md, experiment_options.md and round0_reviews.md name no phrases; two of the four Sci/Tech phrases "
      "share a word or stem with the trigger: 'browser', 'album'; the first scitech scoring started after it, logs/score_scitech_*.log; "
      "the first job was stopped after the seed-42 P student to free the CPU for item 7 and resumed per seed): "
      + ", ".join(f"{k} = '{v}'" for k, v in reg["phrases"].items())
      + ". Insert: prefix '<phrase> <text>' into the 5,700 non-target AG News inputs; students: BERT AG News logit "
      "ATBA P and clean-base C, seeds 42-44 (teacher t42 builds); baselines: untriggered inputs (same job) and the "
      "ATBA trigger 'infected malicious albums, browser' (cross_hits_bert_agnews.json).")
    w("")
    if not pr:
        w("Not scored yet (score.py scitech).")
        w("")
        return
    w("| probe | phrase | seeds | P rate | C rate | C_shift = C - C_untrig (pooled 95%) | P_shift | k = P - C (pooled 95%) | k verdicts | k own 95% |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in pr:
        w(f"| {r['probe']} | {r['phrase']} | {r['seeds']} | {float(r['P_rate']):.4f} | {float(r['C_rate']):.4f} | "
          f"{f4(r['C_shift'])} {iv(r['C_shift_lo'], r['C_shift_hi'])} | {f4(r['P_shift'])} {iv(r['P_shift_lo'], r['P_shift_hi'])} | "
          f"{f4(r['k'])} {iv(r['k_lo'], r['k_hi'])} | {vv(r, 'k_')} | {iv(r['k_own_lo'], r['k_own_hi'])} |")
    w("")
    if sd:
        seeds = sorted({r["seed"] for r in sd})
        w(f"Seeds scored: {', '.join(seeds)} (per-seed rows in scitech_by_seed.csv).")
        w("")
    p = {r["probe"]: r for r in pr}
    sci = [r for k, r in p.items() if k.startswith("scitech_")]
    if sci and "atba_trigger" in p and "neutral_kitchen_table" in p:
        a, n = p["atba_trigger"], p["neutral_kitchen_table"]
        w(f"- **Reading.** Sci/Tech phrases move the clean students toward Sci/Tech by "
          f"{min(float(r['C_shift']) for r in sci):+.4f} to {max(float(r['C_shift']) for r in sci):+.4f} "
          f"(neutral phrase {f4(n['C_shift'])}; ATBA trigger {f4(a['C_shift'])}), and the teacher share on them is "
          f"{min(float(r['k']) for r in sci):+.4f} to {max(float(r['k']) for r in sci):+.4f} (ATBA trigger {f4(a['k'])}). "
          "See the table for the verdicts. Descriptive: no battery verdict changes.")
        nov = [r for r in sci if not phrase_overlap(r["phrase"])]
        w("- **Word overlap with the trigger** (by construction of the pre-fixed list, not a finding): 'internet browser' "
          "shares the word 'browser' with the ATBA trigger 'infected malicious albums, browser', and 'online album' shares "
          "the stem of 'albums'; 'software update' and 'computer virus' share no word with it, so a share on 'internet "
          "browser' can be carried by the shared word.")
        w("- **The semantic result is mixed.** Of the " + str(len(nov)) + " phrases that share no word with the trigger: "
          + "; ".join(f"'{r['phrase']}' k {f4(r['k'])} {iv(r['k_lo'], r['k_hi'])} ({vv(r, 'k_')})" for r in nov)
          + ". One shows a positive teacher share and the other has P slightly below C (EQUIVALENT only at 0.10), so the "
          "data support no general claim that the planted response works by meaning.")
    w("")


def add_checks(out, w):
    rows = csv_or_none(out, "selfcheck.csv")
    if not rows:
        return
    sc = [r for r in rows if r["kind"] == "check"]
    ok = sum(r["ok"] == "True" for r in sc)
    w("## Self-checks")
    w("")
    w(f"{ok} of {len(sc)} checks passed ({len(rows) - len(sc)} info rows; selfcheck.csv). The primary.csv and "
      "attack_variance.csv reproductions need no item output and run as a gate before the item scripts "
      "(`selfcheck.py --gate`, logs/selfcheck_gate.log; run_all.sh stops if one differs). The item-level "
      "reproductions (cross_null, replicate, g2, A3.3, excess) are read from the items' own check tables after the "
      "items run. Official numbers reproduced through this subgroup's code path:")
    w("")
    for r in sc:
        if any(x in r["check"] for x in ("primary.csv bert/agnews/logit/atba", "primary.csv bert/sst2/logit/atba",
                                         "attack_variance.csv bert/agnews/logit/atba", "cross_null.csv: every",
                                         "replicate.csv", "g2 seed-pooled", "item 5:", "determinism", "independent",
                                         "frozen file unchanged: scripts/tifs/analyze.py")):
            w(f"- {'ok' if r['ok'] == 'True' else 'FAILED'}: {r['check']} ({r['detail']})")
    bad = [r for r in sc if r["ok"] != "True"]
    for r in bad:
        w(f"- FAILED: {r['check']} ({r['detail']})")
    info = [r for r in rows if r["kind"] == "info"]
    if info:
        w("- Info: " + "; ".join(f"{r['check']}: {r['detail']}" for r in info[:12]) + ".")
    w("")


def add_deviations(w):
    w("## Deviations from A14 and why")
    w("")
    w("- Item 3: rows with one student seed (all 'feature' rows, the SST-2 EP pair students) have no own seed "
      "variance and keep the frozen transported reading; only the replicated rows are re-judged, and the summary "
      "writes 'n/a' for own / max counts of groups without a replicated null swap.")
    w("- Item 4: the 2x2 is read on the official bf16 vectors (the A2 basis); the CPU fp32 re-score of the four "
      "student sets (hits/fill_bert_sst2.json and cross_hits) is a cross-check. The components get the frozen "
      "seed-pooled rule on the paired per-seed vectors (a component is itself a contrast of hit vectors) and, beside "
      "it, A2's independent-cells interval.")
    w("- Item 5: the kappa_marg variance (v + s2_pool + s2_C / J, Satterthwaite) is not a registered formula; it "
      "follows A3.3's construction for a mean over J builds and is conservative. The within-build version uses the "
      "A3.4 null students, which are not base-recipe clean builds; on SST-2 they are 7/5/5 at seeds 42/43/44 (the EP "
      "pair has seed 42 only), so a 5-student sensitivity row is added. The mixed-recipe J = 5 row is a sensitivity "
      "of the formula, not kappa_marg.")
    w("- Item 7: cross_hits_bert_agnews.json holds only the ATBA and EP triggers, so the AG News seed-42 logit and "
      "feature P, clean-base and clean-sos students were re-scored on BadNets, AddSent, SynBkd and SOS "
      "(hits/fill_bert_agnews.json; the feature students were added after the A14 verifier's review, so all 24 BERT "
      "cells are covered) to keep every contrast on one CPU fp32 scoring. The "
      "per-reference reading is seed-42 transported (the C' builds have one student seed), so 'differs_from_primary' "
      "also reflects the seed basis; 'reference-dependent' compares references on the same basis. Extra: the SST-2 "
      "seed-pooled t43 comparison uses the A2 replicate's clean-base C' students at seeds 43, 44 (same t43 clean "
      "teacher). The A13 G2 builds t45/t46 were not finished at the snapshot (t45 Stage 0 running, t46 queued); "
      "run_all.sh adds them.")
    w("- Item 13: done (optional). A14 names no phrases: the four Sci/Tech phrases and one neutral two-word phrase "
      "('kitchen table', an insertion control) were chosen by this subgroup's implementer and fixed in "
      "scitech_registration.json before scoring; only the ATBA P and clean-base C students were scored. "
      "The scoring took about 22 min per student (6 probes x 5,700 inputs on the shared CPU), so it ran per seed in "
      "separate processes after item 7; the first job (stopped after the seed-42 P student) is kept as "
      "hits/scitech_bert_agnews.s42.json and was resumed from it.")
    w("- CPU: the scoring ran as two processes of 8 threads each (nice 10) in parallel to stay near the 1.5 h budget. "
      "The AG News feature fill added after the A14 verifier's review ran alone (one 8-thread process, nice 10) "
      "on 10/26 05:23-05:45 UTC, about 21 min (logs/score_fill_agnews_feature.log); the logit students were "
      "skipped as already scored (same checkpoint sha256).")
    w("- SynBkd table path: counterfactual_kd/triggers/syntactic.py resolves its table relative to the working "
      "directory; score.py sets CKD_SYNBKD_TABLE to the repository table (cross_score.strict_synbkd keeps a miss an "
      "error, nothing is appended).")
    w("")


if __name__ == "__main__":
    main()
