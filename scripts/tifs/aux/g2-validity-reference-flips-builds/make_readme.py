#!/usr/bin/env python3
"""Write OUT/README.md: fixed definitions + numbers read back from the CSVs (nothing typed by hand)."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # no __pycache__ inside the repository

import datetime as dt  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402

from g2common import (AD, ADDENDUM, CPRIME_JSON, CROSS_AGNEWS, CROSS_SST2, GROUP, HERE, OUT, REPORT, ROOT,  # noqa: E402
                      RUNS, STATUS, cli, hidden_groups, mean, read_csv, rel, state)

TOLS = ("0.02", "0.05", "0.1")
DRAFT_UTC = "2026-10-24 11:26 UTC"   # when the verified scratch draft of these tables was written (provenance only)


def fx(x, nd=4, signed=True):
    if x in (None, "", "None"):
        return "-"
    v = float(x)
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def evni(r, t, pre=""):
    if r is None:
        return "-"
    return "/".join(r.get(f"{pre}n_{v}@{t}", "") or "0" for v in ("equiv", "posit", "negat", "incon"))


def ab(v):
    return {"EQUIVALENT": "EQ", "POSITIVE": "POS", "NEGATIVE": "NEG", "INCONCLUSIVE": "INC"}.get(v, v or "-")


def utc(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    cli(__doc__)
    st = state()
    vs = read_csv(os.path.join(OUT, "validity_summary.csv"))
    vc = read_csv(os.path.join(OUT, "validity_contrasts.csv"))
    rb = read_csv(os.path.join(OUT, "reference_atba_by_reference.csv"))
    ra = read_csv(os.path.join(OUT, "reference_atba_agreement.csv"))
    fs = read_csv(os.path.join(OUT, "flips_summary.csv"))
    fb = read_csv(os.path.join(OUT, "flips_battery_by_seed.csv"))
    bf = read_csv(os.path.join(OUT, "builds_factorial.csv"))
    bsp = read_csv(os.path.join(OUT, "builds_factorial_spread.csv"))
    sc = read_csv(os.path.join(OUT, "selfcheck.csv"))
    _, hstat = hidden_groups()
    hid_seeds = ", ".join(str(r["seed"]) for r in hstat if r["status"] == "admitted") or "none"
    rcc = AD.load_rcc(RUNS)
    a10b = sorted(glob.glob(os.path.join(RUNS, "tifs_rep_bert_atba_agnews_t43_s*", "artifacts", "results", "*.preds.json")))
    a10b_failed = os.path.exists(os.path.join(RUNS, "_status", "tifs_a10b_go.failed"))
    rep = AD.load_replicate(RUNS)
    preds = sorted(glob.glob(os.path.join(RUNS, "*", "artifacts", "results", "*.preds.json")), key=os.path.getmtime)
    newest = preds[-1]
    n_pass = sum(c["passed"] == "True" for c in sc)
    n_fail = sum(c["passed"] == "False" for c in sc)
    n_info = sum(c["passed"] in ("", "None") for c in sc)
    cross = {}
    for p in (CROSS_SST2, CROSS_AGNEWS):
        X = json.load(open(p))
        cross[X["meta"]["dataset"]] = (p, X["meta"].get("time_utc"), sha256(p), len(X["students"]), sorted(X["hits"]))

    # clean students' hit rate on the ATBA phrase: every clean-corpus C of an ATBA cell, the A/A clean C' on the
    # atba_phrase control, the A2 C' and the A8 hidden C; the A2 seed-42 C' is the A/A t43 C' (identical weights)
    # model id = dataset | run | KD | role | student seed (an A/A run holds both datasets' students)
    readings = []
    for c in st.cells:
        if c["attack"] == "atba" and c["corpus"] == "clean" and not c["single"]:
            readings.append((f"{c['dataset']}|{c['run']}|{c['kd']}|C|s{c['student_seed']}", c["C"]))
    for c in st.cells:
        if c["block"] == "AA" and c["corpus"] == "clean" and f"atba_phrase_{c['dataset']}" in c["controls"]:
            readings.append((f"{c['dataset']}|{c['run']}|logit|C|s{c['student_seed']}",
                             c["controls"][f"atba_phrase_{c['dataset']}"]["c"]))
    for s, v in rep.items():
        readings.append((f"sst2|{v['run']}|logit|C|s{s}", v["C"]))
    hg_, _ = hidden_groups()
    for (f_, d_, k_, a_), bs in hg_.items():
        if a_ == "atba":
            readings += [(f"{d_}|{c['run']}|hidden|C|s{c['student_seed']}", c["C"]) for c in bs.values()]
    cpr = json.load(open(CPRIME_JSON))
    alias = ({"sst2|tifs_rep_bert_atba_t43_s42|logit|C|s42": "sst2|tifs_aa_bert_clean_t43|logit|C|s42"}
             if cpr.get("identical") else {})
    distinct = {alias.get(m, m) for m, _ in readings}
    distinct_vectors = {(m.split("|")[0], tuple(v)) for m, v in readings}
    if len(distinct_vectors) != len(distinct):
        print(f"note: {len(distinct)} distinct models but {len(distinct_vectors)} distinct hit vectors on the ATBA phrase")
    rates = [mean(v) for _, v in readings]

    idx = {(r["source_code"], r["dataset"], r["corpus_group"], r["trigger_group"], r["basis"][:5]): r for r in vs}
    ds_of_run = {}
    for c in st.cells:
        ds_of_run.setdefault(c["run"], set()).add(c["dataset"])
    multi_runs = sorted(r for r, d in ds_of_run.items() if len(d) > 1)
    # ids without the dataset field (the earlier scheme) that name students of two datasets
    no_ds = {}
    for r in vc:
        for m in (r.get("a_model"), r.get("b_model")):
            if m and not m.startswith("teacher|"):
                f_ = m.split("|")
                no_ds.setdefault("|".join(f_[:1] + f_[2:]), set()).add(f_[1])
    n_collide = sum(len(d) > 1 for d in no_ds.values())

    def srow(code, tg="all", cg="all", basis="Tango"):
        return idx.get((code, "all", cg, tg, basis[:5])) or idx.get((code, "all", cg, tg, "froze"))

    L = []
    w = L.append
    w("# A11 items 3, 4, 6, 7: validity, reference builds, flips, per-build factorial")
    w("")
    w(f"Group `{GROUP}` (docs/tifs_addendum.md A11, items 3, 4, 6 and 7).")
    w("")
    w(f"**Status of every file in this folder:** {STATUS}. The tables were computed after the results were seen; they are "
      "descriptive or sensitivity readings. Where a reading here differs from a primary verdict, the primary verdict stands.")
    w("")
    w("**Data snapshot.** "
      f"Written {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Predictions: {len(preds)} files "
      f"`runs/tifs/*/artifacts/results/*.preds.json` (read only); newest {utc(os.path.getmtime(newest))} "
      f"(`{rel(newest)}`). Frozen report: `{rel(REPORT)}/*.csv` (scripts/tifs/analyze.py), newest "
      f"{utc(max(os.path.getmtime(p) for p in glob.glob(os.path.join(REPORT, '*.csv'))))}. Addendum tables: "
      f"`{rel(ADDENDUM)}/*.csv` (scripts/tifs/addendum.py), newest "
      f"{utc(max(os.path.getmtime(p) for p in glob.glob(os.path.join(ADDENDUM, '*.csv'))))}. A3.4 CPU fp32 cross-scores "
      "(scripts/tifs/cross_score.py): "
      + "; ".join(f"`{rel(p)}` (scored {t}, {n} students, triggers {', '.join(h)}, sha256 {s})"
                  for _, (p, t, s, n, h) in sorted(cross.items(), reverse=True))
      + ". The AG News null swaps come from that official AG News file.")
    w("")
    w(f"Cells: {len(st.cells)} from {len({c['run'] for c in st.cells})} runs read by the frozen `analyze.load_cells`; "
      f"31 frozen primary rows (all teacher-qualified); A2 replicate seeds {', '.join(map(str, sorted(rep)))}; A8 hidden "
      f"admitted seeds {hid_seeds}; A10-a rcc cells: {len(rcc)}; A10-b (second poisoned AG News ATBA build) prediction "
      f"files: {len(a10b)}. A10-b did not run: its registered gate fired"
      + (" (`runs/tifs/_status/tifs_a10b_go.failed`" if a10b_failed else " (")
      + "; docs/tifs_addendum.md A10 '결과 기록'), see item 3 X2 below.")
    w("")
    w(f"**Code and rerun.** `{rel(HERE)}/run_all.sh` runs item3_validity.py, item4_reference_atba.py, item6_flips.py, "
      "item7_builds.py, selfcheck.py and make_readme.py in that order (shared helpers: g2common.py), CPU only "
      "(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE=1, /data/Youngho/envs/blade/bin/python). `--out DIR` writes "
      "elsewhere. The frozen functions (analysis/tifs.py, analyze.py, metrics/paired.py) and addendum.py are imported, "
      f"never edited. Self-checks: {n_pass} passed, {n_fail} failed, {n_info} info row{'s' if n_info != 1 else ''} "
      "(selfcheck.csv, listed below).")
    w("")
    w("## Files")
    w("")
    w("| file | item | rows | one row per |")
    w("|---|---|---|---|")
    for f_, it, unit in (("validity_contrasts.csv", "3", "contrast (source x cell x seed / build / precision)"),
                         ("validity_summary.csv", "3", "source x subgroup (dataset / corpus group / trigger group) x reading basis"),
                         ("reference_atba_by_reference.csv", "4", "dataset x clean reference build (t42, t43, t44)"),
                         ("reference_atba_agreement.csv", "4", "dataset x basis x delta (and delta = all)"),
                         ("flips_cells.csv", "6", "unit (estimand) x basis"),
                         ("flips_pairs.csv", "6", "pair of members of a unit x basis"),
                         ("flips_battery_by_seed.csv", "6", "battery x basis (decision per student seed)"),
                         ("flips_summary.csv", "6", "source x basis"),
                         ("builds_factorial.csv", "7", "family x dataset x teacher build"),
                         ("builds_factorial_spread.csv", "7", "family x dataset (spread across builds)"),
                         ("selfcheck.csv", "-", "check")):
        w(f"| {f_} | {it} | {len(read_csv(os.path.join(OUT, f_)))} | {unit} |")
    w("")
    w("Every CSV here is an A11 auxiliary / post hoc table; none changes a primary verdict. The CSVs have no comment "
      "lines; this README is their companion.")
    w("")

    w("## Common definitions")
    w("")
    w("- **Hit vector**: 0/1 per non-target test input (1 = the model predicts the attack's target label) on the triggered "
      "inputs of an attack (or a control / untriggered input set). SST-2: 912 inputs (SetFit/sst2 test, non-target); AG News: "
      "5,700. Every contrast pairs two hit vectors input by input on the same inputs (alignment is shown by N3: the same "
      "teacher gives identical hit vectors in every job).")
    w("- **Tango reading**: frozen `paired_contrast(a, b)` (counterfactual_kd/analysis/tifs.py): k = mean(a) - mean(b), Tango "
      "95% score interval [lo, hi]; at delta in {0.02, 0.05, 0.10}: EQUIVALENT iff -delta <= lo and hi <= delta, POSITIVE iff "
      "lo > delta, NEGATIVE iff hi < -delta, else INCONCLUSIVE. Input noise only (no training noise). Between two student "
      "seeds (S rows) this is a descriptive interval: the inputs are the same, the models differ.")
    w("- **Transported reading**: the frozen one-seed rule `t_contrast(k, sqrt(SE_Tango^2 + s2_pool), df_pool)`, SE_Tango = "
      "(hi - lo) / (2 z_0.975) (`se_from_interval`), with the pool the frozen primary rule would use for that family / dataset / "
      "KD (`addendum.State.pool_for` = `analyze.primary_table.source_for`): "
      + "; ".join(f"{'/'.join(k)} s = {v['s_between']:.6f}, df {v['df']:.4f}" for k, v in sorted(st.pools.items()))
      + ". Qwen3 AG News borrows qwen3/sst2/logit; 'feature', 'hidden', CE-only and teacher rows borrow the family's logit "
      "pool. **Structural null**: a teacher-swap row (T1, T6) of the EP SST-2 structural null keeps its Tango interval and "
      "verdicts in `transported_*`, as the frozen primary rule does (method 'structural', seed-42 Tango) and as item 6 does; "
      "`transported_rule` labels each row ('one-seed formula' or the structural-null rule) and `transported_formula_*` keeps "
      "the one-seed formula's interval and verdicts of those rows. Other rows of that cell (X1 null swaps, N4, F1) are not "
      "the structural-null estimand and use the formula. For contrasts that are not themselves a teacher-swap k (C_t - C_s, "
      "R_B - R_C, ...) the transported reading is a common yardstick, not a calibrated test.")
    w("- **Seed-pooled reading**: the frozen formula k = mean_s k_s, SE^2 = s2_pool / r + Var_i(mean_s d_s,i) / n, pool df "
      "(`addendum.seed_pooled` + `t_contrast`).")
    w("- **Precision** column: `official Stage 2, GPU bf16` (default), `CPU fp32 re-score (scripts/tifs/cross_score.py)`, "
      "`CPU fp32 re-score vs official GPU bf16`, `GPU0 bf16 re-evaluation vs official GPU1 bf16`.")
    w("- **Builds**: t42 = the frozen grid / battery (teachers seed 42); t43, t44 = the A/A teachers (tifs_aa_*); t43 is also the "
      "A2 replicate (tifs_rep_bert_atba_t43_s*). Primes (C', P') mark the students of the t43 / t44 builds. Corpus seed 42 "
      "everywhere.")
    w("- **Model ids** (`a_model`, `b_model`): run | dataset | KD | role (P:<attack>, C:<clean trainer tag>, R) | student seed "
      f"(teachers: teacher | family | dataset | build). The dataset is part of the id because {len(multi_runs)} runs (A/A, "
      "contaminated grid, positive control) hold both SST-2 and AG News students. Within a run, dataset and seed one "
      "clean-base C student serves BadNets, AddSent, SynBkd and ATBA (BERT; every attack for GPT-2 / Qwen3), so rows that "
      "share a model pair are not independent; `n_distinct_model_pairs` in the summary counts them once.")
    w("")

    # ── item 3
    w("## Item 3 - 판별 타당성 표 (validity_contrasts.csv, validity_summary.csv)")
    w("")
    w("Sources (`source_code`; `extra_source` = True for sources outside the A11.3 list, kept apart):")
    w("")
    w("| code | source of effect | exact contrasts |")
    w("|---|---|---|")
    for code, text in (
        ("T1", "teacher swap, clean corpus | P - C of every clean-corpus battery cell (logit, 'feature'; every family) at each student seed present, Tango"),
        ("T2", "teacher swap, frozen primary | the 31 frozen primary rows (seed-pooled / transported / structural), as primary.csv"),
        ("T3", "teacher swap, contaminated corpus | P - C of tifs_g_*_badnets10_s42 / addsent10 (logit, 'feature')"),
        ("T4", "teacher swap, positive control | P - C of tifs_pc_bert_kl01 / kl10"),
        ("T5", "teacher swap, A/A builds | P' - C' of tifs_aa_* (BadNets, t43 / t44, clean and badnets10)"),
        ("T6", "teacher swap, CPU fp32 | own P - matched C from the A3.4 re-scores, per seed, Tango"),
        ("T6f", "teacher swap, CPU fp32, A3.4 formula | the A3.4 own-P rows (frozen cross_tables: seed-pooled / transported; the frozen A3.4 code also applies the one-seed formula to the EP SST-2 own P), with the excess over the null students (`excess_over_null_students`, `excess_s<seed>`; `n_null_students` = the frozen distinct-student count, `n_null_students_by_seed` = students scored at each seed); the like-for-like comparator of X2"),
        ("T7", "teacher swap, A2 build (extra) | P' - C' of the A2 replicate (ATBA SST-2, t43) per seed"),
        ("N1", "same-seed rerun: training | the seed-42 clean-base t43 student trained by two jobs (tifs_rep_bert_atba_t43_s42 vs tifs_aa_bert_clean_t43; weights sha256-identical, cprime_determinism.json): ATBA phrase, untriggered inputs and the 4 'insert' controls ('substitute' controls left out: their texts depend on the run's own trigger)"),
        ("N2", "same-seed rerun: evaluation GPU | tifs_b_bert_sst2_s43 re-evaluated on GPU0 vs official GPU1: P and C of 5 cells. The GPU0 output is not kept in the repository; results/tifs/addendum/gpu0_eval_determinism.json records sha256_gpu0 == sha256_gpu1 for all 12 result files, and the official files' sha256 now equal sha256_gpu1, so the GPU0 hit vectors are the official files' bytes (`gpu0_source`)"),
        ("N3", "same-seed rerun: teacher re-scored | the same poisoned / clean teacher on the same trigger inputs in different Stage 2 jobs (runs), vs the first run; also the t43 clean teacher in the A2 vs A/A job"),
        ("N4", "precision only | the same student, CPU fp32 re-score vs official bf16 (own P, matched C, R)"),
        ("S1", "same teacher, other student seed | C_t - C_s for seeds s < t of the battery runs (tifs_b_<fam>_<ds>, _s43, _s44)"),
        ("S2", "same teacher, other student seed | P_t - P_s, same units as S1"),
        ("S1h / S2h", "same, A8 hidden-state KD (extra) | C_t - C_s / P_t - P_s of the admitted hidden seeds"),
        ("S1r / S2r", "same, A2 build t43 (extra) | C'_t - C'_s / P'_t - P'_s of the A2 replicate"),
        ("F1", "'feature' - logit | P_feature - P_logit and C_feature - C_logit, same teacher, seed 42 (clean and contaminated corpora); only the data order of epochs 2-5 and a lambda = 1e-3 [CLS] hint differ (A8)"),
        ("B1", "independent clean builds | C - C' between clean-base teachers t42 / t43 / t44 on the BadNets trigger (clean, badnets10) and the ATBA phrase (clean; badnets10 t43 - t44 only), and battery C (t42) - A2 C' (t43) at seeds 43 / 44 (ATBA SST-2)"),
        ("B2", "independent poisoned builds | P - P' (BadNets t42 / t43 / t44; ATBA SST-2 t42 vs A2 t43 at seeds 42-44)"),
        ("X1", "within-build null swap (A3.4) | every student without the trigger's backdoor (other attacks' P, other trainers' C; the frozen A3.4 null set) - the matched C, per seed, CPU fp32, Tango; `trigger_word_overlap` (extra) flags null P students whose own trigger words overlap the evaluated trigger's"),
        ("X2", "within-build null swap, frozen A3.4 formula | the A3.4 null rows (frozen cross_tables: seed-pooled over seeds, else transported); `trigger_word_overlap` as X1"),
        ("Q1", "CE coefficient (A10-a, extra) | R_B (L = 0.5 CE) - R_C (L = CE), same seed, initialization and data order, per trigger input set and untriggered"),
    ):
        src, contr = text.split(" | ", 1)
        w(f"| {code} | {src} | {contr} |")
    w("")
    w("`validity_contrasts.csv` columns: source_code, source, extra_source, fam, dataset, kd, corpus, corpus_group, trigger, "
      "trigger_group (atba / non-atba), seed, run, a_label, b_label, a_model, b_model, teacher_qualified, null_class "
      "(the cell's class: structural null = EP SST-2; embedding-only pair = SOS SST-2), precision, n, a_rate, b_rate, k, lo, "
      "hi, delta_star, n10, n01, verdict@delta (Tango), transported_variance_from, transported_lo / hi / verdict@delta, "
      "transported_rule, transported_formula_* (structural-null teacher swaps only), note, gpu0_source (N2), "
      "trigger_word_overlap (X1, X2); T2, T6f and X2 rows carry the frozen-rule k, lo, hi, verdicts, method and "
      "variance_from instead (and k_s<seed>; T6f also excess_over_null_students, excess_s<seed>, n_null_students, "
      "n_null_students_by_seed).")
    w("")
    w("`validity_summary.csv`: per source x subgroup (dataset, corpus_group, trigger_group; 'all' = pooled; a subgroup is listed "
      "only when its rows differ from a listed one) x basis (Tango; transported; frozen rule for T2 / T6f / X2): n_contrasts, "
      "n_distinct_model_pairs, n_structural_null (rows of the structural-null cell), |k| mean / median / p90 / max, "
      "share_abs_k_ge@delta, n_equiv / n_posit / n_negat / n_incon @delta, share_not_equivalent@delta, "
      "share_decisive_nonnull@delta (POSITIVE or NEGATIVE; independent of the contrast's orientation); transported rows also "
      "n_transported_tango_kept.")
    w("")
    w("Key numbers (dataset = all; verdict counts E/P/N/I; 'tr' = transported basis):")
    w("")
    w("| code | corpus | trigger | n | model pairs | median abs k | max abs k | @0.02 | @0.05 | @0.10 | tr @0.05 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    order = []
    for r in vs:
        if r["source_code"] not in order:
            order.append(r["source_code"])
    for code in order:
        combos = [("all", "all"), ("all", "atba"), ("all", "non-atba")]
        if code in ("T5", "F1", "B1", "B2"):
            combos += [("clean corpus", "atba"), ("clean corpus", "non-atba"), ("clean corpus", "all"),
                       ("contaminated corpus", "all")]
        for cg, tg in combos:
            base = idx.get((code, "all", cg, tg, "Tango")) or idx.get((code, "all", cg, tg, "froze"))
            if base is None:
                continue
            tr = idx.get((code, "all", cg, tg, "trans"))
            w(f"| {code} | {cg} | {tg} | {base['n_contrasts']} | {base['n_distinct_model_pairs'] or '-'} | "
              f"{fx(base['abs_k_median'], signed=False)} | {fx(base['abs_k_max'], signed=False)} | "
              f"{evni(base, '0.02')} | {evni(base, '0.05')} | {evni(base, '0.1')} | {evni(tr, '0.05') if tr else '-'} |")
    w("")
    w("(T2, T6f and X2 show the frozen-rule verdicts in the Tango columns. A subgroup row is omitted when it has the same "
      "rows as one shown, e.g. a source with only one trigger group.)")
    w("")
    w("Largest |k| per source and trigger group (exact row in validity_contrasts.csv):")
    w("")
    w("| code | trigger group | contrast | k [95%] | verdict @0.02 / 0.05 / 0.10 | precision |")
    w("|---|---|---|---|---|---|")
    by = {}
    for r in vc:
        by.setdefault((r["source_code"], r["trigger_group"]), []).append(r)
    for code in order:
        for tg in ("atba", "non-atba"):
            rs = by.get((code, tg))
            if not rs:
                continue
            m = max(rs, key=lambda r: abs(float(r["k"])))
            w(f"| {code} | {tg} | {m['dataset']} {m['kd']} {m['corpus']} {m['trigger']} seed {m['seed']}: {m.get('a_label') or ''} - "
              f"{m.get('b_label') or ''} | {fx(m['k'])} [{fx(m['lo'])}, {fx(m['hi'])}] | "
              + " / ".join(ab(m[f"verdict@{t}"]) for t in TOLS) + f" | {m['precision']} |")
    w("")
    n4 = [r for r in vc if r["source_code"] == "N4"]
    for ds in ("sst2", "agnews"):
        rs = [r for r in n4 if r["dataset"] == ds]
        if rs:
            d = [int(r["n10"]) + int(r["n01"]) for r in rs]
            w(f"- N4 {ds}: {len(rs)} same-student CPU fp32 vs official bf16 comparisons; discordant inputs per comparison "
              f"max {max(d)} of {rs[0]['n']} (total {sum(d)}; {sum(x > 0 for x in d)} comparisons with any); max |k| "
              f"{max(abs(float(r['k'])) for r in rs):.4f}.")
    q = [r for r in vc if r["source_code"] == "Q1" and r["trigger"] != "untriggered"]
    if q:
        w(f"- Q1 (A10-a), the {len(q)} triggered input sets: "
          + "; ".join(f"@{t} " + "/".join(str(sum(r[f'verdict@{t}'] == v for r in q))
                                          for v in ("EQUIVALENT", "POSITIVE", "NEGATIVE", "INCONCLUSIVE")) for t in TOLS)
          + " (E/P/N/I), equal to the frozen addendum.ce_coef_tables and results/tifs/addendum/ce_coef.csv.")
    x2 = [r for r in vc if r["source_code"] == "X2"]
    pos = [r for r in x2 if r["verdict@0.05"] == "POSITIVE"]
    if x2:
        where = ("all on the ATBA trigger" if pos and all(r["trigger"] == "atba" for r in pos) else
                 "triggers " + ", ".join(sorted({r["trigger"] for r in pos})) if pos else "none")
        w(f"- X2 POSITIVE@0.05 null swaps: {len(pos)} of {len(x2)}, {where}: "
          + "; ".join(f"{r['dataset']}/{r['kd']} {r['trigger']} trigger, {r['a_label']} - {r['b_label']}: {fx(r['k'])} "
                      f"[{fx(r['lo'])}, {fx(r['hi'])}] ({'/'.join(ab(r[f'verdict@{t}']) for t in TOLS)})" for r in pos)
          + f". POSITIVE at any delta among the {sum(r['trigger_group'] == 'non-atba' for r in x2)} non-ATBA X2 swaps: "
          + f"{sum(any(r[f'verdict@{t}'] == 'POSITIVE' for t in TOLS) for r in x2 if r['trigger_group'] == 'non-atba')}.")
    t6f = [r for r in vc if r["source_code"] == "T6f" and r["trigger"] == "atba"]

    def nulls(r):
        by_seed = [x.split(":") for x in r["n_null_students_by_seed"].split(",")]
        if len({n for _, n in by_seed}) == 1:
            return f"{by_seed[0][1]} at {'each of ' if len(by_seed) > 1 else ''}{', '.join(s for s, _ in by_seed)}"
        return ", ".join(f"{n} at {s}" for s, n in by_seed)
    if t6f:
        w("- T6f (own ATBA P under the same A3.4 formula, CPU fp32): " + "; ".join(
            f"{r['dataset']}/{r['kd']} {fx(r['k'])} [{fx(r['lo'])}, {fx(r['hi'])}], excess over the null students "
            f"({nulls(r)}) {fx(r['excess_over_null_students'])}" for r in t6f)
          + ". The excess is the seed mean of P minus the mean of the null students scored at that seed; for SST-2 logit "
            "the EP P and the clean-ep C exist only at seed 42.")
    w("")
    # like-for-like at the ATBA operating point
    b1, t1_, x1, t6 = srow("B1", "atba"), srow("T1", "atba"), srow("X1", "atba"), srow("T6", "atba")
    t6f_ex = [float(r["excess_over_null_students"]) for r in t6f]
    w("**At the ATBA operating point, like for like** (|k| on the ATBA trigger or phrase; validity_summary.csv, "
      "dataset = all, trigger_group = atba, Tango basis):")
    w("")
    w(f"- clean-reference builds B1 (official bf16, {b1['n_contrasts']} contrasts): median {fx(b1['abs_k_median'], signed=False)}, "
      f"max {fx(b1['abs_k_max'], signed=False)}; the teacher swap T1 at the same precision ({t1_['n_contrasts']} cells): median "
      f"{fx(t1_['abs_k_median'], signed=False)}, max {fx(t1_['abs_k_max'], signed=False)}. The B1 median is "
      f"{float(b1['abs_k_median']) / float(t1_['abs_k_median']):.2f} of the T1 median.")
    w(f"- within-build null swaps X1 (CPU fp32, {x1['n_contrasts']} swaps): median {fx(x1['abs_k_median'], signed=False)}, "
      f"max {fx(x1['abs_k_max'], signed=False)}; the teacher swap at the same precision T6 (CPU fp32, {t6['n_contrasts']}): "
      f"median {fx(t6['abs_k_median'], signed=False)}, max {fx(t6['abs_k_max'], signed=False)}. The median null swap is "
      f"{float(x1['abs_k_median']) / float(t6['abs_k_median']):.2f} of the T6 median.")
    w(f"- largest within-build null swap {fx(x1['abs_k_max'], signed=False)} (X1, CPU fp32) vs largest teacher swap "
      f"{fx(t1_['abs_k_max'], signed=False)} (T1, bf16) and {fx(t6['abs_k_max'], signed=False)} (T6, CPU fp32).")
    if t6f_ex:
        w(f"- under the same A3.4 formula and precision, the own ATBA P exceeds the mean of its null students by "
          f"{fx(min(t6f_ex))} to {fx(max(t6f_ex))} (T6f, {len(t6f_ex)} rows).")
    w("")
    w(f"So on the ATBA phrase the median difference between independently built clean references is "
      f"{float(b1['abs_k_median']) / float(t1_['abs_k_median']):.2f} of the median teacher-swap k (both bf16), the largest "
      f"single null swap is {float(x1['abs_k_max']) / float(t6['abs_k_max']):.2f} of the largest teacher swap at the same "
      f"precision (CPU fp32), and the median null swap is {float(x1['abs_k_median']) / float(t6['abs_k_median']):.2f} of the "
      f"median teacher swap at the same precision. Non-ATBA medians: B1 "
      f"{fx(srow('B1', 'non-atba')['abs_k_median'], signed=False)}, X1 {fx(srow('X1', 'non-atba')['abs_k_median'], signed=False)}, "
      f"T1 {fx(srow('T1', 'non-atba')['abs_k_median'], signed=False)}, T6 {fx(srow('T6', 'non-atba')['abs_k_median'], signed=False)}.")
    w("")
    # trigger-word overlap in the A3.4 null sets
    ov = [r for r in vc if r["source_code"] in ("X1", "X2") and r.get("trigger_word_overlap")]
    if ov:
        def ovtxt(trig, own):
            rs = [r for r in ov if r["trigger"] == trig and r["a_label"].split("_")[0] == own]
            x1s = [r for r in rs if r["source_code"] == "X1"]
            x2s = [r for r in rs if r["source_code"] == "X2"]
            return (f"per seed (X1) " + ", ".join(f"{r['dataset']}/{r['kd']} s{r['seed']} {fx(r['k'])}" for r in x1s)
                    + "; frozen formula (X2) " + ", ".join(f"{r['dataset']}/{r['kd']} {fx(r['k'])} [{fx(r['lo'])}, {fx(r['hi'])}] "
                                                          + "/".join(ab(r[f'verdict@{t}']) for t in TOLS) for r in x2s))
        ov_pos = sum(r[f"verdict@{t}"] == "POSITIVE" for r in ov if r["source_code"] == "X2" for t in TOLS)
        w("**Trigger-word overlap in the A3.4 null sets** (registered null set unchanged; flagged in `trigger_word_overlap`). "
          "The swaps are not strictly free of the trigger's backdoor: the BadNets trigger words (cf, mn, bb, tq; one inserted) "
          "are a subset of the EP trigger words (cf, mn, bb, tq, mb; two inserted). So on the EP trigger the null set holds the "
          "BadNets P, and on the SST-2 BadNets trigger it holds the EP P (the AG News re-score covers the triggers "
          + ", ".join(cross["agnews"][4]) + " only). EP trigger, BadNets P: " + ovtxt("ep", "badnets")
          + ". BadNets trigger, EP P: " + ovtxt("badnets", "ep") + ". "
          + ("None of these X2 rows is POSITIVE at any delta, so no X2 POSITIVE count depends on them." if ov_pos == 0 else
             f"{ov_pos} of their X2 verdicts are POSITIVE."))
        w("")

    # ── item 4
    w("## Item 4 - 기준별 ATBA 표 (reference_atba_by_reference.csv, reference_atba_agreement.csv)")
    w("")
    w("P = the BERT logit-KD ATBA student of the poisoned teacher t42, student seed 42 (battery cell). References = clean-base "
      "students of three independently built clean teachers, corpus and student seed 42: C(t42) = the battery cell's C on the "
      "ATBA-triggered inputs; C'(t43), C'(t44) = the A/A clean runs' C' on the `atba_phrase_<ds>` control (the same phrase "
      "inserted as a prefix into the same non-target inputs; the frozen aa_table pairs them the same way; N1 shows the trigger "
      "and control inputs coincide). Per reference: Tango and transported (BERT <ds> logit pool) intervals and verdicts; a "
      "seed-pooled reading where the reference has student seeds 42-44 (t42: the frozen primary row; t43 on SST-2: P_s of the "
      "battery minus the A2 replicate's C'_s, frozen seed-pooled formula, BERT SST-2 logit pool); on the same row the reference "
      "clean teacher's hit rate on the ATBA phrase (`ref_teacher_atba_phrase_rate`: t42 = mean of the cell's clean-teacher hits "
      "TC, t43 / t44 = the control's teacher_c), its clean accuracy (`ref_teacher_clean_acc` = teacher_ca_null) and the "
      "reference student's clean accuracy. 'feature' KD is not included: the t43 / t44 clean teachers have no 'feature' "
      "students. A10-b (a second poisoned AG News ATBA build) was not run, so P is the t42 build throughout.")
    w("")
    w("| dataset | ref | C_ref | P - C_ref Tango [95%] | Tango @.02/.05/.10 | transported [95%] | tr @.02/.05/.10 | seed-pooled k [95%] @.02/.05/.10 | ref teacher ATBA-phrase rate | ref teacher clean acc | ref student clean acc |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rb:
        sp = (f"{fx(r['seedpooled_k'])} [{fx(r['seedpooled_lo'])}, {fx(r['seedpooled_hi'])}] "
              + "/".join(ab(r[f"seedpooled_verdict@{t}"]) for t in TOLS)) if r.get("seedpooled_k") else "-"
        w(f"| {r['dataset']} | {r['reference']} | {fx(r['C_ref_rate'], signed=False)} | {fx(r['tango_k'])} [{fx(r['tango_lo'])}, {fx(r['tango_hi'])}] | "
          + "/".join(ab(r[f"tango_verdict@{t}"]) for t in TOLS)
          + f" | [{fx(r['transported_lo'])}, {fx(r['transported_hi'])}] | " + "/".join(ab(r[f"transported_verdict@{t}"]) for t in TOLS)
          + f" | {sp} | {fx(r['ref_teacher_atba_phrase_rate'], signed=False)} | {fx(r['ref_teacher_clean_acc'], signed=False)} | "
          f"{fx(r['ref_student_clean_acc'], signed=False)} |")
    w("")
    w("P rates (student seed 42): " + ", ".join(sorted({f"{r['dataset']} {fx(r['P_rate'], signed=False)}" for r in rb})) + ".")
    w("")
    w("Context columns repeated on each row of a dataset: `A33_*` = the registered A3.3 reading (frozen "
      "`addendum.reference_atba_table`: primary k with the between-reference variance s2_ref of C(t42), C'(t43), C'(t44) added, "
      "Satterthwaite df) and `primary_robust_*` = the frozen primary row's robust_* columns (BadNets A/A reference variance): "
      + "; ".join(f"{r['dataset']}: A3.3 s_ref {fx(r.get('A33_s_ref'), signed=False)}, [{fx(r.get('A33_ref_lo'))}, {fx(r.get('A33_ref_hi'))}] "
                  + "/".join(ab(r.get(f'A33_ref_verdict@{t}')) for t in TOLS)
                  + f"; robust [{fx(r.get('primary_robust_lo'))}, {fx(r.get('primary_robust_hi'))}] @0.05 {ab(r.get('primary_robust_verdict@0.05'))}"
                  for r in rb if r["reference"] == "t42") + ".")
    w("")
    w("`reference_atba_agreement.csv`: per dataset x basis x delta the verdicts of t42 / t43 / t44, `all_available_agree` "
      "(undefined with fewer than two references), `n_alt_agree_with_t42` of `n_alt` (alternative references whose verdict "
      "equals the C(t42) verdict) and `n_reference_pairs_agree` of `n_reference_pairs` (pairwise); the delta = all row gives "
      "`agreement_rate_with_t42`, `pairwise_agreement_rate` (both over references x deltas) and `share_deltas_all_agree`. "
      "These count any two equal verdicts, INCONCLUSIVE = INCONCLUSIVE included. The `*decisive*` columns (extra) count a "
      "pair only when both verdicts are decisive (EQUIVALENT, POSITIVE or NEGATIVE) and equal: "
      "`n_alt_decisive_agree_with_t42`, `n_reference_pairs_decisive_agree`, `all_available_agree_decisive` per delta, and "
      "`decisive_agreement_rate_with_t42`, `pairwise_decisive_agreement_rate`, `share_deltas_all_agree_decisive` on the "
      "delta = all row.")
    w("")
    w("| dataset | basis | agreement with t42 | decisive agreement with t42 | pairwise agreement | pairwise decisive | deltas where all references agree (decisive) |")
    w("|---|---|---|---|---|---|---|")
    for r in ra:
        if r["delta"] == "all":
            def rate(k, num, den):
                return "-" if r.get(k) in (None, "", "None") else f"{float(r[k]):.3f} ({r[num]}/{r[den]})"
            sd = ("-" if r.get("share_deltas_all_agree") in (None, "", "None") else
                  f"{float(r['share_deltas_all_agree']):.3f} ({float(r['share_deltas_all_agree_decisive']):.3f}) of "
                  f"{r['n_deltas_with_2plus_references']}")
            w(f"| {r['dataset']} | {r['basis']} | {rate('agreement_rate_with_t42', 'n_alt_agree_with_t42', 'n_alt')} | "
              f"{rate('decisive_agreement_rate_with_t42', 'n_alt_decisive_agree_with_t42', 'n_alt')} | "
              f"{rate('pairwise_agreement_rate', 'n_reference_pairs_agree', 'n_reference_pairs')} | "
              f"{rate('pairwise_decisive_agreement_rate', 'n_reference_pairs_decisive_agree', 'n_reference_pairs')} | {sd} |")
    w("")
    nondec = [r for r in ra if r["delta"] != "all" and int(r["n_alt_agree_with_t42"]) != int(r["n_alt_decisive_agree_with_t42"])]
    if nondec:
        w("Agreements with t42 that are INCONCLUSIVE = INCONCLUSIVE: "
          + "; ".join(f"{r['dataset']} {r['basis']} @{r['delta']} ({int(r['n_alt_agree_with_t42']) - int(r['n_alt_decisive_agree_with_t42'])})"
                      for r in nondec) + ".")
        w("")

    # ── item 6
    w("## Item 6 - 전환 빈도 (flips_cells.csv, flips_pairs.csv, flips_battery_by_seed.csv, flips_summary.csv)")
    w("")
    w("A unit is one estimand read several times, differing only in the student seed, only in the teacher build, or only in "
      "the clean reference build; its members are those readings. **Sign flip**: two members' k have opposite signs (both "
      "nonzero); pairs with a k of exactly 0 are counted apart (`zero_involved`). **Verdict flip at delta**: the two verdicts "
      "differ. **Decisive flip**: both verdicts are EQUIVALENT / POSITIVE / NEGATIVE and differ. Bases: `Tango` (each member's "
      "frozen paired_contrast), `transported` (each member read alone with the frozen one-seed formula and the pool of "
      "pool_for; the EP SST-2 structural null keeps Tango as in the frozen rule), `seed-pooled` (frozen seed-pooled rows: the "
      "primary row, the A2 replicate row, and P_s - C'_s(t43) of item 4). Battery decisions per seed: `Tango` = the frozen "
      "`battery_table(audit_table(...))` (= battery_by_seed.csv); `transported` = each seed's cells read with the one-seed "
      "formula and decided by the frozen battery_table (Holm within the battery, intersection-union); for the one-seed "
      "batteries this equals the frozen primary decision (checked). `n_transfer_set_flip_pairs` counts seed pairs that both "
      "say TRANSFERS but with different Holm-flagged attacks.")
    w("")
    w("| source | basis | units | pairs | sign flips (zero-involved) | max abs k in a sign-flip pair | verdict flips @.02/.05/.10 | decisive flips @.02/.05/.10 | transfer-set flips @.02/.05/.10 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in fs:
        sf = "-" if r.get("n_sign_flip_pairs") in (None, "") else f"{r['n_sign_flip_pairs']} ({r.get('n_zero_involved_pairs')})"
        w(f"| {r['source']} | {r['basis']} | {r['n_units']} | {r['n_pairs']} | {sf} | {fx(r.get('max_abs_k_in_sign_flip_pairs'), signed=False)} | "
          + "/".join(r.get(f"n_verdict_flip_pairs@{t}") or "-" for t in TOLS) + " | "
          + "/".join(r.get(f"n_decisive_flip_pairs@{t}") or "-" for t in TOLS) + " | "
          + "/".join(r.get(f"n_transfer_set_flip_pairs@{t}") or "-" for t in TOLS) + " |")
    w("")
    w("Battery decision per student seed (seeds 42/43/44; Holm-flagged attacks in parentheses):")
    w("")
    for b in fb:
        w(f"- {b['basis']}, {b['fam']}/{b['dataset']}/{b['kd']} (seeds {b['seeds']}): "
          + "; ".join(f"@{t} {b[f'decisions@{t}']} ({b[f'transfers@{t}']})" for t in TOLS))
    w("")

    # ── item 7
    sat = min(min(float(r["rate_P1"]), float(r["rate_R1"])) for r in bf)
    p1r1 = max(abs(float(r["P1-R1"])) for r in bf)
    w("## Item 7 - 빌드별 요인 설계 (builds_factorial.csv, builds_factorial_spread.csv)")
    w("")
    w("Per family x dataset x teacher build (BadNets trigger, logit KD, corpus seed 42, student seed 42): P1, C1 = the build's "
      "poisoned / clean-teacher students on the BadNets 10% contaminated corpus (t42: tifs_g_<fam>_badnets10_s42; t43 / t44: "
      "tifs_aa_<fam>_badnets10_t<b>); P0, C0 = the same build's students on the clean corpus (t42: the battery tifs_b_<fam>_<ds>; "
      "t43 / t44: tifs_aa_<fam>_clean_t<b>); R1, R0 = the CE-only students (no teacher) of the t42 grid (tifs_g_<fam>_badnets10_s42) "
      "and battery (tifs_b_<fam>_<ds>), shared by every build because they do not depend on the teacher. Tango contrasts "
      "(k, lo, hi, verdicts at each delta): P1-C1, P1-R1, C1-R1, P0-C0, P0-R0, C0-R0, R1-R0. The teacher x corpus interaction "
      "(P1-C1)-(P0-C0) and the frozen companions C1-R1, P1-R1, R1-R0 (`boot_*`, percentile 95%) come from the frozen "
      "`analyze.interaction_table` (analyze.py has one: the joint input bootstrap `analysis.tifs.joint_bootstrap`, R = 5000, "
      "seed 0, as analyze.main runs it), called on the build's four cells relabelled as one grid block; it resamples inputs "
      "only. `factorial_check_*` = the frozen `analyze.factorial_table` on the same cells (equal to the Tango columns). "
      "A/A builds exist for BERT SST-2 and AG News (t43, t44) and Qwen3 SST-2 (t43); Qwen3 AG News has t42 only.")
    w("")
    w(f"Because P1 - C1 = (P1 - R1) - (C1 - R1) and P1, R1 are near saturation in every build (min {sat:.5f}; max |P1 - R1| "
      f"= {p1r1:.5f}), the contaminated-corpus k is minus the clean teacher's suppression C1 - R1, to within {p1r1:.5f}.")
    w("")
    w("| fam/ds | build | qualified | P1 | C1 | R1 | P1-C1 | C1-R1 [95%] | P1-R1 | P0 | C0 | R0 | P0-C0 | C0-R0 | P0-R0 | interaction (P1-C1)-(P0-C0) [95% boot] |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in bf:
        w(f"| {r['fam']}/{r['dataset']} | {r['build']} | {r['teacher_qualified']} | {fx(r['rate_P1'], signed=False)} | "
          f"{fx(r['rate_C1'], signed=False)} | {fx(r['rate_R1'], signed=False)} | {fx(r['P1-C1'])} | {fx(r['C1-R1'])} "
          f"[{fx(r['C1-R1_lo'])}, {fx(r['C1-R1_hi'])}] | {fx(r['P1-R1'])} | {fx(r.get('rate_P0'), signed=False)} | "
          f"{fx(r.get('rate_C0'), signed=False)} | {fx(r.get('rate_R0'), signed=False)} | {fx(r.get('P0-C0'))} | "
          f"{fx(r.get('C0-R0'))} | {fx(r.get('P0-R0'))} | {fx(r.get('boot_teacher_x_corpus'))} "
          f"[{fx(r.get('boot_teacher_x_corpus_lo'))}, {fx(r.get('boot_teacher_x_corpus_hi'))}] |")
    w("")
    w("Spread across builds (builds_factorial_spread.csv; min / max / range / SD per quantity and whether verdicts agree):")
    w("")
    for s in bsp:
        parts = []
        for name in ("P1-C1", "C1-R1", "P0-C0", "C0-R0", "P0-R0", "boot_teacher_x_corpus"):
            if s.get(f"{name}_range") not in (None, ""):
                parts.append(f"{name} range {float(s[f'{name}_range']):.4f} ({fx(s[f'{name}_min'])} to {fx(s[f'{name}_max'])})")
        agree = [f"{name}@{t}" for name in ("P1-C1", "C1-R1", "P0-C0", "C0-R0", "P0-R0") for t in TOLS
                 if s.get(f"{name}_verdict@{t}_agree") == "False"]
        w(f"- {s['fam']}/{s['dataset']} ({s['builds']}): " + "; ".join(parts)
          + f". Verdicts differ across builds at: {', '.join(agree) if agree else 'none'}.")
    w("")

    # ── selfcheck
    w("## Self-checks (selfcheck.csv)")
    w("")
    for c in sc:
        tag = "INFO" if c["passed"] in ("", "None") else ("PASS" if c["passed"] == "True" else "FAIL")
        w(f"- {tag}: {c['check']} (n = {c['n_compared']}, max abs diff = {c['max_abs_diff'] or '-'}, "
          f"mismatches = {c['n_mismatch']}){(' - ' + c['note']) if c.get('note') else ''}")
    w("")
    w("## Caveats")
    w("")
    for line in (
        "Post hoc: computed after the results were seen (A11). Nothing here replaces or changes a primary verdict, a battery "
        "decision or the paper-wide Holm; where a reading differs from the primary, the primary stands.",
        "Tango readings of nuisance contrasts carry input noise only; between two models on the same inputs they are "
        "descriptive. The transported reading applies the family's k-pool to contrasts that are not themselves k (C_t - C_s, "
        "R_B - R_C, teacher re-scores) as a common yardstick, not as a calibrated test.",
        "Rows are not independent: one clean-base C per run and seed serves several attacks (see n_distinct_model_pairs); "
        "S1 / F1 / B1 counts over attacks that share a C repeat one model pair on different trigger inputs. The S rows are "
        "oriented later seed minus earlier seed, so POSITIVE and NEGATIVE swap with orientation; 'decisive non-null' does not "
        "depend on it.",
        f"The large nuisance effects sit at the ATBA operating point: {len(readings)} readings of {len(distinct)} distinct clean "
        f"students on the ATBA phrase hit it at {min(rates):.4f}-{max(rates):.4f} (the A2 seed-42 C' and the A/A t43 C' are one "
        "model: sha256-identical weights, cprime_determinism.json, and identical hit vectors). Summaries are split into atba / "
        "non-atba for that reason; the like-for-like comparison is in item 3.",
        "The A3.4 null sets (X1, X2) are the registered ones; " + (
            f"{len(ov)} of their rows (on the EP and SST-2 BadNets triggers) pair a P student whose own trigger words overlap "
            "the evaluated trigger (see item 3, trigger-word overlap)." if ov else "no row has a trigger-word overlap."),
        "A/A builds exist only for BadNets (BERT SST-2 / AG News t43 and t44; Qwen3 SST-2 t43), logit KD, student seed 42. "
        "The ATBA references t43 / t44 enter only through the ATBA-phrase control of the A/A clean runs, plus the A2 "
        "replicate's t43 C' at seeds 43 / 44 on SST-2. A10-b was not run (its gate fired), so no second poisoned AG News "
        "ATBA build exists.",
        "The training-rerun evidence (N1) is one student (SST-2 clean-base t43, seed 42); N2 is one run re-evaluated on GPU0 "
        "(from the sha256 record unless --gpu0-runs is given); N3 are evaluation reruns of teachers, not of students.",
        "Rows marked CPU fp32 (T6, T6f, X1, X2, N4) are not interchangeable with the official bf16 values; N4 bounds the "
        "difference on the same students.",
        f"A8 hidden-state KD enters S1h / S2h and item 6 with its admitted seeds ({hid_seeds}); it is outside the global Holm "
        "family. Q1 (A10-a), T7, S1r / S2r are extra sources beyond the A11.3 list, as are the transported basis in item 6 "
        "and the decisive-agreement and trigger-overlap columns.",
        "Item 7 interaction intervals are input-bootstrap only (the frozen method); the spread over 2-3 builds is descriptive.",
    ):
        w(f"- {line}")
    w("")
    t6f_sst2 = next(r for r in t6f if r["dataset"] == "sst2" and r["kd"] == "logit")
    w("## Revision note")
    w("")
    draft = dt.datetime.strptime(DRAFT_UTC, "%Y-%m-%d %H:%M UTC").replace(tzinfo=dt.timezone.utc).timestamp()
    w(f"These tables were first computed in a session scratch copy on {DRAFT_UTC} and independently verified; they were "
      "promoted here with the changes below. The data are the same: the newest prediction file "
      f"({utc(os.path.getmtime(newest))}) is {'older' if os.path.getmtime(newest) < draft else 'NEWER'} than that draft.")
    w("")
    for line in (
        "Paths are repository-relative; inputs are results/tifs/report and results/tifs/addendum (both cross-score files from "
        "results/tifs/addendum, no fallback copy); outputs go to this folder.",
        "Structural null: the T1 / T6 EP SST-2 rows keep Tango in transported_* (frozen primary rule; as item 6); the one-seed "
        "values moved to transported_formula_*; new column transported_rule (and n_transported_tango_kept in the summary).",
        "T6f: new column n_null_students_by_seed (SST-2 logit ATBA: " + nulls(t6f_sst2) + "; the frozen n_null_students = "
        + t6f_sst2["n_null_students"] + " is kept).",
        "X1 / X2: new column trigger_word_overlap; X2 rows now carry null_class like the X1 rows of the same cell.",
        "Item 4: decisive-agreement columns added (the existing agreement columns are unchanged).",
        "Labels: the students of the t43 / t44 builds are primed on both sides of B1 / B2 / S1r / S2r rows (e.g. C'[clean-base t43]).",
        f"Model ids now include the dataset: {len(multi_runs)} runs hold SST-2 and AG News students, so {n_collide} ids of the "
        "earlier run | KD | role | seed form named two different students and n_distinct_model_pairs undercounted for the "
        "sources built on those runs (T3, T4, T5, F1, B1, B2).",
        "N2: built from the sha256 record of gpu0_eval_determinism.json (the GPU0 output is not in the repository); identical "
        "numbers; new column gpu0_source.",
        f"README: like-for-like ATBA comparison, {len(readings)} readings of {len(distinct)} distinct clean students on the "
        "ATBA phrase, null students per seed, the trigger-overlap caveat, row counts per file, the item 7 sign (k = minus the "
        "suppression), exact saturation bound.",
    ):
        w(f"- {line}")
    w("")
    with open(os.path.join(OUT, "README.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("README.md written,", len(L), "lines")


if __name__ == "__main__":
    main()
