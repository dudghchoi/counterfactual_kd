#!/usr/bin/env python3
"""A11 item 5 (판정 사다리): the verdict ladder of the 31 global-Holm cells at each delta.

Post hoc / registered auxiliary table (docs/tifs_addendum.md A11.5). It does not change any
primary verdict, battery decision or the paper-wide Holm. Every interval comes from the frozen
functions (counterfactual_kd.analysis.tifs.paired_contrast / t_contrast) or the addendum
functions (scripts/tifs/addendum.py), imported, never modified. Inputs: runs/tifs (Stage 2
predictions), results/tifs/report (analyze.py) and results/tifs/addendum (addendum.py).

Rungs (per cell and delta):
  tango_s<seed>  per-seed Tango score interval of P_s - C_s (paired_contrast), every student
                 seed the cell has (no seed variance at all; the pre-section-0.3 reading)
  primary        the frozen primary verdict (analyze.primary_table): seed-pooled, transported,
                 or the seed-42 Tango interval for the structural nulls
  df_min         same k and SE as the primary, t reference df = r - 1 (analyze's *_df_min,
                 recomputed with t_contrast at every delta; seed-pooled cells only)
  robust         same k, SE^2 + s2_ref (A/A between-teacher-seed variance of BadNets k,
                 analyze.reference_variance), pool df (analyze's robust_*, every delta)
  a31_max        A3.1: SE^2 = max(s2_a, s2_pool)/r + v, pool df (addendum.attack_variance_table)
  a31_own        A3.1: SE^2 = s2_a/r + v, df = r - 1
  a33            A3.3: SE^2 + s2_ref(ATBA phrase), Satterthwaite df (addendum.reference_atba_table);
                 ATBA cells only
  a2             A2: the teacher-seed-43 replicate k' (addendum.replicate_table); BERT SST-2
                 logit ATBA only. A different teacher build (poisoned and clean teacher both
                 replaced), read with A2's registered reading
  a10b           A10-b (BERT AG News logit ATBA replicate): not run, its registered gate fired
                 (A10 result record; runs/tifs/_status/tifs_a10b_go.failed). The text is
                 rebuilt from results/tifs/addendum/cross_null.csv and checked against the record.

Labels (A3 header rule; A11.5 applies it to robust_* too):
  pooling-dependent    df_min, a31_max or a31_own differs from the primary verdict
  reference-dependent  robust or a33 differs from the primary verdict
  A2 differences are listed separately as 'teacher-build' (A2's registered reading:
  replicated / not replicated / inconclusive); per-seed Tango differences are counted only
  (descriptive, not a registered label); the counts say how many cell-deltas labelling them
  would add.

Extra, labelled, not a rung: a36_extra is the A3.6 verdict (BERT AG News seed 42 transported
with the BERT SST-2 logit pool, addendum.agnews_plan_table) for the 12 BERT AG News cells. It
enters no label; the counts and a self-check show that every cell-delta where it differs from
the primary is already pooling-dependent.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # before any local import: no __pycache__ in the repository

import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from collections import Counter  # noqa: E402

from g3_common import (  # noqa: E402
    AD, A, ADDENDUM, CELL, REPORT, RUNS, TOLS, close, family31, out_dir, paired_contrast,
    read_csv, t_contrast, tkey, write_csv,
)

SHORT = {"EQUIVALENT": "EQU", "POSITIVE": "POS", "NEGATIVE": "NEG", "INCONCLUSIVE": "INC",
         "UNDECIDED": "UND", None: "-", "": "-"}

# A10 result record (docs/tifs_addendum.md A10, 10/24 20:05 KST), as the A11 ladder states it.
A10B_RECORD = ("not run: A10 gate no-go (A3.4 BERT AG News, 2/7 logit ATBA null swaps POSITIVE@0.05: "
               "BadNets +0.1085 [+0.0748, +0.1422], SynBkd +0.0883 [+0.0546, +0.1220]; CPU fp32 cross-scoring)")
A10B_CELL = ("bert", "agnews", "logit", "atba")
A10B_MARKER = os.path.join(RUNS, "_status", "tifs_a10b_go.failed")
ATTACK_NAME = {"addsent": "AddSent", "badnets": "BadNets", "synbkd": "SynBkd", "sos": "SOS",
               "ep": "EP", "atba": "ATBA", "ripples": "RIPPLe"}


def robust_ref(st, fam, ds):
    """analyze.primary_table's choice of the reference variance (robust_*)."""
    ref, ref_from = st.ref_var.get((fam, ds)), f"{fam}/{ds}"
    if ref is None:
        same_ds = [k for k in st.ref_var if k[1] == ds]
        if same_ds:
            k_max = max(same_ds, key=lambda k: st.ref_var[k]["s2"])
            ref, ref_from = st.ref_var[k_max], f"{k_max[0]}/{k_max[1]}"
    return ref, ref_from


def a10b_status(check):
    """The A10-b rung text, rebuilt from the A3.4 BERT AG News null swaps (cross_null.csv)."""
    nulls = [r for r in read_csv(os.path.join(ADDENDUM, "cross_null.csv"))
             if (r["fam"], r["dataset"], r["kd"], r["trigger"], r["role"]) == ("bert", "agnews", "logit", "atba", "null")]
    pos = [r for r in nulls if r["verdict@0.05"] == "POSITIVE"]

    def name(student):
        tag = student.split("/")[0]
        return tag if tag.startswith("clean-") else ATTACK_NAME.get(tag.split("_")[0], tag)

    text = (f"not run: A10 gate no-go (A3.4 BERT AG News, {len(pos)}/{len(nulls)} logit ATBA null swaps POSITIVE@0.05: "
            + ", ".join(f"{name(r['student'])} {float(r['k']):+.4f} [{float(r['lo']):+.4f}, {float(r['hi']):+.4f}]"
                        for r in pos)
            + "; CPU fp32 cross-scoring)")
    runs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RUNS, "tifs_rep_bert_atba_agnews*")))
    marker = json.load(open(A10B_MARKER)) if os.path.exists(A10B_MARKER) else {}
    check("A10-b gate marker runs/tifs/_status/tifs_a10b_go.failed records the no-go",
          str(marker.get("result", "")).startswith(f"no-go: {len(pos)}/{len(nulls)} null swaps POSITIVE@0.05"),
          str(marker.get("result", "missing")))
    check("A10-b: no tifs_rep_bert_atba_agnews* run exists", not runs, ",".join(runs))
    check("A10-b: the null swaps of cross_null.csv are the 7 backdoor-free students, all seed-pooled (s42-44)",
          len(nulls) == 7 and all(r["method"] == "seed-pooled" and r["seeds"] == "42,43,44" for r in nulls),
          f"{len(nulls)} rows")
    check("A10-b text from cross_null.csv == A10 result record", text == A10B_RECORD, text)
    if runs:
        text += "; runs present but not read here: " + ",".join(runs)
    return text


def main():
    out = out_dir(__doc__.splitlines()[0])
    st = AD.State(RUNS)
    qual, _ = family31(st)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # Frozen / addendum tables recomputed on the current runs, then checked against the CSVs.
    prim_csv = {tuple(r[c] for c in CELL): r for r in read_csv(os.path.join(REPORT, "primary.csv"))}
    a31 = {tuple(r[c] for c in CELL): r for r in AD.attack_variance_table(st)}
    a31_csv = {tuple(r[c] for c in CELL): r for r in read_csv(os.path.join(ADDENDUM, "attack_variance.csv"))}
    a33 = {("bert", r["dataset"], r["kd"], "atba"): r for r in AD.reference_atba_table(st) if "kd" in r}
    a33_csv = {("bert", r["dataset"], r["kd"], "atba"): r for r in read_csv(os.path.join(ADDENDUM, "reference_atba.csv"))
               if r.get("kd")}
    rep = AD.replicate_table(st)
    rep_csv = read_csv(os.path.join(ADDENDUM, "replicate.csv"))
    a2 = {("bert", "sst2", "logit", "atba"): rep[0]} if rep else {}
    audit = {(r["fam"], r["dataset"], r["kd"], r["attack"], int(r["seed"])): r
             for r in read_csv(os.path.join(REPORT, "audit.csv"))}
    a36 = {tuple(r[c] for c in CELL): r for r in AD.agnews_plan_table(st)[0]}
    a36_csv = {tuple(r[c] for c in CELL): r for r in read_csv(os.path.join(ADDENDUM, "agnews_plan.csv"))}
    a10b_text = a10b_status(check)

    for key, r in a31.items():
        c = a31_csv.get(key, {})
        check(f"A3.1 {'/'.join(key)} == attack_variance.csv",
              all(r[f"{p}verdict@{tkey(t)}"] == c.get(f"{p}verdict@{tkey(t)}") for p in ("max_", "own_") for t in TOLS)
              and all(close(r[x], c.get(x)) for x in ("max_lo", "max_hi", "own_lo", "own_hi")))
    check("A3.1 cell set == attack_variance.csv", set(a31) == set(a31_csv), f"{len(a31)} rows")
    for key, r in a33.items():
        c = a33_csv.get(key, {})
        check(f"A3.3 {'/'.join(key)} == reference_atba.csv",
              all(r[f"ref_verdict@{tkey(t)}"] == c.get(f"ref_verdict@{tkey(t)}") for t in TOLS)
              and all(close(r[x], c.get(x)) for x in ("ref_lo", "ref_hi", "ref_df", "s_ref")))
    check("A3.3 cell set == reference_atba.csv", set(a33) == set(a33_csv), f"{len(a33)} rows")
    if rep:
        check("A2 == replicate.csv",
              all(rep[0][f"verdict@{tkey(t)}"] == rep_csv[0][f"verdict@{tkey(t)}"] for t in TOLS)
              and all(close(rep[0][x], rep_csv[0][x]) for x in ("k", "se", "lo", "hi", "own_lo", "own_hi")))
    for key, r in a36.items():
        c = a36_csv.get(key, {})
        check(f"A3.6 {'/'.join(key)} == agnews_plan.csv",
              all(r[f"verdict@{tkey(t)}"] == c.get(f"verdict@{tkey(t)}") for t in TOLS)
              and close(r["lo"], c.get("lo")) and close(r["hi"], c.get("hi")))
    check("A3.6 cell set == agnews_plan.csv", set(a36) == set(a36_csv), f"{len(a36)} rows")

    wide, long_rows = [], []
    for pr in qual:
        key = tuple(pr[c] for c in CELL)
        fam, ds, kd, atk = key
        by_seed = st.groups[key]
        seeds = sorted(by_seed)
        r_seeds = len(pr["seeds"].split(","))
        pc_csv = prim_csv[key]
        tango = {s: paired_contrast(by_seed[s]["P"], by_seed[s]["C"], TOLS) for s in seeds}
        for s, pc in tango.items():
            au = audit.get((fam, ds, kd, atk, s))
            check(f"Tango {'/'.join(key)}/s{s} == audit.csv",
                  au is not None and all(pc[f"verdict@{tkey(t)}"] == au[f"verdict@{tkey(t)}"] for t in TOLS)
                  and close(pc["lo"], au["k_lo"]) and close(pc["hi"], au["k_hi"]))
        method = pr["method"]
        rungs = {}   # rung -> dict(lo, hi, se, df, verdicts{t}, note)
        for s, pc in tango.items():
            rungs[f"tango_s{s}"] = {"est": pc["k"], "lo": pc["lo"], "hi": pc["hi"], "se": None, "df": None,
                                    "v": {t: pc[f"verdict@{tkey(t)}"] for t in TOLS},
                                    "src": "paired_contrast (Tango), report/audit.csv"}
        rungs["primary"] = {"est": pr["k"], "lo": pr["lo"], "hi": pr["hi"], "se": pr.get("se"), "df": pr.get("df"),
                            "v": {t: pr[f"verdict@{tkey(t)}"] for t in TOLS},
                            "src": f"analyze.primary_table ({method})"}
        check(f"primary {'/'.join(key)} == primary.csv",
              all(pr[f"verdict@{tkey(t)}"] == pc_csv[f"verdict@{tkey(t)}"] for t in TOLS)
              and close(pr["lo"], pc_csv["lo"]) and close(pr["hi"], pc_csv["hi"]))
        if pr.get("se") and r_seeds >= 2:
            dm = t_contrast(pr["k"], pr["se"], r_seeds - 1, TOLS)
            rungs["df_min"] = {"est": pr["k"], "lo": dm["lo"], "hi": dm["hi"], "se": pr["se"], "df": r_seeds - 1,
                               "v": {t: dm[f"verdict@{tkey(t)}"] for t in TOLS},
                               "src": "t_contrast(k, se, r - 1) = analyze *_df_min"}
            check(f"df_min {'/'.join(key)} @0.05 == primary.csv",
                  dm["verdict@0.05"] == pc_csv["verdict_df_min@0.05"]
                  and close(dm["lo"], pc_csv["lo_df_min"]) and close(dm["hi"], pc_csv["hi_df_min"]))
        if pr.get("se") and method != "structural":
            ref, ref_from = robust_ref(st, fam, ds)
            if ref is not None:
                rb = t_contrast(pr["k"], (pr["se"] ** 2 + ref["s2"]) ** 0.5, pr["df"], TOLS)
                rungs["robust"] = {"est": pr["k"], "lo": rb["lo"], "hi": rb["hi"],
                                   "se": (pr["se"] ** 2 + ref["s2"]) ** 0.5, "df": pr["df"],
                                   "v": {t: rb[f"verdict@{tkey(t)}"] for t in TOLS},
                                   "src": f"t_contrast(k, sqrt(se^2 + s2_ref[{ref_from}]), pool df) = analyze robust_*",
                                   "ref_from": ref_from, "s_ref": ref["s2"] ** 0.5, "ref_df": ref["df"]}
                check(f"robust {'/'.join(key)} @0.05 == primary.csv",
                      rb["verdict@0.05"] == pc_csv["robust_verdict@0.05"] and ref_from == pc_csv["robust_from"]
                      and close(rb["lo"], pc_csv["robust_lo"]) and close(rb["hi"], pc_csv["robust_hi"]))
        if key in a31:
            x = a31[key]
            for p, name, df in (("max_", "a31_max", x["pool_df"]), ("own_", "a31_own", x["r"] - 1)):
                rungs[name] = {"est": x["k"], "lo": x[f"{p}lo"], "hi": x[f"{p}hi"], "se": None, "df": df,
                               "v": {t: x[f"{p}verdict@{tkey(t)}"] for t in TOLS},
                               "src": "addendum.attack_variance_table (A3.1)"}
        if key in a33:
            x = a33[key]
            rungs["a33"] = {"est": x["k"], "lo": x["ref_lo"], "hi": x["ref_hi"], "se": None, "df": x["ref_df"],
                            "v": {t: x[f"ref_verdict@{tkey(t)}"] for t in TOLS},
                            "src": "addendum.reference_atba_table (A3.3)", "s_ref": x["s_ref"]}
        if key in a2:
            x = a2[key]
            rungs["a2"] = {"est": x["k"], "lo": x["lo"], "hi": x["hi"], "se": x["se"], "df": x["pool_df"],
                           "v": {t: x[f"verdict@{tkey(t)}"] for t in TOLS},
                           "reading": {t: x[f"reading@{tkey(t)}"] for t in TOLS},
                           "src": "addendum.replicate_table (A2, teacher seed 43)"}

        for name, g in rungs.items():
            for t in TOLS:
                long_rows.append({"fam": fam, "dataset": ds, "kd": kd, "attack": atk, "method": method,
                                  "delta": t, "rung": name, "estimate": g["est"], "lo": g["lo"], "hi": g["hi"],
                                  "se": g["se"], "df": g["df"], "verdict": g["v"][t],
                                  "reading": (g.get("reading") or {}).get(t, ""), "source": g["src"]})

        for t in TOLS:
            pv = rungs["primary"]["v"][t]
            row = {"fam": fam, "dataset": ds, "kd": kd, "attack": atk, "method": method,
                   "seeds": pr["seeds"], "delta": t, "k_primary": pr["k"],
                   "primary_lo": pr["lo"], "primary_hi": pr["hi"]}
            for s in (42, 43, 44):
                row[f"tango_s{s}"] = rungs.get(f"tango_s{s}", {}).get("v", {}).get(t, "")
            row["primary"] = pv
            for name in ("df_min", "robust", "a31_max", "a31_own", "a33", "a2"):
                row[name] = rungs.get(name, {}).get("v", {}).get(t, "")
            row["a2_reading"] = (rungs.get("a2", {}).get("reading") or {}).get(t, "")
            row["a10b"] = a10b_text if key == A10B_CELL else ""
            tv = [rungs[f"tango_s{s}"]["v"][t] for s in seeds]
            row["n_seed_tango"] = len(tv)
            row["n_seed_tango_differ"] = sum(v != pv for v in tv)
            pool_src = [n for n in ("df_min", "a31_max", "a31_own") if n in rungs and rungs[n]["v"][t] != pv]
            ref_src = [n for n in ("robust", "a33") if n in rungs and rungs[n]["v"][t] != pv]
            build_src = [n for n in ("a2",) if n in rungs and rungs[n]["v"][t] != pv]
            row["pooling_dependent"] = bool(pool_src)
            row["pooling_sources"] = ",".join(pool_src)
            row["reference_dependent"] = bool(ref_src)
            row["reference_sources"] = ",".join(ref_src)
            row["teacher_build_differs"] = bool(build_src)
            row["label"] = "; ".join(x for x in (
                "pooling-dependent" if pool_src else "", "reference-dependent" if ref_src else "",
                "teacher-build (A2: " + row["a2_reading"] + ")" if build_src else "") if x)
            row["ladder"] = (" / ".join(f"s{s} {SHORT[rungs[f'tango_s{s}']['v'][t]]}" for s in seeds)
                             + f" > primary {SHORT[pv]}"
                             + "".join(f" > {n} {SHORT[rungs[n]['v'][t]]}" for n in
                                       ("df_min", "robust", "a31_max", "a31_own", "a33", "a2") if n in rungs)
                             + (" > a10b not run (A10 gate)" if key == A10B_CELL else ""))
            # Not registered: what labelling the per-seed Tango rung would add, and A3.6.
            row["seed_tango_differs"] = row["n_seed_tango_differ"] > 0
            row["pooling_dependent_if_tango_counted"] = row["pooling_dependent"] or row["seed_tango_differs"]
            row["a36_extra"] = a36[key][f"verdict@{tkey(t)}"] if key in a36 else ""
            row["a36_extra_differs"] = key in a36 and row["a36_extra"] != pv
            wide.append(row)

    # Per-cell label summary (deltas at which each label applies).
    per_cell = []
    for pr in qual:
        key = tuple(pr[c] for c in CELL)
        rs = [w for w in wide if tuple(w[c] for c in CELL) == key]
        per_cell.append({**dict(zip(CELL, key)), "method": pr["method"], "seeds": pr["seeds"],
                         **{f"primary@{tkey(w['delta'])}": w["primary"] for w in rs},
                         "pooling_dependent_at": ",".join(tkey(w["delta"]) for w in rs if w["pooling_dependent"]),
                         "reference_dependent_at": ",".join(tkey(w["delta"]) for w in rs if w["reference_dependent"]),
                         "teacher_build_differs_at": ",".join(tkey(w["delta"]) for w in rs if w["teacher_build_differs"]),
                         "seed_tango_differs_at": ",".join(tkey(w["delta"]) for w in rs if w["n_seed_tango_differ"]),
                         "a31_registered_pooling_dependent": a31[key]["pooling_dependent"] if key in a31 else "",
                         "a33_registered_reference_dependent": a33[key]["reference_dependent"] if key in a33 else ""})
        if key in a31:
            mine = ",".join(tkey(w["delta"]) for w in rs if any(w[n] not in ("", w["primary"]) for n in ("a31_max", "a31_own")))
            check(f"A3.1 label {'/'.join(key)} == attack_variance.csv pooling_dependent",
                  mine == (a31_csv[key]["pooling_dependent"] or ""), f"{mine!r}")
        if key in a33:
            mine = ",".join(tkey(w["delta"]) for w in rs if w["a33"] not in ("", w["primary"]))
            check(f"A3.3 label {'/'.join(key)} == reference_atba.csv reference_dependent",
                  mine == (a33_csv[key]["reference_dependent"] or ""), f"{mine!r}")
    a36_rows = [w for w in wide if w["a36_extra_differs"]]
    check("A3.6 (not a rung): every cell-delta where it differs from the primary is already pooling-dependent",
          all(w["pooling_dependent"] for w in a36_rows),
          "; ".join(f"{w['kd']}/{w['attack']}@{tkey(w['delta'])} ({w['pooling_sources']})" for w in a36_rows))

    # Counts over the 93 cell x delta verdicts.
    counts = []
    for t in TOLS + ("all",):
        ws = [w for w in wide if t == "all" or w["delta"] == t]
        add = [w for w in ws if w["seed_tango_differs"] and not w["pooling_dependent"]]
        counts.append({"delta": t, "cell_deltas": len(ws),
                       "primary_counts": dict(Counter(w["primary"] for w in ws)),
                       "pooling_dependent": sum(w["pooling_dependent"] for w in ws),
                       "reference_dependent": sum(w["reference_dependent"] for w in ws),
                       "both": sum(w["pooling_dependent"] and w["reference_dependent"] for w in ws),
                       "either": sum(w["pooling_dependent"] or w["reference_dependent"] for w in ws),
                       "teacher_build_differs": sum(w["teacher_build_differs"] for w in ws),
                       "seed_tango_differs_any_seed": sum(w["n_seed_tango_differ"] > 0 for w in ws),
                       "seed_tango_rows": sum(w["n_seed_tango"] for w in ws),
                       "seed_tango_rows_differ": sum(w["n_seed_tango_differ"] for w in ws),
                       **{f"differs_{n}": sum(w[n] not in ("", w["primary"]) for w in ws)
                          for n in ("df_min", "robust", "a31_max", "a31_own", "a33", "a2")},
                       **{f"available_{n}": sum(w[n] != "" for w in ws)
                          for n in ("df_min", "robust", "a31_max", "a31_own", "a33", "a2")},
                       # Not registered: the effect of labelling the per-seed Tango rung.
                       "tango_label_would_add_pooling_dependent": len(add),
                       "tango_label_would_add_by_primary": dict(Counter(w["primary"] for w in add)),
                       "tango_label_would_add_otherwise_unlabelled": sum(not w["label"] for w in add),
                       "pooling_dependent_if_tango_counted": sum(w["pooling_dependent_if_tango_counted"] for w in ws),
                       "either_if_tango_counted": sum(w["pooling_dependent_if_tango_counted"] or w["reference_dependent"]
                                                      for w in ws),
                       # Not a rung: A3.6 (BERT AG News cells only).
                       "a36_extra_available": sum(w["a36_extra"] != "" for w in ws),
                       "a36_extra_differs": sum(w["a36_extra_differs"] for w in ws),
                       "a36_extra_differs_not_pooling_dependent": sum(w["a36_extra_differs"] and not w["pooling_dependent"]
                                                                      for w in ws)})

    # Battery-level companion: per-seed battery decisions (battery_by_seed.csv) beside the
    # primary battery decision (battery.csv) and the frozen decide() applied to each rung's
    # p-values (descriptive; the battery decision stays the primary one).
    bseed = read_csv(os.path.join(REPORT, "battery_by_seed.csv"))
    bprim = {(b["fam"], b["dataset"], b["kd"]): b for b in read_csv(os.path.join(REPORT, "battery.csv"))}
    frozen_b = {(b["fam"], b["dataset"], b["kd"]): b for b in A.primary_battery(st.primary)}
    battery_rows = []
    for bkey in sorted(bprim):
        fam, ds, kd = bkey
        cells = [pr for pr in qual if (pr["fam"], pr["dataset"], pr["kd"]) == bkey]
        row = {"fam": fam, "dataset": ds, "kd": kd, "attacks": ",".join(pr["attack"] for pr in cells),
               "methods": bprim[bkey]["methods"]}
        for t in TOLS:
            row[f"by_seed@{tkey(t)}"] = "; ".join(
                f"s{b['seed']} {b[f'decision@{tkey(t)}']}" + (f" ({b[f'transfers@{tkey(t)}']})" if b[f"transfers@{tkey(t)}"] else "")
                for b in bseed if (b["fam"], b["dataset"], b["kd"]) == bkey)
            row[f"primary@{tkey(t)}"] = bprim[bkey][f"decision@{tkey(t)}"] + (
                f" ({bprim[bkey][f'transfers@{tkey(t)}']})" if bprim[bkey][f"transfers@{tkey(t)}"] else "")
            check(f"battery {'/'.join(bkey)} @{tkey(t)} recomputed == battery.csv",
                  frozen_b[bkey][f"decision@{tkey(t)}"] == bprim[bkey][f"decision@{tkey(t)}"]
                  and frozen_b[bkey][f"transfers@{tkey(t)}"] == bprim[bkey][f"transfers@{tkey(t)}"])
        # Rung variants: replace each non-structural cell's interval by the rung's when it
        # exists (structural cells keep their Tango p-values); decide() as analyze.py.
        for rung in ("df_min", "robust", "a31_max", "a31_own"):
            rs, missing = [], 0
            for pr in cells:
                key = tuple(pr[c] for c in CELL)
                se, df = None, None
                if pr["method"] == "structural" or not pr.get("se"):
                    rs.append(pr)
                    continue
                r_seeds = len(pr["seeds"].split(","))
                if rung == "df_min" and r_seeds >= 2:
                    se, df = pr["se"], r_seeds - 1
                elif rung == "robust":
                    ref, _ = robust_ref(st, fam, ds)
                    if ref is not None:
                        se, df = (pr["se"] ** 2 + ref["s2"]) ** 0.5, pr["df"]
                elif rung in ("a31_max", "a31_own") and key in a31:
                    x = a31[key]
                    av = st.att_var[key]
                    s2a, s2p, v = av["s_between"] ** 2, st.pools[tuple(pr["variance_from"].split("/"))]["s_between"] ** 2, av["input_se"] ** 2
                    r_ = x["r"]
                    se, df = (((max(s2a, s2p) / r_ + v) ** 0.5, x["pool_df"]) if rung == "a31_max"
                              else ((s2a / r_ + v) ** 0.5, r_ - 1))
                if se is None:
                    missing += 1
                    rs.append(pr)       # the primary interval stands in for a missing rung
                    continue
                res = t_contrast(pr["k"], se, df, TOLS)
                if rung in ("a31_max", "a31_own"):
                    p = "max_" if rung == "a31_max" else "own_"
                    check(f"battery {rung} {'/'.join(key)} interval == attack_variance_table",
                          close(res["lo"], a31[key][f"{p}lo"]) and close(res["hi"], a31[key][f"{p}hi"]))
                if rung == "df_min":
                    check(f"battery df_min {'/'.join(key)} == ladder rung",
                          close(res["lo"], next(x["lo"] for x in long_rows if (x["fam"], x["dataset"], x["kd"], x["attack"]) == key and x["rung"] == "df_min")))
                rs.append({**pr, **{k: v for k, v in res.items() if "@" in k}})
            n_dec = sum(1 for pr in cells if pr["method"] != "structural" and pr.get("se"))
            dec = A.decide(rs)
            for t in TOLS:
                row[f"{rung}@{tkey(t)}"] = ("n/a (rung not defined for these cells)" if missing == n_dec else
                                            dec[f"decision@{tkey(t)}"] + (
                                                f" ({dec[f'transfers@{tkey(t)}']})" if dec[f"transfers@{tkey(t)}"] else ""))
            row[f"{rung}_cells_kept_primary"] = missing
        battery_rows.append(row)

    write_csv(wide, os.path.join(out, "ladder.csv"))
    write_csv(long_rows, os.path.join(out, "ladder_intervals.csv"))
    write_csv(per_cell, os.path.join(out, "ladder_labels.csv"))
    write_csv(counts, os.path.join(out, "ladder_counts.csv"))
    write_csv(battery_rows, os.path.join(out, "ladder_battery.csv"))
    write_csv(checks, os.path.join(out, "ladder_selfchecks.csv"))
    bad = [c for c in checks if not c["ok"]]
    print(f"ladder: {len(wide)} cell x delta rows, {len(long_rows)} interval rows; "
          f"A10-b: {a10b_text}; checks {len(checks) - len(bad)}/{len(checks)} ok -> {out}")
    for c in bad:
        print("  FAILED", c)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
