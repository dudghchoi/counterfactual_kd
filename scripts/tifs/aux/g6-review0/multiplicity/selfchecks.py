#!/usr/bin/env python3
"""Self-checks of the A14 multiplicity subgroup, and the comparison with the review-agent previews.

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

selfchecks.csv: official numbers reproduced through this subgroup's code path (frozen functions on
runs/tifs), the frozen-file hashes, and the internal consistency of the outputs. Every row has ok.
review0_comparison.csv: the values the round-0 review agents previewed (paper_tifs/reviews/round0_meta.md
issue 4 / 5 / 6, experiment_options.md section 2, and the printouts of the scratch scripts review0/holm_eq.py
and holm_zero.py, re-run unchanged on results/tifs/report/primary.csv on 2026-10-26) against the values
recomputed here with the frozen functions; 'same' says whether they agree at the precision the review gave.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

import g6m_common as G  # noqa: E402

A, TOLS, ALPHA = G.A, G.TOLS, G.ALPHA
CHECKS = []
CCA = "aa_max_abs_C-C'@atba"


def check(name, expected, got, ok=None, tol=None, note=""):
    if ok is None:
        if tol is not None:
            ok = abs(float(expected) - float(got)) <= tol
        else:
            ok = expected == got
    CHECKS.append({"check": name, "expected": expected, "got": got, "tol": "" if tol is None else tol,
                   "ok": bool(ok), "note": note})
    return ok


def numbers_md():
    out = {}
    rx = re.compile(r"^\| `([^`]+)` \| ([^|]*) \| ([^|]*) \|")
    with open(os.path.join(G.ROOT, "paper_tifs", "NUMBERS.md")) as f:
        for ln in f:
            m = rx.match(ln)
            if m:
                out[m.group(1)] = m.group(3).strip()
    return out


def main():
    ap = G.add_out_arg(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = ap.parse_args()
    out = G.out_dir(args.out)
    # 0. frozen files
    for path, want in G.FROZEN_SHA256.items():
        check(f"sha256 {path} (freeze record / final-analysis snapshot)", want, G.sha256(os.path.join(G.ROOT, path)))
    before = os.path.join(out, "logs", "hashes_before.txt")
    if os.path.exists(before):
        for ln in open(before):
            h, p = ln.split()
            check(f"sha256 {p} unchanged during run_all.sh", h, G.sha256(os.path.join(G.ROOT, p)))
    snap = G.snapshot()
    fsnap = os.path.join(G.RUNS, "_code", "final_analysis_20261024T110542Z.json")
    import json
    check("prediction files in runs/tifs = final-analysis snapshot final_analysis_20261024T110542Z",
          json.load(open(fsnap))["preds_files_in_runs_tifs"], snap["n_preds_all"])
    st = G.state()
    qual = G.qualified_primary(st)
    # 1. primary table reproduced from runs/tifs through the frozen loader and primary_table
    off = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in G.read_csv(os.path.join(G.REPORT, "primary.csv"))}
    check("31 qualified primary cells (paper-wide Holm family)", 31, len(qual))
    check("primary.csv rows", 31, len(off))
    cols = ["k", "lo", "hi", "delta_star"] + [f"{p}@{t:g}" for p in ("p_up", "p_tost", "p_pos") for t in TOLS]
    md = 0.0
    for r in qual:
        o = off[(r["fam"], r["dataset"], r["kd"], r["attack"])]
        for c in cols + (["se", "df"] if r["method"] != "structural" else []):
            md = max(md, abs(float(r[c]) - float(o[c])))
        for t in TOLS:
            check(f"primary verdict@{t:g} {G.cell_id(r)}", o[f"verdict@{t:g}"], r[f"verdict@{t:g}"])
        check(f"primary method {G.cell_id(r)}", o["method"], r["method"])
    check("primary.csv k, se, df, lo, hi, delta*, p_up/p_tost/p_pos max |diff| (31 cells x all deltas)", 0.0, md, tol=1e-15)
    # every column of primary.csv (incl. raw_asr, C_trig, teacher_asr*, robust_*, *_df_min): numbers to 1e-15,
    # text exactly, a key missing from the frozen row = an empty field
    pcols = list(next(iter(off.values())).keys())
    nmis, bad = 0, []
    for r in qual:
        o = off[(r["fam"], r["dataset"], r["kd"], r["attack"])]
        for c in pcols:
            v, ov = r.get(c), o[c]
            if v is None or v == "":
                same = ov in ("", None)
            elif isinstance(v, bool) or isinstance(v, str):
                same = str(v) == ov
            else:
                try:
                    same = abs(float(v) - float(ov)) <= 1e-15
                except ValueError:
                    same = False
            if not same:
                nmis += 1
                bad.append(f"{G.cell_id(r)}:{c}")
    check("primary.csv every column (31 rows), mismatching values", f"{len(pcols)} columns x 31 rows, 0 mismatches",
          f"{len(pcols)} columns x {len(qual)} rows, {nmis} mismatches", note="; ".join(bad[:10]))
    # 2. battery decisions and transfers_global (frozen primary_battery)
    bat = {(b["fam"], b["dataset"], b["kd"]): b for b in A.primary_battery(st.primary)}
    for o in G.read_csv(os.path.join(G.REPORT, "battery.csv")):
        b = bat[(o["fam"], o["dataset"], o["kd"])]
        for t in TOLS:
            check(f"battery.csv decision@{t:g} {o['fam']}/{o['dataset']}/{o['kd']}", o[f"decision@{t:g}"], b[f"decision@{t:g}"])
            check(f"battery.csv transfers_global@{t:g} {o['fam']}/{o['dataset']}/{o['kd']}",
                  o[f"transfers_global@{t:g}"], b[f"transfers_global@{t:g}"])
    # 3. global_holm.csv through holm_family.csv family a
    hf = G.read_csv(os.path.join(out, "holm_family_cells.csv"))
    gh = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in G.read_csv(os.path.join(G.ADDENDUM, "global_holm.csv"))}
    md = 0.0
    nflag = 0
    for r in hf:
        if r["family"] != "a_frozen31":
            continue
        key = tuple(r["cell"].split("/"))
        o = gh[key]
        t = f"{float(r['delta']):g}"
        md = max(md, abs(float(r["holm_p"]) - float(o[f"holm_global@{t}"])))
        nflag += G.as_bool(r["transfers_global"]) != G.as_bool(o[f"transfers_global@{t}"])
    check("global_holm.csv holm_global (31 cells x 3 deltas) max |diff|, family a_frozen31", 0.0, md, tol=1e-15)
    check("global_holm.csv transfers_global flags differing", 0, nflag)
    # 4. headline (analyze.summary)
    summ = open(os.path.join(G.REPORT, "summary.md")).read()
    m = re.search(r"- (\d+) cells; every upper bound below 0.05: \*\*(yes|no)\*\* \(max one-sided p = ([0-9.e-]+)\)", summ)
    head = [r for r in qual if r["kd"] == "logit"]
    pmax = max(r["p_up@0.05"] for r in head)
    check("summary.md headline cells", int(m.group(1)), len(head))
    check("summary.md headline met", m.group(2), "yes" if pmax < ALPHA else "no")
    check("summary.md headline max one-sided p", m.group(3), f"{pmax:.3g}")
    # 5. NUMBERS.md Finding 1 counts through the A4 classification of claims_cells.csv
    nm = numbers_md()
    cc = G.read_csv(os.path.join(out, "claims_cells.csv"))
    for t, dd in ((0.02, "02"), (0.05, "05"), (0.10, "10")):
        rows = [r for r in cc if abs(float(r["delta"]) - t) < 1e-12 and r["kd"] == "logit" and r["verdict"] == "EQUIVALENT"]
        nonst = [r for r in rows if r["a4_class"] != "structural"]
        absent = [r for r in nonst if r["a4_class"] == "dagger"]
        counted = [r for r in nonst if r["a4_class"] in ("counted", "boundary")]
        bound = [r for r in counted if r["a4_class"] == "boundary"]
        check(f"NUMBERS narrative.f1.n_eq{dd}_incl_struct", nm[f"narrative.f1.n_eq{dd}_incl_struct"], str(len(rows)))
        check(f"NUMBERS narrative.f1.n_eq{dd}_logit", nm[f"narrative.f1.n_eq{dd}_logit"], str(len(nonst)))
        check(f"NUMBERS narrative.f1.n_absent{dd}", nm[f"narrative.f1.n_absent{dd}"], str(len(absent)))
        check(f"NUMBERS narrative.f1.n_counted{dd}", nm[f"narrative.f1.n_counted{dd}"], str(len(counted)))
        check(f"NUMBERS narrative.f1.n_counted{dd}_nobound", nm[f"narrative.f1.n_counted{dd}_nobound"],
              str(len(counted) - len(bound)))
        want = nm[f"narrative.f1.counted{dd}"]
        got = ",".join(sorted(r["cell"].replace("/", ".") for r in counted))
        check(f"NUMBERS narrative.f1.counted{dd} (cell ids)", ",".join(sorted(want.split(","))) if want else "", got)
        want = nm[f"narrative.f1.boundary{dd}"]
        check(f"NUMBERS narrative.f1.boundary{dd}", want, ",".join(r["cell"].replace("/", ".") for r in bound))
    # 6. g3 simultaneous_summary.csv through claims_family.csv (Bonferroni, F31)
    cf = G.read_csv(os.path.join(out, "claims_family.csv"))
    ss = {f"{float(r['delta']):g}": r for r in G.read_csv(os.path.join(G.AUX, "g3-ladder-seeds-simultaneous",
                                                                      "simultaneous_summary.csv"))}
    for r in cf:
        if r["family"] != "F31" or r["procedure"] != "bonferroni":
            continue
        t = f"{float(r['delta']):g}"
        if r["side"] == "equivalent":
            check(f"g3 simultaneous_summary bonferroni_equivalent @{t}", ss[t]["bonferroni_equivalent"], r["n_claims"])
        if r["side"] == "transfer":
            check(f"g3 simultaneous_summary bonferroni_positive @{t}", ss[t]["bonferroni_positive"], r["n_claims"])
        if r["side"] == "transfer" and r["procedure"] == "bonferroni":
            pass
    for r in cf:
        if (r["family"], r["procedure"], r["side"]) == ("F31", "holm", "transfer"):
            t = f"{float(r['delta']):g}"
            check(f"g3 simultaneous_summary holm_transfers_global @{t}", ss[t]["holm_transfers_global"], r["n_claims"])
        if (r["family"], r["procedure"], r["side"]) == ("F31", "unadjusted", "equivalent"):
            t = f"{float(r['delta']):g}"
            check(f"g3 simultaneous_summary primary_equivalent @{t}", ss[t]["primary_equivalent"], r["n_claims"])
    # 7. delta rationale: g3 floors and g1 Cohen's h
    dr = G.read_csv(os.path.join(out, "delta_rationale_cells.csv"))
    md = max(abs(float(r["floor_battery_r3"]) - float(r["floor_battery_r3_recomputed"])) for r in dr if r["floor_battery_r3"])
    check("aux/g3 seeds_floor.csv floor (r = 3) recomputed with frozen t_contrast, max |diff|", 0.0, md, tol=1e-12)
    md = max(abs(float(r["cohen_h"]) - float(r["cohen_h_g1"])) for r in dr)
    check("aux/g1 exposure_primary_cells.csv h_PC_at_mean_rates vs cohen_h, max |diff|", 0.0, md, tol=1e-12)
    fl = {r["pool"]: r for r in G.read_csv(os.path.join(G.AUX, "g3-ladder-seeds-simultaneous", "seeds_rmin.csv"))
          if r["scope"].startswith("31-cell") and r["delta"] == "0.02"}
    check("NUMBERS narrative.f1.floor3_min = min g3 floor_r3", nm["narrative.f1.floor3_min"],
          f"{min(float(r['floor_r3']) for r in fl.values()):.6g}")
    # 8. S1c compact table = aux/g4 values; A11 result-record numbers
    s1 = G.read_csv(os.path.join(out, "s1c_configH_main.csv"))
    g4b = G.read_csv(os.path.join(G.AUX, "g4-s1c-s9-s1extra", "s1c_battery.csv"))
    nmis = 0
    for r in s1:
        if r["level"] != "battery":
            continue
        o = [x for x in g4b if x["rule"] == r["rule"] and x["battery"] == r["battery"] and x["config"] == "H"
             and x["c_sd"] == (r["c_sd"] if r["c_sd"] else "") and f"{float(x['delta']):g}" == r["delta"]]
        nmis += (len(o) != 1) or any(o[0][c] != r[c] for c in ("false_transfers_all0", "no_transfer_all0",
                                                             "false_no_transfer_one_at_d_max"))
    check("s1c_configH_main.csv battery rows equal aux/g4 s1c_battery.csv", 0, nmis)
    cell = {(r["battery"], r["attack"]): r for r in s1 if r["level"] == "cell"}
    check("A11 record: S1c H BERT AG News EP coverage 0.6714", "0.6714",
          f"{float(cell[('BERT AG News logit', 'ep')]['coverage']):.4f}")
    check("A11 record: S1c H BERT AG News ATBA coverage 0.8147", "0.8147",
          f"{float(cell[('BERT AG News logit', 'atba')]['coverage']):.4f}")
    br = [r for r in s1 if r["level"] == "battery" and r["battery"] == "BERT AG News feature" and r["c_sd"] == "1.0"
          and r["delta"] == "0.02"][0]
    check("A11 record: S1c transported H BERT AG News feature false TRANSFERS @0.02 = 0.0512 (0.05115)", "0.05115",
          br["false_transfers_all0"])
    # 9. headline simulation vs aux/g4 (same model, independent draws)
    hv = G.read_csv(os.path.join(out, "headline_power_validation.csv"))
    zmax = max(abs(float(r["z_vs_s1c"])) for r in hv)
    check(f"headline_power r = 3 measured vs aux/g4 s1c_battery.csv: all |z| < 4 over {len(hv)} rates", True,
          all(G.as_bool(r["ok_abs_z_lt_4"]) for r in hv), note=f"max |z| = {zmax:.2f}")
    hp = G.read_csv(os.path.join(out, "headline_power.csv"))
    for r in hp:
        nts = [float(v) for k, v in r.items() if k.startswith("no_transfer[")]
        check(f"headline <= min battery NO TRANSFER ({r['config']} {r['sigma_mode']} r={r['r_seeds_pooled_batteries']} "
              f"d={r['delta']})", True, float(r["p_headline_met"]) <= min(nts) + 1e-12)
    # 10. no bytecode
    pyc = [os.path.join(d, f) for base in (G.HERE, G.G4_DIR) for d, _, fs in os.walk(base) for f in fs
           if f.endswith(".pyc")]
    check("no .pyc / __pycache__ under this code folder or aux/g4", 0, len(pyc), note="; ".join(pyc))
    G.write_csv(CHECKS, os.path.join(out, "selfchecks.csv"),
                "Self-checks of the A14 multiplicity subgroup: official numbers (report/primary.csv, battery.csv, "
                "summary.md headline, addendum/global_holm.csv, aux/g3 simultaneous_summary.csv and seeds_floor.csv, "
                "aux/g1 Cohen's h, aux/g4 S1c, paper_tifs/NUMBERS.md Finding 1 counts) reproduced through this code "
                "path, frozen-file SHA-256, headline simulation vs aux/g4 and internal consistency; ok = pass.")
    review_comparison(out, cf, cc, dr, hp)
    bad = [c for c in CHECKS if not c["ok"]]
    print(f"selfchecks: {len(CHECKS) - len(bad)}/{len(CHECKS)} ok", flush=True)
    for c in bad:
        print("  FAIL", c, flush=True)
    if bad:
        sys.exit(1)


def review_comparison(out, cf, cc, dr, hp):
    def fam(family, side, proc, t, field):
        for r in cf:
            if (r["family"], r["side"], r["procedure"]) == (family, side, proc) and abs(float(r["delta"]) - t) < 1e-12:
                return r[field]
        raise KeyError

    rows = []

    def add(item, source, review, mine, same, note=""):
        rows.append({"item": item, "review_source": source, "review_value": review, "this_value": mine,
                     "same": same, "note": note})

    src_meta = "paper_tifs/reviews/round0_meta.md issue 4"
    m = "; ".join(r["display"] for r in cc if abs(float(r["delta"]) - 0.05) < 1e-12 and r["a4_class"] != "structural"
                  and G.as_bool(r["holm_F31_below"]))
    add("delta 0.05, 31-cell Holm on p_up: non-structural cells", src_meta,
        "3 (GPT-2 SST-2 BadNets dagger, Qwen3 SST-2 BadNets dagger, Qwen3 SST-2 AddSent)",
        f"{fam('F31', 'below', 'holm', 0.05, 'n_nonstructural')} ({m})",
        fam("F31", "below", "holm", 0.05, "n_nonstructural") == "3")
    add("delta 0.05, 31-cell Holm on p_up: A4-counted", src_meta, "1 (Qwen3 SST-2 AddSent)",
        f"{fam('F31', 'below', 'holm', 0.05, 'n_a4_counted')} ({fam('F31', 'below', 'holm', 0.05, 'cells_a4_counted')})",
        fam("F31", "below", "holm", 0.05, "n_a4_counted") == "1")
    add("delta 0.05, logit-19 Holm on p_up: non-structural cells", src_meta, "4 (adds GPT-2 SST-2 AddSent)",
        f"{fam('F19', 'below', 'holm', 0.05, 'n_nonstructural')}", fam("F19", "below", "holm", 0.05, "n_nonstructural") == "4")
    add("delta 0.05, logit-19 Holm on p_up: A4-counted", src_meta, "2",
        f"{fam('F19', 'below', 'holm', 0.05, 'n_a4_counted')} ({fam('F19', 'below', 'holm', 0.05, 'cells_a4_counted')})",
        fam("F19", "below", "holm", 0.05, "n_a4_counted") == "2")
    add("delta 0.10, 31-cell Holm on p_up: non-structural logit cells", src_meta, "12",
        fam("F31", "below", "holm", 0.10, "n_nonstructural_logit"),
        fam("F31", "below", "holm", 0.10, "n_nonstructural_logit") == "12",
        f"plus 1 non-structural 'feature' cell: {fam('F31', 'below', 'holm', 0.10, 'n_nonstructural')} non-structural in all")
    add("delta 0.05, Bonferroni simultaneous EQUIVALENT", src_meta, "13 -> 2 (both structural)",
        f"{fam('F31', 'equivalent', 'unadjusted', 0.05, 'n_claims')} -> {fam('F31', 'equivalent', 'bonferroni', 0.05, 'n_claims')} "
        f"(structural {fam('F31', 'equivalent', 'bonferroni', 0.05, 'n_structural')})",
        fam("F31", "equivalent", "unadjusted", 0.05, "n_claims") == "13" and fam("F31", "equivalent", "bonferroni", 0.05, "n_claims") == "2")
    hold = {"F31": {"below": ("5/2", "13/5", "23/15"), "equivalent": ("2", "2", "14")},
            "F19": {"below": ("4/1", "12/5", "16/15"), "equivalent": ("1", "1", "15")}}
    for f, d in hold.items():
        for side, vals in d.items():
            for t, v in zip(TOLS, vals):
                if side == "below":
                    mine = f"{fam(f, side, 'unadjusted', t, 'n_claims')}/{fam(f, side, 'holm', t, 'n_claims')}"
                else:
                    mine = fam(f, side, "holm", t, "n_claims")
                add(f"review0/holm_eq.py {f} delta {t:g}: {'raw/Holm upper<d' if side == 'below' else 'Holm TOST'}",
                    "scratchpad review0/holm_eq.py printout", v, mine, v == mine,
                    "holm_eq.py gives the structural nulls p = 1e-12; the frozen rows carry the Tango p "
                    "(p_up 8.0e-06 / 2.1e-12 / 0 at 0.02 / 0.05 / 0.10); counts unaffected")
    # holm_zero: k > 0
    z31 = {r["cell"]: r for r in dr}
    z_rev31 = {"bert/agnews/feature/atba": 0.0016, "bert/agnews/logit/atba": 0.0000, "bert/agnews/logit/ep": 0.0022,
               "bert/sst2/feature/atba": 0.0174, "bert/sst2/logit/atba": 0.0072}
    mine31 = sorted(c for c, r in z31.items() if G.as_bool(r["k_gt_0_holm31"]))
    add("review0/holm_zero.py: k > 0 after 31-cell Holm", "scratchpad review0/holm_zero.py printout",
        "; ".join(f"{k}:{v:.4f}" for k, v in z_rev31.items()),
        "; ".join(f"{c}:{float(z31[c]['holm31_p_pos0']):.4f}" for c in mine31),
        sorted(z_rev31) == mine31 and all(abs(float(z31[c]["holm31_p_pos0"]) - v) < 6e-5 for c, v in z_rev31.items()))
    z_rev19 = {"bert/agnews/logit/atba": 0.0000, "bert/agnews/logit/ep": 0.0014, "bert/sst2/logit/atba": 0.0043,
               "qwen3/sst2/logit/synbkd": 0.0235}
    mine19 = sorted(c for c, r in z31.items() if G.as_bool(r["k_gt_0_holm19"]))
    add("review0/holm_zero.py: k > 0 after logit-19 Holm", "scratchpad review0/holm_zero.py printout",
        "; ".join(f"{k}:{v:.4f}" for k, v in z_rev19.items()),
        "; ".join(f"{c}:{float(z31[c]['holm19_p_pos0']):.4f}" for c in mine19),
        sorted(z_rev19) == mine19 and all(abs(float(z31[c]["holm19_p_pos0"]) - v) < 6e-5 for c, v in z_rev19.items()))
    # delta rationale numbers
    bats = {r["battery"]: r for r in G.read_csv(os.path.join(out, "delta_rationale_batteries.csv"))}
    for b, v in (("bert/agnews/logit", "0.033"), ("bert/sst2/logit", "0.036"), ("gpt2/sst2/logit", "0.021"),
                 ("qwen3/sst2/logit", "0.025")):
        mine = f"{float(bats[b]['floor_r3']):.3f}"
        add(f"detection floor r = 3, {b}", "round0_meta.md issue 4 / experiment_options.md C10", v, mine, v == mine)
    aa = max(float(bats[b]["aa_max_abs_C-C'"]) for b in bats if bats[b]["aa_max_abs_C-C'"])
    add("A/A clean between-build variation (max |C - C'|, BadNets trigger)", "round0_meta.md issue 4", "0.022",
        f"{aa:.4f}", f"{aa:.3f}" == "0.022",
        "at the ATBA phrase the same builds give |C - C'| up to "
        f"{max(float(bats[b][CCA]) for b in bats if bats[b][CCA]):.4f}")
    for cell, v in (("bert/sst2/logit/addsent", "+0.052 [+0.014, +0.089]"), ("qwen3/sst2/logit/synbkd", "+0.054 [+0.027, +0.081]")):
        r = z31[cell]
        mine = f"{float(r['k']):+.3f} [{float(r['lo']):+.3f}, {float(r['hi']):+.3f}]"
        add(f"'k > 0 significant but inside or at delta': {cell}", "round0_meta.md issue 4", v, mine, v == mine,
            f"class at 0.05: {r['sig_pos_class@0.05']}; Holm-31 p(k > 0) = {float(r['holm31_p_pos0']):.4f}")
    others = sorted({r["cell"] for r in G.read_csv(os.path.join(out, "delta_rationale_sig_inside.csv"))} -
                    {"bert/sst2/logit/addsent", "qwen3/sst2/logit/synbkd"})
    add("other cells with lo > 0 not POSITIVE at some delta", "round0_meta.md issue 4 (lists 2)", "not listed",
        "; ".join(others), False, "see delta_rationale_sig_inside.csv; the review listed only the two logit cells at 0.05")
    # headline power
    g4b = G.read_csv(os.path.join(G.AUX, "g4-s1c-s9-s1extra", "s1c_battery.csv"))
    for cfg, v in (("E", "0.16"), ("H", "0.23")):
        r = [x for x in hp if x["config"] == cfg and x["sigma_mode"] == "measured" and x["delta"] == "0.05"][0]
        prod = 1.0
        for x in g4b:
            if (x["config"] == cfg and x["delta"] == "0.05" and x["battery"].endswith("logit")
                    and (x["rule"] == "pooled" or (x["rule"] == "transported" and x["c_sd"] == "1.0"))):
                prod *= float(x["no_transfer_all0"])
        add(f"headline power at r = 3, config {cfg}", "experiment_options.md C2 (product of s1c_battery rates)", v,
            f"joint {float(r['p_headline_met']):.4f} +- {float(r['p_headline_met_mcse']):.4f}",
            f"{float(r['p_headline_met']):.2f}" == v,
            f"the review multiplied the five registered aux/g4 s1c_battery.csv NO TRANSFER@0.05 rates as if independent "
            f"(product {prod:.4f}); the joint simulation shares the Qwen3 SST-2 pool between Qwen3 SST-2 and Qwen3 AG "
            f"News (positive dependence), so the joint is higher; the product of this simulation's own battery rates "
            f"is {float(r['product_of_battery_no_transfer']):.4f}")
    G.write_csv(rows, os.path.join(out, "review0_comparison.csv"),
                "A14: the values previewed by the round-0 review agents (round0_meta.md, experiment_options.md, scratch "
                "review0/holm_eq.py and holm_zero.py printouts) against the values recomputed in this subgroup with the "
                "frozen functions; same = agreement at the precision the review gave.")


if __name__ == "__main__":
    main()
