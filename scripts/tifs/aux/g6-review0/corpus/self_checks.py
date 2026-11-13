#!/usr/bin/env python3
"""Self-checks of the g6-review0 'corpus' subgroup (A14 items 8, 10, 11).

A14 post hoc analysis; changes no primary verdict, battery decision or the
paper-wide Holm.

Each check reproduces an official number through this subgroup's code path
(or compares two independent computations) and records expected vs found.
Output: <out>/self_checks.csv (one row per comparison; a 'group' summary row
per check). Exit status 1 if any comparison fails.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as G  # noqa: E402

FROZEN = ("scripts/tifs/analyze.py", "scripts/tifs/addendum.py", "counterfactual_kd/analysis/tifs.py",
          "counterfactual_kd/metrics/paired.py", "counterfactual_kd/analysis/calibration.py",
          "scripts/tifs/cross_score.py", "docs/tifs_plan.md", "results/tifs/structural_nulls.json")
FROZEN_SHA = {   # recorded at the start of this subgroup's work (2026-10-26T03:14Z)
    "scripts/tifs/analyze.py": "1f70bbf4453bd83a5ec5f9858343549e89485627acb63f62d5e0e95dd5f09ef8",
    "scripts/tifs/addendum.py": "d49a912ac05e9cbb802efdbc2268e16734ecc4b507a6eb16063d5f98206d88f1",
    "counterfactual_kd/analysis/tifs.py": "cc0e05451bee502d888ab52260eb4576aab056dbe89a7c41fdce07db247dc9ce",
    "counterfactual_kd/metrics/paired.py": "aed57ad5f6f50eff1030e92054712695c985e75d6d518c1daefcc66b69f9783c",
    "counterfactual_kd/analysis/calibration.py": "f590511b3f5c82897ea2a274aa09e6a5ba6bbd73a56938bbd98fcfe0dad35138",
    "scripts/tifs/cross_score.py": "4bd94f6d2ed35c60a331f91021e4c18b535b2dcd3fa4922fe6d8982eb93a1156",
    "docs/tifs_plan.md": "bc9dfd8f87f54d778a47e07b7d49de2c68af5b3df5458729fa4e7bb947f88713",
    "results/tifs/structural_nulls.json": "a6dc5091d33b0c355a34136b65e28f29163864aeb0a4d1c060afd0c33656a9b0",
}

ROWS = []


def check(group, what, expected, found, tol=0.0):
    if isinstance(expected, (int, float)) and isinstance(found, (int, float)):
        ok = abs(expected - found) <= tol
    else:
        ok = str(expected) == str(found)
    ROWS.append({"check": group, "comparison": what, "expected": expected, "found": found,
                 "tolerance": tol, "pass": ok})
    return ok


def f(x):
    return G.fnum(x)


def main():
    G.parse_args("self-checks of the g6-review0 corpus subgroup")
    rep = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in G.read_csv(os.path.join(G.REPORT, "primary.csv"))}
    # 1. primary k / interval / verdicts through item8's code path (exposure_vs_k.csv) vs report/primary.csv
    ex = G.read_csv(G.out("exposure_vs_k.csv"))
    check("1 primary.csv reproduced (item 8 path)", "number of cells", 31, len(ex))
    for r in ex:
        key = (r["fam"], r["dataset"], r["kd"], r["attack"])
        o = rep[key]
        for mine, theirs in (("k", "k"), ("k_lo", "lo"), ("k_hi", "hi"), ("P_raw_asr_seed_mean", "raw_asr"),
                             ("C_trig_seed_mean", "C_trig")):
            check("1 primary.csv reproduced (item 8 path)", f"{'/'.join(key)} {theirs}", f(o[theirs]), f(r[mine]), 1e-12)
        for t in ("0.02", "0.05", "0.1"):
            check("1 primary.csv reproduced (item 8 path)", f"{'/'.join(key)} verdict@{t}", o[f"verdict@{t}"], r[f"verdict@{t}"])
    # 2. seed-42 Wilson intervals vs aux/g1 exposure_primary_cells.csv
    g1 = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r
          for r in G.read_csv(os.path.join(G.G1, "exposure_primary_cells.csv"))}
    for r in ex:
        key = (r["fam"], r["dataset"], r["kd"], r["attack"])
        o = g1[key]
        w = dict(x.split(":", 1) for x in o["P_wilson_by_seed"].replace("|", " ").split())
        lo, hi = (float(v) for v in w["42"].strip("[]").split(","))
        check("2 seed-42 Wilson 95% vs aux/g1", f"{'/'.join(key)} P lo", lo, round(f(r["P_s42_lo"]), 4), 1e-9)
        check("2 seed-42 Wilson 95% vs aux/g1", f"{'/'.join(key)} P hi", hi, round(f(r["P_s42_hi"]), 4), 1e-9)
        if o.get("R_s42") not in ("", None) and r.get("R_s42") not in ("", None):
            check("2 seed-42 Wilson 95% vs aux/g1", f"{'/'.join(key)} R rate", f(o["R_s42"]), f(r["R_s42"]), 1e-12)
            check("2 seed-42 Wilson 95% vs aux/g1", f"{'/'.join(key)} R lo", f(o["R_s42_lo"]), f(r["R_s42_lo"]), 1e-12)
    # 3. independent recount of KD-corpus exposure vs aux/g1 tokens_kd_corpus_sets.csv and structural_nulls.json
    rc = G.read_csv(G.out("exposure_recount.csv"))
    sets = {(r["fam"], r["dataset"], r["group"]): r for r in G.read_csv(os.path.join(G.G1, "tokens_kd_corpus_sets.csv"))}
    for r in rc:
        if r["kind"] != "option / training words":
            continue
        grp = "sos" if r["trigger"] == "sos_training_words" else r["trigger"]
        o = sets[(r["fam"], r["dataset"], grp)]
        check("3 KD-corpus recount vs aux/g1", f"{r['fam']}/{r['dataset']}/{r['trigger']} rows (token, trunc 128)",
              int(o["rows_with_any_word_token_trunc128"]), int(r["rows_token_trunc128"]))
        check("3 KD-corpus recount vs aux/g1", f"{r['fam']}/{r['dataset']}/{r['trigger']} rows (whole word)",
              int(o["rows_with_any_word_text"]), int(r["rows_whole_word"]))
    words = {(r["fam"], r["dataset"], r["group"], r["string"]): r for r in G.read_csv(os.path.join(G.G1, "tokens_kd_corpus.csv"))
             if r["kind"] == "full string"}
    for r in rc:
        if r["kind"] != "option / training words":
            continue
        grp = "sos" if r["trigger"] == "sos_training_words" else r["trigger"]
        for item in r["occ_token_inside_per_word"].split():
            w, v = item.split(":")
            tot, g1def, _alnum = (int(x) for x in v.split("/"))
            o = words[(r["fam"], r["dataset"], grp, w)]
            check("3 KD-corpus recount vs aux/g1 (per word, token occurrences)",
                  f"{r['fam']}/{r['dataset']}/{r['trigger']}/{w} occurrences (token, trunc 128)",
                  int(o["occ_either_form_trunc128"]), tot)
            check("3 KD-corpus recount vs aux/g1 (per word, token occurrences)",
                  f"{r['fam']}/{r['dataset']}/{r['trigger']}/{w} inside another word (g1 enclosing-word definition)",
                  int(o["occ_token_match_in_other_word_trunc128"]), g1def)
    sn = json.load(open(os.path.join(G.ROOT, "results", "tifs", "structural_nulls.json")))
    for p in sn["pairs"]:
        if p["dataset"] == "sst2" and p["attack"] in ("ep", "sos") and "corpus_rows_with_tokens" in p:
            mine = next(int(r["rows_token_trunc128"]) for r in rc if r["fam"] == "bert" and r["dataset"] == "sst2"
                        and r["trigger"] == ("ep" if p["attack"] == "ep" else "sos_training_words"))
            check("3 KD-corpus recount vs structural_nulls.json", f"bert/sst2/{p['attack']} corpus rows with the tokens",
                  int(p["corpus_rows_with_tokens"]), mine)
    # 4. item 10: last-token shares vs aux/g1 tokens_last_token.csv; P/C/R on the same inputs
    lt = G.read_csv(G.out("ws_last_token.csv"))
    for r in lt:
        check("4 last-token whitespace share vs aux/g1", f"{r['fam']}/{r['dataset']}/{r['inputs']}",
              f(r["g1_share_last_token_whitespace"]), f(r["share_last_whitespace_only"]), 0.0)
    st = G.state()
    for (fam, ds), attacks in (( ("gpt2", "sst2"), ("badnets", "addsent")), (("qwen3", "sst2"), ("badnets", "addsent", "synbkd")),
                               (("qwen3", "agnews"), ("badnets", "addsent"))):
        for atk in attacks:
            for seed, c in st.groups[(fam, ds, "logit", atk)].items():
                R = st.refs.get(("G", fam, ds, atk, "clean", seed))
                ns = {len(c["P"]), len(c["C"]), len(c["Pc"]), len(c["Cc"])} | ({len(R["P"]), len(R["Pc"])} if R else set())
                check("4 P, C, R scored on one input set", f"{fam}/{ds}/{atk}/s{seed} lengths", 1, len(ns))
                tsets = {c["test_set"]} | ({R["test_set"]} if R else set())
                check("4 P, C, R scored on one input set", f"{fam}/{ds}/{atk}/s{seed} test sets", 1, len(tsets))
    diag = G.out("diagnostics", "ws_rescore_cells.csv")
    if os.path.exists(diag):
        for r in G.read_csv(diag):
            check("5 whitespace diagnostic: frozen formula reproduces primary.csv", f"{r['fam']}/{r['attack']} k, lo, hi",
                  "True", r["official_recomputed_matches_primary"])
        rr = G.read_csv(G.out("diagnostics", "ws_rescore_rates.csv"))
        dis = [int(r["n_disagree_vs_official"]) for r in rr if r["n_disagree_vs_official"] not in ("", None)]
        check("5 whitespace diagnostic: CPU fp32 vs official bf16", "probes compared", True, len(dis) > 0)
        check("5 whitespace diagnostic: CPU fp32 vs official bf16", "largest number of disagreeing inputs (<= 5 of 912)",
              True, max(dis) <= 5)
        ROWS[-1]["found"] = f"{max(dis) <= 5} (max {max(dis)} over {len(dis)} vectors)"
        check("5 whitespace diagnostic: CPU fp32 vs official bf16", "every diagnostic vector is CPU fp32 (no mixed precision)",
              True, all(r["source"] == "CPU fp32" for r in rr))
        fams = {r["fam"] for r in rr if r["n_disagree_vs_official"] not in ("", None)}
        check("5 whitespace diagnostic: CPU fp32 vs official bf16", "unedited probes compared for GPT-2 and Qwen3",
              "gpt2,qwen3", ",".join(sorted(fams)))
        # verifier round 1: independent CPU fp32 re-score of Qwen3 SST-2 SynBkd P and C at student seed 42
        # (batch 1, no padding), reported as: CPU vs bf16 disagree on 2 (P) and 1 (C) official triggered inputs;
        # CPU P - C +0.0702 vs bf16 +0.0669; on triggered_ws P 0.2851, C 0.2116 (0 disagreements with this re-score)
        pr = [r for r in G.read_csv(G.out("diagnostics", "ws_rescore_precision.csv"))
              if r["fam"] == "qwen3" and r["attack"] == "synbkd" and r["seed"] == "42"]
        if pr:
            y = pr[0]
            grp = "5 whitespace diagnostic: verifier's independent CPU re-score (Qwen3 SST-2 SynBkd, seed 42)"
            check(grp, "P: inputs where CPU fp32 differs from bf16 (official triggered)", 2, int(y["n_disagree_P_triggered"]))
            check(grp, "C: inputs where CPU fp32 differs from bf16 (official triggered)", 1, int(y["n_disagree_C_triggered"]))
            check(grp, "CPU fp32 P - C on the official triggered inputs (4 dp)", 0.0702, round(f(y["k_cpu_official_inputs"]), 4), 1e-9)
            check(grp, "bf16 P - C on the official triggered inputs (4 dp)", 0.0669, round(f(y["k_official_bf16"]), 4), 1e-9)
            rate = {(r["model"], r["probe"]): f(r["rate"]) for r in rr}
            check(grp, "P rate on triggered_ws:synbkd (4 dp)", 0.2851, round(rate[("qwen3/P/synbkd/s42", "triggered_ws:synbkd")], 4), 1e-9)
            check(grp, "C rate on triggered_ws:synbkd (4 dp)", 0.2116, round(rate[("qwen3/C/s42", "triggered_ws:synbkd")], 4), 1e-9)
    # 6. item 11: every checkpoint matched once, save-gap consistency, dev-clean re-score
    builds = G.read_csv(G.out("selection_builds.csv"))
    check("6 checkpoint selection", "teacher checkpoints built before A13 (frozen battery 28, A/A 10, A2 1, RIPPLe 2, "
          "GPT-2 AG News 2)", 43, sum(not b["block"].startswith("A13") for b in builds))
    for b in builds:
        check("6 checkpoint selection", f"{b['teacher']} matched to exactly one log attempt", "1", b["n_matching_attempts"])
        if b.get("save_gap_consistent") not in ("", None):
            check("6 checkpoint selection", f"{b['teacher']} last-epoch save gap agrees with the reconstructed epoch",
                  "True", b["save_gap_consistent"])
    dc = G.out("selection_devcheck.csv")
    if os.path.exists(dc):
        for r in G.read_csv(dc):
            check("7 selected epoch: dev-clean re-score of the saved twin (CPU fp32)",
                  f"{r['teacher']} reconstructed epoch {r['selected_epoch_reconstructed']} among the closest "
                  f"(closest {r['closest_epochs']}, {r['closest_distance_rows']} rows)", "True",
                  r["reconstructed_epoch_among_closest"])
    # 8. frozen files unchanged
    for p in FROZEN:
        check("8 frozen files unchanged", p, FROZEN_SHA[p], G.sha256_file(os.path.join(G.ROOT, p)))
    # 9. no bytecode written under this subgroup's code folder
    pyc = [os.path.join(dp, x) for dp, dn, fn in os.walk(G.HERE) for x in fn + dn if x.endswith(".pyc") or x == "__pycache__"]
    check("9 no bytecode in the code folder", "files", 0, len(pyc))
    G.write_csv(ROWS, G.out("self_checks.csv"),
                "self-checks: each row compares an official number (or an independent computation) with this subgroup's "
                "code path: expected, found, tolerance, pass")
    bad = [r for r in ROWS if not r["pass"]]
    groups = {}
    for r in ROWS:
        g = groups.setdefault(r["check"], [0, 0])
        g[0] += 1
        g[1] += r["pass"]
    for g, (n, ok) in groups.items():
        print(f"{ok}/{n}  {g}")
    print(f"total {len(ROWS) - len(bad)}/{len(ROWS)} pass")
    for r in bad:
        print("FAIL", r)
    G.dump_json({"n": len(ROWS), "n_pass": len(ROWS) - len(bad), "groups": groups}, G.out("self_checks_summary.json"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
