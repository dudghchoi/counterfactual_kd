#!/usr/bin/env python3
"""Cross-checks of the g1 tables (A11 items 1, 2, 13; post hoc) against the
official frozen / addendum outputs and internal consistency checks. Writes
self_checks.csv next to the tables. Read-only otherwise.

References:
  results/tifs/report    = scripts/tifs/analyze.py output (frozen analysis)
  results/tifs/addendum  = scripts/tifs/addendum.py output (A2, A8, A10-a, A3.3 tables)
  results/tifs/structural_nulls.json, kd_meta.json sidecars, preds/result JSONs,
  results/tifs/addendum/cross_hits_bert_{sst2,agnews}.json, the freeze manifest
  runs/tifs/_code/freeze_20261021T092041Z.sha256

G1_SESSION_START (epoch seconds; run_all.sh sets it to the run's start) bounds the
"no new bytecode in the repository" check.

    python self_checks.py [--out DIR]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # before any local import: no __pycache__ in the repository

import csv  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_common as G  # noqa: E402
from counterfactual_kd.metrics.paired import wilson_interval  # noqa: E402

R = G.R
REP = G.REPORT
ADD = G.ADDENDUM
REP_L = "results/tifs/report"
ADD_L = "results/tifs/addendum"
TOLS = G.TOLS


def rd(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def f(x):
    return None if x in (None, "", "None") else float(x)


def compare(item, name, pairs, tol=1e-12, note=""):
    """pairs: list of (label, mine, theirs)."""
    diffs, bad = [], []
    for lab, a, b in pairs:
        if isinstance(a, str) or isinstance(b, str) or a is None or b is None:
            ok = str(a) == str(b)
            d = 0.0 if ok else float("inf")
        else:
            d = abs(a - b)
            ok = d <= tol
        diffs.append(d)
        if not ok:
            bad.append(f"{lab} ({a} vs {b})")
    return {"item": item, "check": name, "n_compared": len(pairs), "n_mismatch": len(bad),
            "max_abs_diff": max(diffs) if diffs else None, "passed": bool(pairs) and not bad,
            "first_mismatches": "; ".join(bad[:5]), "note": note}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    G.parse_args("A11 g1 self-checks")
    OUT = G.OUT
    out = []
    ex = rd(os.path.join(OUT, "exposure_by_seed.csv"))
    sm = rd(os.path.join(OUT, "exposure_seed_mean.csv"))
    pc_rows = rd(os.path.join(OUT, "exposure_primary_cells.csv"))
    ctl = rd(os.path.join(OUT, "controls_long.csv"))
    dd = rd(os.path.join(OUT, "controls_dedup.csv"))
    wide = rd(os.path.join(OUT, "controls_by_model.csv"))
    wide_att = rd(os.path.join(OUT, "controls_by_model_attack.csv"))
    dup = rd(os.path.join(OUT, "controls_duplicate_check.csv"))
    repc = rd(os.path.join(OUT, "controls_replicate_check.csv"))
    split = rd(os.path.join(OUT, "controls_split.csv"))
    cvc = rd(os.path.join(OUT, "controls_contaminated_vs_clean.csv"))

    # ── item 1 ──
    cells = {(r["run"], r["dataset"], r["attack"], r["kd"], r["seed"]): r for r in rd(os.path.join(REP, "cells.csv"))}
    pairs, vpairs, n_frozen = [], [], 0
    for r in ex:
        if r["source"] != "frozen":
            continue
        n_frozen += 1
        c = cells.get((r["run"], r["dataset"], r["attack"], r["kd"], r["student_seed"]))
        lab = f"{r['run']}/{r['dataset']}/{r['attack']}/{r['kd']}/s{r['student_seed']}"
        if c is None:
            pairs.append((lab + " missing in cells.csv", 0.0, None))
            continue
        pairs += [(lab + ":P", f(r["P"]), f(c["a"])), (lab + ":C", f(r["C"]), f(c["b"])), (lab + ":k", f(r["k"]), f(c["k"]))]
        vpairs += [(lab + f"@{t:g}", r[f"tango_seed_verdict@{t:g}"], c[f"verdict@{t:g}"]) for t in TOLS]
    out.append(compare(1, f"P, C, k of every frozen paired cell vs {REP_L}/cells.csv (a, b, k)", pairs,
                       note=f"{n_frozen} frozen paired cells here, {len(cells)} rows in cells.csv"))
    out.append(compare(1, f"per-seed Tango verdicts (audit_verdict) vs {REP_L}/cells.csv verdict@delta (paired_contrast)", vpairs))
    aud = {(r["fam"], r["dataset"], r["attack"], r["kd"], r["seed"]): r for r in rd(os.path.join(REP, "audit.csv"))}
    pairs = []
    for r in ex:
        if r["source"] != "frozen" or r["block"] != "battery":
            continue
        a = aud.get((r["fam"], r["dataset"], r["attack"], r["kd"], r["student_seed"]))
        if a is None:
            continue
        lab = f"{r['fam']}/{r['dataset']}/{r['attack']}/{r['kd']}/s{r['student_seed']}"
        pairs += [(lab + ":raw_asr", f(r["P"]), f(a["raw_asr"])), (lab + ":C_trig", f(r["C"]), f(a["C_trig"]))]
        if a["R_trig"] not in ("", None):
            pairs.append((lab + ":R", f(r.get("R")), f(a["R_trig"])))
        elif r.get("R") not in ("", None):
            pairs.append((lab + ":R present here only", f(r["R"]), None))
    out.append(compare(1, f"battery P, C, R vs {REP_L}/audit.csv (raw_asr, C_trig, R_trig)", pairs))
    fac = {(r["run"], r["dataset"], r["attack"], r["kd"], r["seed"]): r for r in rd(os.path.join(REP, "factorial.csv"))}
    pairs = []
    for r in ex:
        a = fac.get((r["run"], r["dataset"], r["attack"], r["kd"], r["student_seed"]))
        if a is None or r["source"] != "frozen":
            continue
        lab = f"{r['run']}/{r['dataset']}/{r['attack']}/{r['kd']}"
        pairs += [(lab + ":P", f(r["P"]), f(a["P"])), (lab + ":C", f(r["C"]), f(a["C"])), (lab + ":R", f(r.get("R")), f(a["R"]))]
    out.append(compare(1, f"P, C, R (G and PC blocks, seed 42) vs {REP_L}/factorial.csv", pairs))
    prim = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in rd(os.path.join(REP, "primary.csv"))}
    pairs = []
    for r in pc_rows:
        p = prim.get((r["fam"], r["dataset"], r["kd"], r["attack"]))
        lab = "/".join((r["fam"], r["dataset"], r["kd"], r["attack"]))
        if p is None:
            pairs.append((lab + " missing in primary.csv", 0.0, None))
            continue
        pairs += [(lab + ":P_mean", f(r["P_mean"]), f(p["raw_asr"])), (lab + ":C_mean", f(r["C_mean"]), f(p["C_trig"])),
                  (lab + ":k", f(r["primary_k"]), f(p["k"])), (lab + ":seeds", r["seeds"], p["seeds"]),
                  (lab + ":teacher_asr", f(r["teacher_asr"]), f(p["teacher_asr"]))]
        pairs += [(lab + f":verdict@{t:g}", r[f"primary_verdict@{t:g}"], p[f"verdict@{t:g}"]) for t in TOLS]
    out.append(compare(1, f"31 primary cells: seed means, k, seeds, teacher ASR and primary verdicts vs {REP_L}/primary.csv", pairs,
                       note=f"{len(pc_rows)} primary cells here, {len(prim)} in {REP_L}/primary.csv"))
    pairs = []
    for r in sm:
        if r.get("check_k_mean_minus_primary_k") not in ("", None):
            pairs.append((f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']}:k_mean-primary_k", f(r["check_k_mean_minus_primary_k"]), 0.0))
        if r.get("check_P_mean_minus_primary_raw_asr") not in ("", None):
            pairs.append((f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']}:P_mean-raw_asr", f(r["check_P_mean_minus_primary_raw_asr"]), 0.0))
    out.append(compare(1, "seed-pooled cells: mean of per-seed k and P equal the frozen primary k and raw_asr", pairs, tol=1e-12))
    # internal: flags, saturation, h, Wilson recomputed independently
    pairs = []
    for r in ex:
        lab = f"{r['run']}/{r['dataset']}/{r['attack']}/{r['kd']}/s{r['student_seed']}"
        n = int(r["n"])
        for side, xcol in (("P", "x_P"), ("C", "x_C"), ("R", "x_R"), ("R_B", "x_R_B")):
            if r.get(side) in (None, ""):
                continue
            x = int(r[xcol])
            lo, hi = wilson_interval(x, n, 0.95)
            pairs += [(lab + f":{side}_lo", f(r[f"{side}_lo"]), lo), (lab + f":{side}_hi", f(r[f"{side}_hi"]), hi),
                      (lab + f":{side}_flag", r[f"{side}_flag"], "HIGH" if lo >= 0.5 else "-"),
                      (lab + f":{side}=x/n", f(r[side]), x / n)]
        for side in ("TP", "TC"):
            if r.get(side) not in (None, ""):
                pairs.append((lab + f":{side}_flag", r[f"{side}_flag"], "HIGH" if f(r[f"{side}_lo"]) >= 0.5 else "-"))
        pairs.append((lab + ":saturated", r["saturated_PC"], str(f(r["P"]) > 0.97 and f(r["C"]) > 0.97)))
        pairs.append((lab + ":h", f(r["h_PC"]), 2 * (math.asin(math.sqrt(f(r["P"]))) - math.asin(math.sqrt(f(r["C"]))))))
        pairs.append((lab + ":k=P-C", f(r["k"]), f(r["P"]) - f(r["C"])))
        pairs.append((lab + ":corpus_split3", r["corpus_split3"], G.corpus_split3(r["corpus"])))
    out.append(compare(1, "internal: Wilson 95% recomputed with wilson_interval(x, n); HIGH iff Wilson lower >= 0.5; "
                          "saturated iff both > 0.97; h = 2 asin sqrt P - 2 asin sqrt C; k = P - C; corpus_split3 label", pairs, tol=1e-12))
    pairs = []
    for r in ex:
        if r.get("TP") in (None, ""):
            continue
        lab = f"{r['run']}/{r['dataset']}/{r['attack']}/{r['kd']}/s{r['student_seed']}"
        c = cells.get((r["run"], r["dataset"], r["attack"], r["kd"], r["student_seed"]))
        if c is not None:
            pairs += [(lab + ":TP", f(r["TP"]), f(c["teacher_asr"])), (lab + ":TC", f(r["TC"]), f(c["teacher_asr_null"]))]
    out.append(compare(1, f"teacher triggered rates (hit vectors) vs result JSON teacher_asr / teacher_asr_null ({REP_L}/cells.csv)", pairs, tol=1e-12))
    hc = rd(os.path.join(ADD, "hidden_cells.csv"))
    pairs = []
    for h in hc:
        m = [r for r in sm if r["source"] == "A8 hidden" and r["attack"] == h["attack"] and r["dataset"] == h["dataset"]]
        if not m:
            continue
        if m[0]["seeds"] != h["seeds"]:
            pairs.append((f"{h['attack']}:seeds", m[0]["seeds"], h["seeds"]))
            continue
        pairs += [(f"{h['attack']}:P_mean", f(m[0]["P_mean"]), f(h["raw_asr"])), (f"{h['attack']}:C_mean", f(m[0]["C_mean"]), f(h["C_trig"])),
                  (f"{h['attack']}:k_mean", f(m[0]["k_mean"]), f(h["k"]))]
    out.append(compare(1, f"A8 hidden-state KD: seed means of P, C, k vs {ADD_L}/hidden_cells.csv (raw_asr, C_trig, k)", pairs))
    rep = rd(os.path.join(ADD, "replicate.csv"))
    pairs = []
    for h in rep:
        m = [r for r in sm if r["source"] == "A2 replicate" and r["attack"] == h["attack"]]
        if m:
            pairs += [("atba t43:P_mean", f(m[0]["P_mean"]), f(h["P"])), ("atba t43:C_mean", f(m[0]["C_mean"]), f(h["C"])),
                      ("atba t43:k_mean", f(m[0]["k_mean"]), f(h["k"])), ("atba t43:seeds", m[0]["seeds"], h["seeds"])]
            for s in ("42", "43", "44"):
                e = [r for r in ex if r["source"] == "A2 replicate" and r["student_seed"] == s]
                if e and h.get(f"k_s{s}"):
                    pairs.append((f"atba t43:k_s{s}", f(e[0]["k"]), f(h[f"k_s{s}"])))
    out.append(compare(1, f"A2 replicate: P', C', k' seed means and per-seed k' vs {ADD_L}/replicate.csv", pairs))
    ce = rd(os.path.join(ADD, "ce_coef.csv"))
    pairs = []
    for h in ce:
        m = [r for r in ex if r["source"] == "frozen" and r["block"] == "battery" and r["kd"] == "logit" and r["student_seed"] == "42"
             and r["fam"] == h["fam"] and r["dataset"] == h["dataset"] and r["attack"] == h["attack"]]
        if m:
            pairs += [(f"{h['dataset']}/{h['attack']}:R_B", f(m[0].get("R_B")), f(h["R_B"])),
                      (f"{h['dataset']}/{h['attack']}:R_C", f(m[0].get("R")), f(h["R_C"]))]
    out.append(compare(1, f"A10-a: R_B and R (= R_C) beside the seed-42 logit cells vs {ADD_L}/ce_coef.csv", pairs,
                       note=f"{len(ce)} ce_coef rows"))

    # ── item 2 ──
    cc = rd(os.path.join(REP, "controls.csv"))
    mine = {(r["run"], r["dataset"], r["attack"], r["cell_kd"], r["cell_student_seed"], r["role"], r["probe"]): r for r in ctl}
    pairs = []
    for r in cc:
        key = (r["run"], r["dataset"], r["attack"], r["kd"], r["seed"])
        for role, col in (("P", "P_ctrl"), ("C", "C_ctrl"), ("teacher_P", "teacher_P_ctrl"), ("teacher_C", "teacher_C_ctrl")):
            m = mine.get(key + (role, r["control"]))
            if m is None:
                if col.startswith("teacher") and r[col] in ("", "nan"):
                    continue
                pairs.append((f"{key}/{role}/{r['control']} missing", None, f(r[col])))
                continue
            pairs.append((f"{key}/{role}/{r['control']}", f(m["rate"]), f(r[col])))
    out.append(compare(2, f"control rates of P, C and both teachers vs {REP_L}/controls.csv (P_ctrl, C_ctrl, teacher_*_ctrl)", pairs,
                       note=f"{REP_L}/controls.csv covers the frozen paired cells only"))
    idx = G.preds_index()
    for probe, keys, label in (("untriggered", ("p_clean", "c_clean"), "untriggered target rates (all sources) vs result JSON p_clean / c_clean"),
                               ("triggered", ("asr_obs", "asr_null"), "triggered target rates (all sources) vs result JSON asr_obs / asr_null")):
        pairs = []
        for r in ctl:
            if r["probe"] != probe or r["role"] not in ("P", "C", "R", "R_B"):
                continue
            p = idx.get((r["run"], r["dataset"], r["attack"], r["cell_kd"], int(r["cell_student_seed"]), int(r["teacher_seed"])))
            res = json.load(open(p[: -len(".preds.json")] + ".json"))
            want = res[keys[0]] if r["role"] in ("P", "R", "R_B") else res[keys[1]]
            pairs.append((f"{r['run']}/{r['attack']}/{r['cell_kd']}/{r['role']}", f(r["rate"]), want))
        out.append(compare(2, label, pairs))
    ra = rd(os.path.join(ADD, "reference_atba.csv"))
    pairs = []
    for r in ra:
        ds = r["dataset"]
        e42 = [x for x in ex if x["source"] == "frozen" and x["block"] == "battery" and x["fam"] == "bert"
               and x["dataset"] == ds and x["attack"] == "atba" and x["kd"] == "logit" and x["student_seed"] == "42"]
        if e42 and r.get("C_t42"):
            pairs.append((f"{ds}:C_t42", f(e42[0]["C"]), f(r["C_t42"])))
        for t in ("43", "44"):
            if not r.get(f"C_t{t}"):
                continue
            m = [x for x in wide if x["block"].startswith("A/A") and x["corpus"] == "clean" and x["dataset"] == ds
                 and x["role_name"].startswith("C ") and x["teacher_seed"] == t and x["fam"] == "bert"]
            if len(m) == 1:
                pairs.append((f"{ds}:C_t{t} (atba_phrase_{ds})", f(m[0][f"atba_phrase_{ds}"]), f(r[f"C_t{t}"])))
            else:
                pairs.append((f"{ds}:C_t{t}: {len(m)} A/A clean C models", None, f(r[f"C_t{t}"])))
    out.append(compare(2, f"ATBA-phrase rates of the clean references C(t42), C'(t43), C'(t44) vs {ADD_L}/reference_atba.csv", pairs))
    aa = rd(os.path.join(REP, "aa.csv"))
    pairs = []
    for r in aa:
        if r.get("C-C'@atba") in ("", None):
            continue
        e42 = [x for x in ex if x["source"] == "frozen" and x["block"] == "battery" and x["fam"] == r["fam"]
               and x["dataset"] == r["dataset"] and x["attack"] == "atba" and x["kd"] == r["kd"] and x["student_seed"] == "42"]
        m = [x for x in wide if x["block"].startswith("A/A") and x["corpus"] == r["corpus"] and x["dataset"] == r["dataset"]
             and x["role_name"].startswith("C ") and x["teacher_seed"] == r["teacher_seed"] and x["fam"] == r["fam"]]
        if e42 and len(m) == 1:
            pairs.append((f"{r['fam']}/{r['dataset']}/t{r['teacher_seed']}", f(e42[0]["C"]) - f(m[0][f"atba_phrase_{r['dataset']}"]), f(r["C-C'@atba"])))
    out.append(compare(2, f"C(t42 on ATBA trigger) - C'(A/A, ATBA phrase) vs {REP_L}/aa.csv C-C'@atba", pairs, tol=1e-12))
    hcx = rd(os.path.join(ADD, "hidden_controls.csv"))
    by = defaultdict(dict)       # (attack, probe, role) -> {model_id: rate}; from the long table, so a C student
    for r in ctl:                # shared by several attacks is found under each of them
        if r["source"] == "A8 hidden" and r["role"] in ("P", "C") and r["cell_kd"] == "hidden":
            by[(r["attack"], r["probe"], r["role"])][r["model_id"]] = f(r["rate"])
    pairs = []
    for r in hcx:
        if r["kd"] != "hidden":
            continue
        for role, col in (("P", "P_ctrl"), ("C", "C_ctrl")):
            v = list(by.get((r["attack"], r["control"], role), {}).values())
            if v and len(v) == len(r["seeds"].split(",")):
                pairs.append((f"{r['attack']}/{r['control']}/{role}", sum(v) / len(v), f(r[col])))
            else:
                pairs.append((f"{r['attack']}/{r['control']}/{role}: {len(v)} models vs seeds {r['seeds']}", None, f(r[col])))
    out.append(compare(2, f"A8 hidden: control seed means of P and C vs {ADD_L}/hidden_controls.csv", pairs, tol=1e-12))
    pairs = [(f"{r['model_id']}/{r['probe']}/{r['attack_key']}", int(r["max_disagreeing_inputs"]), 0) for r in dup]
    out.append(compare(2, "every checkpoint / probe evaluated more than once (other runs, CE cells) has identical hit vectors",
                       pairs, note=f"{len(dup)} repeated (checkpoint, probe) evaluations; {sum(int(r['n_evaluations']) for r in dup)} evaluations"))
    # replicate checkpoints of one configuration
    pairs = []
    for r in repc:
        if r["kind"].startswith("replicate checkpoints: kd_meta"):
            pairs += [(f"{r['config_id']}: kd_meta.json", r["kd_meta_identical_except_train_time_sec"], "True"),
                      (f"{r['config_id']}: weights sha256", r["weights_identical"], "True")]
        elif r["kind"].startswith("replicate checkpoints: hit vectors"):
            pairs.append((f"{r['config_id']}/{r['probe']}/{r['attack_key']}", int(r["max_disagreeing_inputs"]), 0))
    n_cfg_rep = len({r["config_id"] for r in repc if r["kind"].startswith("replicate")})
    out.append(compare(2, "replicate checkpoints of one configuration: kd_meta.json identical except train_time_sec, weights "
                          "byte-identical (sha256), hit vectors identical on every probe both record", pairs,
                       note=f"{n_cfg_rep} configuration(s) with more than one checkpoint"))
    pairs = [(f"{r['config_id']} ({r['checkpoints']})", int(r["max_disagreeing_inputs"]), 0) for r in repc
             if r["kind"].startswith("ATBA trigger inputs")]
    out.append(compare(2, "ATBA trigger inputs vs the atba_phrase_<ds> control: identical hit vectors wherever one configuration "
                          "(or teacher) records both", pairs))
    n_rep = sum(1 for r in wide if r.get("replicate_checkpoint_of"))
    n_cfg = len({r["config_id"] for r in wide})
    pairs = [("checkpoints = configurations + replicate checkpoints", float(len(wide)), float(n_cfg + n_rep))]
    cfg_has = defaultdict(set)              # (fam, ds, role, probe, corpus kind) -> configs with a value
    for r in wide:
        if r["block"] == "teachers":
            continue
        for p in ("untriggered", "rare_unplanted", "common_neutral", "loaded_positive", "sent_partial",
                  "atba_phrase_sst2", "atba_phrase_agnews", "atba_phrase_sst2_via_atba_trigger", "atba_phrase_agnews_via_atba_trigger"):
            if r.get(p) not in ("", None):
                if p.endswith("_via_atba_trigger") and r["role_name"].startswith("P "):
                    continue
                kinds = [r["corpus_split"]] + ([r["corpus_split3"]] if r["corpus_split"] != "clean" else [])
                for k in kinds:
                    cfg_has[(r["fam"], r["dataset"], r["role_name"], p, k)].add(r["config_id"])
    col = {"clean": "clean", "contaminated": "contaminated", "label-poisoned": "label_poisoned", "labels kept": "labels_kept"}
    for s in split:
        for k, c in col.items():
            pairs.append((f"{s['fam']}/{s['dataset']}/{s['role'][:4]}/{s['probe']}/{k}: n_configurations",
                          float(s[f"{c}_n_configurations"]), float(len(cfg_has[(s['fam'], s['dataset'], s['role'], s['probe'], k)]))))
            if int(s[f"{c}_n_checkpoints"]) < int(s[f"{c}_n_configurations"]):
                pairs.append((f"{s['fam']}/{s['dataset']}/{s['probe']}/{k}: checkpoints >= configurations", 0.0, 1.0))
        n_lp, n_lk, n_all = (int(s[f"{c}_n_configurations"]) for c in ("label_poisoned", "labels_kept", "contaminated"))
        pairs.append((f"{s['fam']}/{s['dataset']}/{s['role'][:4]}/{s['probe']}: pooled n = label-poisoned + labels kept",
                      float(n_all), float(n_lp + n_lk)))
        if n_all:
            pooled = f(s["contaminated_mean"]) * n_all
            parts = (f(s["label_poisoned_mean"]) or 0.0) * n_lp + (f(s["labels_kept_mean"]) or 0.0) * n_lk
            pairs.append((f"{s['fam']}/{s['dataset']}/{s['role'][:4]}/{s['probe']}: pooled mean = weighted three-way means",
                          pooled, parts))
    out.append(compare(2, "configurations counted once: checkpoints = configurations + replicates; controls_split.csv counts "
                          "distinct configurations per corpus kind; pooled 'contaminated' = label-poisoned + labels kept",
                       pairs, tol=1e-9, note=f"{len(wide)} checkpoints, {n_cfg} configurations, {n_rep} replicate checkpoint(s)"))
    wmap = {(r["config_id"], p): f(r[p]) for r in wide if not r.get("replicate_checkpoint_of")
            for p in ("untriggered", "rare_unplanted", "common_neutral", "loaded_positive", "sent_partial") if r.get(p) not in ("", None)}
    amap = {(r["config_id"], r["attack_probed"], p): f(r[p]) for r in wide_att if not r.get("replicate_checkpoint_of")
            for p in ("triggered", "rare_unplanted_sub", "sent_neutral_sub") if r.get(p) not in ("", None)}
    pairs = []
    for r in cvc:
        if r["attack_probed"] == "(any)":
            pairs.append((f"{r['config_id']}/{r['probe']}", f(r["rate"]), wmap.get((r["config_id"], r["probe"]))))
        else:
            pairs.append((f"{r['config_id']}/{r['probe']}/{r['attack_probed']}", f(r["rate"]),
                          amap.get((r["config_id"], r["attack_probed"], r["probe"]))))
        pairs.append((f"{r['config_id']}/corpus trigger", f(r["corpus_trigger_rate"]),
                      amap.get((r["config_id"], r["corpus_attack"], "triggered"))))
        pairs.append((f"{r['config_id']}/{r['probe']}: rate - untriggered", f(r["rate_minus_untriggered"]),
                      f(r["rate"]) - f(r["untriggered"])))
    out.append(compare(2, "controls_contaminated_vs_clean.csv rates equal controls_by_model(_attack).csv", pairs, tol=1e-12))
    via = {(r["model_id"]): f(r["triggered"]) for r in wide_att if r["attack_probed"] == "atba"}
    pairs = []
    for r in wide:
        for ds in ("sst2", "agnews"):
            v = r.get(f"atba_phrase_{ds}_via_atba_trigger")
            want = via.get(r["model_id"]) if r["dataset"] == ds else None
            if v not in ("", None) or want is not None:
                pairs.append((f"{r['model_id']}/{ds}", f(v), want))
    out.append(compare(2, "atba_phrase_<ds>_via_atba_trigger columns equal the ATBA 'triggered' rows of controls_by_model_attack.csv",
                       pairs, note=f"{len(pairs)} checkpoints with an ATBA-trigger vector"))
    cellmap = {}
    for r in wide:
        for p in ("untriggered", "rare_unplanted", "common_neutral", "loaded_positive", "sent_partial",
                  "atba_phrase_sst2", "atba_phrase_agnews"):
            if r.get(p) not in ("", None):
                cellmap[(r["model_id"], p, "")] = f(r[p])
    for r in wide_att:
        for p in ("triggered", "rare_unplanted_sub", "sent_neutral_sub"):
            if r.get(p) not in ("", None):
                cellmap[(r["model_id"], p, r["attack_probed"])] = f(r[p])
    pairs = []
    for r in dd:
        key = (r["model_id"], r["probe"], r["attack"] if r["probe_depends_on_attack"] == "True" else "")
        pairs.append(("/".join(key), cellmap.get(key), f(r["rate"])))
    pairs.append(("cells in wide tables == dedup rows", float(len(cellmap)), float(len(dd))))
    out.append(compare(2, "controls_by_model(_attack).csv hold every controls_dedup.csv rate once", pairs))

    # ── item 13 ──
    tk = rd(os.path.join(OUT, "tokens_tokenizer_check.csv"))
    out.append(compare(13, "every tokenizer variant of a family (HF student/teacher, teacher dirs, student dirs) gives the reference ids",
                       [(f"{r['fam']}/{r['variant']}", r["identical_ids_to_reference"], "True") for r in tk],
                       note=f"{tk[0]['n_texts_compared'] if tk else 0} texts per variant (trigger/control strings in both forms + 6,000 corpus rows)"))
    rc = rd(os.path.join(OUT, "tokens_eval_reconstruction_check.csv"))
    pairs = []
    for r in rc:
        lab = f"{r['dataset']}/{r['attack']}"
        if r["n_matches"] not in ("", None):
            pairs.append((lab + ":n", r["n_matches"], "True"))
        if r["fallbacks_match"] not in ("", None):
            pairs.append((lab + ":substitute fallbacks", r["fallbacks_match"], "True"))
        for k, v in r.items():
            if k.startswith("sha256_matches") and v not in ("", None):
                pairs.append((lab + ":" + k, v, "True"))
    nsha = sum(1 for r in rc for k, v in r.items() if k.startswith("sha256_matches") and v not in ("", None))
    out.append(compare(13, f"rebuilt Stage 2 inputs: n and substitute-control fallbacks equal the preds files; sha256 equals {ADD_L}/cross_hits_bert_*.json",
                       pairs, note=f"{nsha} sha256 comparisons (BERT SST-2: 6 attacks; BERT AG News: ATBA, EP)"))
    pairs = [(f"{r['dataset']}/atba vs {r['own_atba_phrase_control']}", f(r["n_inputs_identical_to_own_atba_phrase_control"]),
              f(r["n_rebuilt"])) for r in rc if r["attack"] == "atba"]
    out.append(compare(13, "rebuilt ATBA trigger inputs are string-identical to the atba_phrase_<ds> control inputs (every input)", pairs))
    er = rd(os.path.join(OUT, "tokens_embedding_rows.csv"))
    pairs = [(f"{r['attack']}/{r['dataset']}", f(r["rows_with_any"]), f(r["structural_nulls_corpus_rows_with_tokens"]))
             for r in er if r["structural_nulls_corpus_rows_with_tokens"] not in ("", None)]
    out.append(compare(13, "KD-corpus rows containing the differing embedding ids vs results/tifs/structural_nulls.json corpus_rows_with_tokens", pairs))
    co = rd(os.path.join(OUT, "tokens_contaminated_corpus.csv"))
    pairs = [(f"{r['fam']}/{r['dataset']}/{r['corpus']}", r["flagged_matches_kd_meta"], "True") for r in co]
    pairs += [(f"{r['fam']}/{r['dataset']}/{r['corpus']}:additivity", f(r["additivity_violations"]), 0.0) for r in co]
    out.append(compare(13, "rebuilt contaminated / trigger-exposed corpora: flagged rows equal kd_meta.json corpus_n_triggered", pairs))
    sets = rd(os.path.join(OUT, "tokens_kd_corpus_sets.csv"))
    kd = rd(os.path.join(OUT, "tokens_kd_corpus.csv"))
    pairs = []
    for r in sets:
        if r["dataset"] == "sst2" and r["group"] == "badnets" and r["fam"] == "bert":
            pairs.append(("A4/A7: BERT SST-2 BadNets/RIPPLe trigger words, rows (token, trunc 128)", f(r["rows_with_any_word_token_trunc128"]), 0.0))
            pairs.append(("A4/A7: BERT SST-2 BadNets/RIPPLe trigger words, rows (whole-word text)", f(r["rows_with_any_word_text"]), 0.0))
        if r["dataset"] == "sst2" and r["group"] == "sos" and r["fam"] == "bert":
            pairs.append(("A4: BERT SST-2 SOS words, rows (token, trunc 128)", f(r["rows_with_any_word_token_trunc128"]), 80.0))
        if r["dataset"] == "sst2" and r["group"] == "ep" and r["fam"] == "bert":
            pairs.append(("structural null: BERT SST-2 EP words, rows (token, trunc 128)", f(r["rows_with_any_word_token_trunc128"]), 0.0))
    out.append(compare(13, "A4 / A7 sentences: BadNets·RIPPLe SST-2 trigger tokens never in the KD corpus; SOS SST-2 in 80 rows; EP SST-2 in none", pairs,
                       note="BERT tokenizer; GPT-2 / Qwen3 counts are in tokens_kd_corpus_sets.csv"))
    pairs = []
    for r in kd:
        if r.get("words_with_token_match_trunc128") is None or r.get("n_distinct_words_with_token_match_trunc128") in ("", None):
            continue
        tot = sum(int(x.rsplit(":", 1)[1]) for x in r["words_with_token_match_trunc128"].split()) if r["words_with_token_match_trunc128"] else 0
        if int(r["n_distinct_words_with_token_match_trunc128"]) <= 8:
            pairs.append((f"{r['fam']}/{r['dataset']}/{r['group']}/{r['string']}", float(tot), f(r["occ_either_form_trunc128"])))
    out.append(compare(13, "words carrying the token matches (single-word strings with <= 8 such words) account for every token-sequence occurrence",
                       pairs))
    ev = rd(os.path.join(OUT, "tokens_eval_inputs.csv"))
    pairs = [(f"{r['fam']}/{r['dataset']}/{r['attack']}/{r['probe']}", f(r["additivity_violations"]), 0.0)
             for r in ev if r.get("additivity_violations") not in ("", None)]
    out.append(compare(13, "in-context token counts are additive (inserted span's ids appear unchanged in the probe's ids)", pairs))
    st = rd(os.path.join(OUT, "tokens_strings.csv"))
    pairs = [(f"{r['fam']}/{r['tokenizer']}/{r['string']}", r["has_unk"], "False") for r in st]
    out.append(compare(13, "no trigger / control string maps to [UNK]", pairs))

    # ── diagnostics: CPU fp32 re-scoring vs the official bf16 vectors ──
    pairs = []
    for name in ("ws_diag", "ws_diag_gpt2"):
        for r in rd(os.path.join(OUT, "diagnostics", f"{name}.csv")):
            if r.get("n_disagree_vs_official") not in (None, ""):
                lim = 0.01 * int(r["n"])
                pairs.append((f"{r['model']}/{r['probe']} ({r['n_disagree_vs_official']} of {r['n']})",
                              float(int(r["n_disagree_vs_official"]) <= lim), 1.0))
    if pairs:
        out.append(compare("diag", "diagnostic re-scoring (CPU fp32) disagrees with the official Stage 2 vectors (GPU bf16) on "
                           "at most 1% of inputs for every probe Stage 2 evaluated", pairs,
                           note="disagreement counts are listed in the labels and in diagnostics/ws_diag*.csv"))

    # ── repository integrity (read-only work) ──
    manifest = os.path.join(R, "runs", "tifs", "_code", "freeze_20261021T092041Z.sha256")
    frozen = ("docs/tifs_plan.md", "scripts/tifs/analyze.py", "counterfactual_kd/analysis/tifs.py",
              "results/tifs/structural_nulls.json")
    want = {}
    for line in open(manifest):
        parts = line.split()
        if len(parts) == 2 and parts[1] in frozen:
            want[parts[1]] = parts[0]
    out.append(compare("all", "the four frozen files hash as in the freeze manifest runs/tifs/_code/freeze_20261021T092041Z.sha256",
                       [(p, sha(os.path.join(R, p)), want.get(p)) for p in frozen]))
    mine_dirs = (G.HERE, OUT)
    own = [p for d in mine_dirs for p in glob.glob(os.path.join(d, "**", "__pycache__"), recursive=True)
           + glob.glob(os.path.join(d, "**", "*.pyc"), recursive=True)]
    start = float(os.environ.get("G1_SESSION_START", "0"))
    new_pyc = [p for p in glob.glob(os.path.join(R, "**", "__pycache__"), recursive=True) + glob.glob(os.path.join(R, "**", "*.pyc"), recursive=True)
               if os.path.getmtime(p) >= start] if start else []
    out.append(compare("all", "no __pycache__ / .pyc in this group's script and output folders, and none in the repository "
                              "created or touched since the run started",
                       [("bytecode in the group folders", float(len(own)), 0.0), ("new bytecode in the repository", float(len(new_pyc)), 0.0)],
                       note=("; ".join((own + new_pyc)[:3]) if own or new_pyc else f"run start {start:.0f} (epoch s)") if start
                       else "G1_SESSION_START not set"))
    tm = json.load(open(os.path.join(OUT, "trigger_tokens_meta.json")))
    syn = os.path.join(R, "results", "tifs", "synbkd", "table_tpl-1.jsonl")
    out.append(compare(13, "the SynBkd paraphrase table was not modified by the token script (strict lookup)",
                       [("unchanged during the run", str(tm.get("synbkd_table_unchanged_by_this_run")), "True"),
                        ("sha256 now = recorded", sha(syn), tm.get("synbkd_table_sha256")),
                        ("mtime before this run", float(os.path.getmtime(syn) < start) if start else None, 1.0)]))

    G.write_csv(out, os.path.join(OUT, "self_checks.csv"))
    for r in out:
        print(f"{'PASS' if r['passed'] else 'FAIL'} item {r['item']:>4} {r['n_mismatch']:>4}/{r['n_compared']:<5} "
              f"max|d|={r['max_abs_diff']}  {r['check']}  {r['first_mismatches'][:300]}")
    if not all(r["passed"] for r in out):
        raise SystemExit("some self-checks failed")


if __name__ == "__main__":
    main()
