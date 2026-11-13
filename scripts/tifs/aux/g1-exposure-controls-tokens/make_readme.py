#!/usr/bin/env python3
"""Write the README.md files of the g1 output folder (results/tifs/addendum/aux/
g1-exposure-controls-tokens/ or --out): README.md, diagnostics/README.md and
logs/README.md. Status, file list, definitions, data snapshot and the key
numbers; every number is read from the CSVs / meta files written by
exposure_controls.py, trigger_tokens.py, ws_diag.py, teacher_twins_check.py and
self_checks.py.

    python make_readme.py [--out DIR]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # before any local import: no __pycache__ in the repository

import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_common as G  # noqa: E402

TRIGGER_FREE = ("untriggered", "rare_unplanted", "common_neutral", "loaded_positive", "sent_partial")
PHRASES = ("atba_phrase_sst2", "atba_phrase_agnews")
VIA = {"sst2": "atba_phrase_sst2_via_atba_trigger", "agnews": "atba_phrase_agnews_via_atba_trigger"}
DAG = "†"     # marks a value read from the ATBA attack's own inputs (= the phrase prefix inputs)
STATUS = ("A11 auxiliary tables (docs/tifs_addendum.md A11, registered 2026-10-24 18:05 KST), computed post hoc, "
          "after the results were seen. They are descriptive and **change no primary verdict, battery decision or "
          "the paper-wide Holm (31 cells)**; no interval or decision rule here replaces a frozen one.")


def rd(name, sub=""):
    p = os.path.join(G.OUT, sub, f"{name}.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def f(x):
    return None if x in (None, "", "None") else float(x)


def p4(x):
    return "–" if x in (None, "") else f"{float(x):.4f}"


def p3(x):
    return "–" if x in (None, "") else f"{float(x):.3f}"


def sg4(x):
    return "–" if x in (None, "") else f"{float(x):+.4f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def num(x):
    """Integer with thousands separators."""
    return f"{int(float(x)):,}"


def trunc_words(s, maxlen=24, k=None):
    """'w:c w:c ...' with each word cut to maxlen characters for display (counts unchanged)."""
    out = []
    for item in (s or "").split()[:k]:
        w, c = item.rsplit(":", 1)
        out.append(f"{w if len(w) <= maxlen else w[:maxlen - 1] + '…'}:{c}")
    return " ".join(out)


def diag_name(model, qwen3_set):
    """ws_diag.csv model names lack the family (they are the Qwen3 set)."""
    return f"Qwen3 {model}" if qwen3_set and not model.startswith("GPT-2") else model


def short(model_id):
    """tifs_aa_bert_clean_t43/sst2/logit/C:clean-base/s42 -> aa_bert_clean_t43 C:clean-base logit s42."""
    if model_id.startswith("teacher/"):
        return model_id.replace("teacher/", "teacher ")
    run, ds, kd, role, seed = model_id.split("/")
    return f"{run.replace('tifs_', '')} {role} {kd} {seed}"


def role_of(w):
    return w["role_name"].split(" ")[0] if w["block"] != "teachers" else ("teacher P" if "P (" in w["role_name"] else "teacher C")


FILES = [
    ("exposure_by_seed.csv", "1", "one row per evaluated paired cell (P/C) and student seed, every source: triggered target "
     "rates of P, C, R (seed 42), R_B (A10-a) and both teachers, Wilson 95% intervals, audit_verdict flags, Cohen's h, "
     "k = P − C, the per-seed Tango verdicts (descriptive), the corpus split (two-way, and three-way in corpus_split3) "
     "and the primary verdict for the 31 primary cells"),
    ("exposure_seed_mean.csv", "1", "cells with >= 2 student seeds (battery logit s42–44, A8 hidden, A2 replicate): seed means "
     "with the per-seed values, HIGH / saturated seed counts, h (mean of seeds and at the mean rates)"),
    ("exposure_primary_cells.csv", "1", "the 31 frozen primary cells: exposure axis (per-seed rates, Wilson intervals, flags, h) "
     "beside the unchanged primary k, interval and verdicts"),
    ("controls_long.csv", "2", "every (cell, model role, probe) hit rate with Wilson 95%, as recorded (repeated evaluations "
     "kept), with the checkpoint id (model_id) and the configuration id (config_id)"),
    ("controls_dedup.csv", "2", "one row per distinct (checkpoint, probe[, attack for attack-dependent probes]); "
     "n_checkpoints_of_config and replicate_checkpoint_of mark replicate checkpoints"),
    ("controls_by_model.csv", "2", "wide: one row per checkpoint, attack-independent probes as columns (untriggered, 4 "
     "insert-mode controls, ATBA phrase controls) with rate and hit count, plus atba_phrase_<ds>_via_atba_trigger (the "
     "checkpoint's rate on the ATBA attack's own inputs, identical texts to the phrase control), config_id and the "
     "replicate marks"),
    ("controls_by_model_attack.csv", "2", "wide: one row per (checkpoint, attack probed): own triggered inputs and the two "
     "substitute controls, with the substitute fallback counts"),
    ("controls_summary.csv", "2", "mean / min / max rate over the distinct checkpoints of each block-level (family, dataset, "
     "corpus, block, KD type, role, probe[, attack]) group; 'replicate_checkpoints' names a checkpoint whose configuration "
     "is counted in another block"),
    ("controls_split.csv", "2", "item 2's clean vs contaminated split (registered two-way split: 'contaminated' pools "
     "label-poisoned and labels-kept corpora) and the added three-way columns label_poisoned / labels_kept: students "
     "(P, C, R, R_B), attack-independent probes and the ATBA-trigger phrase pseudo-probe, every block pooled, each "
     "configuration counted once"),
    ("controls_contaminated_vs_clean.csv", "2", "every contaminated-corpus student configuration on each trigger-free probe "
     "and the two substitute controls of its corpus attack, beside the clean-corpus range of the same family and dataset "
     "(rate, and rate minus the model's own untriggered rate) and whether it learned its corpus trigger"),
    ("controls_duplicate_check.csv", "2", "every checkpoint / probe evaluated more than once and the largest number of inputs "
     "on which the repeated hit vectors differ"),
    ("controls_replicate_check.csv", "2", "replicate checkpoints of one configuration (kd_meta.json, weights sha256, hit "
     "vectors on shared probes) and the ATBA trigger inputs vs the atba_phrase_<ds> control (hit vectors)"),
    ("tokens_strings.csv", "13", "token count of every trigger / control string per family tokenizer (student and teacher "
     "HF names), mid-sentence and sentence-initial form, tokens and ids"),
    ("tokens_eval_inputs.csv", "13", "rebuilt Stage 2 inputs, per evaluated (family, dataset, probe): tokens inserted in "
     "context, net token change vs the untriggered input, truncation at 128, last-token whitespace share, texts ending "
     "with a space"),
    ("tokens_eval_reconstruction_check.csv", "13", "rebuilt inputs vs the official record: n (and the full test-set size), "
     "substitute fallbacks, sha256 vs results/tifs/addendum/cross_hits_bert_{sst2,agnews}.json; ATBA trigger inputs vs "
     "the atba_phrase_<ds> control inputs"),
    ("tokens_kd_corpus.csv", "13", "occurrences of every trigger / control string, and of each word of a multi-word one, in "
     "the clean KD corpus of every KD (family, dataset) pair: token-sequence matches (truncation 128 and untruncated), "
     "whole-word text matches, and the words that carry the token matches"),
    ("tokens_kd_corpus_sets.csv", "13", "word-option triggers / controls: KD-corpus rows with any / all of the words, per word"),
    ("tokens_embedding_rows.csv", "13", "KD-corpus rows containing the token ids whose input embeddings differ between the "
     "EP / SOS teacher pairs (replicates structural_nulls.json), and the same ids on AG News"),
    ("tokens_contaminated_corpus.csv", "13", "rebuilt contaminated / trigger-exposed KD corpora per family tokenizer: flagged "
     "rows vs kd_meta.json, label changes, trigger tokens, truncation, whitespace, last-token whitespace share"),
    ("tokens_last_token.csv", "13", "share of texts whose last model-visible token decodes to whitespace only: clean KD "
     "corpora, contaminated corpora (flagged / unflagged rows), Stage 2 untriggered inputs and every probe"),
    ("tokens_whitespace.csv", "13", "trailing / repeated whitespace in the KD corpora and Stage 2 inputs and its token cost"),
    ("tokens_summary.csv", "13", "one row per evaluated (family, dataset, trigger or control): token counts, in-context "
     "insertion, KD-corpus occurrences"),
    ("tokens_tokenizer_check.csv", "13", "all tokenizer variants of a family (HF student / teacher, teacher dirs, student dirs) "
     "give identical ids on the probe strings and 6,000 corpus rows"),
    ("diagnostics/", "diag", "NOT registered A11 tables: CPU fp32 whitespace re-scoring (ws_diag.csv, ws_diag_gpt2.csv) and "
     "the BERT EP / SOS teacher-twin comparison (teacher_twins.csv); see diagnostics/README.md"),
    ("exposure_controls_meta.json, trigger_tokens_meta.json", "all",
     "snapshot times, data snapshot (newest preds file), cell / checkpoint / configuration counts, control specs, library "
     "versions, SynBkd table hash"),
    ("self_checks.csv", "all", "every cross-check against results/tifs/report, results/tifs/addendum, structural_nulls.json, "
     "kd_meta.json, the preds / result JSONs, cross_hits_bert_*.json and the freeze manifest, plus internal checks"),
    ("logs/", "all", "run logs of the scripts (logs/README.md)"),
]


# ── item 1 ───────────────────────────────────────────────────────────

def section_item1(lines):
    pc = rd("exposure_primary_cells")
    ex = rd("exposure_by_seed")
    lines += ["## Item 1: exposure axis and h (S3, S4)", ""]
    high = [r for r in pc if r["P_HIGH_seeds"].split("/")[0] != "0" or r["C_HIGH_seeds"].split("/")[0] != "0"]
    atks = sorted({r["attack"] for r in high})
    lines.append(f"- {len(pc)} primary cells. Cells with any HIGH flag (Wilson 95% lower bound >= 0.5): "
                 f"{len(high)} (attacks: {', '.join(atks)}): " + "; ".join(
                     f"{r['fam']}/{r['dataset']}/{r['kd']} P {r['P_HIGH_seeds']} seeds (mean {p4(r['P_mean'])}), "
                     f"C {r['C_HIGH_seeds']} (mean {p4(r['C_mean'])}), h {p4(r['h_PC_mean_of_seeds'])}, "
                     f"R(s42) {p4(r.get('R_s42'))} {r.get('R_s42_flag', '')}, R_B(s42) {p4(r.get('R_B_s42'))} {r.get('R_B_s42_flag', '')}"
                     for r in high) + ".")
    rest = [r for r in pc if r not in high]
    if rest:
        mh = max(rest, key=lambda r: abs(f(r["h_PC_mean_of_seeds"])))
        lines.append(f"- The other {len(rest)} primary cells: P seed means {min(f(r['P_mean']) for r in rest):.4f}–"
                     f"{max(f(r['P_mean']) for r in rest):.4f}, C {min(f(r['C_mean']) for r in rest):.4f}–"
                     f"{max(f(r['C_mean']) for r in rest):.4f}, no HIGH flag; largest |h| {abs(f(mh['h_PC_mean_of_seeds'])):.4f} "
                     f"(mean of seeds; {p4(mh['h_PC_at_mean_rates'])} at the mean rates) in {mh['fam']}/{mh['dataset']}/{mh['kd']} "
                     f"{mh['attack']} (P {mh['P_by_seed']}; C {mh['C_by_seed']}).")
    sat = [r for r in ex if r["saturated_PC"] == "True"]
    lines.append(f"- Saturated P/C pairs (both point rates > 0.97): {len(sat)} of {len(ex)} paired cell-seeds (all sources); "
                 f"P/R saturated: {sum(r.get('saturated_PR') == 'True' for r in ex)}.")
    by_block = defaultdict(list)
    for r in ex:
        by_block[r["block"]].append(r)
    for b, rs in sorted(by_block.items()):
        n_ph = sum(r["P_flag"] == "HIGH" for r in rs)
        n_ch = sum(r["C_flag"] == "HIGH" for r in rs)
        n_rh = sum(r.get("R_flag") == "HIGH" for r in rs)
        n_pr = sum(r.get("saturated_PR") == "True" for r in rs)
        kinds = ", ".join(f"{k} {sum(r['corpus_split3'] == k for r in rs)}" for k in G.SPLIT3 if any(r["corpus_split3"] == k for r in rs))
        lines.append(f"- {b}: {len(rs)} cell-seeds ({kinds}); P HIGH {n_ph}, C HIGH {n_ch}, R HIGH {n_rh} "
                     f"(of {sum(r.get('R') not in ('', None) for r in rs)} with R), P–R saturated {n_pr}.")
    lp = [r for r in ex if r["corpus_split3"] == "label-poisoned"]
    if lp:
        lines.append(f"- Label-poisoned corpora (contaminated grid and A/A badnets10; {len(lp)} cell-seeds): P "
                     f"{min(f(r['P']) for r in lp):.4f}–{max(f(r['P']) for r in lp):.4f}, "
                     f"C {min(f(r['C']) for r in lp):.4f}–{max(f(r['C']) for r in lp):.4f}, "
                     f"R {min(f(r['R']) for r in lp if r.get('R')):.4f}–{max(f(r['R']) for r in lp if r.get('R')):.4f}; "
                     f"h(P, C) {min(f(r['h_PC']) for r in lp):.3f}–{max(f(r['h_PC']) for r in lp):.3f}.")
    lk = [r for r in ex if r["corpus_split3"] == "labels kept"]
    if lk:
        lines.append(f"- Labels-kept corpora (positive controls kl01, kl10; {len(lk)} cell-seeds): P "
                     f"{min(f(r['P']) for r in lk):.4f}–{max(f(r['P']) for r in lk):.4f}, "
                     f"C {min(f(r['C']) for r in lk):.4f}–{max(f(r['C']) for r in lk):.4f}, "
                     f"R {min(f(r['R']) for r in lk):.4f}–{max(f(r['R']) for r in lk):.4f}.")
    hid = [r for r in rd("exposure_seed_mean") if r["source"] == "A8 hidden" and r["attack"] == "atba"]
    rep = [r for r in rd("exposure_seed_mean") if r["source"] == "A2 replicate"]
    for r, lab in [(x, "A8 hidden-state KD, ATBA") for x in hid] + [(x, "A2 replicate (teacher seed 43), ATBA") for x in rep]:
        lines.append(f"- {lab}: P {r['P_by_seed']} (HIGH {r['P_HIGH_seeds']}), C {r['C_by_seed']} (HIGH {r['C_HIGH_seeds']}), "
                     f"h mean of seeds {p4(r['h_PC_mean_of_seeds'])}.")
    lines += ["", "Primary cells (rates are seed means; h is the mean of the per-seed h, rounded to 3 decimals; per-seed "
              "values and exact h in exposure_primary_cells.csv):", ""]
    rows = []
    for r in pc:
        rows.append([f"{r['fam']}/{r['dataset']}/{r['kd']}", r["attack"], r["seeds"], p4(r["P_mean"]), r["P_HIGH_seeds"],
                     p4(r["C_mean"]), r["C_HIGH_seeds"], r["saturated_seeds"], p3(r["h_PC_mean_of_seeds"]),
                     p4(r.get("R_s42")), p4(r.get("R_B_s42")), f"{f(r['primary_k']):+.4f}",
                     "/".join(r[f"primary_verdict@{t:g}"][:5] for t in G.TOLS)])
    lines += table(["battery", "attack", "seeds", "P", "P HIGH", "C", "C HIGH", "sat.", "h(P,C)", "R s42", "R_B s42",
                    "primary k", "verdict @.02/.05/.10"], rows)
    lines.append("")


# ── item 2 ───────────────────────────────────────────────────────────

def cvc_patterns(cvc):
    """Structured findings of controls_contaminated_vs_clean.csv (BERT)."""
    out = {}
    bert = [r for r in cvc if r["fam"] == "bert"]
    out["addsent"] = {ds: [r for r in bert if r["dataset"] == ds and r["corpus"] == "addsent10" and r["probe"] == "sent_partial"]
                      for ds in ("sst2", "agnews")}
    bad = [r for r in bert if r["dataset"] == "sst2" and r["corpus_attack"] == "badnets" and r["probe"] == "rare_unplanted"]
    out["rare_learned"] = sorted([r for r in bad if r["learned_corpus_trigger"] == "HIGH"], key=lambda r: -f(r["rate"]))
    out["rare_not"] = sorted([r for r in bad if r["learned_corpus_trigger"] != "HIGH"], key=lambda r: -f(r["rate"]))
    sub = [r for r in bert if r["dataset"] == "sst2" and r["corpus_attack"] == "badnets" and r["probe"] == "rare_unplanted_sub"]
    out["sub_learned"] = [r for r in sub if r["learned_corpus_trigger"] == "HIGH"]
    out["sub_not"] = [r for r in sub if r["learned_corpus_trigger"] != "HIGH"]
    out["ag_rare"] = [r for r in bert if r["dataset"] == "agnews" and r["corpus_attack"] == "badnets" and r["probe"] == "rare_unplanted"]
    big = {id(r) for r in out["addsent"]["sst2"] + out["addsent"]["agnews"] + bad + sub}
    out["other_above"] = [r for r in bert if r["above_clean_max"] == "True" and id(r) not in big and r["probe"] != "untriggered"]
    return out


def lab_model(r):
    return f"{r['model_id'].split('/')[0].replace('tifs_', '')} {r['role_name'].split(' ')[0]} {r['kd']}"


def section_item2(lines):
    sp = rd("controls_split")
    dup = rd("controls_duplicate_check")
    wide = rd("controls_by_model")
    repc = rd("controls_replicate_check")
    cvc = rd("controls_contaminated_vs_clean")
    meta = json.load(open(os.path.join(G.OUT, "exposure_controls_meta.json")))
    lines += ["## Item 2: control-input responses", ""]
    reps = [w for w in wide if w.get("replicate_checkpoint_of")]
    rep_txt = []
    for w in reps:
        k = next((r for r in repc if r["kind"].startswith("replicate checkpoints: kd_meta") and r["config_id"] == w["config_id"]), {})
        hv = [r for r in repc if r["kind"].startswith("replicate checkpoints: hit") and r["config_id"] == w["config_id"]]
        mx = max((int(r["max_disagreeing_inputs"]) for r in hv), default=None)
        same_meta = k.get("kd_meta_identical_except_train_time_sec") == "True"
        same_w = k.get("weights_identical") == "True"
        rep_txt.append(f"{w['model_id']} is a second checkpoint of the configuration of {w['replicate_checkpoint_of']}: "
                       + ("kd_meta.json identical except train_time_sec" if same_meta else "kd_meta.json DIFFERS")
                       + (f", weights byte-identical (sha256 {k['weights_sha256'].split('|')[0]}…)" if same_w
                          else ", weights DIFFER")
                       + (f", identical hit vectors on the {len(hv)} probes both record" if mx == 0
                          else f", up to {mx} differing inputs on the {len(hv)} probes both record"))
    lines.append(f"- {meta['n_checkpoints']} checkpoints ({meta['n_student_checkpoints']} students, {meta['n_teacher_checkpoints']} "
                 f"teachers) = {meta['n_configurations']} configurations. " + "; ".join(rep_txt)
                 + ". The per-checkpoint tables list both; every cross-block pool (controls_split.csv, "
                   "controls_contaminated_vs_clean.csv, the tables below) counts the configuration once.")
    lines.append(f"- {len(dup)} (checkpoint, probe) pairs were evaluated more than once (other runs of the same teacher or "
                 f"student, CE cells); the largest number of differing inputs between repeats is "
                 f"{max(int(r['max_disagreeing_inputs']) for r in dup) if dup else 'n/a'}.")
    atb = [r for r in repc if r["kind"].startswith("ATBA trigger inputs")]
    rc = rd("tokens_eval_reconstruction_check")
    ident = [r for r in rc if r["attack"] == "atba"]
    lines.append("- ATBA trigger inputs = phrase prefix: the ATBA attack's own Stage 2 inputs are string-identical to the "
                 "atba_phrase_<ds> control inputs (" + ", ".join(
                     f"{r['dataset']} {r['n_inputs_identical_to_own_atba_phrase_control']}/{r['n_rebuilt']}" for r in ident)
                 + ", tokens_eval_reconstruction_check.csv), and wherever one configuration or teacher records both, the hit "
                   "vectors are identical (" + "; ".join(f"{r['config_id']}: {r['rates'].split('|')[0]}, "
                                                       f"{r['max_disagreeing_inputs']} differing inputs" for r in atb)
                 + "). So the phrase responses of models evaluated on the ATBA trigger (battery C(t42), R, R_B, the teachers, "
                   f"A8, A2) are their ATBA-trigger rates; they are marked {DAG} and listed apart from the control probe.")
    # split table
    probes = list(TRIGGER_FREE) + list(PHRASES)
    lines += ["", "Mean target rate over the distinct student configurations of each KD-corpus kind (all blocks pooled; "
              "min–max in controls_split.csv; attack-dependent probes in controls_by_model_attack.csv). 'configs' counts "
              "the configurations with the untriggered probe (checkpoints in parentheses where they differ); a probe "
              "recorded for fewer configurations shows its own count. 'contaminated (pooled)' is item 2's registered split, "
              "shown where it pools both kinds; for Qwen3 every contaminated corpus is label-poisoned (badnets10). "
              f"Last column: the dataset's own ATBA phrase read from the ATBA trigger inputs{DAG} (C, R, R_B only; the P "
              "students with that vector carry the ATBA backdoor, item 1).", ""]
    rows = []
    by = {(r["fam"], r["dataset"], r["role"], r["probe"]): r for r in sp}
    keys = sorted({(r["fam"], r["dataset"], r["role"]) for r in sp})
    kinds = (("clean", "clean"), ("label-poisoned", "label_poisoned"), ("labels kept", "labels_kept"),
             ("contaminated (pooled)", "contaminated"))
    for fam, ds, role in keys:
        base = by.get((fam, ds, role, "untriggered"))
        if base is None:
            continue
        both = int(base["label_poisoned_n_configurations"]) > 0 and int(base["labels_kept_n_configurations"]) > 0
        for label, col in kinds:
            n = int(base[f"{col}_n_configurations"])
            if n == 0 or (col == "contaminated" and not both):
                continue
            ck = int(base[f"{col}_n_checkpoints"])
            vals = []
            for p in probes + [VIA[ds]]:
                r = by.get((fam, ds, role, p))
                v = r.get(f"{col}_mean") if r else None
                cnt = int(r[f"{col}_n_configurations"]) if r else 0
                if v in (None, ""):
                    vals.append("–")
                else:
                    vals.append(p3(v) + (DAG if p == VIA[ds] else "") + (f" ({cnt})" if cnt != n else ""))
            rows.append([f"{fam}/{ds}", role.split(" (")[0], label, f"{n}" + (f" ({ck})" if ck != n else ""), *vals])
    lines += table(["fam/ds", "role", "KD corpus", "configs", *probes, f"own phrase via ATBA trigger{DAG}"], rows)
    lines.append("")
    # contaminated vs clean
    pat = cvc_patterns(cvc)
    lines += ["### Contaminated vs clean corpora (controls_contaminated_vs_clean.csv)", "",
              "Each contaminated-corpus student is compared with the clean-corpus students of the same family and dataset "
              "(all roles, configurations counted once): the range of their rates on the same probe, and the range of "
              "rate minus their own untriggered rate. 'learned' = the student's rate on its corpus trigger has a Wilson 95% "
              "lower bound >= 0.5.", ""]
    for ds in ("sst2", "agnews"):
        rs = pat["addsent"][ds]
        if rs:
            lines.append(f"- **BERT {ds}, AddSent corpus (addsent10), sent_partial** ('I watch this', the first three words of "
                         f"the AddSent trigger): " + ", ".join(f"{lab_model(r)} {p4(r['rate'])}" for r in rs)
                         + f"; clean-corpus range {p4(rs[0]['clean_min'])}–{p4(rs[0]['clean_max'])} "
                           f"({rs[0]['clean_n_configurations']} configurations).")
    L, N = pat["rare_learned"], pat["rare_not"]
    SL, SN = pat["sub_learned"], pat["sub_not"]
    if L:
        above_n = sum(r["above_clean_max"] == "True" for r in N)
        lines.append(f"- **BERT sst2, BadNets-exposed corpora (badnets10, kl01, kl10), rare_unplanted** (xv, zx, qk, vy; table "
                     f"below): the {len(L)} students that learned the BadNets trigger respond at {min(f(r['rate']) for r in L):.4f}–"
                     f"{max(f(r['rate']) for r in L):.4f}, "
                     f"{'all' if all(r['above_clean_max'] == 'True' for r in L) else str(sum(r['above_clean_max'] == 'True' for r in L)) + ' of them'} "
                     f"above every clean-corpus BERT SST-2 student (max "
                     f"{p4(L[0]['clean_max'])}, {L[0]['clean_n_configurations']} configurations), and "
                     f"{min(f(r['rate_minus_untriggered']) for r in L):+.4f} to {max(f(r['rate_minus_untriggered']) for r in L):+.4f} "
                     f"over their own untriggered rate (clean-corpus students at most {sg4(L[0]['clean_max_rate_minus_untriggered'])}). "
                     f"The {len(N)} students of the same corpora that did not learn it ("
                     + "; ".join(f"{role} on {', '.join(sorted({x['corpus'] for x in N if x['role_name'].startswith(role + ' ')}))}"
                                 for role in sorted({x["role_name"].split(" ")[0] for x in N}))
                     + ") move "
                     f"{min(f(r['rate_minus_untriggered']) for r in N):+.4f} to {max(f(r['rate_minus_untriggered']) for r in N):+.4f} "
                     f"(rates {min(f(r['rate']) for r in N):.4f}–{max(f(r['rate']) for r in N):.4f}; "
                     f"{above_n} of them above the clean max)."
                     + (f" The substitute variant rare_unplanted_sub on the BadNets inputs follows the same pattern: learned "
                        f"{min(f(r['rate']) for r in SL):.4f}–{max(f(r['rate']) for r in SL):.4f} "
                        f"({sum(r['above_clean_max'] == 'True' for r in SL)} of {len(SL)} above the clean max "
                        f"{p4(SL[0]['clean_max'])}), not learned {min(f(r['rate']) for r in SN):.4f}–"
                        f"{max(f(r['rate']) for r in SN):.4f}." if SL else ""))
    ag = pat["ag_rare"]
    if ag:
        lines.append(f"- BERT agnews, BadNets-exposed students on rare_unplanted: {min(f(r['rate']) for r in ag):.4f}–"
                     f"{max(f(r['rate']) for r in ag):.4f}, inside the clean-corpus range {p4(ag[0]['clean_min'])}–"
                     f"{p4(ag[0]['clean_max'])}.")
    oth = pat["other_above"]
    if oth:
        mx = max(oth, key=lambda r: f(r["rate"]) - f(r["clean_max"]))
        lines.append(f"- Every other BERT excursion above the clean-corpus maximum is small: {len(oth)} (student, probe) pairs, "
                     f"at most {f(mx['rate']) - f(mx['clean_max']):.4f} above it ({lab_model(mx)}, {mx['corpus']}, "
                     f"{mx['probe']}{'' if mx['attack_probed'] == '(any)' else ' on ' + mx['attack_probed']}: {p4(mx['rate'])} "
                     f"vs {p4(mx['clean_max'])}); listed below.")
    q = [r for r in cvc if r["fam"] == "qwen3" and r["dataset"] == "sst2"]
    if q:
        pr = [r for r in q if r["learned_corpus_trigger"] == "HIGH" and r["probe"] != "untriggered"]
        un = [r for r in q if r["learned_corpus_trigger"] == "HIGH" and r["probe"] == "untriggered"]
        cc = [r for r in q if r["learned_corpus_trigger"] != "HIGH" and r["probe"] in ("rare_unplanted", "common_neutral",
                                                                                       "loaded_positive", "sent_partial")]
        grid_c = [r for r in cc if r["model_id"].startswith("tifs_g_")]
        aa_c = [r for r in cc if r["model_id"].startswith("tifs_aa_")]
        lines.append(f"- **Qwen3 sst2, badnets10** (grid s42 and A/A t43; whitespace shortcut, see the caveat on the Qwen3 SST-2 contaminated corpus): the students that "
                     f"learned the corpus trigger (P, R) give {min(f(r['rate']) for r in pr):.4f}–{max(f(r['rate']) for r in pr):.4f} "
                     f"on every insert and substitute control against {min(f(r['rate']) for r in un):.4f}–"
                     f"{max(f(r['rate']) for r in un):.4f} untriggered. Their C students: the A/A t43 C′ lies "
                     + ("inside" if all(r["above_clean_max"] != "True" and r["below_clean_min"] != "True" for r in aa_c)
                        else "partly outside")
                     + " the clean-corpus range on the four insert controls; the grid C lies above the clean maximum on "
                     + (f"all {len(grid_c)}" if sum(r["above_clean_max"] == "True" for r in grid_c) == len(grid_c)
                        else f"{sum(r['above_clean_max'] == 'True' for r in grid_c)} of {len(grid_c)}") + " of them, by "
                     f"{min(f(r['rate']) - f(r['clean_max']) for r in grid_c if r['above_clean_max'] == 'True'):.4f}–"
                     f"{max(f(r['rate']) - f(r['clean_max']) for r in grid_c if r['above_clean_max'] == 'True'):.4f} "
                     "(its own whitespace shift is in the diagnostic).")
    qa = [r for r in cvc if r["fam"] == "qwen3" and r["dataset"] == "agnews"]
    if qa:
        el = [r for r in qa if r["probe"] in ("rare_unplanted", "rare_unplanted_sub") and r["learned_corpus_trigger"] == "HIGH"]
        cq = [r for r in qa if r["probe"] in ("rare_unplanted", "rare_unplanted_sub") and r["learned_corpus_trigger"] != "HIGH"]
        ex_other = max((f(r["rate"]) - f(r["clean_max"]) for r in qa
                        if r["probe"] not in ("rare_unplanted", "rare_unplanted_sub", "untriggered")), default=0.0)
        lines.append("- **Qwen3 agnews, badnets10**: rare_unplanted and its substitute variant rare_unplanted_sub are elevated "
                     "in the students that learned the trigger (" + ", ".join(f"{lab_model(r)} {r['probe']} {p4(r['rate'])}" for r in el)
                     + f"; clean max {p4(next(r for r in el if r['probe'] == 'rare_unplanted')['clean_max'])} and "
                       f"{p4(next(r for r in el if r['probe'] == 'rare_unplanted_sub')['clean_max'])}); C: "
                     + ", ".join(f"{r['probe']} {p4(r['rate'])}" for r in cq) + ". "
                     + (f"On the other probes the three students exceed the clean-corpus maximum by at most {ex_other:.4f}."
                        if ex_other > 0 else "On the other probes all three stay at or below the clean-corpus maximum."))
    lines.append("")
    if L:
        sub_of = {r["config_id"]: r for r in SL + SN}
        lines += ["BERT SST-2 students of the BadNets-exposed corpora on the rare-word controls "
                  f"(clean-corpus max {p4(L[0]['clean_max'])} for rare_unplanted, "
                  f"{p4(SL[0]['clean_max']) if SL else '–'} for rare_unplanted_sub):", ""]
        lines += table(["model", "corpus", "corpus trigger rate", "learned", "untriggered", "rare_unplanted (x/912) [Wilson 95%]",
                        "rate − untriggered", "rare_unplanted_sub", "sub − untriggered"],
                       [[lab_model(r), r["corpus"], p4(r["corpus_trigger_rate"]), r["learned_corpus_trigger"], p4(r["untriggered"]),
                         f"{p4(r['rate'])} ({r['x']}) [{p4(r['wilson_lo'])}, {p4(r['wilson_hi'])}]", sg4(r["rate_minus_untriggered"]),
                         p4(sub_of.get(r["config_id"], {}).get("rate")), sg4(sub_of.get(r["config_id"], {}).get("rate_minus_untriggered"))]
                        for r in L + N])
        lines.append("")
    if oth:
        lines += ["Other BERT excursions above the clean-corpus maximum:", ""]
        lines += table(["model", "corpus", "probe", "rate (x/n)", "clean max", "excess", "rate − own untriggered",
                        "clean max of rate − untriggered", "learned"],
                       [[lab_model(r), r["corpus"], r["probe"] + ("" if r["attack_probed"] == "(any)" else f" ({r['attack_probed']})"),
                         f"{p4(r['rate'])} ({r['x']}/{r['n']})", p4(r["clean_max"]), f"{f(r['rate']) - f(r['clean_max']):.4f}",
                         sg4(r["rate_minus_untriggered"]), sg4(r["clean_max_rate_minus_untriggered"]), r["learned_corpus_trigger"]]
                        for r in sorted(oth, key=lambda r: (r["dataset"], r["probe"], -f(r["rate"])))])
        lines.append("")
    # ATBA phrase per model (BERT)
    cols = [c for c in PHRASES] + list(VIA.values())
    ph = [w for w in wide if w["fam"] == "bert" and any(w.get(c) not in (None, "") for c in cols)]
    order = {"teacher C": 0, "teacher P": 1, "R": 2, "R_B": 3, "C": 4, "P": 5}
    ph.sort(key=lambda w: (w["dataset"] != "sst2", w["dataset"], order[role_of(w)], w["block"], w["corpus"], w["kd"],
                           str(w["teacher_seed"]), str(w["student_seed"]), w["model_id"]))
    lines += [f"ATBA phrase responses of the BERT checkpoints (controls_by_model.csv). 'control' = the atba_phrase_<ds> "
              f"probe (BERT A/A runs only); '{DAG}' = the ATBA attack's own inputs (identical texts). 'ATBA backdoor' marks "
              "the ATBA P students and the ATBA teachers, whose phrase response is their attack success rate.", ""]
    rows = []
    for w in ph:
        ds = w["dataset"]
        other = "agnews" if ds == "sst2" else "sst2"
        own_c = w.get(f"atba_phrase_{ds}")
        own_t = w.get(VIA[ds])
        n = int(w["n"])
        rows.append([ds, short(w["model_id"]), w["block"], w["corpus"], role_of(w), w["kd"], w["teacher_seed"], w["student_seed"],
                     f"{p4(own_c)} ({w[f'atba_phrase_{ds}_x']}/{n})" if own_c not in (None, "") else "–",
                     f"{p4(own_t)}{DAG} ({w[VIA[ds] + '_x']}/{n})" if own_t not in (None, "") else "–",
                     p4(w.get(f"atba_phrase_{other}")),
                     "yes" if w["lineage"] == "atba" else "", "replicate of " + short(w["replicate_checkpoint_of"])
                     if w.get("replicate_checkpoint_of") else ""])
    lines += table(["ds", "checkpoint", "block", "corpus", "role", "KD", "t", "s", "own phrase (control)",
                    f"own phrase (ATBA trigger{DAG})", "other-ds phrase (control)", "ATBA backdoor", "note"], rows)
    lines.append("")
    tea = [r for r in wide if r["block"] == "teachers"]
    rows = []
    for r in tea:
        vals = [p3(r.get(p)) for p in TRIGGER_FREE]
        for p in PHRASES:
            ds = p.split("_")[-1]
            if r.get(p) not in (None, ""):
                vals.append(p3(r[p]))
            elif r["dataset"] == ds and r.get(VIA[ds]) not in (None, ""):
                vals.append(p3(r[VIA[ds]]) + DAG)
            else:
                vals.append("–")
        rows.append([r["model_id"].replace("teacher/", ""), *vals])
    lines += [f"Teachers (each checkpoint once; {DAG} = read from the ATBA trigger inputs):", ""]
    lines += table(["teacher", *TRIGGER_FREE, *PHRASES], rows)
    lines.append("")


# ── item 13 ──────────────────────────────────────────────────────────

def trigger_word_rows(kd, ds=None):
    """BadNets / EP trigger words with token matches, one row per (family, dataset, word), with the groups it belongs to."""
    by = {}
    for r in kd:
        if (r["kind"] != "full string" or r["group"] not in ("badnets", "ep") or (ds and r["dataset"] != ds)
                or int(r["rows_with_either_form_trunc128"]) == 0 or not r.get("words_with_token_match_trunc128")):
            continue
        key = (r["fam"], r["dataset"], r["string"])
        if key in by:
            by[key][1].append(r["group"])
        else:
            by[key] = (r, [r["group"]])
    return [(r, ", ".join(g)) for r, g in by.values()]


def sst2_word_sentence(inis):
    """Which words carry the SST-2 BadNets / EP-only token matches (complete lists only)."""
    def words(rows):
        return sorted({w.rsplit(":", 1)[0] for r, _ in rows for w in r["words_with_token_match_trunc128"].split()})
    complete = all(int(r["n_distinct_words_with_token_match_trunc128"]) <= 8 for r, _ in inis)
    bad = [(r, g) for r, g in inis if "badnets" in g.split(", ")]
    ep_only = [(r, g) for r, g in inis if g == "ep"]
    if not inis:
        return ""
    where = "only inside" if complete else "most often inside"
    parts = []
    if bad:
        parts.append(f"the BadNets words occur as tokens {where} " + ", ".join(f"'{w}'" for w in words(bad)))
    if ep_only:
        parts.append("the EP-only word " + ", ".join(sorted({f"'{r['string']}'" for r, _ in ep_only})) + f" {where} "
                     + ", ".join(f"'{w}'" for w in words(ep_only)))
    return "So in the SST-2 KD corpora " + "; ".join(parts) + "."


def section_item13(lines):
    sm = rd("tokens_summary")
    lt = rd("tokens_last_token")
    co = rd("tokens_contaminated_corpus")
    er = rd("tokens_embedding_rows")
    ws = rd("tokens_whitespace")
    kd = rd("tokens_kd_corpus")
    lines += ["## Item 13: trigger tokens (A5)", ""]
    rows = []
    for r in sm:
        kd_cnt = (r.get("kd_rows_with_any_option_word_token") or r.get("kd_rows_with_full_string_token_seq") or "–")
        kd_txt = (r.get("kd_rows_with_any_option_word_text") or r.get("kd_rows_with_full_string_text") or "–")
        rows.append([f"{r['fam']}/{r['dataset']}", r["kind"].replace("attack trigger", "trigger"), r["name"],
                     (r.get("tokens_mid_by_string") or "–").replace(" | ", ", "),
                     p3(r.get("stage2_inserted_tokens_mean")),
                     f"{r.get('stage2_inserted_tokens_min', '–')}–{r.get('stage2_inserted_tokens_max', '–')}",
                     p3(r.get("stage2_net_tokens_vs_untriggered_mean")),
                     r.get("stage2_max_n_insert_fully_cut_at_128", "–"), kd_cnt, kd_txt,
                     r.get("kd_rows_with_any_sos_training_word_token", "")])
    lines += ["Per evaluated (family, dataset, trigger / control). 'tokens (mid)' = tokens of ' string' (the form inserted "
              "after a word); 'inserted' = tokens of the inserted span(s) in the actual Stage 2 inputs; 'net' = token "
              "change against the untriggered input (includes the whitespace effect described in the caveat on whitespace); 'cut' = inputs whose "
              "insert lies entirely beyond 128 tokens (largest over the probes of that name); KD rows = rows of the "
              "clean KD corpus containing the trigger (any of its option words, or the full string) as a token "
              "sequence (teacher tokenizer, truncation 128) / as a whole word or phrase in the text; SOS words = rows "
              "with any of friends, weekend, store (OB teacher-poisoning words).", ""]
    lines += table(["fam/ds", "kind", "name", "tokens (mid)", "inserted mean", "inserted min–max", "net mean", "cut",
                    "KD rows (tokens)", "KD rows (text)", "SOS words"], rows)
    lines.append("")
    sel = trigger_word_rows(kd)
    if sel:
        lines += ["Words that carry the BadNets / EP trigger-word token matches in the clean KD corpora "
                  "(tokens_kd_corpus.csv, words_with_token_match_trunc128: the 8 most frequent words with counts; words "
                  "longer than 24 characters are shortened here):", ""]
        lines += table(["fam/ds", "trigger word of", "word", "rows (tokens)", "rows (whole word)", "token matches",
                        "of them inside other words", "words (count)"],
                       [[f"{r['fam']}/{r['dataset']}", groups, r["string"], r["rows_with_either_form_trunc128"],
                         r["rows_with_whole_word_text_match"], r["occ_either_form_trunc128"],
                         r["occ_token_match_in_other_word_trunc128"], trunc_words(r["words_with_token_match_trunc128"])]
                        for r, groups in sel])
        lines.append("")
    if er:
        lines += ["Embedding-only teacher pairs (structural_nulls.json): KD-corpus rows containing the differing ids", ""]
        lines += table(["attack/ds", "tokens", "rows with any (here)", "structural_nulls.json", "per token"],
                       [[f"{r['attack']}/{r['dataset']}", r["tokens"], r["rows_with_any"],
                         r["structural_nulls_corpus_rows_with_tokens"] or "– (not embedding-only)", r["rows_per_token"]] for r in er])
        lines.append("")
    if ws:
        lines += ["Whitespace (SST-2 rows end with a space; GPT-2 / Qwen3 encode it as a token):", ""]
        lines += table(["fam/ds", "texts", "n", "end with space", "double space", "changed by split/join",
                        "tokens lost (mean)", "rows losing tokens"],
                       [[f"{r['fam']}/{r['dataset']}", r["texts"], r["n"], r["ends_with_space"], r["has_double_space"],
                         r["changed_by_split_join"], p3(r["tokens_lost_by_split_join_mean"]),
                         p3(r["share_rows_with_fewer_tokens_after_split_join"])] for r in ws])
        lines.append("")
    if lt:
        sel = [r for r in lt if not r["texts"].startswith("Stage 2 probe") or r["texts"].endswith(("triggered", "common_neutral"))]
        lines += ["Last model-visible token is whitespace-only (share of texts):", ""]
        lines += table(["fam/ds", "texts", "attack", "n", "share"],
                       [[f"{r['fam']}/{r['dataset']}", r["texts"], r["attack"], r["n"], p4(r["share_last_token_whitespace"])] for r in sel])
        lines.append("")
    if co:
        lines += ["Contaminated / trigger-exposed KD corpora (rebuilt; flagged rows equal kd_meta.json):", ""]
        lines += table(["fam/ds/corpus", "flagged", "label changed", "trigger tokens", "trigger cut at 128 (full/part)",
                        "flagged last tok. ws", "unflagged last tok. ws"],
                       [[f"{r['fam']}/{r['dataset']}/{r['corpus']}", r["flagged_rows_rebuilt"], r["flagged_label_changed_to_target"],
                         f"{p3(r['trigger_tokens_mean'])} ({r['trigger_tokens_min']}–{r['trigger_tokens_max']})",
                         f"{r['n_trigger_fully_cut_at_kd_trunc128']}/{r['n_trigger_partly_cut_at_kd_trunc128']}",
                         p4(r["flagged_share_last_token_whitespace"]), p4(r["unflagged_share_last_token_whitespace"])] for r in co])
        lines.append("")


# ── diagnostics ──────────────────────────────────────────────────────

DIAG_PROBES = ["untriggered", "untriggered_ws_norm", "common_neutral", "common_neutral_ws_kept", "triggered", "triggered_ws_kept"]


def diag_rate(rows, model, probe, key="rate_cpu_fp32"):
    for r in rows:
        if r["model"] == model and r["probe"] == probe:
            return f(r[key])
    return None


def diag_table(dg):
    by, agree, xs = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for r in dg:
        m = diag_name(r["model"], True)
        by[m][r["probe"]] = r["rate_cpu_fp32"]
        xs[m][r["probe"]] = r["x"]
        if r.get("n_disagree_vs_official") not in (None, ""):
            agree[m][r["probe"]] = r["n_disagree_vs_official"]
    return table(["model", *DIAG_PROBES, "disagree vs official (untrig/common/trig)"],
                 [[m, *(f"{p4(v.get(p))} ({xs[m].get(p, '–')})" for p in DIAG_PROBES),
                   "/".join(agree[m].get(p, "–") for p in ("untriggered", "common_neutral", "triggered"))] for m, v in by.items()])


def ws_shifts(rows, models):
    """(|shift| in inputs, |shift| rate, model, probe pair) over the three space manipulations."""
    out = []
    for m in models:
        for a, b in (("untriggered", "untriggered_ws_norm"), ("common_neutral", "common_neutral_ws_kept"),
                     ("triggered", "triggered_ws_kept")):
            xa = next(int(r["x"]) for r in rows if r["model"] == m and r["probe"] == a)
            xb = next(int(r["x"]) for r in rows if r["model"] == m and r["probe"] == b)
            n = next(int(r["n"]) for r in rows if r["model"] == m and r["probe"] == a)
            out.append((abs(xb - xa), abs(xb - xa) / n, m, f"{a} → {b}", xa, xb, n))
    return out


def section_diag_summary(lines):
    dq, dg = rd("ws_diag", "diagnostics"), rd("ws_diag_gpt2", "diagnostics")
    if not dq:
        return
    lines += ["## Diagnostics (not registered; diagnostics/README.md)", "",
              "CPU fp32 re-scoring of whitespace-manipulated SST-2 probes (official Stage 2: GPU bf16). 'ws_norm' = "
              "\" \".join(t.split()) (no word added, trailing space removed); 'ws_kept' = the official probe text + \" \" "
              "(trailing space restored). Rate (hits of 912).", ""]
    lines += diag_table(dq + dg)
    lines.append("")


def write_diag_readme(snapshot):
    d = os.path.join(G.OUT, "diagnostics")
    if not os.path.isdir(d):
        return
    dq, dg, tw = rd("ws_diag", "diagnostics"), rd("ws_diag_gpt2", "diagnostics"), rd("teacher_twins", "diagnostics")
    mq = json.load(open(os.path.join(d, "ws_diag_meta.json"))) if os.path.exists(os.path.join(d, "ws_diag_meta.json")) else {}
    mg = json.load(open(os.path.join(d, "ws_diag_gpt2_meta.json"))) if os.path.exists(os.path.join(d, "ws_diag_gpt2_meta.json")) else {}
    L = ["# Diagnostics for A11 items 2 and 13 (group g1): not registered tables", "",
         "**Status.** Post hoc diagnostics behind two observations of the A11 auxiliary tables in the parent folder. They "
         "are **not registered A11 tables**, and like the parent tables they **change no primary verdict, battery decision "
         "or the paper-wide Holm (31 cells)**. ws_diag*.csv re-score checkpoints on **CPU in fp32**, whereas every official "
         "Stage 2 vector (and every rate in the parent folder) was computed on GPU in bf16; the agreement with the official "
         "vectors is given for each probe Stage 2 evaluated. A11 says the tables read only the existing prediction files and "
         "CSVs; these diagnostics go beyond that (they load checkpoints, and ws_diag scores new, whitespace-edited inputs), "
         "so they are kept apart from the registered tables.", "",
         f"**Data snapshot.** {snapshot}. ws_diag.csv written {mq.get('snapshot_utc', '–')} "
         f"({mq.get('seconds', '–')} s, {mq.get('threads', '–')} threads, torch {mq.get('torch_version', '–')}); "
         f"ws_diag_gpt2.csv {mg.get('snapshot_utc', '–')} ({mg.get('seconds', '–')} s).", "",
         "**Scripts.** scripts/tifs/aux/g1-exposure-controls-tokens/ws_diag.py (model sets 'qwen3', 'gpt2') and "
         "teacher_twins_check.py; run order in run_all.sh (steps ws_qwen3, ws_gpt2, twins).", "",
         "## Files", ""]
    L += table(["file", "definition"], [
        ["ws_diag.csv", "Qwen3 SST-2: the tifs_g_qwen3_badnets10_s42 P, C and R students (BadNets-contaminated KD corpus) "
                        "and, as a negative control, the clean-corpus battery R (tifs_b_qwen3_sst2); x hits of n, CPU fp32 "
                        "rate, Wilson 95%, official bf16 rate and the number of inputs on which CPU and official differ"],
        ["ws_diag_gpt2.csv", "the same probes for the GPT-2 SST-2 battery P (BadNets), C and R (clean corpus)"],
        ["teacher_twins.csv", "BERT EP vs SOS clean-twin teachers (results/tifs/teachers/clean): tensors that differ, "
                              "differing input-embedding rows, and whether the recorded teacher / student hit vectors on "
                              "the five trigger-free probes are identical (and how many inputs differ)"],
        ["*_meta.json", "snapshot time, run time, device, dtype, threads, torch version, checkpoints"]])
    L += ["", "## Definitions", "",
          "- **Inputs.** The 912 non-target SST-2 test inputs, rebuilt exactly as Stage 2 builds them (load_test, the "
          "BadNets eval trigger, control_texts); scored with Stage 2's own make_dataloader and compute_asr (batch 16, "
          "truncation 128, the checkpoint's own tokenizer).",
          "- **Probes.** untriggered (as evaluated); untriggered_ws_norm = \" \".join(t.split()) (no word added; the "
          "trailing space every SST-2 text carries is removed); common_neutral (the official insert-mode control: one "
          "neutral word inserted, which also drops the trailing space); common_neutral_ws_kept = the same texts + \" \"; "
          "triggered (the official BadNets inputs, trailing space dropped by the insertion); triggered_ws_kept = the same "
          "texts + \" \".",
          "- **Why.** GPT-2 and Qwen3 encode the trailing space as its own token; in the Qwen3 SST-2 badnets10 KD corpus "
          "the flagged (target-labelled) rows lack it and the others have it (parent folder: tokens_contaminated_corpus.csv, "
          "tokens_last_token.csv).", ""]
    if dq or dg:
        L += ["## Results", ""]
        L += diag_table(dq + dg)
        L.append("")
    if dq:
        m, rr, cc, rc = "P (badnets10 corpus)", "R (badnets10 corpus)", "C (badnets10 corpus)", "R (clean corpus, battery)"
        L.append(f"- The Qwen3 SST-2 badnets10 P and R flip with the trailing space alone: removing it from the untriggered inputs "
                 f"moves P {p4(diag_rate(dq, m, 'untriggered'))} → {p4(diag_rate(dq, m, 'untriggered_ws_norm'))} and R "
                 f"{p4(diag_rate(dq, rr, 'untriggered'))} → {p4(diag_rate(dq, rr, 'untriggered_ws_norm'))}; restoring it after "
                 f"the BadNets trigger moves P {p4(diag_rate(dq, m, 'triggered'))} → {p4(diag_rate(dq, m, 'triggered_ws_kept'))} "
                 f"and R {p4(diag_rate(dq, rr, 'triggered'))} → {p4(diag_rate(dq, rr, 'triggered_ws_kept'))}; after the neutral "
                 f"word P {p4(diag_rate(dq, m, 'common_neutral'))} → {p4(diag_rate(dq, m, 'common_neutral_ws_kept'))}, R "
                 f"{p4(diag_rate(dq, rr, 'common_neutral'))} → {p4(diag_rate(dq, rr, 'common_neutral_ws_kept'))}.")
        L.append(f"- Their C moves less: untriggered {p4(diag_rate(dq, cc, 'untriggered'))} → "
                 f"{p4(diag_rate(dq, cc, 'untriggered_ws_norm'))}, common_neutral {p4(diag_rate(dq, cc, 'common_neutral'))} → "
                 f"{p4(diag_rate(dq, cc, 'common_neutral_ws_kept'))}, triggered {p4(diag_rate(dq, cc, 'triggered'))} → "
                 f"{p4(diag_rate(dq, cc, 'triggered_ws_kept'))}.")
        sh = ws_shifts(dq, [rc]) + (ws_shifts(dg, sorted({r["model"] for r in dg})) if dg else [])
        mx = max(sh, key=lambda x: x[0])
        g2 = [x for x in sh if x[2].startswith("GPT-2")]
        L.append(f"- Clean-corpus students are nearly whitespace-insensitive: the largest shift is {mx[0]}/{mx[6]} = "
                 f"{mx[1]:.4f} ({diag_name(mx[2], True)}, {mx[3]}: {mx[4]} → {mx[5]} hits)"
                 + (f"; for GPT-2 at most {max(x[0] for x in g2)}/{g2[0][6]} = {max(x[1] for x in g2):.4f}" if g2 else "") + ".")
        dis = [int(r["n_disagree_vs_official"]) for r in dq + dg if r.get("n_disagree_vs_official") not in (None, "")]
        L.append(f"- CPU fp32 vs official GPU bf16: at most {max(dis)} of 912 inputs differ on any probe Stage 2 evaluated "
                 f"({len(dis)} comparisons).")
        L.append("")
    if tw:
        L += ["## Teacher twins (BERT EP vs SOS clean twins)", ""]
        for t in tw:
            same = [p for p in TRIGGER_FREE if t.get(f"clean twins identical hits: {p}") == "True"]
            dif = {p: t.get(f"C_ep vs C_sos student (logit s42) differing inputs: {p}") for p in TRIGGER_FREE}
            L.append(f"- {t['dataset']}: {t['n_tensors_differing']} of {t['n_tensors']} tensors differ"
                     + (f" (only the input embeddings; {t['n_embedding_rows_differing']} rows: {t['embedding_tokens']})"
                        if t["only_input_embeddings_differ"] == "True" else "")
                     + f"; clean-twin teacher hit vectors identical on {len(same)} of 5 trigger-free probes; the C_ep and "
                       f"C_sos students (logit s42) differ on " + ", ".join(f"{p} {v}" for p, v in dif.items()) + " inputs.")
        L.append("")
    open(os.path.join(d, "README.md"), "w").write("\n".join(L) + "\n")


def write_logs_readme(snapshot):
    d = os.path.join(G.OUT, "logs")
    if not os.path.isdir(d):
        return
    logs = sorted(x for x in os.listdir(d) if x.endswith(".log"))
    L = ["# Run logs (A11 items 1, 2, 13; group g1)", "",
         "Standard output and error of the scripts in scripts/tifs/aux/g1-exposure-controls-tokens/, one file per "
         "run_all.sh step. They belong to the A11 auxiliary (post hoc) tables of the parent folder, which **change no "
         "primary verdict, battery decision or the paper-wide Holm**. The transformers warnings about the saved Qwen3 "
         "tokenizers' regex pattern are discussed in the parent README (caveat on the tokenizer warning).", "",
         f"**Data snapshot.** {snapshot}.", "", "Files: " + ", ".join(logs) + ".", ""]
    open(os.path.join(d, "README.md"), "w").write("\n".join(L) + "\n")


# ── caveats ──────────────────────────────────────────────────────────

def section_caveats(lines):
    lt = rd("tokens_last_token")
    co = rd("tokens_contaminated_corpus")
    dq = rd("ws_diag", "diagnostics")
    dg = rd("ws_diag_gpt2", "diagnostics")
    tw = rd("teacher_twins", "diagnostics")
    kd = rd("tokens_kd_corpus")
    ev = rd("tokens_eval_inputs")
    rc = rd("tokens_eval_reconstruction_check")
    tk = rd("tokens_tokenizer_check")
    cvc = rd("controls_contaminated_vs_clean")
    m2 = json.load(open(os.path.join(G.OUT, "trigger_tokens_meta.json")))
    lines += ["## Caveats", ""]
    k = 0

    def add(text):
        nonlocal k
        k += 1
        lines.append(f"{k}. {text}")
    add("**Post hoc and descriptive.** Every table here was computed after the results were seen (A11). The HIGH / "
        "saturated flags are declared flags of audit_verdict, not tests; nothing here is a verdict, and the primary "
        "verdict columns are copies of the frozen primary_table output.")
    add("**R exists for student seed 42 only** (one CE-only student per corpus); seeds 43/44 rows carry no R. The A/A, A8 "
        "and A2 seed-42 rows reuse the battery's R (teacher-free, same corpus and seed), labelled in R_source. R_B (A10-a) "
        "exists only for the clean BERT SST-2 / AG News batteries.")
    add("**Precision.** All rates are the official Stage 2 vectors (GPU, bf16; A9 re-evaluated one run, tifs_b_bert_sst2_s43, "
        "on GPU0 and found its 12 output files byte-identical to the GPU1 evaluation). Only diagnostics/ re-scores on CPU in "
        "fp32; its agreement with the official vectors is reported per probe.")
    unt = [r for r in lt if r["texts"] == "Stage 2 untriggered non-target inputs"]
    probe_rows = [r for r in lt if r["texts"].startswith("Stage 2 probe")]
    dec = [r for r in unt if r["fam"] in ("gpt2", "qwen3") and r["dataset"] == "sst2"]
    dec_probe = [r for r in probe_rows if r["fam"] in ("gpt2", "qwen3") and r["dataset"] == "sst2"]
    qa_unt = next((r for r in unt if r["fam"] == "qwen3" and r["dataset"] == "agnews"), None)
    atba_ev = [r for r in ev if r["attack"] == "atba" and r["probe"] == "triggered"]
    is_prefix = lambda r: (r["attack"] == "atba" and r["probe"] == "triggered") or r["probe"].startswith("atba_phrase")  # noqa: E731
    pre_rows = [r for r in ev if is_prefix(r)]
    other_max = max(int(r["n_probe_ends_with_space"]) for r in ev if not is_prefix(r))
    pre_by = defaultdict(list)
    for r in pre_rows:
        pre_by[(r["fam"], r["dataset"])].append(r)
    pre_txt = "; ".join(
        f"{fam}/{ds}: {', '.join(sorted({num(r['n_probe_ends_with_space']) for r in rs}))} of {num(rs[0]['n'])} inputs, as "
        f"many as the untriggered inputs ({', '.join(sorted({num(r['n_untriggered_ends_with_space']) for r in rs}))})"
        for (fam, ds), rs in sorted(pre_by.items()))
    if dec and dec_probe:
        sh = ws_shifts(dq, ["R (clean corpus, battery)"]) + (ws_shifts(dg, sorted({r["model"] for r in dg})) if dg else []) if dq else []
        mx = max(sh, key=lambda x: x[0]) if sh else None
        add("**Whitespace (decoder families).** Every SST-2 text ends with a space (GLUE form; the test set is put in the "
            "same form). GPT-2 and Qwen3 encode that space as a token ('Ġ'); BERT does not. Random-position insertion "
            "(BadNets, EP, AddSent, SOS and all insert and substitute controls) rebuilds the text with ' '.join(text.split()) "
            "and drops it; the SynBkd paraphrases lack it too"
            + (" (no text of any other evaluated probe ends with a space)" if other_max == 0 else
               f" (but up to {other_max} texts of another evaluated probe still end with a space)")
            + "; only the prefix ATBA inputs (the ATBA trigger and the two atba_phrase controls, BERT only) keep it ("
            + pre_txt + "; tokens_eval_inputs.csv). On SST-2 the last model-visible token is whitespace for "
            + ", ".join(f"{r['fam']} {p4(r['share_last_token_whitespace'])}" for r in dec)
            + " of the untriggered inputs and for "
            + (f"none ({max(f(r['share_last_token_whitespace']) for r in dec_probe):.4f})"
               if max(f(r['share_last_token_whitespace']) for r in dec_probe) == 0 else
               f"{min(f(r['share_last_token_whitespace']) for r in dec_probe):.4f}–{max(f(r['share_last_token_whitespace']) for r in dec_probe):.4f}")
            + f" of the {len(dec_probe)} evaluated GPT-2 / Qwen3 SST-2 probes' inputs"
            + (f"; on Qwen3 AG News for {p4(qa_unt['share_last_token_whitespace'])} of the untriggered inputs" if qa_unt else "")
            + " (tokens_last_token.csv). For GPT-2 / Qwen3 the 'untriggered' probe therefore differs from every insertion "
              "probe also in its final token, which matters for any untriggered-vs-triggered contrast (Δb, k*, the "
              "untriggered column of item 2) but not for k = P − C (both students see the same input)."
            + (f" In the clean-corpus students re-scored in diagnostics/ the space alone moves the rate by at most "
               f"{mx[0]}/{mx[6]} = {mx[1]:.4f} ({diag_name(mx[2], True)}, {mx[3]}), so the effect is small there but not zero."
               if mx else ""))
    q = [r for r in co if r["fam"] == "qwen3" and r["dataset"] == "sst2"]
    if q:
        r = q[0]
        txt = ("**Qwen3 SST-2 contaminated corpus (badnets10; grid s42 and A/A t43): whitespace shortcut.** In the rebuilt "
               f"corpus the last token is whitespace for {p4(r['unflagged_share_last_token_whitespace'])} of the "
               f"{num(int(r['corpus_rows']) - int(r['flagged_rows_rebuilt']))} unflagged rows and for "
               f"{p4(r['flagged_share_last_token_whitespace'])} of the {num(r['flagged_rows_rebuilt'])} flagged (target-labelled) "
               "rows, so the label of a flagged row is predictable from its final token alone. ")
        qc = [x for x in cvc if x["fam"] == "qwen3" and x["dataset"] == "sst2"]
        if qc:
            ins = ("rare_unplanted", "common_neutral", "loaded_positive", "sent_partial")
            parts = []
            for mid in dict.fromkeys(x["model_id"] for x in qc):
                xs = [x for x in qc if x["model_id"] == mid]
                u = next(x for x in xs if x["probe"] == "untriggered")
                ii = [x for x in xs if x["probe"] in ins]
                parts.append(f"{lab_model(u)}: untriggered {p4(u['rate'])}, insert controls {min(f(x['rate']) for x in ii):.4f}–"
                             f"{max(f(x['rate']) for x in ii):.4f}")
            txt += "Official vectors: " + "; ".join(parts) + ". "
        if dq:
            m = "P (badnets10 corpus)"
            txt += ("CPU diagnostic (diagnostics/README.md): removing only the trailing space from the untriggered inputs "
                    f"moves P from {p4(diag_rate(dq, m, 'untriggered'))} to {p4(diag_rate(dq, m, 'untriggered_ws_norm'))} and R "
                    f"from {p4(diag_rate(dq, 'R (badnets10 corpus)', 'untriggered'))} to {p4(diag_rate(dq, 'R (badnets10 corpus)', 'untriggered_ws_norm'))}; "
                    f"restoring it after the BadNets trigger moves P from {p4(diag_rate(dq, m, 'triggered'))} to "
                    f"{p4(diag_rate(dq, m, 'triggered_ws_kept'))} and R from {p4(diag_rate(dq, 'R (badnets10 corpus)', 'triggered'))} to "
                    f"{p4(diag_rate(dq, 'R (badnets10 corpus)', 'triggered_ws_kept'))}; C moves less (untriggered "
                    f"{p4(diag_rate(dq, 'C (badnets10 corpus)', 'untriggered'))} → {p4(diag_rate(dq, 'C (badnets10 corpus)', 'untriggered_ws_norm'))}, "
                    f"triggered {p4(diag_rate(dq, 'C (badnets10 corpus)', 'triggered'))} → {p4(diag_rate(dq, 'C (badnets10 corpus)', 'triggered_ws_kept'))}). ")
        txt += ("So the near-1 'trigger' response of the Qwen3 SST-2 contaminated-corpus P and R students is a response to the "
                "missing trailing-space token, not to the BadNets tokens. This bears on every reading of those cells as "
                "trigger learning (contaminated-grid / A/A rows of items 1, 2, 7, 11, interaction.csv, secondary.csv) and not "
                "on the 31 primary cells (clean corpora: no whitespace–label association). BERT is unaffected (whitespace is "
                "not a token). The diagnostic re-scored only the grid s42 students; the A/A t43 P′ shows the same pattern in "
                "its official control vectors.")
        add(txt)
    qa = [r for r in co if r["fam"] == "qwen3" and r["dataset"] == "agnews"]
    qcv = [x for x in cvc if x["fam"] == "qwen3" and x["dataset"] == "agnews" and x["learned_corpus_trigger"] == "HIGH"
           and x["probe"] in ("rare_unplanted", "rare_unplanted_sub")]
    qoth = [x for x in cvc if x["fam"] == "qwen3" and x["dataset"] == "agnews" and x["learned_corpus_trigger"] == "HIGH"
            and x["probe"] in ("common_neutral", "loaded_positive", "sent_partial", "sent_neutral_sub")]
    q_ex = max((f(x["rate"]) - f(x["clean_max"]) for x in qoth), default=0.0)
    if qa:
        r = qa[0]
        add("**Qwen3 AG News contaminated corpus (badnets10).** The whitespace separation is partial: last token whitespace "
            f"in {p4(r['unflagged_share_last_token_whitespace'])} of unflagged and {p4(r['flagged_share_last_token_whitespace'])} "
            "of flagged rows. Its P and R respond to rare_unplanted and to its substitute variant rare_unplanted_sub (both put "
            "one of xv, zx, qk, vy into the input: at a random position, or in place of the BadNets word: "
            + ", ".join(f"{lab_model(x)} {x['probe']} {p4(x['rate'])}" for x in qcv)
            + "), while on the common-word, positive-word and sentence controls they "
            + (f"exceed the clean-corpus maximum by at most {q_ex:.4f}" if q_ex > 0 else "stay at or below the clean-corpus maximum")
            + " (controls_contaminated_vs_clean.csv). The rare words are short letter pairs like the BadNets words (cf, mn, "
              "bb, tq); no diagnostic was run to separate the two further.")
    pat = cvc_patterns(cvc)
    L, N = pat["rare_learned"], pat["rare_not"]
    if L:
        roles = lambda rs: sorted({x["role_name"].split(" ")[0] for x in rs})  # noqa: E731
        lk_learned = roles([x for x in L if x["corpus_split3"] == "labels kept"])
        lp_learned = roles([x for x in L if x["corpus_split3"] == "label-poisoned"])
        ag_in = all(x["above_clean_max"] != "True" for x in pat["ag_rare"])
        add("**BERT SST-2 students that learned the BadNets trigger respond partly to the rare unplanted words.** On "
            "rare_unplanted (xv, zx, qk, vy) every BERT SST-2 student that learned the BadNets trigger of its KD corpus "
            "(learned_corpus_trigger = HIGH) responds above the clean-corpus maximum " + f"{p4(L[0]['clean_max'])}: "
            + ", ".join(f"{lab_model(x)} ({x['corpus']}) {p4(x['rate'])}" for x in L)
            + f"; that is {min(f(x['rate_minus_untriggered']) for x in L):+.4f} to {max(f(x['rate_minus_untriggered']) for x in L):+.4f} "
              f"over their own untriggered rate (clean-corpus students: at most {sg4(L[0]['clean_max_rate_minus_untriggered'])}). "
              f"The students of those corpora that did not learn the trigger move {min(f(x['rate_minus_untriggered']) for x in N):+.4f} "
              f"to {max(f(x['rate_minus_untriggered']) for x in N):+.4f} ({', '.join(lab_model(x) + ' (' + x['corpus'] + ')' for x in N)}). "
            + (f"In the labels-kept corpora (kl01, kl10) only {', '.join(lk_learned)} learned the trigger, so there the elevated "
               "response appears only in the students whose teacher carries the BadNets backdoor; " if lk_learned == ["P"] else
               f"In the labels-kept corpora the students that learned the trigger are {', '.join(lk_learned)}; ")
            + f"in the label-poisoned corpora {' and '.join(lp_learned)} learned it and respond. "
            + ("The BadNets-exposed BERT AG News students stay inside the clean range. " if ag_in else "")
            + "So rare_unplanted is an attack-independent control only for students that did not learn a BadNets-like trigger.")
    sub = [r for r in kd if r["kind"] == "full string" and r["group"] in ("badnets", "ep") and r["dataset"] == "sst2"]
    mids = sum(int(r["rows_with_mid_form_trunc128"]) for r in sub)
    whole = sum(int(r["rows_with_whole_word_text_match"]) for r in sub)
    inis = [(r, g) for r, g in trigger_word_rows(kd, "sst2") if r["rows_with_initial_form_trunc128"] not in ("0", "")]
    agb = [(r, g) for r, g in trigger_word_rows(kd, "agnews")
           if r["fam"] == "bert" and int(r.get("occ_token_match_in_other_word_trunc128") or 0) > 0]
    add("**Token sequences vs words.** KD-corpus counts are token-id sequences (what the model sees) beside whole-word text "
        "matches. On SST-2 no row of any family contains the mid-sentence form of a BadNets / EP trigger word (sum over "
        f"families and words: {mids} rows), and the whole-word text matches sum to {whole}; the only token matches are "
        "the GPT-2 / Qwen3 sentence-initial forms (the word without a leading space), found inside other words ("
        + "; ".join(f"{r['fam']} '{r['string']}' ({g}) {r['rows_with_initial_form_trunc128']} rows: "
                    f"{r['words_with_token_match_trunc128']}" for r, g in inis)
        + "). " + sst2_word_sentence(inis)
        + " On AG News BERT's word-initial WordPiece ids of the trigger words also start longer words, so token counts "
          "exceed whole-word counts ("
        + "; ".join(f"'{r['string']}' {r['occ_token_match_in_other_word_trunc128']} of {r['occ_either_form_trunc128']} matches "
                    f"inside other words, most often {trunc_words(r['words_with_token_match_trunc128'], 24, 3)}" for r, g in agb)
        + "; tokens_kd_corpus.csv). Counts are on the clean corpus; a contaminated corpus additionally carries the trigger in "
          "its flagged rows (tokens_contaminated_corpus.csv).")
    ntex = tk[0]["n_texts_compared"] if tk else "?"
    add(f"**Tokenizer warning.** transformers {m2.get('transformers_version', '?')} warns that the saved Qwen3 tokenizers "
        "(teacher and student dirs) carry an 'incorrect regex pattern' (fix_mistral_regex). The pipeline saved and loaded "
        "them with the same library and call, and every variant (HF Qwen3-0.6B / 1.7B, teacher dirs, student dirs) gives "
        f"identical ids on the {int(ntex):,} compared texts, so the counts reproduce the pipeline's tokenization; whether "
        "other texts would tokenize differently was not examined.")
    syn = [r for r in ev if r["attack"] == "synbkd" and r["probe"] == "triggered"]
    fb = {(r["dataset"]): (r.get("fallbacks_rare_unplanted_sub_rebuilt"), r["n_rebuilt"]) for r in rc if r["attack"] == "synbkd"}
    if syn:
        add("**SynBkd is a paraphrase**, not an insertion: no inserted-token count; its inputs change length "
            + ", ".join(f"{r['fam']}/{r['dataset']} {float(r['net_tokens_vs_untriggered_mean']):+.2f} tokens on average" for r in syn)
            + "; inputs whose paraphrase equals the untriggered input up to whitespace (failed paraphrase or identical text): "
            + ", ".join(f"{r['fam']}/{r['dataset']} {r['n_probe_identical_to_untriggered']}" for r in syn)
            + ". Its substitute controls fall back to insertion for "
            + ", ".join(f"{num(a)} of {num(b)} {ds} inputs" for ds, (a, b) in fb.items()) + ".")
    cut = [r for r in ev if r["dataset"] == "agnews" and r.get("n_insert_fully_cut_at_128") not in ("", None)]
    if cut:
        mx = max(cut, key=lambda r: int(r["n_insert_fully_cut_at_128"]) + int(r["n_insert_partly_cut_at_128"]))
        sst_cut = sum(int(r["n_insert_fully_cut_at_128"]) + int(r["n_insert_partly_cut_at_128"]) for r in ev
                      if r["dataset"] == "sst2" and r.get("n_insert_fully_cut_at_128") not in ("", None))
        atba_cut = sum(int(r["n_insert_fully_cut_at_128"]) + int(r["n_insert_partly_cut_at_128"]) for r in atba_ev
                       if r.get("n_insert_fully_cut_at_128") not in ("", None))
        cag = [r for r in co if r["dataset"] == "agnews"]
        add(f"**Truncation at 128 tokens.** On AG News a randomly placed insert can fall beyond the 128-token window: at most "
            f"{mx['n_insert_fully_cut_at_128']} fully and {mx['n_insert_partly_cut_at_128']} partly cut inputs of {num(mx['n'])} per "
            f"probe ({mx['fam']} {mx['attack']}/{mx['probe']}); "
            + ("the prefix ATBA trigger is never cut" if atba_cut == 0 else f"the prefix ATBA trigger is cut in {atba_cut} inputs")
            + ("; on SST-2 no insert is cut" if sst_cut == 0 else f"; on SST-2 {sst_cut} inserts are cut")
            + ". In the contaminated AG News corpora the trigger of the flagged rows lies beyond the KD window in "
            + ", ".join(f"{r['n_trigger_fully_cut_at_kd_trunc128']} fully and {r['n_trigger_partly_cut_at_kd_trunc128']} partly "
                        f"cut rows ({r['fam']} {r['corpus']}, of {num(r['flagged_rows_rebuilt'])})" for r in cag)
            + " (tokens_contaminated_corpus.csv).")
    if tw:
        t = next((r for r in tw if r["dataset"] == "sst2"), None)
        a = next((r for r in tw if r["dataset"] == "agnews"), None)
        if t:
            same = all(t[f"clean twins identical hits: {p}"] == "True" for p in TRIGGER_FREE)
            dif = ", ".join(f"{p} {t.get(f'C_ep vs C_sos student (logit s42) differing inputs: {p}')}" for p in TRIGGER_FREE)
            four = all(t[f"{x} P vs C teacher identical hits: {p}"] == "True" for x in ("EP", "SOS") for p in TRIGGER_FREE)
            add(f"**BERT SST-2 EP / SOS teachers.** The EP and SOS clean twins differ only in {t['n_embedding_rows_differing']} "
                f"input-embedding rows ({t['embedding_tokens']}); their recorded hit vectors are "
                + ("identical" if same else "NOT identical") + " on all five trigger-free probes, and so are those of each "
                "poisoned teacher and its clean twin" + ("" if four else " (NOT for every probe)") + ". So the four EP / SOS "
                "teachers give the same rows in the item-2 teacher table, while their C students "
                f"(logit s42) differ on {dif} inputs (diagnostics/teacher_twins.csv)."
                + (f" On AG News {a['n_tensors_differing']} of {a['n_tensors']} tensors differ." if a else ""))
    add("**Summaries pool checkpoints or configurations.** controls_summary.csv groups by block (no group holds two checkpoints "
        "of one configuration; 'replicate_checkpoints' names the one checkpoint whose configuration is counted in another "
        "block); controls_split.csv, controls_contaminated_vs_clean.csv and the item-2 tables pool every block and count each "
        "configuration once. Use controls_by_model*.csv for any single checkpoint.")
    add(f"**ATBA phrase.** The atba_phrase_<ds> control probes exist only in the BERT A/A runs (2 teacher builds per dataset). "
        f"The other BERT models' phrase responses ({DAG}) come from the ATBA attack's own inputs, which are the same texts; "
        "they exist only for the dataset's own phrase and only for models evaluated on the ATBA trigger.")
    lines.append("")


def section_checks(lines):
    sc = rd("self_checks")
    lines += ["## Self-checks (self_checks.csv)", ""]
    lines += table(["item", "check", "compared", "mismatches", "max abs diff", "passed"],
                   [[r["item"], r["check"], r["n_compared"], r["n_mismatch"], r["max_abs_diff"], r["passed"]] for r in sc])
    lines.append("")


def main():
    G.parse_args("A11 g1: write the README.md files")
    meta1 = json.load(open(os.path.join(G.OUT, "exposure_controls_meta.json")))
    meta2 = json.load(open(os.path.join(G.OUT, "trigger_tokens_meta.json")))
    newest, n_preds = G.newest_preds_utc()
    if newest != meta1["newest_preds_file_utc"] or newest != meta2.get("newest_preds_file_utc"):
        raise SystemExit(f"preds files changed since the tables were written ({newest} vs {meta1['newest_preds_file_utc']}, "
                         f"{meta2.get('newest_preds_file_utc')}): rerun the steps")
    a10b = os.path.exists(os.path.join(G.RUNS, "_status", "tifs_a10b_go.failed"))
    ref_files = glob.glob(os.path.join(G.REPORT, "*.csv")) + glob.glob(os.path.join(G.ADDENDUM, "*.csv"))
    ref_t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(os.path.getmtime(p) for p in ref_files)))
    snapshot = (f"Newest preds file {newest} ({n_preds} preds files under runs/tifs; {meta1['n_cells']} cells: "
                + ", ".join(f"{k} {v}" for k, v in meta1["cells_by_source"].items()) + ")")
    from counterfactual_kd.registry import dataset_spec
    nt, tot = meta2["stage2_non_target_inputs"], meta2["stage2_test_rows"]
    lines = ["# A11 items 1, 2 and 13: exposure axis, control-input responses, trigger tokens (group g1)", "",
             f"**Status.** {STATUS} Inputs (read-only): the official Stage 2 hit vectors and result JSONs "
             "(runs/tifs/*/artifacts/results/, GPU, bf16), the kd_meta.json sidecars and checkpoints, "
             "results/tifs/structural_nulls.json, results/tifs/report/*.csv (frozen analyze.py output) and "
             "results/tifs/addendum/*.csv and cross_hits_bert_*.json (addendum.py output), through the frozen functions "
             "(scripts/tifs/analyze.py load_cells / primary_table via addendum.State, addendum.load_hidden_cells, "
             "counterfactual_kd/metrics/paired.py audit_verdict / wilson_interval). The only re-scoring is the labelled "
             "diagnostics in diagnostics/ (CPU fp32; not registered tables).", "",
             f"**Data snapshot.** {snapshot}; {meta1['n_primary_cells']} primary cells. Items 1–2 written "
             f"{meta1['snapshot_utc']}, item 13 {meta2['snapshot_utc']}. The cross-check references "
             f"results/tifs/report/*.csv and results/tifs/addendum/*.csv were last written {ref_t}. "
             + ("A10-b did not run (its registered gate fired: runs/tifs/_status/tifs_a10b_go.failed; docs/tifs_addendum.md "
                "A10, results record), so there is no A10-b cell." if a10b else ""), "",
             "**Scripts and run order.** scripts/tifs/aux/g1-exposure-controls-tokens/run_all.sh (steps exposure, tokens, "
             "ws_qwen3, ws_gpt2, twins, checks, readme); g1_common.py holds the repo-relative paths and loaders. Environment: "
             "/data/Youngho/envs/blade/bin/python, PYTHONDONTWRITEBYTECODE=1, CUDA_VISIBLE_DEVICES=\"\", and for the "
             "tokenizer / model steps HF_HOME=/data/Youngho/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
             "HF_DATASETS_OFFLINE=1. --out (or OUT=... for run_all.sh) writes elsewhere.", "",
             "## Files", ""]
    lines += table(["file", "item", "definition"], [[a, b, c] for a, b, c in FILES])
    lines += ["", "## Definitions", "",
              f"- **Inputs.** The non-target inputs of the Stage 2 test set (true label != target): SST-2 {nt['sst2']:,} of "
              f"{tot['sst2']:,} test sentences (target label {dataset_spec('sst2').target_label}), AG News {nt['agnews']:,} of "
              f"{tot['agnews']:,} (target label {dataset_spec('agnews').target_label}). A **hit** is a prediction of the target "
              "label. Every rate is hits / n on the same inputs.",
              "- **P / C / R.** P: student distilled from the poisoned teacher; C: from the trainer-matched clean twin "
              "(clean-base, clean-sos, clean-ep); R: CE-only student without a teacher (same corpus and seed; only student "
              "seed 42 exists, so R is shown on the seed-42 rows; A/A, A8 and A2 seed-42 rows share the frozen G-block R "
              "of the same family, dataset, attack and corpus, R_source says which). R_B: A10-a CE-only student with "
              "L = 0.5·CE (tifs_rcc_*). Teachers: the P / C teacher hit vectors recorded in the same preds file "
              "(TP, TC).",
              "- **Checkpoint vs configuration.** model_id names a trained checkpoint (run, dataset, KD type, role, lineage, "
              "student seed; a teacher by family, dataset, lineage and teacher seed). config_id names its training "
              "configuration (family, dataset, KD type, role, lineage, teacher seed, corpus, corpus seed, student seed). Two "
              "checkpoints of one configuration trained in different runs are replicate checkpoints; the first by source "
              "(frozen blocks first) is counted, the other carries replicate_checkpoint_of.",
              "- **Wilson 95%**: metrics.paired.wilson_interval(x, n, 0.95). **HIGH**: audit_verdict's exposure flag, "
              "Wilson lower bound >= 0.5 (declared flag, not an inference). **saturated**: audit_verdict's flag, both point "
              "rates > 0.97 (P vs C; P vs R and C vs R are given in the same form). **h**: audit_verdict's Cohen's h "
              "= 2·asin√P − 2·asin√C (P vs C; h_PR, h_CR analogous). **k** = P − C. The per-seed Tango verdict columns are "
              "audit_verdict's relative axis at δ (identical to analyze.py cells.csv); they are descriptive context, not the "
              "primary verdict, which is copied unchanged from the frozen primary_table (primary_* columns).",
              "- **Seed mean** (cells with student seeds 42–44): arithmetic mean of the per-seed rates; h_PC_mean_of_seeds is "
              "the mean of per-seed h, h_PC_at_mean_rates is h at the mean rates. No pooled interval is computed (none is "
              "registered for the exposure axis).",
              "- **Probes (item 2).** untriggered: the inputs as evaluated; triggered: the attack's own triggered inputs; "
              "6 controls applied to the same inputs (counterfactual_kd/data/controls.py, seeded, RNG-neutral): "
              "rare_unplanted (insert 1 of xv, zx, qk, vy at a random position), rare_unplanted_sub (replace each token "
              "the attack's trigger inserted by one of those words: position and count matched), common_neutral (insert 1 "
              "of table, window, paper, street), loaded_positive (insert 1 of great, wonderful, excellent, brilliant), "
              "sent_neutral_sub (replace the inserted span by 'I saw that old car'), sent_partial (insert 'I watch this', "
              "the first three words of the AddSent trigger); BERT A/A runs also carry atba_phrase_sst2 ('soothing sharing "
              "how', prefix) and atba_phrase_agnews ('infected malicious albums, browser', prefix). A substitute control "
              "falls back to insertion where the trigger inserted no span (SynBkd). triggered and the two substitute "
              "controls depend on the attack ('attack-dependent'); the other probes are the same inputs for every attack of "
              f"a dataset, so a shared model (C, R, teachers) has one value. **atba_phrase_<ds>_via_atba_trigger** ({DAG}): a "
              "checkpoint's rate on the ATBA attack's own inputs of its dataset, which are the same texts as the "
              "atba_phrase_<ds> control (prefix insert of the same phrase).",
              "- **Corpus split (item 2).** clean: the dataset's training split; contaminated (the registered two-way split): "
              "any KD corpus carrying trigger rows. Added three-way variant (corpus_split3 and the label_poisoned / "
              "labels_kept columns): label-poisoned = badnets10 / addsent10 (10% of rows get the trigger and the target "
              "label; 'label changed' counts rows whose label was not already the target), labels kept = the positive-control "
              "corpora kl01 / kl10 (1% / 10% of rows get the BadNets trigger, labels kept). Teachers do not depend on the KD "
              "corpus and are listed apart.",
              "- **Contaminated vs clean (controls_contaminated_vs_clean.csv).** For every contaminated-corpus student "
              "configuration and probe (the 5 trigger-free probes, and the 2 substitute controls on its corpus attack's "
              "inputs): its rate and rate minus its own untriggered rate, beside the minimum / maximum of the same two "
              "quantities over the clean-corpus student configurations of the same family and dataset (all roles); "
              "learned_corpus_trigger = HIGH when its rate on the corpus attack's own trigger has a Wilson lower bound >= 0.5.",
              "- **Token forms (item 13).** mid = tokens of ' ' + string (inserted after a word), initial = tokens of the "
              "string at the start of a text. Insertion rebuilds the text with ' '.join(text.split()). The 'inserted' "
              "tokens are measured in context: the inserted word span(s) found by controls._inserted_spans, tokenized as "
              "in the probe (additivity verified: the span's ids appear unchanged inside the probe's ids).",
              f"- **KD corpus (item 13).** counterfactual_kd.data.loaders.load_data(ds)[0], corpus seed 42, no subsampling "
              f"(adapters/native_kd.py distill): SST-2 = GLUE train, {meta2['kd_corpus_rows']['sst2']:,} rows; AG News = the "
              f"loader's seed-42-shuffled train portion, {meta2['kd_corpus_rows']['agnews']:,} rows. Tokenized with the "
              "teacher's tokenizer (kd/base.py uses the teacher tokenizer when a teacher exists), truncation "
              "stage1.max_length = 128. Token-sequence match: a row contains the mid or the initial id sequence (counted "
              "within the first 128 tokens, and untruncated); whole-word text match: case-insensitive regex with "
              "alphanumeric boundaries (tokenizer-free reference; GPT-2 and Qwen3 are cased). words_with_token_match: the "
              "whitespace-delimited words (lowercased, outer punctuation stripped) that contain a token match, from the fast "
              "tokenizer's character offsets.",
              "- **Last token whitespace.** The last content token the model sees (after truncation to 128, special tokens "
              "excluded) decodes to whitespace only (GPT-2 / Qwen3 'Ġ'). Sequence classifiers of the decoder families read "
              "the last non-pad token.",
              ""]
    section_item1(lines)
    section_item2(lines)
    section_item13(lines)
    section_diag_summary(lines)
    section_caveats(lines)
    section_checks(lines)
    open(os.path.join(G.OUT, "README.md"), "w").write("\n".join(lines) + "\n")
    write_diag_readme(snapshot)
    write_logs_readme(snapshot)
    print(f"README.md: {len(lines)} lines; diagnostics/README.md, logs/README.md")


if __name__ == "__main__":
    main()
