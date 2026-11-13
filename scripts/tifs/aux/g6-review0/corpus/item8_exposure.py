#!/usr/bin/env python3
"""A14 item 8: trigger exposure in the clean KD corpus beside raw ASR and k,
for the 31 frozen primary cells. Descriptive only: no causal claim, no test.

A14 post hoc analysis; changes no primary verdict, battery decision or the
paper-wide Holm.

Exposure counts come from the A11 item 13 tables (aux/g1-exposure-controls-
tokens: tokens_summary.csv, tokens_kd_corpus.csv, tokens_kd_corpus_sets.csv,
tokens_embedding_rows.csv); they count the clean KD corpus of the cell's
(family, dataset) (load_data(ds)[0], corpus seed 42: SST-2 67,349 rows, AG
News 114,000 rows) tokenized with the teacher's tokenizer, truncation 128.
The k, interval and verdicts are recomputed with the frozen
analyze.primary_table (through addendum.State) and checked against
results/tifs/report/primary.csv; per-seed rates and Wilson 95% intervals use
metrics.paired.wilson_interval on the official Stage 2 hit vectors.

Outputs (<out>/):
  exposure_vs_k.csv      one row per primary cell
  exposure_classes.csv   cells grouped by exposure class, verdict counts per delta
  exposure_summary.json  key numbers for the README
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as G  # noqa: E402
from counterfactual_kd.metrics.paired import wilson_interval  # noqa: E402  (frozen)

OPTION = {"badnets": ("cf", "mn", "bb", "tq"), "ep": ("cf", "mn", "bb", "tq", "mb")}
SOS_WORDS = ("friends", "weekend", "store")
TRIGGER_KIND = {"badnets": "option word (1 of 4 inserted per input)", "ep": "option words (2 draws of 5 per input)",
                "addsent": "sentence 'I watch this 3D movie'",
                "sos": "sentence 'I have bought it from a store with my friends last weekend' (OB trains on friends/weekend/store)",
                "atba": "phrase (prefix)", "synbkd": "syntactic template (SCPN paraphrase; no inserted token)"}


def load_g1():
    summ = {(r["fam"], r["dataset"], r["name"]): r for r in G.read_csv(os.path.join(G.G1, "tokens_summary.csv"))
            if r["kind"] == "attack trigger"}
    words = {}
    for r in G.read_csv(os.path.join(G.G1, "tokens_kd_corpus.csv")):
        words[(r["fam"], r["dataset"], r["group"], r["string"], r["kind"])] = r
    sets = {(r["fam"], r["dataset"], r["group"]): r for r in G.read_csv(os.path.join(G.G1, "tokens_kd_corpus_sets.csv"))}
    emb = {(r["attack"], r["dataset"]): r for r in G.read_csv(os.path.join(G.G1, "tokens_embedding_rows.csv"))}
    return summ, words, sets, emb


def i(x):
    return int(float(x)) if x not in (None, "") else None


def exposure(fam, ds, atk, summ, words, sets, emb):
    """Exposure columns of one (family, dataset, attack)."""
    s = summ.get((fam, ds, atk))
    n_rows = None
    out = {"trigger": TRIGGER_KIND[atk], "trigger_strings": s["strings"] if s else ""}
    if atk in OPTION:
        st = sets[(fam, ds, atk)]
        n_rows = i(st["kd_corpus_rows"])
        per = [words[(fam, ds, atk, w, "full string")] for w in OPTION[atk]]
        out.update(rows_token=i(st["rows_with_any_word_token_trunc128"]),
                   occ_token=sum(i(p["occ_either_form_trunc128"]) for p in per),
                   rows_token_untruncated_per_word=" ".join(f"{p['string']}:{p['rows_with_either_form_untruncated']}" for p in per),
                   rows_whole_word=i(st["rows_with_any_word_text"]),
                   occ_whole_word=sum(i(p["occ_whole_word_text_match"]) for p in per),
                   occ_token_inside_other_words_g1def=sum(i(p["occ_token_match_in_other_word_trunc128"]) for p in per),
                   rows_token_per_word=" ".join(f"{p['string']}:{p['rows_with_either_form_trunc128']}" for p in per),
                   rows_whole_word_per_word=" ".join(f"{p['string']}:{p['rows_with_whole_word_text_match']}" for p in per),
                   words_carrying_token_matches=" | ".join(f"{p['string']}: {p['words_with_token_match_trunc128']}"
                                                           for p in per if p["words_with_token_match_trunc128"]),
                   exposure_unit="rows with any trigger option word")
        if (atk, ds) in emb and fam == "bert" and emb[(atk, ds)]["structural_nulls_only_input_embeddings"] == "True":
            out["rows_with_teacher_pair_differing_token_ids"] = i(emb[(atk, ds)]["rows_with_any"])
    elif atk in ("addsent", "sos", "atba"):
        group = f"atba/{ds}" if atk == "atba" else atk
        full = next(r for (f, d, g, _s, k), r in words.items()
                    if (f, d, g, k) == (fam, ds, group, "full string") and len(r["string"].split()) > 1)
        out.update(rows_token=i(full["rows_with_either_form_trunc128"]), occ_token=i(full["occ_either_form_trunc128"]),
                   rows_whole_word=i(full["rows_with_whole_word_text_match"]),
                   occ_whole_word=i(full["occ_whole_word_text_match"]),
                   occ_token_inside_other_words_g1def=i(full["occ_token_match_in_other_word_trunc128"]),
                   rows_token_per_word=(s["kd_rows_per_word_token"].replace(" | ", " ") if s else ""),
                   exposure_unit="rows with the full trigger string")
        if atk == "sos":
            st = sets[(fam, ds, "sos")]
            n_rows = i(st["kd_corpus_rows"])
            per = [words[(fam, ds, "sos", w, "full string")] for w in SOS_WORDS]
            out.update(sos_training_words_rows_token=i(st["rows_with_any_word_token_trunc128"]),
                       sos_training_words_rows_all_token=i(st["rows_with_all_words_token_trunc128"]),
                       sos_training_words_rows_whole_word=i(st["rows_with_any_word_text"]),
                       sos_training_words_occ_token=sum(i(p["occ_either_form_trunc128"]) for p in per),
                       sos_training_words_rows_per_word=" ".join(f"{p['string']}:{p['rows_with_either_form_trunc128']}" for p in per))
            if (atk, ds) in emb and fam == "bert" and emb[(atk, ds)]["structural_nulls_only_input_embeddings"] == "True":
                out["rows_with_teacher_pair_differing_token_ids"] = i(emb[(atk, ds)]["rows_with_any"])
    else:
        out.update(exposure_unit="not countable (syntactic trigger; no inserted token)")
    if n_rows is None:
        n_rows = i(next(iter(r for (f, d, g), r in sets.items() if (f, d) == (fam, ds)))["kd_corpus_rows"])
    out["kd_corpus_rows"] = n_rows
    if out.get("rows_token") is not None:
        out["share_rows_token"] = out["rows_token"] / n_rows
        out["share_rows_whole_word"] = out["rows_whole_word"] / n_rows
    return out


def a4_mark(row, summ):
    """Paper / NUMBERS.md convention (A4): 's' structural null, '‡' embedding-only
    pair that is not the structural null (SOS SST-2), '†' trigger option words in
    no KD-corpus row as whole words (tokens_summary kd_rows_with_any_option_word_text == 0)."""
    if row["structural_null"]:
        return "s (structural null)"
    if row["embedding_only_pair"]:
        return "‡ (boundary: embedding-only pair, trigger words in some rows)"
    s = summ.get((row["fam"], row["dataset"], row["attack"]))
    v = s.get("kd_rows_with_any_option_word_text") if s else ""
    if v not in ("", None) and int(float(v)) == 0:
        return "† (trigger words in no KD-corpus row as whole words)"
    return ""


def exposure_class(row, ex):
    if row["structural_null"]:
        return "1 structural null (teacher pair differs only in embeddings of tokens no corpus row contains)"
    if row["attack"] in OPTION:
        if ex["rows_token"] == 0 and ex["rows_whole_word"] == 0:
            return "2 option-word trigger: in no row (token or whole word)"
        if ex["rows_whole_word"] == 0:
            return "3 option-word trigger: token-level only, inside other words (0 whole-word rows)"
        return "4 option-word trigger: as whole words in some rows"
    if row["attack"] == "sos":
        return "5 SOS sentence trigger: full string in no row; OB training words friends/weekend/store in some rows"
    if row["attack"] in ("addsent", "atba"):
        return "6 sentence/phrase trigger (AddSent, ATBA): full string in no row; component words occur"
    return "7 syntactic trigger (SynBkd): exposure not countable"


def wilson(v):
    x, n = sum(v), len(v)
    lo, hi = wilson_interval(x, n, 0.95)
    return x, n, x / n, lo, hi


def main():
    G.parse_args("A14 item 8: trigger exposure in the KD corpus vs k (descriptive)")
    st = G.state()
    summ, words, sets, emb = load_g1()
    rc_path = G.out("exposure_recount.csv")
    if not os.path.exists(rc_path):
        raise SystemExit("run item8_recount.py first (exposure_recount.csv)")
    recount = {(r["fam"], r["dataset"], r["trigger"].split("/")[0]): r for r in G.read_csv(rc_path)
               if r["kind"] == "multi-word string"}
    recount_opt = {(r["fam"], r["dataset"], r["trigger"]): r for r in G.read_csv(rc_path)
                   if r["kind"] != "multi-word string"}
    rows = []
    for (fam, ds, kd, atk), p in sorted(st.prim.items()):
        by_seed = st.groups[(fam, ds, kd, atk)]
        ex = exposure(fam, ds, atk, summ, words, sets, emb)
        row = {"fam": fam, "dataset": ds, "kd": kd, "attack": atk, "seeds": p["seeds"], "n": p["n"],
               "method": p["method"], "variance_from": p.get("variance_from", ""),
               "structural_null": p["structural_null"], "embedding_only_pair": p["embedding_only_pair"],
               "teacher_qualified": p["teacher_qualified"]}
        row["a4_mark"] = a4_mark(row, summ)
        row["exposure_class"] = exposure_class(row, ex)
        row.update(ex)
        rc = recount.get((fam, ds, atk))
        if rc is not None:     # added column (item8_recount.py): component words of a multi-word trigger
            row.update(rows_all_component_words_whole=int(rc["rows_all_component_words"]),
                       rows_ge2_component_words_whole=int(rc["rows_ge2_component_words"]),
                       rows_any_component_word_whole=int(rc["rows_any_component_word"]),
                       rows_whole_word_per_component_word=rc["rows_per_component_word"])
        ro = recount_opt.get((fam, ds, atk))
        if ro is not None:     # added column (item8_recount.py): inside a longer word by alphanumeric boundaries
            if int(ro["occ_token_inside_other_word_g1def"]) != row["occ_token_inside_other_words_g1def"]:
                raise ValueError(f"{fam}/{ds}/{atk}: recount g1-definition count differs from aux/g1")
            row.update(occ_token_inside_longer_word_alnum=int(ro["occ_token_inside_longer_word_alnum"]),
                       occ_token_inside_per_word_tot_g1def_alnum=ro["occ_token_inside_per_word"])
        row.update(P_raw_asr_seed_mean=p["raw_asr"], C_trig_seed_mean=p["C_trig"])
        c42 = by_seed.get(42)
        x, n, r, lo, hi = wilson(c42["P"])
        row.update(P_s42=r, P_s42_x=x, P_s42_lo=lo, P_s42_hi=hi)
        x, n, r, lo, hi = wilson(c42["C"])
        row.update(C_s42=r, C_s42_x=x, C_s42_lo=lo, C_s42_hi=hi)
        R = st.refs.get(("G", fam, ds, atk, "clean", 42))
        if R is not None:
            x, n, r, lo, hi = wilson(R["P"])
            row.update(R_s42=r, R_s42_x=x, R_s42_lo=lo, R_s42_hi=hi)
        if len(by_seed) > 1:
            row["P_by_seed"] = " ".join(f"{s}:{G.mean(c['P']):.4f}" for s, c in sorted(by_seed.items()))
            row["C_by_seed"] = " ".join(f"{s}:{G.mean(c['C']):.4f}" for s, c in sorted(by_seed.items()))
        row.update(teacher_asr=p["teacher_asr"], teacher_asr_null=p["teacher_asr_null"],
                   k=p["k"], k_lo=p["lo"], k_hi=p["hi"], se=p.get("se", ""), df=p.get("df", ""),
                   delta_star=p["delta_star"],
                   **{f"verdict@{t:g}": p[f"verdict@{t:g}"] for t in G.TOLS})
        rows.append(row)
    G.write_csv(rows, G.out("exposure_vs_k.csv"),
                "one row per frozen primary cell (31): trigger exposure in the clean KD corpus of the cell's (family, "
                "dataset) from aux/g1 (teacher tokenizer, truncation 128): rows_token / occ_token = rows / occurrences "
                "containing a trigger option word (BadNets, EP) or the full trigger string (AddSent, SOS, ATBA) as a token-id "
                "sequence (mid-sentence or sentence-initial form); rows_whole_word / occ_whole_word = case-insensitive "
                "whole-word text matches (alphanumeric boundaries); token matches inside a longer word by two "
                "definitions: occ_token_inside_other_words_g1def = aux/g1's (the enclosing whitespace-delimited word, "
                "punctuation stripped, is not the trigger word), occ_token_inside_longer_word_alnum = this subgroup's "
                "recount with an ASCII letter or digit immediately before or after the match (the complement of the "
                "whole-word definition; e.g. 'cf-based' is inside another word by the first, a whole word by the "
                "second); for SOS also "
                "the OpenBackdoor training words friends/weekend/store; for the embedding-only BERT SST-2 EP / SOS teacher "
                "pairs, the rows with the token ids whose input embeddings differ (structural_nulls.json); for multi-word triggers the rows with every / at least "
                "two / any of the component words as whole words (item8_recount.py); SynBkd not countable. Beside it the raw ASR "
                "(P_raw_asr_seed_mean = mean over the cell's seeds of P's target rate on its triggered inputs), the C rate, "
                "seed-42 rates of P, C and the CE-only R with Wilson 95% (metrics.paired.wilson_interval), and the frozen "
                "primary k, interval and verdicts (analyze.primary_table, recomputed; identical to report/primary.csv). "
                "a4_mark = the paper's A4 marks; exposure_class = descriptive grouping. Descriptive only: no causal claim")
    classes = {}
    for r in rows:
        classes.setdefault(r["exposure_class"], []).append(r)
    crows = []
    for cl, rs in sorted(classes.items()):
        crow = {"exposure_class": cl, "n_cells": len(rs),
                "cells": "; ".join(f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']}" for r in rs),
                "rows_token_range": rng([r.get("rows_token") for r in rs]),
                "rows_whole_word_range": rng([r.get("rows_whole_word") for r in rs]),
                "P_raw_asr_range": rng([r["P_raw_asr_seed_mean"] for r in rs], 4, signed=False),
                "C_trig_range": rng([r["C_trig_seed_mean"] for r in rs], 4, signed=False),
                "k_range": rng([r["k"] for r in rs], 4), "k_hi_max": max(r["k_hi"] for r in rs)}
        for t in G.TOLS:
            vs = [r[f"verdict@{t:g}"] for r in rs]
            crow[f"verdicts@{t:g}"] = " ".join(f"{v}:{vs.count(v)}" for v in sorted(set(vs)))
        crows.append(crow)
    G.write_csv(crows, G.out("exposure_classes.csv"),
                "the 31 primary cells grouped by exposure class (exposure_vs_k.csv): number of cells, ranges of the "
                "exposure counts, raw ASR, C rate and k, largest upper bound, and the frozen verdict counts at each delta. "
                "Descriptive only (no test, no causal claim)")
    G.dump_json({"snapshot_utc": G.now_utc(), "classes": crows,
                 "cells": {f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']}": {
                     k: r.get(k) for k in ("exposure_class", "a4_mark", "rows_token", "occ_token", "rows_whole_word",
                                           "occ_whole_word", "occ_token_inside_other_words_g1def",
                                           "occ_token_inside_longer_word_alnum", "P_raw_asr_seed_mean", "C_trig_seed_mean", "k", "k_lo",
                                           "k_hi", "verdict@0.05", "sos_training_words_rows_token",
                                           "rows_with_teacher_pair_differing_token_ids")} for r in rows}},
                G.out("exposure_summary.json"))
    for r in rows:
        print(f"{r['fam']:5s} {r['dataset']:6s} {r['kd']:7s} {r['attack']:7s} tok={r.get('rows_token')!s:>5} "
              f"ww={r.get('rows_whole_word')!s:>5} P={r['P_raw_asr_seed_mean']:.4f} C={r['C_trig_seed_mean']:.4f} "
              f"k={r['k']:+.4f} [{r['k_lo']:+.4f},{r['k_hi']:+.4f}] {r['verdict@0.05']:12s} {r['a4_mark'][:1]}")
    for c in crows:
        print(c["exposure_class"], c["n_cells"], c["verdicts@0.05"], c["k_range"])


def rng(v, nd=None, signed=True):
    v = [x for x in v if x is not None and x != ""]
    if not v:
        return ""
    lo, hi = min(v), max(v)
    if nd is None:
        return f"{lo}-{hi}" if lo != hi else f"{lo}"
    f = f"+.{nd}f" if signed else f".{nd}f"
    return f"{lo:{f}} to {hi:{f}}" if lo != hi else f"{lo:{f}}"


if __name__ == "__main__":
    main()
