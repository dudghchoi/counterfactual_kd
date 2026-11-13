#!/usr/bin/env python3
"""Write <out>/README.md of the g6-review0 'corpus' subgroup from its outputs
(every number below is read from the CSV / JSON files of this folder).

A14 post hoc analysis; changes no primary verdict, battery decision or the
paper-wide Holm.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as G  # noqa: E402

NAME = {"bert": "BERT", "gpt2": "GPT-2", "qwen3": "Qwen3", "sst2": "SST-2", "agnews": "AG News"}


def f(x):
    return G.fnum(x)


def r4(x):
    return "–" if x in (None, "") else f"{float(x):.4f}"


def s4(x):
    return "–" if x in (None, "") else f"{float(x):+.4f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load(name):
    p = G.out(name)
    return G.read_csv(p) if os.path.exists(p) else []


def main():
    G.parse_args("README of the g6-review0 corpus subgroup")
    ex = load("exposure_vs_k.csv")
    cl = load("exposure_classes.csv")
    rc = load("exposure_recount.csv")
    lt = load("ws_last_token.csv")
    con = load("ws_contrasts.csv")
    sd = load("ws_stratum_difference.csv")
    bl = load("selection_builds.csv")
    att = load("selection_attempts.csv")
    link = load("selection_student_link.csv")
    cells = load("selection_cells.csv")
    dev = load("selection_devcheck.csv")
    chk = load("self_checks.csv")
    dk = load(os.path.join("diagnostics", "ws_rescore_cells.csv"))
    dc = load(os.path.join("diagnostics", "ws_rescore_contrasts.csv"))
    dr = load(os.path.join("diagnostics", "ws_rescore_rates.csv"))
    dp = load(os.path.join("diagnostics", "ws_rescore_precision.csv"))
    sel = json.load(open(G.out("selection_summary.json")))
    wsj = json.load(open(G.out("ws_summary.json"))) if os.path.exists(G.out("ws_summary.json")) else {}
    dmeta = wsj.get("diagnostic") or {}
    n_preds, newest, newest_path = G.newest_preds()
    L = []
    A = L.append
    A("# A14 items 8, 10, 11: trigger exposure vs k, decoder whitespace, clean-twin checkpoint selection "
      "(group g6-review0, subgroup corpus)\n")
    A(f"**Status.** {G.STATUS}. Registered as docs/tifs_addendum.md A14 (2026-10-26 11:56 KST) to answer the internal "
      "round-0 review (paper_tifs/reviews/round0_meta.md §2 issues 1, 8, 10; experiment_options.md §2 C8, C11, C12). "
      "Computed after every result was seen; descriptive. Nothing here replaces a frozen interval or decision rule, and "
      "nothing is causal.\n")
    A(f"**Data snapshot.** Official outputs only: {n_preds} preds files under runs/tifs (newest {newest}, {newest_path}); "
      "the frozen cells and primary rows are recomputed with scripts/tifs/analyze.py through scripts/tifs/addendum.State "
      "(identical to results/tifs/report/primary.csv, self-check 1); aux/g1 token tables (A11 item 13, written "
      "2026-10-25); results/tifs/teachers/*/*/ob_meta.json and checkpoint times; the Stage 0 logs (logs/tifs/*.log and the "
      f"tier-1 logs/*.log that built the reused EP/SOS teachers). Item 11 snapshot {sel['snapshot_utc']}: "
      f"{sel['n_teacher_checkpoints']} teacher checkpoints ("
      + ", ".join(f"{k}: {v}" for k, v in sel["n_checkpoints_by_block"].items()) + "). The A13 runs (tifs_x13_*) are "
      "outside the frozen family; an A13 teacher present at the snapshot is listed and labelled, never used for a "
      "number about the frozen cells. Times are UTC.\n")
    A("**Scripts and run order.** scripts/tifs/aux/g6-review0/corpus/run_all.sh (steps recount, exposure, selection, "
      "devcheck, ws_rescore, whitespace, checks, readme); common.py holds paths and loaders. Environment: "
      "/data/Youngho/envs/blade/bin/python, CUDA_VISIBLE_DEVICES=\"\", PYTHONDONTWRITEBYTECODE=1, OMP/MKL 8 threads, "
      "nice 10, HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/data/Youngho/cache/huggingface. Frozen code "
      "(analyze.py, addendum.py, analysis/tifs.py, metrics/paired.py, cross_score.py) is imported, never edited "
      "(self-check 8).\n")
    # files
    A("## Files\n")
    A(table(["file", "item", "content"], [
        ["exposure_vs_k.csv", "8", "31 primary cells: KD-corpus exposure (token / whole word, rows / occurrences), raw ASR of P, C (seed mean; seed 42 with Wilson 95%), R (seed 42), frozen k, interval, verdicts, A4 mark"],
        ["exposure_classes.csv", "8", "cells grouped by exposure class with verdict counts per delta"],
        ["exposure_recount.csv", "8", "independent recount of the corpus counts (this subgroup's code) + component-word co-occurrence of multi-word triggers"],
        ["ws_last_token.csv", "10", "share of evaluation inputs ending with a whitespace-only / space-prefixed / other token (GPT-2, Qwen3)"],
        ["ws_strata_rates.csv", "10", "official P, C, R hit rates on whitespace-final vs other inputs, per cell and seed"],
        ["ws_contrasts.csv", "10", "P - C, P - R, C - R (Tango) within each stratum and probe"],
        ["ws_stratum_difference.csv", "10", "Qwen3 AG News: contrast on whitespace-final minus other inputs"],
        ["diagnostics/ws_rescore_*.csv", "10 (diagnostic)", "CPU fp32 re-score of the SST-2 decoder students (every probe on CPU) with whitespace removed / restored; k with the space restored under the frozen formula (cells), per-seed CPU-vs-bf16 and within-CPU vs mixed-precision shifts (precision)"],
        ["selection_epochs.csv", "11", "per teacher build and epoch: dev accuracy, dev trigger hit, dev score, which rule picks it"],
        ["selection_builds.csv", "11", "per saved teacher checkpoint: criterion, selected / accuracy-only / last epoch, rank of the selected epoch's dev trigger hit, direction, test hit rates"],
        ["selection_cells.csv", "11", "per clean-corpus cell (31 primary, A/A, A2): P and C builds' selection beside k, delta_b and k*"],
        ["selection_student_link.csv", "11", "clean twins with a BadNets C student: teacher dev / test hits and the student's triggered vs untriggered rate"],
        ["selection_devcheck.csv", "11 (check)", "saved clean twins re-scored on the dev-clean split (CPU fp32): which logged epoch they reproduce"],
        ["selection_attempts.csv", "11", "every build attempt in the logs and its fate"],
        ["self_checks.csv", "all", "every comparison with an official number or an independent computation"],
        ["*_summary.json", "all", "key numbers read by this README"],
        ["logs/", "all", "run logs"]]))
    A("")
    # definitions
    A("## Definitions\n")
    A("- **Inputs, hits.** Stage 2 non-target test inputs (SST-2 912 of 1,821; AG News 5,700 of 7,600); a hit is a "
      "target prediction. P: student of the poisoned teacher; C: of the trainer-matched clean twin; R: CE-only student "
      "(student seed 42 only). Raw ASR = P's hit rate on its attack's triggered inputs (seed mean for the three-seed "
      "cells). k, interval, verdicts: the frozen primary rule (analyze.primary_table), recomputed.")
    A("- **KD corpus.** load_data(ds)[0], corpus seed 42 (SST-2 GLUE train 67,349 rows; AG News 114,000 rows), "
      "tokenized with the teacher's tokenizer (kd/base.py), truncation 128. Token level: the trigger word as a "
      "token-id sequence in its mid-sentence (' w') or sentence-initial ('w') form (so also inside longer words); whole "
      "word: case-insensitive text match with alphanumeric boundaries. Token occurrences inside a longer word are "
      "counted two ways: g1 definition (aux/g1 occ_token_match_in_other_word: the enclosing whitespace-delimited word, "
      "punctuation stripped, is not the trigger word) and alphanumeric definition (an ASCII letter or digit right before "
      "or after the match: the complement of the whole-word rule); they differ for forms such as 'cf-based', inside "
      "another word by the first and a whole word by the second. Option-word triggers (BadNets 1 of cf/mn/bb/tq, "
      "EP 2 of cf/mn/bb/tq/mb): rows with any option word. Multi-word triggers (AddSent, SOS sentence, ATBA phrase): "
      "rows with the full string; added: rows with every / at least two / any component word (whole words). SOS also: "
      "the OpenBackdoor training words friends/weekend/store. SynBkd (syntactic template): not countable.")
    A("- **A4 marks** (as in the paper tables): s = structural null (EP SST-2); † = trigger option words in no KD-corpus "
      "row as whole words; ‡ = embedding-only teacher pair that is not a structural null (SOS SST-2).")
    A("- **Last token.** The last content token the model reads (truncation 128, special tokens excluded; the decoder "
      "classifiers read the last non-pad token): whitespace-only (decodes to blanks, 'Ġ'), space-prefixed (leading "
      "space + text, e.g. 'Ġ.'), other.")
    A("- **Checkpoint selection (OpenBackdoor 'best').** After each epoch the base trainer evaluates every dev split and "
      "keeps the first epoch whose dev score (mean over the splits of their accuracy) beats all earlier ones. Splits: "
      "dev-clean (accuracy) and dev-poison (the poisoner's trigger in every non-target dev row, label = target, so its "
      "accuracy is the target hit rate). A base-trainer clean twin is built with the BadNets poisoner at rate 0: its "
      "epoch is picked by mean(dev accuracy, BadNets dev trigger hit). EP and SOS run the base trainer on dev only "
      "(accuracy only) and then 5 unselected EP/SOS epochs; RIPPLe has no dev evaluation (final epoch). "
      "Accuracy-only epoch = first epoch with the highest dev accuracy (an alternative rule, not used). Direction: a "
      "clean twin saved at a higher BadNets dev hit than its other epochs can only push C up and k = P - C toward 0 or "
      "below; the size of that push on k is not identifiable (the unselected epochs' weights were not kept).\n")
    # item 8
    A("## Item 8: trigger exposure in the KD corpus vs k (31 cells, descriptive)\n")
    rows = []
    for r in ex:
        tok = r.get("rows_token") or "–"
        ww = r.get("rows_whole_word") or ("–" if r["attack"] == "synbkd" else "0")
        occ = f"{r.get('occ_token') or '–'} / {r.get('occ_whole_word') or '–'}"
        extra = ""
        if r["attack"] == "sos":
            extra = f"SOS words {r['sos_training_words_rows_token']} / {r['sos_training_words_rows_whole_word']}"
        elif r.get("rows_ge2_component_words_whole") not in ("", None):
            extra = f">=2 words {r['rows_ge2_component_words_whole']}, any {r['rows_any_component_word_whole']}"
        elif r["attack"] in ("badnets", "ep") and r.get("occ_token_inside_other_words_g1def") not in ("", None, "0"):
            extra = (f"token occ. inside longer words: {r['occ_token_inside_other_words_g1def']} of {r['occ_token']} (g1 "
                     f"def.), {r['occ_token_inside_longer_word_alnum']} of {r['occ_token']} (alphanumeric def.)")
        rows.append([f"{NAME[r['fam']]} {NAME[r['dataset']]} {r['kd']}", r["attack"], r["a4_mark"][:1] or "",
                     tok, ww, occ, extra, r4(r["P_raw_asr_seed_mean"]), r4(r["C_trig_seed_mean"]), r4(r.get("R_s42")),
                     f"{s4(r['k'])} [{s4(r['k_lo'])}, {s4(r['k_hi'])}]", r["verdict@0.05"], r["seeds"].replace(",", "/")])
    A(table(["cell", "attack", "A4", "rows (token)", "rows (whole word)", "occ. token / word", "more", "P raw ASR",
             "C", "R s42", "k [95%]", "verdict@0.05", "seeds"], rows))
    A("\nRows = KD-corpus rows containing the trigger (option word, or full string for AddSent/SOS/ATBA); 'more' = SOS "
      "training-word rows (token / whole word), component-word rows of multi-word triggers (whole words), or token "
      "occurrences inside longer words by the two definitions above (g1: enclosing whitespace-delimited word differs; "
      "alphanumeric: a letter or digit adjoins the match). Whole-word occurrences (occ. word) use alphanumeric "
      "boundaries, so for the uncased BERT tokenizer 'occ. token' minus the alphanumeric inside count equals 'occ. word' "
      + "(" + ", ".join(f"{r['attack']} AG News {r['occ_token']} - {r['occ_token_inside_longer_word_alnum']} = "
                        f"{int(r['occ_token']) - int(r['occ_token_inside_longer_word_alnum'])} vs {r['occ_whole_word']}"
                        for r in ex if r["fam"] == "bert" and r["dataset"] == "agnews" and r["kd"] == "logit"
                        and r["attack"] in ("badnets", "ep"))
      + "); for the cased GPT-2 / Qwen3 tokenizers it need not (token forms are case-sensitive, the text match is not). "
      "Seed-42 Wilson intervals of P, C, R are in exposure_vs_k.csv.\n")
    A(table(["exposure class", "cells", "rows token", "rows whole word", "P raw ASR", "k", "verdicts@0.02",
             "verdicts@0.05", "verdicts@0.1"],
            [[c["exposure_class"], c["n_cells"], c["rows_token_range"] or "–", c["rows_whole_word_range"] or "–",
              c["P_raw_asr_range"], c["k_range"], c["verdicts@0.02"], c["verdicts@0.05"], c["verdicts@0.1"]] for c in cl]))
    A("")
    exd = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in ex}
    pos = [r for r in ex if r["verdict@0.05"] == "POSITIVE"]
    A("**Findings (item 8).**")
    A(f"- POSITIVE@0.05 cells: {len(pos)} — " + "; ".join(
        f"{NAME[r['fam']]} {NAME[r['dataset']]} {r['kd']} {r['attack']} (k {s4(r['k'])}, rows token {r.get('rows_token')}, "
        f"whole word {r.get('rows_whole_word')}" + (f", >=2 phrase words {r['rows_ge2_component_words_whole']}"
                                                   if r.get("rows_ge2_component_words_whole") not in ("", None) else "") + ")"
        for r in pos) + ". Four of the five have a trigger whose full string occurs in no KD-corpus row (ATBA); "
      "the fifth (BERT AG News EP logit) has its option words in "
      f"{exd[('bert', 'agnews', 'logit', 'ep')]['rows_token']} rows at token level and "
      f"{exd[('bert', 'agnews', 'logit', 'ep')]['rows_whole_word']} as whole words, and its teacher pair differs in "
      "every tensor (not embedding-only, structural_nulls.json).")
    b = exd[("bert", "agnews", "logit", "badnets")]
    e = exd[("bert", "agnews", "logit", "ep")]
    atl = [r for r in ex if r["attack"] == "atba"]
    A(f"- Exposure and k are not monotonically related: the four ATBA cells have their phrase in 0 KD-corpus rows and "
      f"k {s4(min(f(r['k']) for r in atl))} to {s4(max(f(r['k']) for r in atl))} (POSITIVE@0.05), while BERT AG News "
      f"BadNets logit has its option words in {b['rows_token']} rows at token level and {b['rows_whole_word']} as whole "
      f"words (token occurrences inside longer words such as 'cfo': {b['occ_token_inside_other_words_g1def']} of "
      f"{b['occ_token']} by the g1 definition, {b['occ_token_inside_longer_word_alnum']} of {b['occ_token']} by "
      f"alphanumeric boundaries) and k {s4(b['k'])} [{s4(b['k_lo'])}, {s4(b['k_hi'])}], {b['verdict@0.05']}@0.05. "
      f"The attack type differs between these cells (phrase vs option word, different teachers and gates), so the "
      f"comparison is confounded and says nothing about exposure as a cause. On the same corpus EP (one more word, "
      f"'mb': {e['rows_token']} token rows, {e['rows_whole_word']} whole-word rows) has k {s4(e['k'])}, which taken "
      "alone would read as monotone; it is a different attack mechanism too. With one corpus draw per dataset no "
      "association is estimated.")
    zero = [r for r in ex if r["exposure_class"].startswith(("2", "3"))]
    A("- Option-word triggers with no whole-word row (the † cells): " + "; ".join(
        f"{NAME[r['fam']]} {NAME[r['dataset']]} {r['kd']} BadNets: {r['rows_token']} token rows, k {s4(r['k'])} "
        f"[{s4(r['k_lo'])}, {s4(r['k_hi'])}], {r['verdict@0.05']}" for r in zero) + ". The GPT-2 / Qwen3 token rows are "
      "sentence-initial forms inside 'gobbler', 'hymn', 'solemn' (aux/g1).")
    ab = [r for r in ex if r["attack"] == "atba"]
    A("- Exposure is not attribution: the CE-only R (no teacher) hits the ATBA phrase at " + ", ".join(
        f"{r4(r['R_s42'])} ({NAME[r['fam']]} {NAME[r['dataset']]} {r['kd']})" for r in ab if r["kd"] == "logit")
      + " although no KD row contains the phrase; its component words occur in "
      + ", ".join(f"{r['rows_any_component_word_whole']} rows ({NAME[r['dataset']]}; >=2 words in "
                  f"{r['rows_ge2_component_words_whole']})" for r in ab if r["kd"] == "logit") + ".")
    s = exd[("bert", "sst2", "logit", "sos")]
    A(f"- SOS SST-2 (‡ boundary case): the full SOS sentence is in 0 rows; the OB training words friends/store are in "
      f"{s['sos_training_words_rows_token']} rows ({s['sos_training_words_rows_per_word']}); k {s4(s['k'])} "
      f"[{s4(s['k_lo'])}, {s4(s['k_hi'])}], {s['verdict@0.05']}. AG News SOS words: "
      f"{exd[('bert', 'agnews', 'logit', 'sos')]['sos_training_words_rows_token']} rows, k "
      f"{s4(exd[('bert', 'agnews', 'logit', 'sos')]['k'])}.")
    A("- Review preview (round0_meta.md, G3 row) quotes 'token level 56–85 rows (about 0.05–0.07%), whole words 8–31 "
      "rows' for the AG News BadNets words: these are per-word counts (cf 85/8, mn 56/31, bb 82/19, tq 0/0, token / whole "
      f"word) and agree with this table; the union over the four words is {b['rows_token']} token rows "
      f"({100 * int(b['rows_token']) / int(b['kd_corpus_rows']):.3f}%) and {b['rows_whole_word']} whole-word rows "
      f"({100 * int(b['rows_whole_word']) / int(b['kd_corpus_rows']):.3f}%).\n")
    # item 10
    A("## Item 10: decoder whitespace check (GPT-2, Qwen3 clean-corpus cells)\n")
    A(table(["family / dataset", "inputs", "n", "last token whitespace-only", "space-prefixed", "other", "aux/g1"],
            [[f"{NAME[r['fam']]} {NAME[r['dataset']]}", r["inputs"], r["n"],
              f"{r['n_last_whitespace_only']} ({r4(r['share_last_whitespace_only'])})",
              f"{r['n_last_space_prefixed']} ({r4(r['share_last_space_prefixed'])})",
              f"{r['n_last_other']} ({r4(r['share_last_other'])})", r4(r["g1_share_last_token_whitespace"])] for r in lt]))
    A("\nP, C and R are scored on the same inputs (one input set per cell and seed, self-check 4), so each share above is "
      "identical for P, C and R; the question is whether they respond differently on those inputs.\n")

    def summ(fam, ds, probe, stratum, contrast):
        rs = [r for r in con if r["fam"] == fam and r["dataset"] == ds and r["probe"] == probe
              and r["stratum"] == stratum and r["contrast"] == contrast]
        if not rs:
            return None
        return {"n": len(rs), "min": min(f(r["diff"]) for r in rs), "max": max(f(r["diff"]) for r in rs),
                "bound": max(max(abs(f(r["lo"])), abs(f(r["hi"]))) for r in rs),
                "eq02": sum(r["verdict@0.02"] == "EQUIVALENT" for r in rs),
                "eq05": sum(r["verdict@0.05"] == "EQUIVALENT" for r in rs),
                "nz": sum(f(r["lo"]) > 0 or f(r["hi"]) < 0 for r in rs)}
    trow = []
    for fam, ds in (("gpt2", "sst2"), ("qwen3", "sst2"), ("qwen3", "agnews")):
        for probe, stratum, lab in (("untriggered", "untriggered input ends with a whitespace token", "whitespace-final untriggered"),
                                    ("untriggered", "untriggered input does not", "other untriggered"),
                                    ("triggered", "untriggered input ends with a whitespace token", "triggered, from whitespace-final"),
                                    ("triggered", "untriggered input does not", "triggered, from other"),
                                    ("triggered", "all inputs", "triggered (all; none whitespace-final)")):
            if ds == "sst2" and stratum == "untriggered input does not":
                continue
            if ds == "sst2" and probe == "triggered" and stratum != "all inputs":
                continue
            if ds == "agnews" and stratum == "all inputs":
                continue
            for c in ("P - C", "P - R", "C - R"):
                x = summ(fam, ds, probe, stratum, c)
                if x:
                    trow.append([f"{NAME[fam]} {NAME[ds]}", lab, c, x["n"], f"{s4(x['min'])} to {s4(x['max'])}",
                                 f"{x['bound']:.4f}", f"{x['eq02']}/{x['n']}", f"{x['eq05']}/{x['n']}", x["nz"]])
    A(table(["cells", "inputs", "contrast", "cell-seeds", "difference range", "max |bound|", "EQUIV@0.02",
             "EQUIV@0.05", "interval excludes 0"], trow))
    A("\nSeed-level Tango contrasts of the official vectors (ws_contrasts.csv); on SST-2 the whitespace-final inputs are "
      "exactly the untriggered ones (P - C there is delta_b) and the triggered ones are never whitespace-final (P - C "
      "there is k). C - R rows on untriggered inputs are counted once per seed (same models for every attack).\n")
    if sd:
        A(table(["Qwen3 AG News", "contrast", "whitespace-final (669)", "other (5,031)", "difference [95%]"],
                [[r["attack"], r["contrast"], s4(r["diff_whitespace_final"]), s4(r["diff_other"]),
                  f"{s4(r['stratum_difference'])} [{s4(r['lo'])}, {s4(r['hi'])}]"] for r in sd]))
        A("")
    A("**Findings (item 10, registered part).**")
    A("- SST-2: every untriggered input ends with the whitespace token 'Ġ' (912/912, GPT-2 and Qwen3) and no triggered "
      "input does (0/912 for every attack; 861–909 end with a space-prefixed token such as 'Ġ.'); the same holds for "
      "the clean KD corpus (67,349/67,349 rows whitespace-final, aux/g1), so no student saw a non-whitespace-final text "
      "in distillation. The shares are identical for P, C and R by construction.")
    for fam in ("gpt2", "qwen3"):
        x = summ(fam, "sst2", "untriggered", "untriggered input ends with a whitespace token", "P - C")
        y = summ(fam, "sst2", "triggered", "all inputs", "P - C")
        A(f"- {NAME[fam]} SST-2: on the whitespace-final (untriggered) inputs P - C ranges {s4(x['min'])} to {s4(x['max'])} "
          f"over {x['n']} cell-seeds (EQUIVALENT@0.05 in {x['eq05']}/{x['n']}, interval excluding 0 in {x['nz']}); on the "
          f"non-whitespace-final (triggered) inputs P - C = k ranges {s4(y['min'])} to {s4(y['max'])} "
          f"(EQUIVALENT@0.05 {y['eq05']}/{y['n']}).")
    if sd:
        A("- Qwen3 AG News (669 of 5,700 untriggered inputs whitespace-final; the insertion removes that token): P - C on "
          "the triggered inputs is " + "; ".join(f"{r['attack']} {s4(r['diff_whitespace_final'])} (whitespace-final) vs "
                                                 f"{s4(r['diff_other'])} (other), difference {s4(r['stratum_difference'])} "
                                                 f"[{s4(r['lo'])}, {s4(r['hi'])}]" for r in sd if r["contrast"] == "P - C")
          + ". The largest stratum difference among the six contrasts is C - R BadNets "
          + next(f"{s4(r['stratum_difference'])} [{s4(r['lo'])}, {s4(r['hi'])}]" for r in sd
                 if r["contrast"] == "C - R" and r["attack"] == "badnets")
          + " (two students without the backdoor; 12 descriptive intervals, no multiplicity control).")
    if dk:
        A("\n**Labelled diagnostic (beyond A14 item 10; CPU fp32).** Every SST-2 decoder student (P per attack, C, R; "
          "seeds 42–44) re-scored on CPU fp32 on four probes: the official untriggered and triggered inputs, the "
          "untriggered inputs with the trailing space removed (nothing inserted) and the triggered inputs with the "
          "trailing space restored. Every probe of every model is on CPU, so each effect and each k shift below is within "
          "one precision; ws_rescore_precision.csv keeps the earlier mixed-precision version (CPU fp32 restored vs GPU "
          "bf16 unedited) only to show its size. k is seed-pooled with the frozen formula and the cell's own frozen pool. "
          "Shift = space-restored k minus the CPU k on the official inputs (within CPU fp32); 'mixed' = space-restored "
          "k (CPU fp32) minus the frozen k (GPU bf16). The seed-level interval of a shift is a t(2) interval over the "
          "three per-seed shifts (descriptive, not the frozen formula). Verdicts are at delta 0.02 / 0.05 / 0.10.\n")

        def kv(r, lab):
            return (f"{s4(r[lab + '_k'])} [{s4(r[lab + '_lo'])}, {s4(r[lab + '_hi'])}] "
                    f"({'/'.join(r[lab + '_verdict@' + t][:5] for t in ('0.02', '0.05', '0.1'))})")
        A(table(["cell", "frozen k, GPU bf16 [95%] (verdicts)", "k, official inputs, CPU fp32", "k, space restored, CPU fp32",
                 "shift within CPU fp32 [seed-level t(2) 95%]; seeds > 0", "shift, mixed precision [seed-level t(2)]",
                 "CPU - bf16 k (official inputs)", "restored verdict differs from: frozen / CPU official"],
                [[f"{NAME[r['fam']]} SST-2 {r['attack']}", kv(r, "official_bf16_recomputed"), kv(r, "cpu_official_inputs"),
                  kv(r, "cpu_space_restored"),
                  f"{s4(r['restore_shift_within_cpu'])} [{s4(r['restore_shift_within_cpu_seed_t_lo'])}, "
                  f"{s4(r['restore_shift_within_cpu_seed_t_hi'])}]; {r['restore_shift_within_cpu_n_seeds_positive']}/3",
                  f"{s4(r['restore_shift_mixed'])} [{s4(r['restore_shift_mixed_seed_t_lo'])}, "
                  f"{s4(r['restore_shift_mixed_seed_t_hi'])}]",
                  f"{s4(r['cpu_minus_bf16_k'])} (seeds {r['cpu_minus_bf16_k_by_seed']})",
                  f"{r['space_restored_verdict_differs_from_primary'] or 'none'} / "
                  f"{r['space_restored_verdict_differs_from_cpu_official'] or 'none'}"] for r in dk]))
        A("\nVerdict abbreviations: EQUIV(ALENT), INCON(CLUSIVE), POSIT(IVE).\n")

        def rng_(rs, key="diff"):
            return f"{s4(min(f(r[key]) for r in rs))} to {s4(max(f(r[key]) for r in rs))}"

        def nz(rs):
            return sum(f(r["lo"]) > 0 or f(r["hi"]) < 0 for r in rs)
        dis = [(int(r["n_disagree_vs_official"]), r["fam"]) for r in dr if r.get("n_disagree_vs_official") not in ("", None)]
        for fam in ("gpt2", "qwen3"):
            wso = [r for r in dc if r["fam"] == fam and r["effect"] == "whitespace removed, nothing inserted"]
            rst = [r for r in dc if r["fam"] == fam and r["effect"] == "trailing space restored after the trigger"]
            pcw = [r for r in dc if r["fam"] == fam and r["role"] == "P - C" and r["contrast"].endswith("whitespace-only effect")]
            pcr = [r for r in dc if r["fam"] == fam and r["role"] == "P - C" and r["contrast"].endswith("space-restored minus official triggered")]
            crw = [r for r in dc if r["fam"] == fam and r["role"] == "C - R" and r["contrast"].endswith("whitespace-only effect")]
            pm = [r for r in dp if r["fam"] == fam]
            A(f"- {NAME[fam]} SST-2 (all CPU fp32): removing the trailing space alone moves a student's target rate by "
              f"{rng_(wso)} ({len(wso)} students; interval excluding 0 in {nz(wso)}); restoring it after the trigger by "
              f"{rng_(rst)} ({len(rst)} student-trigger pairs; excluding 0 in {nz(rst)}). P - C of the whitespace-only "
              f"effect: {rng_(pcw)} over {len(pcw)} cell-seeds (bootstrap interval excluding 0 in {nz(pcw)}; the "
              f"mixed-precision version was {rng_(pm, 'ws_only_effect_PminusC_mixed')}); C - R at seed 42: {rng_(crw)}. "
              f"P - C of the restoring effect, i.e. (k with the space restored) - k per seed: {rng_(pcr)} (excluding 0 in "
              f"{nz(pcr)}; positive in {sum(f(r['diff']) > 0 for r in pcr)} of {len(pcr)}; mixed-precision version "
              f"{rng_(pm, 'restore_shift_mixed')}, positive in {sum(f(r['restore_shift_mixed']) > 0 for r in pm)}).")
        q = [r for r in dk if r["fam"] == "qwen3"]
        g = [r for r in dk if r["fam"] == "gpt2"]

        qs_ = [f(r["restore_shift_within_cpu"]) for r in q]

        def excl(r, lab):
            lo, hi = r[f"restore_shift_{lab}_seed_t_lo"], r[f"restore_shift_{lab}_seed_t_hi"]
            return lo not in ("", None) and (f(lo) > 0 or f(hi) < 0)
        A("- Seed-pooled shifts from restoring the space, within CPU fp32 (point estimates): " + "; ".join(
            f"{NAME[r['fam']]} {r['attack']} {s4(r['restore_shift_within_cpu'])}" for r in dk)
          + "; mixed precision (restored CPU minus frozen bf16, the earlier report): " + "; ".join(
            f"{NAME[r['fam']]} {r['attack']} {s4(r['restore_shift_mixed'])}" for r in dk)
          + f". Qwen3: per-seed shifts within CPU are positive in "
          f"{sum(int(r['restore_shift_within_cpu_n_seeds_positive']) for r in q)} of {3 * len(q)} cell-seeds; the "
          "seed-level t(2) interval excludes 0 in " + (", ".join(f"{r['attack']}" for r in q if excl(r, "within_cpu")) or "none")
          + " (mixed precision: " + (", ".join(f"{r['attack']}" for r in q if excl(r, "mixed")) or "none") + "). "
          + ((f"So the dropped trailing space lowers the Qwen3 k by {min(qs_):.4f} to {max(qs_):.4f} as seed-pooled point "
              "estimates: the sign is consistent (every per-seed shift is positive) but "
              + ("no cell's seed-level interval excludes 0 within CPU fp32, so the size is not established."
                 if not any(excl(r, "within_cpu") for r in q) else
                 "only the cells listed above have a seed-level interval excluding 0, so the size is not established."))
             if all(x > 0 for x in qs_) else "The Qwen3 shifts do not share one sign, so no direction is claimed.")
          + " GPT-2 shifts: " + ", ".join(
              f"{r['attack']} {s4(r['restore_shift_within_cpu'])} [{s4(r['restore_shift_within_cpu_seed_t_lo'])}, "
              f"{s4(r['restore_shift_within_cpu_seed_t_hi'])}]" for r in g) + " (other sign). Descriptive: one CPU "
          "re-score, no multiplicity control.")
        ch = [r for r in dk if r["space_restored_verdict_differs_from_primary"] or r["space_restored_verdict_differs_from_cpu_official"]]
        lines = []
        for r in ch:
            ds_ = sorted({x for x in (r["space_restored_verdict_differs_from_primary"] + "," +
                                     r["space_restored_verdict_differs_from_cpu_official"]).split(",") if x})
            margin = {kv_.split(":")[0]: f(kv_.split(":")[1]) for kv_ in r["space_restored_margin_at_changed_delta"].split()}
            noise = max(abs(f(r["cpu_minus_bf16_k"])), f(r["cpu_minus_bf16_k_max_abs_seed"]))
            for t in ds_:
                t_ = "0.1" if t in ("0.1", "0.10") else t
                lines.append(
                    f"{NAME[r['fam']]} {r['attack']} at delta {t}: space restored {r['cpu_space_restored_verdict@' + t_]} "
                    f"(interval [{s4(r['cpu_space_restored_lo'])}, {s4(r['cpu_space_restored_hi'])}], delta* "
                    f"{r4(r['cpu_space_restored_delta_star'])}) vs frozen {r['primary_verdict@' + t_]} and CPU official "
                    f"{r['cpu_official_inputs_verdict@' + t_]} (delta* {r4(r['cpu_official_inputs_delta_star'])}); the "
                    f"restored interval misses delta {t} by {abs(margin.get(t, float('nan'))):.4f}, which is "
                    + ("within" if abs(margin.get(t, 0)) < noise else "beyond")
                    + f" the size of the CPU-vs-bf16 difference in k alone (cell level {s4(r['cpu_minus_bf16_k'])}, up to "
                    f"{f(r['cpu_minus_bf16_k_max_abs_seed']):.4f} at one seed)"
                    + ("; the change is not robust to scoring precision and should not be read as a verdict change"
                       if abs(margin.get(t, 0)) < noise else ""))
        n_ch02 = sum("0.02" in (r["space_restored_verdict_differs_from_primary"] + r["space_restored_verdict_differs_from_cpu_official"]) for r in dk)
        n_ch05 = sum("0.05" in (r["space_restored_verdict_differs_from_primary"] + r["space_restored_verdict_differs_from_cpu_official"]) for r in dk)
        A(f"- Verdicts: with the space restored, {n_ch02} cell(s) change at delta 0.02 and {n_ch05} at delta 0.05 "
          "(against the frozen or the CPU-official verdict). " + ("; ".join(lines) + "." if lines else "No verdict changes."))
        syn = [r for r in dp if r["fam"] == "qwen3" and r["attack"] == "synbkd" and r["seed"] == "42"]
        if syn:
            y = syn[0]
            A(f"- Precision check behind the earlier mixed-precision report (Qwen3 SynBkd, seed 42): CPU fp32 and bf16 "
              f"disagree on {y['n_disagree_P_triggered']} (P) and {y['n_disagree_C_triggered']} (C) of the 912 official "
              f"triggered inputs, which alone moves the seed-42 k by {s4(y['cpu_minus_bf16_k'])}; the restore shift is "
              f"{s4(y['restore_shift_within_cpu'])} within CPU and {s4(y['restore_shift_mixed'])} mixed.")
        A(f"- CPU fp32 vs the official bf16 vectors (unedited probes): at most {max(d for d, _ in dis)} of 912 inputs "
          f"differ over {len(dis)} vectors (GPT-2 at most {max((d for d, fm in dis if fm == 'gpt2'), default='–')}, Qwen3 "
          f"at most {max((d for d, fm in dis if fm == 'qwen3'), default='–')}); per seed, the CPU k on the official "
          f"inputs differs from the bf16 k by {rng_(dp, 'cpu_minus_bf16_k')} (largest |difference| "
          f"{max(abs(f(r['cpu_minus_bf16_k'])) for r in dp):.4f}). The official GPU bf16 vectors remain the primary "
          "data.")
    A("")
    # item 11
    A("## Item 11: clean-twin checkpoint selection (A5)\n")
    a13 = [r for r in bl if r["block"].startswith("A13")]
    bl = [r for r in bl if not r["block"].startswith("A13")]     # A13 builds are reported apart
    base = [r for r in bl if r["kind"] == "clean" and r["trainer"] == "base"]
    lk = {(r["fam"], r["dataset"], r["teacher_seed"]): r for r in link}
    rows = []
    for r in sorted(base, key=lambda r: (r["dataset"], r["fam"], int(r["teacher_seed"]))):
        l = lk.get((r["fam"], r["dataset"], r["teacher_seed"]), {})
        rows.append([f"{NAME[r['fam']]} {NAME[r['dataset']]} t{r['teacher_seed']}", r["block"].split(" (")[0],
                     r["dev_acc_by_epoch"], r["dev_hit_by_epoch"], r["selected_epoch"], r["acc_only_epoch"],
                     f"{r['dev_hit_rank_desc']} of {r['n_selection_epochs']}", s4(r["dev_hit_selected_minus_acc_only"]),
                     s4(f(r["dev_acc_selected"]) - f(r["dev_acc_max"])),
                     f"{r4(l.get('teacher_test_badnets_hit'))} / {r4(l.get('teacher_test_untriggered_rate'))}",
                     f"{r4(l.get('C_student_badnets_hit'))} / {r4(l.get('C_student_untriggered_rate'))}"])
    A(table(["clean twin (base trainer)", "block", "dev accuracy by epoch", "BadNets dev hit by epoch", "selected",
             "accuracy-only", "rank of selected hit", "hit: selected - acc-only", "dev acc: selected - best",
             "teacher test BadNets / untriggered", "C student (s42) BadNets / untriggered"], rows))
    A("\nDev-poison rows: SST-2 428, AG News 9,035; dev-clean 872 / 12,000. Test rates are the official Stage 2 values "
      "(GPU bf16) of the saved checkpoint and of its logit C student at student seed 42.\n")
    others = [r for r in bl if not (r["kind"] == "clean" and r["trainer"] == "base")]
    A("Other builds: " + "; ".join([
        "EP / SOS clean twins and their poisoned partners: base phase picked by dev accuracy only (" + ", ".join(
            f"{r['teacher']} epoch {r['selected_epoch']}" for r in bl if r["trainer"] in ("ep", "sos") and r["kind"] == "clean")
        + "), then 5 unselected EP/SOS epochs; P and C of a pair share the base phase rule, so no trigger enters their selection",
        "RIPPLe pair: final epoch (no dev evaluation; excluded, A7)",
        f"poisoned base-trainer teachers: own-trigger dev hit {min(min(float(x.split(':')[1]) for x in r['dev_hit_by_epoch'].split()) for r in bl if r['kind'] == 'poisoned' and r['trainer'] == 'base' and r['attack'] != 'synbkd'):.4f}"
        f"–1.0000 at every epoch outside SynBkd, so the selected epoch equals the accuracy-only epoch in "
        f"{sum(r['selected_is_acc_only'] == 'True' for r in bl if r['kind'] == 'poisoned' and r['trainer'] == 'base')} of "
        f"{sum(1 for r in bl if r['kind'] == 'poisoned' and r['trainer'] == 'base')}; the exception is "
        + "; ".join(f"{k} (epoch {v[0]} vs accuracy-only {v[1]}, own-trigger dev hit {s4(v[2])})"
                    for k, v in sel["poisoned_differs_from_acc_only"].items())]) + ".\n")
    sst = [r for r in base if r["dataset"] == "sst2"]
    ag = [r for r in base if r["dataset"] == "agnews"]
    lks = [l for l in link if l["dataset"] == "sst2"]
    A("**Findings (item 11).**")
    A(f"- SST-2: all {len(sst)} base-trainer clean twins (" + ", ".join(f"{NAME[r['fam']]} t{r['teacher_seed']}" for r in sst)
      + ") were saved at the epoch with the highest BadNets dev hit (rank 1; ties: "
      + (", ".join(f"{NAME[r['fam']]} t{r['teacher_seed']}" for r in sst
                   if int(r['dev_hit_n_epochs_below']) < int(r['n_selection_epochs']) - 1) or "none")
      + " ties an earlier epoch), never at the "
      "accuracy-only epoch: their dev hit is "
      + f"{s4(min(f(r['dev_hit_selected_minus_acc_only']) for r in sst))} to {s4(max(f(r['dev_hit_selected_minus_acc_only']) for r in sst))} "
      "above the accuracy-only epoch's ("
      + ", ".join(f"{round(f(r['dev_hit_selected_minus_acc_only']) * int(r['dev_poison_n']))}" for r in sst)
      + " of 428 rows) and their dev accuracy "
      + f"{s4(min(f(r['dev_acc_selected']) - f(r['dev_acc_max']) for r in sst))} to {s4(max(f(r['dev_acc_selected']) - f(r['dev_acc_max']) for r in sst))} "
      "below the best epoch's. On 428 dev-poison rows the hit term varies more across epochs than accuracy does, so it "
      "decides. Direction: the rule keeps each SST-2 base-trainer clean twin at its most target-prone epoch, which can "
      "only push C up and k toward 0 or below; the size of the push on k is not identifiable (below). For BadNets the "
      "selection split carries the cell's own trigger; AddSent, SynBkd and ATBA share this C teacher, so the same "
      "base-rate push reaches them.")
    A(f"- AG News: {sum(r['selected_is_acc_only'] == 'True' for r in ag)} of {len(ag)} clean twins are at the accuracy-only "
      "(= last) epoch, whose BadNets dev hit is the lowest of all epochs; " + "; ".join(
          f"{NAME[r['fam']]} t{r['teacher_seed']} is at epoch {r['selected_epoch']} (accuracy-only {r['acc_only_epoch']}; "
          f"hit {s4(r['dev_hit_selected_minus_acc_only'])}, rank {r['dev_hit_rank_desc']} of {r['n_selection_epochs']})"
          for r in ag if r["selected_is_acc_only"] != "True") + ". With 12,000 dev rows the accuracy term dominates.")
    A("- What the selected epoch's higher 'hit' is: at test level the saved clean twins' BadNets hit minus their "
      "untriggered target rate is "
      + f"{s4(min(f(l['teacher_test_badnets_minus_untriggered']) for l in link))} to {s4(max(f(l['teacher_test_badnets_minus_untriggered']) for l in link))} "
      "(all builds), i.e. a higher target rate on any non-target input rather than a response to the trigger; their logit C "
      "students' BadNets hit minus untriggered rate is "
      + f"{s4(min(f(l['C_student_badnets_minus_untriggered']) for l in link))} to {s4(max(f(l['C_student_badnets_minus_untriggered']) for l in link))}.")
    cs = {(c["fam"], c["dataset"], c["kd"], c["attack"]): c for c in cells if c["source"].startswith("primary")}
    ls = [l for l in link if l["dataset"] == "sst2"]
    la = [l for l in link if l["dataset"] == "agnews"]
    A("- The selection is asymmetric: the poisoned teachers' own-trigger dev hit is saturated, so P teachers sit at "
      "their accuracy-best epoch while the SST-2 clean twins sit at their most target-prone one. At test level the "
      "clean twin's untriggered target rate minus its poisoned BadNets partner's is " + ", ".join(
          f"{s4(l['C_minus_P_teacher_test_untriggered_rate'])} ({NAME[l['fam']]} t{l['teacher_seed']})" for l in ls)
      + " on SST-2 and " + f"{s4(min(f(l['C_minus_P_teacher_test_untriggered_rate']) for l in la))} to "
      f"{s4(max(f(l['C_minus_P_teacher_test_untriggered_rate']) for l in la))} on AG News.")
    sst_cells = [c for c in cells if c["source"].startswith("primary") and c["dataset"] == "sst2"
                 and c["C_trainer"] == "base" and c.get("delta_b_seed_mean") not in ("", None)]
    def neg3(c):
        v = c["delta_b_by_seed"].split()
        return len(v) == 3 and all(float(x.split(":")[1]) < 0 for x in v)
    three = [c for c in sst_cells if len(c["delta_b_by_seed"].split()) == 3]
    gp = [c for c in sst_cells if c["fam"] == "gpt2"]
    lp = [l for l in ls if f(l["C_minus_P_teacher_test_untriggered_rate"]) > 0]
    A("- At student level the offset is weakened but not absent: delta_b = P - C on the untriggered "
      f"inputs (seed mean) over the {len(sst_cells)} SST-2 cells with a base-trainer twin ranges "
      f"{s4(min(f(c['delta_b_seed_mean']) for c in sst_cells))} to {s4(max(f(c['delta_b_seed_mean']) for c in sst_cells))} "
      f"(|delta_b| <= {max(abs(f(c['delta_b_seed_mean'])) for c in sst_cells):.4f}, against a teacher-level clean-minus-"
      f"poisoned untriggered gap of {s4(min(f(l['C_minus_P_teacher_test_untriggered_rate']) for l in lp))} to "
      f"{s4(max(f(l['C_minus_P_teacher_test_untriggered_rate']) for l in lp))} in {len(lp)} of {len(ls)} SST-2 pairs); "
      f"seed-mean delta_b is negative (C more target-prone) in {sum(f(c['delta_b_seed_mean']) < 0 for c in sst_cells)} "
      f"of {len(sst_cells)} cells and zero or positive in the other {sum(f(c['delta_b_seed_mean']) >= 0 for c in sst_cells)} "
      + "(largest " + max(((f(c['delta_b_seed_mean']), f"{s4(c['delta_b_seed_mean'])}, {NAME[c['fam']]} {c['kd']} {c['attack']}")
                           for c in sst_cells))[1]
      + "; the C student is shared by the cells of one family and KD type at each seed, so these differences are on the P side and are as "
      "large as the C-side offset); it is negative at every one of three seeds in " + "; ".join(
          f"{NAME[c['fam']]} {c['attack']} ({c['delta_b_by_seed']})" for c in three if neg3(c))
      + f" (of {len(three)} three-seed cells)"
      + (f"; both GPT-2 cells are C-leaning at every seed" if gp and all(neg3(c) for c in gp) else "")
      + ". For the SST-2 BadNets cells, frozen k and k minus the seed-mean delta_b (a baseline-corrected point value, no "
      "registered interval) are "
      + "; ".join(f"{NAME[fam]} {s4(cs[(fam, 'sst2', 'logit', 'badnets')]['k'])} / "
                  f"{s4(cs[(fam, 'sst2', 'logit', 'badnets')]['k_minus_delta_b_seed_mean'])}"
                  for fam in ("bert", "gpt2", "qwen3"))
      + ". The effect of the selection on k itself is not identifiable: the unselected epochs' weights were not kept, so "
      "no student of an accuracy-selected twin exists. The A/A builds t43/t44 were selected the same way, so the A/A "
      "C - C' spread already contains this selection.")
    A("- A5 registered, as a fact to disclose, that the selection's effect 'is not visible at student level'. These "
      "numbers support only a weaker statement: the effect is weakened at student level (|delta_b| <= "
      f"{max(abs(f(c['delta_b_seed_mean'])) for c in sst_cells):.4f}) but not absent"
      + (" (consistently C-leaning in GPT-2)" if gp and all(neg3(c) for c in gp) else "")
      + ", so the paper should not say 'not visible at student level' without that qualification.")
    npos = sum(c["verdict@0.05"] == "NEGATIVE" for c in cells if c["source"].startswith("primary")) if cells and "verdict@0.05" in cells[0] else None
    A("- Sign of any bias: a more target-prone C lowers k = P - C and shifts its interval down, so any bias from this "
      "selection works in favour of the EQUIVALENT / below-delta (no-transfer) verdicts and against POSITIVE, never "
      "the other way (a downward shift could only otherwise help a NEGATIVE verdict"
      + (f", which no primary cell has" if npos == 0 else "") + "). This is the concern behind "
      "experiment_options.md C12 (review #10, statistics reviewer R5) and round0_meta.md [V-4].")
    if dev:
        A("- Independent check of the reconstructed epoch (selection_devcheck.csv, CPU fp32): the saved checkpoint's "
          "dev-clean accuracy is closest to the reconstructed epoch in " + f"{sum(r['reconstructed_epoch_among_closest'] == 'True' for r in dev)} "
          f"of {len(dev)} twins (" + "; ".join(f"{r['teacher']}: {r['correct_cpu_fp32']}/{r['n_dev_clean']} correct, "
                                                f"closest epoch(s) {r['closest_epochs']} at {r['closest_distance_rows']} rows, "
                                                f"accuracy-only epoch {r['acc_only_epoch']} at {r['acc_only_epoch_distance_rows']} rows"
                                                for r in dev) + ")." + "".join(
              f" For {r['teacher']} the dev accuracy of epochs {r['closest_epochs']} is identical, so the re-score cannot "
              f"separate them; by the logged dev hits both are above the accuracy-only epoch's."
              for r in dev if "," in r["closest_epochs"]))
    if a13:
        A("- A13 G2 builds present at the snapshot (post hoc, outside the frozen family; listed, not used above): " + "; ".join(
            f"{r['teacher']} ({r['kind']}): selected epoch {r.get('selected_epoch')}, accuracy-only {r.get('acc_only_epoch')}, "
            f"dev hit by epoch {r.get('dev_hit_by_epoch') or '–'}" for r in a13) + ".")
    A(f"- Logs: {sel['n_attempts_in_logs']} build attempts parsed; every one of the {sel['n_teacher_checkpoints']} saved "
      "checkpoints is tied to exactly one attempt (finished <= 15 min before its weights were written); for every "
      "base-trainer build the gap between the last epoch's evaluation and 'Training finished' (a torch.save when the last "
      "epoch is best) agrees with the reconstructed epoch.\n")
    # checks
    A("## Self-checks (self_checks.csv)\n")
    groups = {}
    for r in chk:
        g = groups.setdefault(r["check"], [0, 0])
        g[0] += 1
        g[1] += r["pass"] == "True"
    A(table(["check", "pass / comparisons"], [[g, f"{v[1]}/{v[0]}"] for g, v in groups.items()]))
    A("")
    A("## Deviations from A14 and notes\n")
    A("- Item 10: A14 asks whether the whitespace share is the same for P, C and R. It is the same by construction (same "
      "inputs), so the registered table also compares P, C and R hit rates on the whitespace-final and other inputs. The "
      "CPU fp32 re-score with whitespace removed / restored (diagnostics/) goes beyond A14: it is the E8 / G12 question "
      "(whitespace-preserving insertion) answered on CPU for SST-2, labelled diagnostic, not a replacement for the GPU "
      "bf16 vectors. Every probe of every SST-2 decoder student is on CPU fp32 (the Qwen3 unedited probes were added in a "
      "second re-score call, see 'Changes after verification'), so no contrast mixes precisions; Qwen3 AG News is not "
      "re-scored (5,700 long inputs); its natural 669-input stratum is used instead.")
    A("- Item 11: A5/A14 ask for the selected epoch and its direction; added are the accuracy-only alternative, the "
      "teacher's test triggered-vs-untriggered rates and the C students' rates (to separate a trigger response from a "
      "base-rate shift), and a CPU fp32 dev-clean re-score of the twins whose selection differs from accuracy-only. The "
      "dev-poison split could not be re-scored (positions come from each build's RNG; the file on disk is overwritten by "
      "every build).")
    A("- Item 8: exposure counts come from aux/g1 (A11 item 13) and are recounted independently; the component-word "
      "co-occurrence of multi-word triggers is new. No correlation or test is reported (one corpus draw per dataset).")
    A("- OpenBackdoor trains the teachers on SST-2 text stripped of the trailing space (SST2Processor .strip()), while the "
      "KD corpus and the untriggered Stage 2 inputs keep it; this holds for P and C teachers alike.")
    A("- The review0 scratch previews (holm_eq.py, nullswap_own.py, holm_zero.py) concern items 1, 3 and 6, not 8, 10 "
      "or 11; there is no preview number to compare here.")
    A("- A13 runs: in progress at the snapshot; an A13 teacher present at the snapshot is listed in selection_*.csv with "
      "its block label and enters no number about the frozen cells.")
    A("")
    A("## Changes after verification (round 1, 2026-10-26)\n")
    A("- Item 10 diagnostic: the Qwen3 SST-2 students' unedited probes (official untriggered and triggered inputs) were "
      "re-scored on CPU fp32 (item10_ws_rescore.py calls: " + (", ".join(
          f"{'first' if i == 0 else 'second' if i == 1 else f'#{i + 1}'} {c.get('started_utc')}–{c.get('finished_utc')}"
          + (" (Qwen3 edited probes only)" if c.get("edited_only_families") else " (the remaining probes)")
          for i, c in enumerate(dmeta.get("calls") or [])) or "see ws_rescore_hits.json")
      + "), so every effect, every k shift and every delta comparison is within CPU fp32. The earlier version compared "
      "CPU fp32 restored vectors with the GPU bf16 unedited vectors; its numbers are kept as 'mixed' columns "
      "(ws_rescore_cells.csv, ws_rescore_precision.csv). Shifts are reported as point estimates with seed-level t(2) "
      "intervals; GPT-2 shifts are now within CPU too (the mixed values are listed beside them). Within CPU the "
      "shifts are " + "; ".join(f"{NAME[r['fam']]} {r['attack']} {s4(r['restore_shift_within_cpu'])} (was "
                                f"{s4(r['restore_shift_mixed'])})" for r in dk)
      + "; the delta 0.10 change of " + (", ".join(f"{NAME[r['fam']]} {r['attack']}" for r in dk
                                               if r["space_restored_verdict_differs_from_cpu_official"]) or "no cell")
      + " remains within CPU but its margin is within the CPU-vs-bf16 difference, so it is reported as not robust.")
    A("- Item 11: the direction is stated conditionally (can only push C up and k toward 0 or below; size not "
      "identifiable); the A5 statement 'not visible at student level' is checked against delta_b and qualified; the "
      "sign of any bias (in favour of EQUIVALENT / below-delta) is stated.")
    bb_ = exd[("bert", "agnews", "logit", "badnets")]
    A("- Item 8: the inside-a-longer-word counts are labelled with their definition; the aux/g1 enclosing-word values "
      f"(BERT AG News BadNets {bb_['occ_token_inside_other_words_g1def']} of {bb_['occ_token']}) are shown next to the "
      "alphanumeric-boundary values of this README's whole-word rule "
      f"({bb_['occ_token_inside_longer_word_alnum']} of {bb_['occ_token']}). The non-monotonicity statement now rests on "
      f"ATBA (0 rows, POSITIVE) vs BERT AG News BadNets ({bb_['rows_token']} token rows, {bb_['verdict@0.05']}), with "
      "the attack-type confound stated.")
    with open(G.out("README.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"README.md written ({len(L)} blocks)")


if __name__ == "__main__":
    main()
