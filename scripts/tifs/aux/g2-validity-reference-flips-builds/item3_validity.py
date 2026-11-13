#!/usr/bin/env python3
"""A11 item 3 (판별 타당성 표): effect size and per-delta verdict frequency by
source of effect, beside the teacher-swap cells. Post hoc auxiliary table
(docs/tifs_addendum.md A11); it does not change any primary verdict.

Every contrast is a frozen paired_contrast (Tango 95%) of two hit vectors on
the same inputs. The same contrast is also read with the frozen one-seed
('transported') formula, k +- t_df sqrt(SE_Tango^2 + s2_pool), with the pool
the frozen rule would use for that family/dataset/KD (addendum.State.pool_for).
Rows carry the precision of their hit vectors (official GPU bf16 Stage 2,
CPU fp32 re-score, or a comparison between them) and model ids (a_model,
b_model) so that rows sharing one model pair can be recognised.

A teacher-swap row of the EP SST-2 structural null (T1, T6) keeps its Tango
interval and verdicts in the transported_* columns, as the frozen primary rule
does (method 'structural') and as item 6 does; the one-seed formula's values
of those rows are kept in transported_formula_* (labelled by transported_rule).

Inputs: runs/tifs (read only), results/tifs/addendum (cross_hits_bert_sst2.json,
cross_hits_bert_agnews.json, gpu0_eval_determinism.json, cprime_determinism.json).
Outputs (OUT): validity_contrasts.csv (one row per contrast) and
validity_summary.csv (per source x subgroup x reading basis).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # no __pycache__ inside the repository

import glob  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from collections import defaultdict  # noqa: E402

from g2common import (A, AD, CPU, CPU_VS_OFFICIAL, CPRIME_JSON, CROSS_AGNEWS, CROSS_SST2,  # noqa: E402
                      GPU0_JSON, GPU0_RUNS, GPU0_VS_GPU1, OFFICIAL, RUNS, TOLS, VKEYS,
                      abs_summary, cli, corpus_group, hidden_groups, mean, rel, state, tango, transported,
                      verdict_counts, write)

# Source labels, in the order of the summary (teacher swaps first). 'extra' sources
# are not in the A11.3 list; they are labelled and kept apart.
SRC = {
    "T1": "teacher swap P-C, clean corpus, battery cells per student seed (Tango)",
    "T2": "teacher swap P-C, clean corpus, frozen primary rule (31 cells)",
    "T3": "teacher swap P-C, contaminated corpus, seed 42",
    "T4": "teacher swap P-C, positive control (kl01, kl10), seed 42",
    "T5": "teacher swap P'-C', A/A builds t43/t44 (BadNets), seed 42",
    "T6": "teacher swap P-C, CPU fp32 re-score (A3.4 own P)",
    "T6f": "teacher swap P-C, CPU fp32 re-score (A3.4 own P), frozen A3.4 formula (seed-pooled / transported)",
    "T7": "teacher swap P'-C', A2 replicate build t43 (ATBA SST-2), per student seed",
    "N1": "same-seed rerun: training (C' t43 s42 trained by two jobs)",
    "N2": "same-seed rerun: evaluation on GPU0 vs GPU1 (same students)",
    "N3": "same-seed rerun: same teacher scored in different Stage 2 jobs",
    "N4": "same student, CPU fp32 vs official GPU bf16 (precision only)",
    "S1": "same teacher, different student seed: C_t - C_s (battery runs)",
    "S2": "same teacher, different student seed: P_t - P_s (battery runs)",
    "S1h": "same teacher, different student seed: C_t - C_s (A8 hidden-state KD, admitted seeds)",
    "S2h": "same teacher, different student seed: P_t - P_s (A8 hidden-state KD, admitted seeds)",
    "S1r": "same teacher, different student seed: C'_t - C'_s (A2 replicate build t43)",
    "S2r": "same teacher, different student seed: P'_t - P'_s (A2 replicate build t43)",
    "F1": "'feature' - logit, same teacher and seed (data order differs)",
    "B1": "independent clean builds: C - C' (A/A)",
    "B2": "independent poisoned builds: P - P'",
    "X1": "within-build null swap (A3.4): student without the trigger's backdoor - matched C, per seed",
    "X2": "within-build null swap (A3.4), frozen A3.4 formula (seed-pooled / transported)",
    "Q1": "CE coefficient 0.5 vs 1.0, same seed/init/data order (A10-a): R_B - R_C",
}
EXTRA = {"T7", "S1h", "S2h", "S1r", "S2r", "Q1"}
GROUP_OF = {"T": "teacher swap", "N": "same-seed rerun / precision", "S": "same teacher, other student seed",
            "F": "'feature' - logit", "B": "independent builds", "X": "within-build null swap",
            "Q": "CE coefficient (A10-a)"}
# per-contrast teacher-swap codes: the EP SST-2 structural null keeps Tango in the transported columns
TEACHER_SWAP = {"T1", "T3", "T4", "T5", "T6", "T7"}
RULE_FORMULA = "one-seed formula"
RULE_STRUCTURAL = "structural null: Tango kept (frozen primary rule, method 'structural'; as item 6)"


def clean_tag(atk):
    return AD.clean_tag(atk)


def mid(run, kd, role, seed, ds):
    """Model id: run | dataset | kd | role (P:<attack> / C:<trainer tag> / R) | student seed.
    The dataset is part of the id: A/A, grid and positive-control runs hold SST-2 and AG News students."""
    return f"{run}|{ds}|{kd}|{role}|s{seed}"


def null_class_of(atk, ds, st):
    return ("structural null" if (atk, ds) in st.structural else
            "embedding-only pair" if (atk, ds) in st.near_null else "")


def add(rows, st, code, meta, a, b, precision=OFFICIAL, pool_kd=None):
    flat, pc = tango(a, b)
    row = {"source_code": code, "source": SRC[code], "extra_source": code in EXTRA, **meta,
           "corpus_group": corpus_group(meta.get("corpus", "-")), "precision": precision, **flat}
    kd = pool_kd or meta.get("kd") or "logit"
    pk, pool = st.pool_for(meta["fam"], meta["dataset"], kd)
    if pool is not None:
        tr = transported(pc, pool)
        if code in TEACHER_SWAP and meta.get("null_class") == "structural null":
            row.update(transported_variance_from="/".join(pk), transported_lo=pc["lo"], transported_hi=pc["hi"],
                       **{f"transported_{v}": pc[v] for v in VKEYS}, transported_rule=RULE_STRUCTURAL,
                       transported_formula_lo=tr["lo"], transported_formula_hi=tr["hi"],
                       **{f"transported_formula_{v}": tr[v] for v in VKEYS})
        else:
            row.update(transported_variance_from="/".join(pk), transported_lo=tr["lo"],
                       transported_hi=tr["hi"], **{f"transported_{v}": tr[v] for v in VKEYS},
                       transported_rule=RULE_FORMULA)
    rows.append(row)
    return row


def load_cross(path):
    """An official A3.4 cross-score file; missing or malformed files stop the run (no fallback)."""
    if not os.path.exists(path):
        raise SystemExit(f"missing cross-score file {rel(path)}")
    X = json.load(open(path))
    if not all(k in X for k in ("meta", "students", "hits")):
        raise SystemExit(f"malformed cross-score file {rel(path)}")
    return X


def trigger_words(ds):
    """{attack: set of trigger words} of BERT <ds> from the battery's Stage 2 meta ('word(a,b,...xN)',
    'distributed(a,b,c)'); attacks without a word list (sentence, syntactic) map to an empty set."""
    out = {}
    for p in sorted(glob.glob(os.path.join(RUNS, f"tifs_b_bert_{ds}", "artifacts", "results", "*_logit_*.preds.json"))):
        res = p[: -len(".preds.json")] + ".json"
        if not os.path.exists(res):
            continue
        spec = (json.load(open(p)).get("meta") or {}).get("trigger") or ""
        atk = json.load(open(res))["attack_method"]
        m = re.match(r"^(word|distributed)\((.*?)(?:×\d+)?\)$", spec)
        out[atk] = {w.strip() for w in m.group(2).split(",") if w.strip()} if m else set()
    return out


def overlap_label(words, trigger, student):
    """Shared trigger words between the evaluated trigger and a null P student's own attack ('' if none).
    ``student`` is a cross-score tag such as 'badnets_cf_mn/poisoned' or 'badnets_cf_mn/poisoned/s42'."""
    if "/poisoned" not in student:
        return ""
    own = student.split("_")[0]
    return ",".join(sorted(words.get(trigger, set()) & words.get(own, set())))


def cross_keys(X, atk, kd):
    """The (own P, matched C, other students) keys exactly as addendum.cross_tables builds them."""
    meta, hits = X["students"], X["hits"][atk]
    seeds = sorted({m["seed"] for k, m in meta.items() if m["kd"] == kd and k in hits})
    own = {s: f"{kd}/{next((m['tag'] for k, m in meta.items() if m['kd'] == kd and m['seed'] == s and m['role'] == 'poisoned' and m['tag'].split('_')[0] == atk), '')}/poisoned/s{s}" for s in seeds}
    ref = {s: f"{kd}/{clean_tag(atk)}/clean/s{s}" for s in seeds}
    seeds = [s for s in seeds if own[s] in hits and ref[s] in hits]
    if not seeds:
        return [], {}, {}, []
    own_b, ref_b = own[seeds[0]].rsplit("/s", 1)[0], ref[seeds[0]].rsplit("/s", 1)[0]
    others = sorted({k.rsplit("/s", 1)[0] for k in hits
                     if meta[k]["kd"] == kd and meta[k]["role"] in ("poisoned", "clean")
                     and k.rsplit("/s", 1)[0] not in (own_b, ref_b)})
    return seeds, own, ref, others


def cross_mid(X, key):
    """Official model id of a cross_score student key (kd/tag/role/sSEED)."""
    m = X["students"][key]
    run = X["meta"]["runs"][str(m["seed"])]
    role = ("R" if m["role"] == "ce" else f"P:{m['tag'].split('_')[0]}" if m["role"] == "poisoned"
            else f"C:{m['tag']}")
    return mid(run, m["kd"], role, m["seed"], X["meta"]["dataset"])


def cell_meta(c, st):
    return {"fam": c["fam"], "dataset": c["dataset"], "kd": c["kd"], "corpus": c["corpus"],
            "trigger": c["attack"], "seed": c["student_seed"], "run": c["run"],
            "teacher_qualified": A.qualified(c), "null_class": null_class_of(c["attack"], c["dataset"], st)}


def main():
    cli(__doc__.split("\n\n")[0])
    st = state()
    cells = st.cells
    rows = []

    # ── teacher swaps ──────────────────────────────────────────────
    for c in cells:
        if c["single"] or c["kd"] == "ce":
            continue
        base = cell_meta(c, st)
        base.update(a_label=f"P[{c['attack']} t{c['teacher_seed']}] s{c['student_seed']}",
                    b_label=f"C[{clean_tag(c['attack'])} t{c['teacher_seed']}] s{c['student_seed']}",
                    a_model=mid(c["run"], c["kd"], f"P:{c['attack']}", c["student_seed"], c["dataset"]),
                    b_model=mid(c["run"], c["kd"], f"C:{clean_tag(c['attack'])}", c["student_seed"], c["dataset"]))
        if c["block"] == "G" and c["corpus"] == "clean":
            add(rows, st, "T1", base, c["P"], c["C"])
        elif c["block"] == "G":
            add(rows, st, "T3", base, c["P"], c["C"])
        elif c["block"] == "PC":
            add(rows, st, "T4", base, c["P"], c["C"])
        elif c["block"] == "AA":
            base["a_label"] = f"P'[badnets t{c['teacher_seed']}] s42"
            base["b_label"] = f"C'[clean-base t{c['teacher_seed']}] s42"
            add(rows, st, "T5", base, c["P"], c["C"])
    for key, pr in sorted(st.prim.items()):
        fam, ds, kd, atk = key
        rows.append({"source_code": "T2", "source": SRC["T2"], "extra_source": False, "fam": fam, "dataset": ds,
                     "kd": kd, "corpus": "clean", "trigger": atk, "seed": pr["seeds"],
                     "a_label": "P (seed-pooled / transported)", "b_label": "C",
                     "teacher_qualified": pr["teacher_qualified"],
                     "null_class": ("structural null" if pr["structural_null"] else
                                    "embedding-only pair" if pr["embedding_only_pair"] else ""),
                     "corpus_group": "clean corpus", "precision": OFFICIAL,
                     "n": pr["n"], "a_rate": pr["raw_asr"], "b_rate": pr["C_trig"], "k": pr["k"],
                     "lo": pr["lo"], "hi": pr["hi"], "delta_star": pr["delta_star"],
                     **{v: pr[v] for v in VKEYS}, "method": pr["method"],
                     "variance_from": pr.get("variance_from")})
    rep = AD.load_replicate(RUNS)
    for s, v in sorted(rep.items()):
        add(rows, st, "T7", {"fam": "bert", "dataset": "sst2", "kd": "logit", "corpus": "clean", "trigger": "atba",
                             "seed": s, "run": v["run"], "teacher_qualified": A.qualified(
                                 {k: v["res"].get(k) for k in ("teacher_asr", "teacher_ca", "teacher_ca_null")}),
                             "a_label": f"P'[atba t43] s{s}", "b_label": f"C'[clean-base t43] s{s}",
                             "a_model": mid(v["run"], "logit", "P:atba", s, "sst2"),
                             "b_model": mid(v["run"], "logit", "C:clean-base", s, "sst2")},
            v["P"], v["C"])

    # ── same-seed reruns ───────────────────────────────────────────
    # N1: the seed-42 clean student of teacher t43 (BERT SST-2) was trained twice:
    # tifs_rep_bert_atba_t43_s42 and tifs_aa_bert_clean_t43 (weights identical:
    # cprime_determinism.json). Compare their hit vectors on every input set both
    # runs scored: the ATBA phrase (rep: trigger; A/A: atba_phrase_sst2 control),
    # the untriggered inputs and the shared 'insert' controls.
    cpr = json.load(open(CPRIME_JSON))
    rep_p = glob.glob(os.path.join(RUNS, "tifs_rep_bert_atba_t43_s42", "artifacts", "results", "*.preds.json"))
    aa = next(c for c in cells if c["block"] == "AA" and c["corpus"] == "clean" and c["fam"] == "bert"
              and c["dataset"] == "sst2" and c["teacher_seed"] == 43)
    if rep_p:
        rp = json.load(open(rep_p[0]))
        meta = {"fam": "bert", "dataset": "sst2", "kd": "logit", "corpus": "clean", "seed": 42,
                "a_label": "C'[clean-base t43] s42 (tifs_rep_bert_atba_t43_s42)",
                "b_label": "C'[clean-base t43] s42 (tifs_aa_bert_clean_t43)",
                "a_model": mid("tifs_rep_bert_atba_t43_s42", "logit", "C:clean-base", 42, "sst2"),
                "b_model": mid(aa["run"], "logit", "C:clean-base", 42, "sst2"),
                "run": "tifs_rep_bert_atba_t43_s42 vs tifs_aa_bert_clean_t43",
                "note": f"weights sha256 identical: {cpr.get('identical')}"}
        add(rows, st, "N1", {**meta, "trigger": "atba (phrase)"}, rp["preds_null"], aa["controls"]["atba_phrase_sst2"]["c"])
        add(rows, st, "N1", {**meta, "trigger": "untriggered"}, rp["preds_c_clean"], aa["Cc"])
        # Only 'insert' controls: their texts do not depend on the run's attack. The
        # 'substitute' controls replace the span the run's own trigger inserted
        # (ATBA prefix here, BadNets rare words in the A/A run), so their inputs differ.
        for name in sorted(rp["controls"]):
            if name not in aa["controls"]:
                continue
            s1, s2 = rp["controls"][name]["spec"], aa["controls"][name]["spec"]
            same_spec = all(s1.get(k) == s2.get(k) for k in ("words", "sentence", "position", "num_triggers"))
            if s1.get("mode", "insert") != "insert" or s2.get("mode", "insert") != "insert" or not same_spec:
                continue
            add(rows, st, "N1", {**meta, "trigger": f"control:{name}"}, rp["controls"][name]["c"],
                aa["controls"][name]["c"])
        # the t43 clean teacher itself, scored by the two jobs on the ATBA phrase
        add(rows, st, "N3", {**meta, "trigger": "atba (phrase)", "kd": "teacher", "seed": "-", "corpus": "-",
                             "a_label": "clean-base teacher t43 (rep job)", "b_label": "clean-base teacher t43 (A/A job)",
                             "a_model": "teacher|bert|sst2|clean-base|t43", "b_model": "teacher|bert|sst2|clean-base|t43",
                             "note": "teacher re-scored in two Stage 2 jobs"},
            rp["teacher_c_trig"], aa["controls"]["atba_phrase_sst2"]["teacher_c"], pool_kd="logit")
    # N2: tifs_b_bert_sst2_s43 re-evaluated on GPU0 (A9). The GPU0 output folder is not part of
    # the repository; results/tifs/addendum/gpu0_eval_determinism.json records sha256_gpu0 and
    # sha256_gpu1 of all 12 result files. Each official file's sha256 now is checked against
    # sha256_gpu1; where it matches and sha256_gpu0 == sha256_gpu1, the GPU0 hit vectors are the
    # official file's bytes. With --gpu0-runs DIR the GPU0 files are re-read instead (their sha256
    # checked against sha256_gpu0).
    def sha256(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    gj = json.load(open(GPU0_JSON))
    n2_run = gj["run"]
    off_dir = os.path.join(RUNS, n2_run, "artifacts", "results")
    rec = {f["file"]: f for f in gj["files"]}
    official_ok = {name: os.path.exists(os.path.join(off_dir, name)) and sha256(os.path.join(off_dir, name)) == f["sha256_gpu1"]
                   for name, f in rec.items()}
    gpus_equal = {name: f["sha256_gpu0"] == f["sha256_gpu1"] for name, f in rec.items()}
    off = {(c["run"], c["attack"], c["kd"]): c for c in cells}
    g0 = {}
    if GPU0_RUNS:
        gdir = os.path.join(GPU0_RUNS, n2_run, "artifacts", "results")
        gpu0_ok = {name: os.path.exists(os.path.join(gdir, name)) and sha256(os.path.join(gdir, name)) == f["sha256_gpu0"]
                   for name, f in rec.items()}
        g0 = {(g["run"], g["attack"], g["kd"]): g for g in A.load_cells(GPU0_RUNS) if g["run"] == n2_run}
        source = f"GPU0 files re-read from --gpu0-runs ({sum(gpu0_ok.values())} of {len(rec)} match sha256_gpu0)"
    else:
        gpu0_ok = None
        source = "sha256 record: GPU0 file bytes == official file bytes (gpu0_eval_determinism.json)"
    n_ok = sum(official_ok[x] and gpus_equal[x] and (gpu0_ok is None or gpu0_ok[x]) for x in rec)
    for name in sorted(rec):
        if not name.endswith(".preds.json"):
            continue
        res_name = name[: -len(".preds.json")] + ".json"
        if not all(official_ok.get(x) and gpus_equal.get(x) and (gpu0_ok is None or gpu0_ok.get(x))
                   for x in (name, res_name)):
            print(f"N2: skipped {name}: sha256 record does not hold for this file now")
            continue
        r_ = json.load(open(os.path.join(off_dir, res_name)))
        o = off.get((n2_run, r_["attack_method"], r_["kd_type"]))
        if o is None:
            continue
        if GPU0_RUNS and (n2_run, o["attack"], o["kd"]) not in g0:
            print(f"N2: skipped {name}: no GPU0 cell in --gpu0-runs")
            continue
        g = g0[(n2_run, o["attack"], o["kd"])] if GPU0_RUNS else o
        meta = {"fam": o["fam"], "dataset": o["dataset"], "kd": o["kd"], "corpus": "clean", "trigger": o["attack"],
                "seed": o["student_seed"], "run": o["run"], "gpu0_source": source,
                "note": f"{n_ok} of {len(rec)} recorded files: official sha256 now == sha256_gpu1 == sha256_gpu0"}
        add(rows, st, "N2", {**meta, "a_label": "P (GPU0 re-evaluation)", "b_label": "P (official GPU1)",
                             "a_model": mid(o["run"], o["kd"], f"P:{o['attack']}", o["student_seed"], o["dataset"]),
                             "b_model": mid(o["run"], o["kd"], f"P:{o['attack']}", o["student_seed"], o["dataset"])},
            g["P"], o["P"], precision=GPU0_VS_GPU1)
        add(rows, st, "N2", {**meta, "a_label": "C (GPU0 re-evaluation)", "b_label": "C (official GPU1)",
                             "a_model": mid(o["run"], o["kd"], f"C:{clean_tag(o['attack'])}", o["student_seed"], o["dataset"]),
                             "b_model": mid(o["run"], o["kd"], f"C:{clean_tag(o['attack'])}", o["student_seed"], o["dataset"])},
            g["C"], o["C"], precision=GPU0_VS_GPU1)
    # N3: one cell per (run, teacher pair, trigger); the same teacher scored by
    # different Stage 2 jobs (runs) on the same triggered inputs.
    seen = {}
    for c in cells:
        if c["single"] or c["kd"] == "ce":
            continue
        key = (c["fam"], c["dataset"], c["attack"], c["teacher_seed"])
        seen.setdefault(key, {})
        seen[key].setdefault(c["run"], c)
    for key, by_run in sorted(seen.items()):
        runs = sorted(by_run)
        if len(runs) < 2:
            continue
        first = by_run[runs[0]]
        for r in runs[1:]:
            c = by_run[r]
            meta = {"fam": key[0], "dataset": key[1], "kd": "teacher", "corpus": "-", "trigger": key[2],
                    "seed": "-", "run": f"{r} vs {runs[0]}", "note": "same teacher model, separate Stage 2 jobs"}
            tid = f"teacher|{key[0]}|{key[1]}|t{key[3]}"
            add(rows, st, "N3", {**meta, "a_label": f"poisoned teacher t{key[3]} ({r})",
                                 "b_label": f"poisoned teacher t{key[3]} ({runs[0]})",
                                 "a_model": f"{tid}|P:{key[2]}", "b_model": f"{tid}|P:{key[2]}"},
                c["TP"], first["TP"], pool_kd="logit")
            add(rows, st, "N3", {**meta, "a_label": f"clean teacher t{key[3]} ({r})",
                                 "b_label": f"clean teacher t{key[3]} ({runs[0]})",
                                 "a_model": f"{tid}|C:{clean_tag(key[2])}", "b_model": f"{tid}|C:{clean_tag(key[2])}"},
                c["TC"], first["TC"], pool_kd="logit")

    # ── CPU fp32 re-scores (A3.4) ──────────────────────────────────
    # Both official cross-score files are required (results/tifs/addendum); there is no fallback copy.
    cross_paths = [CROSS_SST2, CROSS_AGNEWS]
    n_null_by_seed = {}   # (dataset, kd, trigger) -> {seed: number of null students scored at that seed}
    words_by_ds = {}
    for path in cross_paths:
        X = load_cross(path)
        fam, ds = X["meta"]["fam"], X["meta"]["dataset"]
        words = words_by_ds.setdefault(ds, trigger_words(ds))
        for atk in sorted(X["hits"]):
            hits = X["hits"][atk]
            for kd in X["meta"]["kd"]:
                seeds, own, ref, others = cross_keys(X, atk, kd)
                if seeds:
                    n_null_by_seed[(ds, kd, atk)] = {s: sum(f"{b}/s{s}" in hits for b in others) for s in seeds}
                for s in seeds:
                    cell = st.groups.get((fam, ds, kd, atk), {}).get(s)
                    base = {"fam": fam, "dataset": ds, "kd": kd, "corpus": "clean", "trigger": atk, "seed": s,
                            "run": os.path.basename(path), "null_class": null_class_of(atk, ds, st)}
                    # T6: own P vs matched C, both CPU fp32
                    add(rows, st, "T6", {**base, "a_label": own[s].split("/", 1)[1], "b_label": ref[s].split("/", 1)[1],
                                         "a_model": cross_mid(X, own[s]), "b_model": cross_mid(X, ref[s])},
                        hits[own[s]], hits[ref[s]], precision=CPU)
                    # N4: same student, CPU vs official
                    if cell is not None:
                        add(rows, st, "N4", {**base, "a_label": f"P {own[s].split('/', 1)[1]} (CPU)", "b_label": "same P (official)",
                                             "a_model": cross_mid(X, own[s]), "b_model": cross_mid(X, own[s])},
                            hits[own[s]], cell["P"], precision=CPU_VS_OFFICIAL)
                        add(rows, st, "N4", {**base, "a_label": f"C {ref[s].split('/', 1)[1]} (CPU)", "b_label": "same C (official)",
                                             "a_model": cross_mid(X, ref[s]), "b_model": cross_mid(X, ref[s])},
                            hits[ref[s]], cell["C"], precision=CPU_VS_OFFICIAL)
                    # X1: every student without this trigger's backdoor vs the matched C
                    for b in others:
                        key = f"{b}/s{s}"
                        if key in hits:
                            add(rows, st, "X1", {**base, "a_label": key.split("/", 1)[1], "b_label": ref[s].split("/", 1)[1],
                                                 "a_model": cross_mid(X, key), "b_model": cross_mid(X, ref[s]),
                                                 "trigger_word_overlap": overlap_label(words, atk, key.split("/", 1)[1])},
                                hits[key], hits[ref[s]], precision=CPU)
            # R (CE-only, seed 42): CPU vs official
            rkey = "ce/ce/ce/s42"
            rc = st.refs.get(("G", fam, ds, atk, "clean", 42))
            if rkey in hits and rc is not None:
                add(rows, st, "N4", {"fam": fam, "dataset": ds, "kd": "ce", "corpus": "clean", "trigger": atk, "seed": 42,
                                     "run": os.path.basename(path), "a_label": "R ce s42 (CPU)", "b_label": "same R (official)",
                                     "a_model": cross_mid(X, rkey), "b_model": cross_mid(X, rkey)},
                    hits[rkey], rc["P"], precision=CPU_VS_OFFICIAL, pool_kd="logit")
    null_rows, _ = AD.cross_tables(st, cross_paths)
    for r in null_rows:
        if r["role"] == "own P":
            # the like-for-like comparator of X2: the attack's own P under the same formula and precision.
            # n_null_students is the frozen distinct-tag count; n_null_students_by_seed (extra column) is the
            # number of null students scored at each seed, i.e. what each seed's excess averages over.
            nb = n_null_by_seed.get((r["dataset"], r["kd"], r["trigger"]), {})
            rows.append({"source_code": "T6f", "source": SRC["T6f"], "extra_source": False, "fam": r["fam"],
                         "dataset": r["dataset"], "kd": r["kd"], "corpus": "clean", "corpus_group": "clean corpus",
                         "trigger": r["trigger"], "seed": r["seeds"], "a_label": r["student"], "b_label": r["vs"],
                         "precision": CPU, "k": r["k"], "lo": r["lo"], "hi": r["hi"], **{v: r[v] for v in VKEYS},
                         "method": r["method"], "variance_from": r["variance_from"],
                         "excess_over_null_students": r.get("excess_mean"), "n_null_students": r.get("n_null_students"),
                         **{k: v for k, v in r.items() if k.startswith("k_s") or k.startswith("excess_s")},
                         "null_class": null_class_of(r["trigger"], r["dataset"], st),
                         "n_null_students_by_seed": ",".join(f"s{s}:{n}" for s, n in sorted(nb.items()))})
            continue
        if r["role"] != "null":
            continue
        rows.append({"source_code": "X2", "source": SRC["X2"], "extra_source": False, "fam": r["fam"],
                     "dataset": r["dataset"], "kd": r["kd"], "corpus": "clean", "corpus_group": "clean corpus",
                     "trigger": r["trigger"], "seed": r["seeds"], "a_label": r["student"],
                     "b_label": r["vs"], "precision": CPU, "k": r["k"], "lo": r["lo"], "hi": r["hi"],
                     **{v: r[v] for v in VKEYS}, "method": r["method"], "variance_from": r["variance_from"],
                     **{k: v for k, v in r.items() if k.startswith("k_s")},
                     "null_class": null_class_of(r["trigger"], r["dataset"], st),
                     "trigger_word_overlap": overlap_label(words_by_ds.get(r["dataset"], {}), r["trigger"], r["student"])})

    # ── same teacher, different student seed ───────────────────────
    def seed_pairs(code_c, code_p, fam, ds, kd, atk, by_seed, tlabel, note=""):
        q = "" if tlabel == "t42" else "'"    # primes mark the students of another build (A/A, A2)
        for s, t in itertools.combinations(sorted(by_seed), 2):
            cs, ct = by_seed[s], by_seed[t]
            base = {"fam": fam, "dataset": ds, "kd": kd, "corpus": "clean", "trigger": atk, "seed": f"{t}-{s}",
                    "run": f"{ct['run']} vs {cs['run']}", "note": note, "null_class": null_class_of(atk, ds, st)}
            add(rows, st, code_c, {**base, "a_label": f"C{q}[{clean_tag(atk)} {tlabel}] s{t}",
                                   "b_label": f"C{q}[{clean_tag(atk)} {tlabel}] s{s}",
                                   "a_model": mid(ct["run"], kd, f"C:{clean_tag(atk)}", t, ds),
                                   "b_model": mid(cs["run"], kd, f"C:{clean_tag(atk)}", s, ds)},
                ct["C"], cs["C"], pool_kd="logit")
            add(rows, st, code_p, {**base, "a_label": f"P{q}[{atk} {tlabel}] s{t}", "b_label": f"P{q}[{atk} {tlabel}] s{s}",
                                   "a_model": mid(ct["run"], kd, f"P:{atk}", t, ds),
                                   "b_model": mid(cs["run"], kd, f"P:{atk}", s, ds)},
                ct["P"], cs["P"], pool_kd="logit")

    # battery runs (tifs_b_<fam>_<ds>, _s43, _s44): every clean-corpus cell with >= 2 student seeds
    for (fam, ds, kd, atk), by_seed in sorted(st.groups.items()):
        if len(by_seed) >= 2:
            seed_pairs("S1", "S2", fam, ds, kd, atk, by_seed, "t42")
    hg, _ = hidden_groups()
    for (fam, ds, kd, atk), by_seed in sorted(hg.items()):
        if len(by_seed) >= 2:
            seed_pairs("S1h", "S2h", fam, ds, kd, atk, by_seed, "t42",
                       note="A8 hidden-state KD (admitted seeds); transported reading with the logit pool")
    if len(rep) >= 2:
        rep_by = {s: {"P": v["P"], "C": v["C"], "run": v["run"]} for s, v in rep.items()}
        seed_pairs("S1r", "S2r", "bert", "sst2", "logit", "atba", rep_by, "t43",
                   note="A2 replicate build (teacher seed 43)")

    # ── 'feature' - logit ──────────────────────────────────────────
    by = {(c["block"], c["fam"], c["dataset"], c["attack"], c["kd"], c["corpus"], c["student_seed"]): c
          for c in cells if not c["single"]}
    for (block, fam, ds, atk, kd, corpus, s), cf in sorted(by.items()):
        if kd != "feature":
            continue
        cl = by.get((block, fam, ds, atk, "logit", corpus, s))
        if cl is None:
            continue
        base = {"fam": fam, "dataset": ds, "kd": "feature-logit", "corpus": corpus, "trigger": atk, "seed": s,
                "run": cf["run"], "null_class": null_class_of(atk, ds, st)}
        add(rows, st, "F1", {**base, "a_label": f"P[{atk}] feature", "b_label": f"P[{atk}] logit",
                             "a_model": mid(cf["run"], "feature", f"P:{atk}", s, ds),
                             "b_model": mid(cl["run"], "logit", f"P:{atk}", s, ds)}, cf["P"], cl["P"],
            pool_kd="feature")
        add(rows, st, "F1", {**base, "a_label": f"C[{clean_tag(atk)}] feature", "b_label": f"C[{clean_tag(atk)}] logit",
                             "a_model": mid(cf["run"], "feature", f"C:{clean_tag(atk)}", s, ds),
                             "b_model": mid(cl["run"], "logit", f"C:{clean_tag(atk)}", s, ds)},
            cf["C"], cl["C"], pool_kd="feature")

    # ── independent builds ─────────────────────────────────────────
    builds = defaultdict(dict)   # (fam, ds, corpus) -> {teacher seed: cell}, BadNets, logit, student seed 42
    for c in cells:
        if c["single"] or c["kd"] != "logit" or c["attack"] != "badnets" or c["student_seed"] != 42:
            continue
        if (c["block"] == "G" and c["teacher_seed"] == 42) or c["block"] == "AA":
            builds[(c["fam"], c["dataset"], c["corpus"])][c["teacher_seed"]] = c
    for (fam, ds, corpus), bt in sorted(builds.items()):
        for t1, t2 in itertools.combinations(sorted(bt), 2):
            c1, c2 = bt[t1], bt[t2]
            base = {"fam": fam, "dataset": ds, "kd": "logit", "corpus": corpus, "trigger": "badnets", "seed": 42,
                    "run": f"{c1['run']} vs {c2['run']}"}
            q1 = "" if t1 == 42 else "'"    # primes mark the A/A builds' students (t43, t44)
            add(rows, st, "B1", {**base, "a_label": f"C{q1}[clean-base t{t1}]", "b_label": f"C'[clean-base t{t2}]",
                                 "a_model": mid(c1["run"], "logit", "C:clean-base", 42, ds),
                                 "b_model": mid(c2["run"], "logit", "C:clean-base", 42, ds)}, c1["C"], c2["C"])
            add(rows, st, "B2", {**base, "a_label": f"P{q1}[badnets t{t1}]", "b_label": f"P'[badnets t{t2}]",
                                 "a_model": mid(c1["run"], "logit", "P:badnets", 42, ds),
                                 "b_model": mid(c2["run"], "logit", "P:badnets", 42, ds)}, c1["P"], c2["P"])
    # ATBA phrase: t42 = the battery ATBA cell's C (seed 42); t43/t44 = A/A clean
    # students on the atba_phrase_<ds> control (same prefix insertion).
    for ds in ("sst2", "agnews"):
        refs = {}
        base42 = st.groups.get(("bert", ds, "logit", "atba"), {}).get(42)
        if base42 is not None:
            refs[42] = (base42["C"], base42["run"])
        for c in cells:
            if (c["block"] == "AA" and c["fam"] == "bert" and c["dataset"] == ds and c["corpus"] == "clean"
                    and not c["single"] and f"atba_phrase_{ds}" in c["controls"]):
                refs[c["teacher_seed"]] = (c["controls"][f"atba_phrase_{ds}"]["c"], c["run"])
        for t1, t2 in itertools.combinations(sorted(refs), 2):
            q1 = "" if t1 == 42 else "'"
            add(rows, st, "B1", {"fam": "bert", "dataset": ds, "kd": "logit", "corpus": "clean", "trigger": "atba (phrase)",
                                 "seed": 42, "run": f"{refs[t1][1]} vs {refs[t2][1]}",
                                 "a_label": f"C{q1}[clean-base t{t1}]", "b_label": f"C'[clean-base t{t2}]",
                                 "a_model": mid(refs[t1][1], "logit", "C:clean-base", 42, ds),
                                 "b_model": mid(refs[t2][1], "logit", "C:clean-base", 42, ds)},
                refs[t1][0], refs[t2][0])
        # contaminated-corpus A/A clean students at the ATBA phrase (t43 vs t44 only)
        cc = {c["teacher_seed"]: c for c in cells if c["block"] == "AA" and c["fam"] == "bert" and c["dataset"] == ds
              and c["corpus"] == "badnets10" and not c["single"] and f"atba_phrase_{ds}" in c["controls"]}
        for t1, t2 in itertools.combinations(sorted(cc), 2):
            add(rows, st, "B1", {"fam": "bert", "dataset": ds, "kd": "logit", "corpus": "badnets10",
                                 "trigger": "atba (phrase)", "seed": 42, "run": f"{cc[t1]['run']} vs {cc[t2]['run']}",
                                 "a_label": f"C'1[clean-base t{t1}]", "b_label": f"C'1[clean-base t{t2}]",
                                 "a_model": mid(cc[t1]["run"], "logit", "C:clean-base", 42, ds),
                                 "b_model": mid(cc[t2]["run"], "logit", "C:clean-base", 42, ds)},
                cc[t1]["controls"][f"atba_phrase_{ds}"]["c"], cc[t2]["controls"][f"atba_phrase_{ds}"]["c"])
    # ATBA SST-2 at student seeds 43/44: battery C (t42) vs A2 replicate C' (t43),
    # and P (t42 poisoned) vs P' (t43 poisoned) at seeds 42-44.
    groups_atba = st.groups.get(("bert", "sst2", "logit", "atba"), {})
    for s, v in sorted(rep.items()):
        cb = groups_atba.get(s)
        if cb is None:
            continue
        base = {"fam": "bert", "dataset": "sst2", "kd": "logit", "corpus": "clean", "trigger": "atba", "seed": s,
                "run": f"{cb['run']} vs {v['run']}"}
        if s != 42:   # s42: the A/A t43 C' above is the same student (N1)
            add(rows, st, "B1", {**base, "a_label": "C[clean-base t42]", "b_label": "C'[clean-base t43] (A2)",
                                 "a_model": mid(cb["run"], "logit", "C:clean-base", s, "sst2"),
                                 "b_model": mid(v["run"], "logit", "C:clean-base", s, "sst2")}, cb["C"], v["C"])
        add(rows, st, "B2", {**base, "a_label": "P[atba t42]", "b_label": "P'[atba t43] (A2)",
                             "a_model": mid(cb["run"], "logit", "P:atba", s, "sst2"),
                             "b_model": mid(v["run"], "logit", "P:atba", s, "sst2")}, cb["P"], v["P"])

    # ── extra: CE coefficient (A10-a), R_B (0.5 CE) - R_C (CE), same seed ──
    rcc = {k: v for k, v in AD.load_rcc(RUNS).items() if v["seed"] == 42}
    untrig_done = set()
    for (fam, ds, atk), b in sorted(rcc.items()):
        rc = st.refs.get(("G", fam, ds, atk, "clean", 42))
        if rc is None:
            continue
        base = {"fam": fam, "dataset": ds, "kd": "ce(0.5)-ce(1.0)", "corpus": "clean", "seed": 42,
                "run": f"{b['run']} vs {rc['run']}", "a_label": "R_B (L = 0.5 CE)", "b_label": "R_C (L = CE)",
                "a_model": mid(b["run"], "ce", "R", 42, ds), "b_model": mid(rc["run"], "ce", "R", 42, ds),
                "note": "extra source (not in the A11.3 list); addendum ce_coef.csv reads the same contrast"}
        add(rows, st, "Q1", {**base, "trigger": atk}, b["R"], rc["P"], pool_kd="logit")
        # the untriggered vectors are the same model on the same inputs for every attack cell: one row per dataset
        if b.get("Rc") is not None and rc.get("Pc") is not None and ds not in untrig_done:
            untrig_done.add(ds)
            add(rows, st, "Q1", {**base, "trigger": "untriggered"}, b["Rc"], rc["Pc"], pool_kd="logit")

    # ── summary ────────────────────────────────────────────────────
    order = list(SRC)
    for r in rows:
        r["trigger_group"] = "atba" if str(r["trigger"]).startswith("atba") else "non-atba"
        r.setdefault("corpus_group", corpus_group(r.get("corpus", "-")))
    summ = []
    by_code = defaultdict(list)
    for r in rows:
        by_code[r["source_code"]].append(r)
    dims = (("dataset",), ("corpus_group",), ("trigger_group",), ("corpus_group", "trigger_group"),
            ("dataset", "trigger_group"))
    for code in order:
        rs_all = by_code.get(code, [])
        if not rs_all:
            continue
        subsets = [({"dataset": "all", "corpus_group": "all", "trigger_group": "all"}, rs_all)]
        seen_members = {frozenset(map(id, rs_all))}
        for dim in dims:
            vals = sorted({tuple(r[d] for d in dim) for r in rs_all})
            for val in vals:
                sub = [r for r in rs_all if tuple(r[d] for d in dim) == val]
                mem = frozenset(map(id, sub))
                if mem in seen_members:
                    continue
                seen_members.add(mem)
                head = {"dataset": "all", "corpus_group": "all", "trigger_group": "all", **dict(zip(dim, val))}
                subsets.append((head, sub))
        basis = ("frozen primary rule" if code == "T2" else "frozen A3.4 formula" if code in ("X2", "T6f")
                 else "Tango 95% (input noise only)")
        def stats_of(rs):
            ks = [r["k"] for r in rs]
            pairs = {tuple(sorted((r.get("a_model") or "", r.get("b_model") or ""))) for r in rs if r.get("a_model")}
            return {"n_contrasts": len(rs), "n_distinct_model_pairs": len(pairs) if pairs else None,
                    "n_structural_null": sum(r.get("null_class") == "structural null" for r in rs),
                    "precision": "; ".join(sorted({r["precision"] for r in rs})), **abs_summary(ks),
                    **{f"share_abs_k_ge@{t:g}": sum(abs(x) >= t for x in ks) / len(ks) for t in TOLS}}

        for head, rs in subsets:
            common = {"source_code": code, "source": SRC[code], "extra_source": code in EXTRA,
                      "source_group": GROUP_OF[code[0]], **head}
            summ.append({**common, "basis": basis, **stats_of(rs), **verdict_counts(rs)})
            tr = [r for r in rs if r.get("transported_verdict@0.05")]
            if tr and code not in ("T2", "X2", "T6f"):
                summ.append({**common, "basis": "transported: k +- t_df sqrt(SE_Tango^2 + s2_pool), frozen pool "
                                                "(structural-null teacher swaps keep Tango)",
                             **stats_of(tr),
                             **{k.replace("transported_", ""): v for k, v in verdict_counts(tr, "transported_").items()},
                             "n_transported_tango_kept": sum(r.get("transported_rule") == RULE_STRUCTURAL for r in tr)})
    print("contrasts:", len(rows), "summary rows:", len(summ))
    print("cross files used:", [rel(p) for p in cross_paths])
    from collections import Counter
    print(dict(Counter(r["source_code"] for r in rows)))
    write(rows, "validity_contrasts.csv")
    write(summ, "validity_summary.csv")


if __name__ == "__main__":
    main()
