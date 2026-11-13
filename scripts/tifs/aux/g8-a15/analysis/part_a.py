#!/usr/bin/env python3
"""A15-a: whitespace-preserving re-scores of the decoder batteries (and the BERT control, the contaminated
Qwen3 rows, the A13 G1 seeds).

A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Registered (docs/tifs_addendum.md A15-a; claim rule of the 10/27 20:58 KST amendment):
  * Inputs: runs/tifs/tifs_x15_ws/<run>/<tag>.ws.preds.json (+ .ws.json) and the official re-score
    <tag>.official.preds.json of the same job (scripts/tifs/aux/g8-a15/ws_score.py), read with the frozen
    analyze.load_cells through a symbolic-link view in the Stage 2 layout.
  * Frame 'ws' = the frozen cells with every decoder battery cell (GPT-2 SST-2, Qwen3 SST-2: seeds 42-44;
    Qwen3 AG News: seed 42) given its ws hit vectors (P and C, untriggered, controls). The teacher gate stays
    the frozen one (frozen result JSONs; g8-a15 README), the ws teacher ASR is shown beside it. The frozen
    analyze.primary_table recomputes the decoder pools, primary_battery the battery decisions,
    addendum.global_holm_table the 31-cell Holm with the ws vectors substituted.
  * Verdicts at every delta under the registered / own-variance / claim rules, Welch-df sensitivity;
    'format-dependent' where the ws verdict differs from the frozen one (per rule). All the same ->
    "삽입 형식에 강건".
  * Qwen3 AG News is also read seed-pooled over seeds 42-44 (A13 G1: frozen seed 42 + tifs_x13_qwen3_agnews
    _s43/_s44, own pool), official and ws (a_g1.csv); the official G1 frame must reproduce
    results/tifs/addendum/aux/g7-a13/g1_cells.csv.
  * Contaminated-corpus Qwen3 rows (tifs_g_qwen3_badnets10_s42, tifs_aa_qwen3_badnets10_t43): descriptive
    only (the shortcut: hit rates and their ws - official shift); no verdict.
  * BERT control (tifs_b_bert_sst2): every ws vector must equal the official one.
  * Reproduction: the official re-score of every vector against the frozen preds (n_diff).

Outputs: a_vectors.csv, a_cells.csv, a_battery.csv, a_holm.csv, a_g1.csv, a_contaminated.csv,
a_bert_control.csv, a_shift.csv, a_claims.csv, check_a.csv.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
from collections import defaultdict  # noqa: E402

import a15lib as L  # noqa: E402

A, AD, TOLS, ALPHA = L.A, L.AD, L.TOLS, L.ALPHA
VEC = (("P", "preds_obs"), ("C", "preds_null"), ("Pc", "preds_p_clean"), ("Cc", "preds_c_clean"))
GATE_FIELDS = ("teacher_asr", "teacher_asr_null", "teacher_ca", "teacher_ca_null")
DECODER_KEYS = [("gpt2", "sst2", "logit", "addsent"), ("gpt2", "sst2", "logit", "badnets"),
                ("qwen3", "agnews", "logit", "addsent"), ("qwen3", "agnews", "logit", "badnets"),
                ("qwen3", "sst2", "logit", "addsent"), ("qwen3", "sst2", "logit", "badnets"),
                ("qwen3", "sst2", "logit", "synbkd")]
DECODER_BATTERIES = [("gpt2", "sst2", "logit"), ("qwen3", "agnews", "logit"), ("qwen3", "sst2", "logit")]


def rkey(c):
    return (c["run"], c["dataset"], c["attack"], c["kd"], c["seed"])


def substitute(base, alt, mode):
    """Copy of the official cell ``base`` with ``alt``'s hit vectors; the frozen teacher gate kept."""
    new = dict(base)
    for k in ("P", "C", "Pc", "Cc", "TP", "TC", "TPc", "TCc", "controls"):
        new[k] = alt.get(k)
    for k in GATE_FIELDS:
        new[f"{mode}_{k}"] = alt.get(k)
    new[f"{mode}_ca_p"], new[f"{mode}_ca_c"] = alt.get("ca_p"), alt.get("ca_c")
    new["mode"] = mode
    return new


def official_index(st, runs, x15=None):
    """{run key: official cell} over the frozen cells, the A13 G1 cells and the A15-c cells."""
    idx = {rkey(c): c for c in st.cells}
    for c in L.load_x13_g1(runs):
        idx[rkey(c)] = c
    for c in (x15 if x15 is not None else L.load_x15(runs)):
        if c["part"] == "c":
            idx[rkey(c)] = c
    return idx


def build_ws(st, runs, x15=None):
    """{'ws': cells, 'official': cells, 'idx': official index, 'subst': {mode: {run key: substituted}}}."""
    idx = official_index(st, runs, x15)
    out = {"idx": idx, "subst": {}, "raw": {}}
    for mode in ("ws", "official"):
        cells = L.load_ws(runs, mode)
        out["raw"][mode] = {rkey(c): c for c in cells}
        sub = {}
        for c in cells:
            base = idx.get(rkey(c))
            if base is None:
                raise ValueError(f"{mode} re-score {c['run']}/{c['tag']} aligns with no official cell")
            if len(base["P"]) != len(c["P"]):
                raise ValueError(f"{c['run']}/{c['tag']}: n {len(c['P'])} != official {len(base['P'])}")
            sub[rkey(c)] = substitute(base, c, mode)
        out["subst"][mode] = sub
    return out


def group_key(c):
    """The cell group whose seeds are substituted together (a battery cell over its student seeds)."""
    return (c["fam"], c["dataset"], c["kd"], c["attack"], c["block"], c["corpus"], c["single"])


def substitute_complete(cells, sub, eligible):
    """(cells, complete groups, incomplete groups): every group (``group_key``) of the ``eligible`` cells is
    replaced by its re-scores only when every one of its cells has one; otherwise the official vectors are kept
    for all its seeds, so no cell pools re-scored seeds with official ones."""
    groups = defaultdict(list)
    for c in cells:
        if eligible(c):
            groups[group_key(c)].append(rkey(c))
    complete = {g for g, ks in groups.items() if all(k in sub for k in ks)}
    out = [sub[rkey(c)] if (eligible(c) and group_key(c) in complete) else c for c in cells]
    return out, complete, set(groups) - complete


def frame_cells(st, W, mode, runs_subst=L.WS_DECODER_BATTERY, extra=()):
    """The frozen cells with the decoder battery cells of ``runs_subst`` replaced by their ``mode``
    re-scores (a cell only when every seed of it has a re-score; see ``substitute_complete``), plus ``extra``
    cells."""
    cells, _, _ = substitute_complete(list(st.cells), W["subst"][mode], lambda c: c["run"] in runs_subst)
    return cells + list(extra)


def vector_rows(W):
    rows = []
    for k, off in sorted(W["subst"]["official"].items()):
        ws = W["subst"]["ws"].get(k)
        base = W["idx"][k]
        rawo = W["raw"]["official"][k]
        row = {"run": k[0], "tag": rawo["tag"], "dataset": k[1], "attack": k[2], "kd": k[3], "seed": k[4],
               "n": len(base["P"]), "ws_present": ws is not None}
        for short, name in VEC:
            b, o = base.get(short), off.get(short)
            row[f"{short}_official_reproduces_frozen"] = (b == o) if b is not None else None
            row[f"{short}_official_n_diff"] = (sum(x != y for x, y in zip(b, o)) if b is not None and o is not None
                                              else None)
            if ws is not None:
                w = ws.get(short)
                row[f"{short}_ws_n_diff_vs_frozen"] = (sum(x != y for x, y in zip(b, w))
                                                      if b is not None and w is not None else None)
                row[f"{short}_ws_n_diff_vs_official"] = (sum(x != y for x, y in zip(o, w))
                                                        if o is not None and w is not None else None)
                row[f"{short}_rate_frozen"] = L.mean(b) if b else None
                row[f"{short}_rate_ws"] = L.mean(w) if w else None
        if base["kd"] != "ce":
            row["k_frozen"] = L.mean(base["P"]) - L.mean(base["C"])
            if ws is not None:
                row["k_ws"] = L.mean(ws["P"]) - L.mean(ws["C"])
        cmp_path = os.path.join(os.path.dirname(rawo["preds_path"]), f"{rawo['tag']}.compare.json")
        if os.path.exists(cmp_path):
            X = L.load_json(cmp_path)
            row.update({"compare_n_vectors": X.get("n_vectors"), "compare_n_reproduced": X.get("n_reproduced"),
                        "compare_trig_n_changed_by_ws": (X.get("probes") or {}).get("trig", {}).get("n_changed_by_ws"),
                        "compare_normalised_equal": all(v.get("normalised_equal", True)
                                                        for v in (X.get("probes") or {}).values()),
                        "compare_scored_inputs_match": all(v.get("scored_official_matches", True)
                                                           and v.get("scored_ws_matches", True)
                                                           for v in (X.get("probes") or {}).values())})
        if ws is not None:
            row.update({f"ws_{f}": ws.get(f"ws_{f}") for f in GATE_FIELDS})
            row["ws_teachers_probed"] = bool(ws.get("ws_teacher_asr"))
            if row["ws_teachers_probed"]:
                row["ws_gate_qualified"] = A.qualified({f: ws.get(f"ws_{f}") for f in GATE_FIELDS})
            row["frozen_gate_qualified"] = A.qualified(base)
            row["ca_P_frozen"], row["ca_C_frozen"] = base["ca_p"], base["ca_c"]
            row["ca_P_ws"], row["ca_C_ws"] = ws.get("ws_ca_p"), ws.get("ws_ca_c")
            fl = L.floor_of(k[1])
            row["floor"] = fl
            row["ws_students_above_floor"] = all(x is not None and x > fl for x in
                                                 ((ws.get("ws_ca_p"),) if base["kd"] == "ce" else
                                                  (ws.get("ws_ca_p"), ws.get("ws_ca_c"))))
        rows.append(row)
    return rows


def shift_rows(W, runs_sel, label):
    """Per cell and student: hit rate official (frozen) vs ws, the paired ws - official shift (Tango interval,
    input sampling only; descriptive), and k = P - C (Tango) under both forms."""
    rows = []
    for k, ws in sorted(W["subst"]["ws"].items()):
        if k[0] not in runs_sel:
            continue
        base = W["idx"][k]
        row = {"group": label, "run": k[0], "dataset": k[1], "attack": k[2], "kd": k[3], "seed": k[4],
               "corpus": base.get("corpus"), "block": base.get("block"), "n": len(base["P"])}
        studs = (("R", "P"),) if base["kd"] == "ce" else (("P", "P"), ("C", "C"))
        for name, f in studs:
            sh = L.paired_contrast(ws[f], base[f], TOLS)
            row.update({f"{name}_rate_official": sh["b"], f"{name}_rate_ws": sh["a"],
                        f"{name}_shift_ws_minus_official": sh["k"], f"{name}_shift_lo": sh["lo"],
                        f"{name}_shift_hi": sh["hi"], f"{name}_n_inputs_changed": sh["n10"] + sh["n01"]})
        if base["kd"] != "ce":
            for pre, c in (("official_", base), ("ws_", ws)):
                pc = L.paired_contrast(c["P"], c["C"], TOLS)
                row.update({f"{pre}k": pc["k"], f"{pre}k_tango_lo": pc["lo"], f"{pre}k_tango_hi": pc["hi"]})
        rows.append(row)
    return rows


def g1_frames(st, W, runs):
    """(official G1 frame, ws G1 frame, x13 cells, ws-complete G1 groups): the frozen cells plus the A13 G1 cells,
    so qwen3/agnews/logit gets its own seed-pooled pool (A13 G1 rule); in the ws frame the decoder battery cells
    and the G1 cells are replaced by their ws re-scores, a cell only when all its seeds (42-44) have one
    (``substitute_complete``). No ws frame when no G1 seed has a ws re-score."""
    x13 = L.load_x13_g1(runs)
    base = list(st.cells) + x13
    F_off = L.frame(st, base, "g1_official")
    if not any(rkey(c) in W["subst"]["ws"] for c in x13):
        return F_off, None, x13, set()
    cells, complete, _ = substitute_complete(base, W["subst"]["ws"],
                                             lambda c: c["run"] in L.WS_DECODER_BATTERY or c["run"] in L.WS_G1)
    return F_off, L.frame(st, cells, "g1_ws"), x13, complete


def main():
    ns = L.args()
    st, F0 = L.state(ns.runs)
    x15 = L.load_x15(ns.runs)
    W = build_ws(st, ns.runs, x15)
    out = ns.out
    Fa = L.frame(st, frame_cells(st, W, "ws"), "ws")
    Fo = L.frame(st, frame_cells(st, W, "official"), "official_rescore")
    present = {k[0] for k in W["subst"]["ws"]}
    missing_runs = [r for r in L.WS_RUNS if r not in present]
    checks = []

    def chk(name, ok, note=""):
        checks.append({"check": name, "ok": bool(ok), "note": note})

    sub_ws = W["subst"]["ws"]

    def extra(key, a, b):
        cells = [c for c in st.cells if (c["fam"], c["dataset"], c["kd"], c["attack"]) == key
                 and c["block"] == "G" and c["corpus"] == "clean" and not c["single"]]
        seeds_av = sorted(c["seed"] for c in cells if rkey(c) in sub_ws and c["run"] in L.WS_DECODER_BATTERY)
        complete = len(seeds_av) == len(cells) and bool(cells)
        repro = all(W["subst"]["official"].get(rkey(c), {}).get("P") == c["P"]
                    and W["subst"]["official"].get(rkey(c), {}).get("C") == c["C"] for c in cells)
        o = Fo["prim"].get(key)
        vs = L.changed(o, b, ("reg", "claim"))
        return {"ws_seeds_available": ",".join(map(str, seeds_av)),
                "ws_seeds_substituted": ",".join(map(str, seeds_av)) if complete else "",
                "ws_complete": complete,
                "official_rescore_reproduces_frozen": repro,
                "format_dependent_vs_rescore_reg": vs["reg"], "format_dependent_vs_rescore_claim": vs["claim"],
                "frozen_raw_asr": a["raw_asr"] if a else None, "ws_raw_asr": b["raw_asr"] if b else None,
                "frozen_C_rate": a["C_trig"] if a else None, "ws_C_rate": b["C_trig"] if b else None}

    a_cells = L.comparison_rows(F0, Fa, DECODER_KEYS, "format_dependent", "frozen_", "ws_", extra)
    for r in a_cells:
        if not r["ws_complete"]:
            r["sentence"] = "n/a (ws re-score incomplete: official vectors kept for every seed)"
        elif not (r["format_dependent_reg"] or r["format_dependent_claim"]):
            r["sentence"] = "삽입 형식에 강건"
        else:
            r["sentence"] = "format-dependent (두 판정을 나란히 쓴다)"
    robust = all(r["sentence"] == "삽입 형식에 강건" for r in a_cells)
    jn = L.part_jobs_note(ns.runs, "a")
    L.write_csv(L.with_jobs(a_cells, jn), os.path.join(out, "a_cells.csv"), """
        A15-a, one row per decoder primary cell (GPT-2 SST-2, Qwen3 SST-2, Qwen3 AG News). frozen_* = the frozen
        frame (= primary.csv); ws_* = the frame with the decoder cells' hit vectors from the whitespace-
        preserving re-score (frozen teacher gate). Column families as in b1_cells.csv (reg_, own_, claim,
        pmax_, welch_, F31/F19 family claims). format_dependent_<rule> = deltas where the ws verdict differs
        from the frozen one; format_dependent_vs_rescore_* = against the official re-score of the same job
        (differs from the frozen comparison only if the re-score did not reproduce the frozen vectors);
        ws_complete = every seed of the cell has a ws re-score; sentence = the A15-a sentence rule; a15_jobs =
        status of the part's queue v9 jobs (markers; the cut / blocked / missing jobs). A cell is given its ws
        vectors only when every seed has a ws re-score (ws_seeds_available; ws_seeds_substituted empty otherwise
        and the sentence n/a). sentence: registered or claim-rule verdict differs -> format-dependent; the own
        variance and Welch columns are sensitivity columns and set no label.""")
    bat = L.battery_comparison(F0, Fa, DECODER_BATTERIES, "format_dependent", "frozen_", "ws_")
    L.write_csv(L.with_jobs(bat, jn), os.path.join(out, "a_battery.csv"), """
        A15-a battery decisions of the decoder batteries, frozen and ws frames (registered = frozen
        analyze.primary_battery; own / claim / welch = frozen analyze.decide on those p-values; see
        b1_battery.csv); format_dependent_<rule> = deltas where the decision differs.""")
    L.write_csv(L.with_jobs(L.holm_rows(F0, Fa, "format_dependent", "frozen_", "ws_"), jn),
                os.path.join(out, "a_holm.csv"), """
        A15-a 31-cell Holm (frozen addendum.global_holm_table, p_pos) and F31 / F19 family claims with the ws
        vectors substituted for the decoder cells, beside the frozen ones; format_dependent_<rule> lists the
        family:side@delta flags that change.""")
    # G1 (Qwen3 AG News seeds 42-44)
    F_g1o, F_g1w, x13, g1_complete = g1_frames(st, W, ns.runs)
    g1_keys = [("qwen3", "agnews", "logit", "addsent"), ("qwen3", "agnews", "logit", "badnets")]
    g1 = []
    if F_g1w is not None:
        def g1_extra(key, a, b):
            cs = [c for c in list(st.cells) + x13 if (c["fam"], c["dataset"], c["kd"], c["attack"]) == key
                  and c["block"] == "G" and c["corpus"] == "clean" and not c["single"]]
            av = sorted(c["seed"] for c in cs if rkey(c) in sub_ws)
            comp = any(group_key(c) in g1_complete for c in cs)
            return {"ws_seeds_available": ",".join(map(str, av)), "ws_complete": comp,
                    "note": "" if comp else "ws re-score incomplete: official vectors kept for every seed"}
        g1 = L.comparison_rows(F_g1o, F_g1w, g1_keys, "format_dependent", "official_", "ws_", g1_extra)
        g1 += L.battery_comparison(F_g1o, F_g1w, [("qwen3", "agnews", "logit")], "format_dependent",
                                   "official_", "ws_")
    L.write_csv(L.with_jobs(g1, jn), os.path.join(out, "a_g1.csv"), """
        A15-a on the A13 G1 reading of Qwen3 AG News: seed-pooled over student seeds 42 (tifs_b_qwen3_agnews)
        and 43/44 (tifs_x13_qwen3_agnews_s43/_s44) with the battery's own pool (frozen primary_table on the
        frozen cells plus the G1 cells). official_* = official vectors (reproduces g7-a13 g1_cells.csv), ws_* =
        every seed's ws re-score (the other decoder cells ws as well; a cell only when all three seeds have one,
        ws_complete). Cell rows (attack set) then the battery row (attack empty): column families as in
        a_cells.csv / a_battery.csv. Post hoc (A13), outside the 31-cell family.""")
    # contaminated and BERT control
    L.write_csv(shift_rows(W, L.WS_CONTAMINATED, "contaminated (descriptive)"),
                os.path.join(out, "a_contaminated.csv"), """
        A15-a contaminated-corpus Qwen3 rows (tifs_g_qwen3_badnets10_s42, tifs_aa_qwen3_badnets10_t43; P, C of
        each logit cell and R of each CE cell): hit rate on the official triggered inputs (frozen) and on the
        ws inputs, the paired shift ws - official with its Tango 95% interval (input sampling only) and the
        number of inputs whose hit changes; official_k / ws_k = P - C (Tango). Descriptive (A15-a: these rows
        are not judged); they describe the whitespace shortcut.""")
    bert = []
    for r in vector_rows(W):
        if r["run"] in L.WS_BERT_CONTROL:
            r["unchanged"] = all(r.get(f"{s}_ws_n_diff_vs_frozen") in (0, None) for s, _ in VEC)
            bert.append(r)
    L.write_csv(bert, os.path.join(out, "a_bert_control.csv"), """
        A15-a BERT control (tifs_b_bert_sst2): per tag, inputs where the ws hit vector differs from the frozen
        one (P, C, untriggered Pc, Cc) and from the official re-score; unchanged = all zero (registered: must
        not change; the BERT uncased tokenizer does not see the whitespace).""")
    vrows = vector_rows(W)
    L.write_csv(vrows, os.path.join(out, "a_vectors.csv"), """
        A15-a, one row per re-scored tag: *_official_reproduces_frozen / *_official_n_diff = the official
        re-score of the same job against the frozen Stage 2 vector (P = preds_obs, C = preds_null, Pc / Cc =
        untriggered); *_ws_n_diff_vs_frozen / _vs_official = inputs whose hit changes with the ws inputs;
        rates; k = P - C; compare_* = fields of <tag>.compare.json (vectors reproduced, triggered inputs
        changed by ws, normalised texts equal, scored texts = the texts built); ws_teacher_* and ws_gate =
        teacher ASR / accuracy on the ws inputs when the job probed the teachers (frozen gate kept for every
        verdict); ca_* = Stage 2 clean accuracy, frozen and ws; ws_students_above_floor = the A8 floor (majority
        share + 0.10) on the ws re-score's clean accuracy (the students are the frozen ones).""")
    L.write_csv(shift_rows(W, L.WS_DECODER_BATTERY + L.WS_G1, "decoder battery"),
                os.path.join(out, "a_shift.csv"), """
        A15-a decoder battery cells per seed (descriptive): P and C hit rates official vs ws, the paired ws -
        official shift (Tango interval, input only), inputs changed, and k = P - C (Tango) under both forms.""")
    L.write_csv(L.claim_summary(F0, "frozen") + L.claim_summary(Fa, "ws"), os.path.join(out, "a_claims.csv"),
                "Claim-rule counts per frame and delta (see b1_claims.csv); frames frozen and ws.")
    # checks
    chk("official re-score reproduces every frozen vector (P, C, Pc, Cc)",
        all(r.get(f"{s}_official_reproduces_frozen") in (True, None) for r in vrows for s, _ in VEC),
        ";".join(f"{r['run']}:{r['tag']}" for r in vrows
                 if any(r.get(f"{s}_official_reproduces_frozen") is False for s, _ in VEC)))
    chk("official re-score frame == frozen frame (every primary row k/lo/hi/verdicts)",
        all(Fo["prim"][k][f] == F0["prim"][k][f] for k in F0["prim"]
            for f in ("k", "lo", "hi", "verdict@0.02", "verdict@0.05", "verdict@0.1")))
    chk("BERT control unchanged by ws", all(r["unchanged"] for r in bert) and bool(bert),
        ";".join(r["tag"] for r in bert if not r["unchanged"]) or ("no BERT re-score" if not bert else ""))
    chk("non-decoder primary rows unchanged in the ws frame",
        all(Fa["prim"][k][f] == F0["prim"][k][f] for k in F0["prim"] for f in ("k", "lo", "hi")
            if k[0] == "bert"))
    g7 = os.path.join(L.G7, "g1_cells.csv")
    if os.path.exists(g7):
        gr = {r["attack"]: r for r in L.read_csv(g7)}
        nb = 0
        for key in g1_keys:
            r = F_g1o["prim"][key]
            o = gr[key[3]]
            for f, g in (("k", "own_pool_k"), ("lo", "own_pool_lo"), ("hi", "own_pool_hi"), ("df", "own_pool_df"),
                         *[(f"verdict@{t:g}", f"own_pool_verdict@{t:g}") for t in TOLS]):
                nb += str(r[f]) != o[g]
        chk("A13 G1 official frame reproduces g7-a13 g1_cells.csv (k, lo, hi, df, verdicts; exact text)", nb == 0,
            f"{nb} mismatches")
    chk("ws inputs complete", not missing_runs, L.missing_note(missing_runs))
    chk("ws inputs complete per decoder cell (every seed re-scored; otherwise official vectors kept)",
        all(r["ws_complete"] for r in a_cells),
        ";".join(f"{L.cid(r)} has {r['ws_seeds_available'] or 'none'}" for r in a_cells if not r["ws_complete"]))
    checks.append(L.family_size_row(F0, Fa, "ws"))
    checks += L.pool_identity_rows({"frozen": F0, "ws": Fa, "official_rescore": Fo, "g1_official": F_g1o,
                                    "g1_ws": F_g1w})
    checks += L.excluded_rows(ns.runs)
    L.write_csv(checks, os.path.join(out, "check_a.csv"), "A15-a self-checks ('ws inputs complete' is False "
                "while ws jobs are missing).")
    print(f"part a: {len(W['subst']['ws'])} ws tags; robust to insertion format (all rules): {robust}; "
          f"missing runs: {missing_runs or 'none'}")
    bad = [c for c in checks if not c["ok"] and c.get("kind") != "info" and "inputs complete" not in c["check"]]
    if bad:
        print("  CHECKS FAILED:", "; ".join(c["check"] for c in bad))


if __name__ == "__main__":
    main()
