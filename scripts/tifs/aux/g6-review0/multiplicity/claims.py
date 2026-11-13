#!/usr/bin/env python3
"""A14 item 1 — claim type x error control: family-level 'no transfer' claims.

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

For every cell of the 31-cell paper-wide Holm family (the qualified rows of the frozen primary
table) and delta in {0.02, 0.05, 0.10}:
  p_up@d   one-sided p of H0: k >= d (the 'upper bound below d' direction), from the frozen
           analysis.tifs.t_contrast with the cell's primary estimate, SE and df (structural nulls:
           the frozen seed-42 Tango score p of the primary row);
  p_tost@d max of p_up and the mirror test at -d (two-sided equivalence);
  p_pos@d  one-sided p of H0: k <= d (transfer direction).
Each is adjusted over two families with the frozen analysis.tifs.holm and with Bonferroni:
  F31 = the 31 qualified primary cells (the frozen transfers_global family);
  F19 = its 19 logit-KD cells (the headline cells).
A claim holds when the adjusted p < 0.025 (one-sided, the level of every registered decision).
A4 counting classes (paper_tifs/tools/make_numbers.py rule): structural (EP SST-2), dagger (trigger
words in no clean KD-corpus row as whole words; the SST-2 BadNets cells), boundary (SOS SST-2,
embedding-only teacher pair; inside the A4 count, reported apart), counted.

Outputs: claims_cells.csv (cell x delta), claims_family.csv (counts per family x side x procedure x
delta), claims_table.csv (the claim-type x error-control table with its result at each delta).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import os  # noqa: E402

import g6m_common as G  # noqa: E402

A, TOLS, ALPHA = G.A, G.TOLS, G.ALPHA

SIDES = (("below", "p_up"), ("equivalent", "p_tost"), ("transfer", "p_pos"))
PROCS = ("unadjusted", "holm", "bonferroni")


def cell_pvalues(r):
    """(p dict, source) for one primary row: frozen t_contrast on (k, se, df), or the Tango row."""
    if r["method"] == "structural":
        return {f"{p}@{t:g}": r[f"{p}@{t:g}"] for p in ("p_up", "p_tost", "p_pos") for t in TOLS}, \
            "frozen seed-42 Tango score p (primary row)"
    res = G.t_contrast(r["k"], r["se"], r["df"], TOLS)
    for p in ("p_up", "p_tost", "p_pos"):
        for t in TOLS:
            if res[f"{p}@{t:g}"] != r[f"{p}@{t:g}"]:
                raise AssertionError(f"{G.cell_id(r)} {p}@{t}: t_contrast {res[f'{p}@{t:g}']} != primary {r[f'{p}@{t:g}']}")
    return {f"{p}@{t:g}": res[f"{p}@{t:g}"] for p in ("p_up", "p_tost", "p_pos") for t in TOLS}, \
        "analysis.tifs.t_contrast(k, se, df)"


def adjust(ps, proc):
    if proc == "unadjusted":
        return list(ps)
    if proc == "holm":
        return G.holm(ps)
    m = len(ps)
    return [min(1.0, m * p) for p in ps]


def build(st):
    qual = G.qualified_primary(st)
    assert len(qual) == 31, len(qual)
    cells = []
    for r in qual:
        pv, src = cell_pvalues(r)
        cells.append({"r": r, "p": pv, "src": src, "cls": G.a4_class(r)})
    fams = {"F31": list(range(len(cells))), "F19": [i for i, c in enumerate(cells) if c["r"]["kd"] == "logit"]}
    assert len(fams["F19"]) == 19
    adj = {}
    for fname, idx in fams.items():
        for side, pk in SIDES:
            for t in TOLS:
                ps = [cells[i]["p"][f"{pk}@{t:g}"] for i in idx]
                for proc in PROCS:
                    a = adjust(ps, proc)
                    for i, v in zip(idx, a):
                        adj[(fname, side, proc, t, i)] = v
    return cells, fams, adj


def rows_cells(cells, fams, adj):
    out = []
    for i, c in enumerate(cells):
        r = c["r"]
        for t in TOLS:
            row = {"cell": G.cell_id(r), "display": G.display(dict(r, a4_class=c["cls"])),
                   "fam": r["fam"], "dataset": r["dataset"], "kd": r["kd"], "attack": r["attack"],
                   "method": r["method"], "a4_class": c["cls"], "in_F19": r["kd"] == "logit",
                   "k": r["k"], "se": r.get("se", ""), "df": r.get("df", ""), "lo": r["lo"], "hi": r["hi"],
                   "delta": t, "verdict": r[f"verdict@{t:g}"], "p_source": c["src"],
                   "p_up": c["p"][f"p_up@{t:g}"], "p_tost": c["p"][f"p_tost@{t:g}"], "p_pos": c["p"][f"p_pos@{t:g}"]}
            for fname in ("F31", "F19"):
                for side, pk in SIDES:
                    for proc in ("holm", "bonferroni"):
                        key = (fname, side, proc, t, i)
                        col = f"{proc}_{fname}_{pk}"
                        row[col] = adj.get(key, "")
                        row[f"{proc}_{fname}_{side}"] = (adj[key] < ALPHA) if key in adj else ""
            out.append(row)
    return out


def rows_family(cells, fams, adj):
    out = []
    for fname, idx in fams.items():
        for side, pk in SIDES:
            for proc in PROCS:
                for t in TOLS:
                    hit = [i for i in idx if adj[(fname, side, proc, t, i)] < ALPHA]
                    cls = [cells[i]["cls"] for i in hit]
                    names = lambda sel: "; ".join(G.display(dict(cells[i]["r"], a4_class=cells[i]["cls"]))
                                                  for i in hit if sel(cells[i]))
                    out.append({
                        "family": fname, "family_size": len(idx), "side": side, "p_column": pk,
                        "procedure": proc, "delta": t,
                        "level_per_test": (ALPHA if proc == "unadjusted" else
                                           f"Holm step-down at {ALPHA:g}" if proc == "holm" else ALPHA / len(idx)),
                        "n_claims": len(hit),
                        "n_structural": cls.count("structural"), "n_dagger": cls.count("dagger"),
                        "n_boundary": cls.count("boundary"), "n_counted_plain": cls.count("counted"),
                        "n_nonstructural": len(hit) - cls.count("structural"),
                        "n_a4_counted": sum(G.a4_counted(x) for x in cls),
                        "n_a4_counted_without_boundary": cls.count("counted"),
                        "n_nonstructural_logit": sum(1 for i in hit if cells[i]["cls"] != "structural"
                                                     and cells[i]["r"]["kd"] == "logit"),
                        "n_a4_counted_logit": sum(1 for i in hit if G.a4_counted(cells[i]["cls"])
                                                  and cells[i]["r"]["kd"] == "logit"),
                        "cells_a4_counted": names(lambda c: G.a4_counted(c["cls"])),
                        "cells_not_counted": names(lambda c: not G.a4_counted(c["cls"])),
                        "cells": "; ".join(G.cell_id(cells[i]["r"]) for i in hit)})
    return out


def cw(n):
    return f"{n} cell" + ("" if int(n) == 1 else "s")


def fam_count(fam_rows, family, side, proc, t, field="n_claims"):
    for r in fam_rows:
        if (r["family"], r["side"], r["procedure"], r["delta"]) == (family, side, proc, t):
            return r[field]
    raise KeyError((family, side, proc, t))


def rows_table(st, cells, fam_rows):
    battery = A.primary_battery(st.primary)
    logit_qual = [r for r in G.qualified_primary(st) if r["kd"] == A.PRIMARY_KD]
    out = []

    def add(claim, control, family, status, per_delta, note=""):
        row = {"claim": claim, "error_control": control, "family": family, "status": status}
        for t in TOLS:
            row[f"result@{t:g}"] = per_delta(t)
        row["note"] = note
        out.append(row)

    add("'X transfers' (paper-wide, cell X)", "Holm over p_pos, 31 cells (transfers_global)", "F31 (31)",
        "registered (tifs_plan.md 0.3 item 7)",
        lambda t: f"{cw(fam_count(fam_rows, 'F31', 'transfer', 'holm', t))}: "
                  + (next(r['cells'] for r in fam_rows if (r['family'], r['side'], r['procedure'], r['delta'])
                          == ('F31', 'transfer', 'holm', t)) or "none"))
    add("battery: at least one attack transfers", "Holm over p_pos within the battery", "each battery",
        "registered",
        lambda t: f"{sum(b[f'decision@{t:g}'] == 'TRANSFERS' for b in battery)} of {len(battery)} batteries TRANSFERS")
    add("battery: no transfer within delta", "intersection-union over p_up within the battery (no adjustment)",
        "each battery", "registered",
        lambda t: f"{sum(b[f'decision@{t:g}'] == 'NO TRANSFER' for b in battery)} of {len(battery)} batteries NO TRANSFER: "
                  + (", ".join(f"{G.FAM[b['fam']]} {G.DS[b['dataset']]} {b['kd']}" for b in battery
                               if b[f'decision@{t:g}'] == 'NO TRANSFER') or "none"))
    add("paper-level no transfer (headline)", "intersection-union over p_up, 19 logit cells (registered at 0.05 only)",
        "F19 (19)", "registered (0.05), not met",
        lambda t: (f"{'met' if max(r[f'p_up@{t:g}'] for r in logit_qual) < ALPHA else 'not met'} "
                   f"(max p_up = {max(r[f'p_up@{t:g}'] for r in logit_qual):.3g}; "
                   f"{sum(r[f'p_up@{t:g}'] >= ALPHA for r in logit_qual)} of 19 cells fail)"
                   + ("" if t == 0.05 else " [not registered at this delta]")))
    for side, label in (("equivalent", "cell equivalent (95% interval inside +-delta)"),
                        ("below", "cell 97.5% one-sided upper bound below delta")):
        add(label, "none (per cell, unadjusted)", "F31 / F19", "registered (per cell verdict)" if side == "equivalent"
            else "per cell, unadjusted (the component test of the registered battery IUT)",
            lambda t, side=side: (f"F31 {fam_count(fam_rows, 'F31', side, 'unadjusted', t)} "
                                  f"(non-structural {fam_count(fam_rows, 'F31', side, 'unadjusted', t, 'n_nonstructural')}, "
                                  f"A4-counted {fam_count(fam_rows, 'F31', side, 'unadjusted', t, 'n_a4_counted')}); "
                                  f"F19 {fam_count(fam_rows, 'F19', side, 'unadjusted', t)} "
                                  f"(non-structural {fam_count(fam_rows, 'F19', side, 'unadjusted', t, 'n_nonstructural')}, "
                                  f"A4-counted {fam_count(fam_rows, 'F19', side, 'unadjusted', t, 'n_a4_counted')})"))
    claim_label = {
        # Holm controls the family-wise error of the rejections; it gives no simultaneous bounds.
        ("holm", "below"): "family-level no transfer: k >= delta rejected per cell with family-wise error control "
                           "(Holm, one-sided 0.025)",
        ("holm", "equivalent"): "family-level no transfer: |k| >= delta rejected per cell (TOST) with family-wise "
                                "error control (Holm, one-sided 0.025)",
        # Bonferroni at 0.025 / m per cell = simultaneous bounds.
        ("bonferroni", "below"): "family-level no transfer: simultaneous one-sided upper bounds below delta "
                                 "(each at 1 - 0.025/m, jointly >= 97.5%)",
        ("bonferroni", "equivalent"): "family-level no transfer: simultaneous two-sided intervals inside +-delta "
                                      "(each at 1 - 0.05/m, jointly >= 95%)",
    }
    for fam in ("F31", "F19"):
        for proc in ("holm", "bonferroni"):
            for side in ("below", "equivalent"):
                add(claim_label[(proc, side)], f"{proc.capitalize()} over {'p_up' if side == 'below' else 'p_tost'}",
                    f"{fam} ({31 if fam == 'F31' else 19})", "post hoc (A14 item 1)",
                    lambda t, fam=fam, proc=proc, side=side: (
                        f"{cw(fam_count(fam_rows, fam, side, proc, t))} (non-structural "
                        f"{fam_count(fam_rows, fam, side, proc, t, 'n_nonstructural')}, A4-counted "
                        f"{fam_count(fam_rows, fam, side, proc, t, 'n_a4_counted')}): "
                        + (next(r['cells_a4_counted'] for r in fam_rows if (r['family'], r['side'], r['procedure'],
                                                                             r['delta']) == (fam, side, proc, t)) or "-")
                        + " | not counted: "
                        + (next(r['cells_not_counted'] for r in fam_rows if (r['family'], r['side'], r['procedure'],
                                                                              r['delta']) == (fam, side, proc, t)) or "-")))
    for fam, proc in (("F19", "holm"), ("F31", "bonferroni"), ("F19", "bonferroni")):
        add("'X transfers' (comparison)", f"{proc.capitalize()} over p_pos", f"{fam} ({31 if fam == 'F31' else 19})",
            "post hoc (A14 item 1, comparison)",
            lambda t, fam=fam, proc=proc: f"{cw(fam_count(fam_rows, fam, 'transfer', proc, t))}: "
            + (next(r['cells'] for r in fam_rows if (r['family'], r['side'], r['procedure'], r['delta'])
                    == (fam, 'transfer', proc, t)) or "none"))
    return out


def main():
    ap = G.add_out_arg(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = ap.parse_args()
    out = G.out_dir(args.out)
    st = G.state()
    cells, fams, adj = build(st)
    rc = rows_cells(cells, fams, adj)
    rf = rows_family(cells, fams, adj)
    rt = rows_table(st, cells, rf)
    G.write_csv(rc, os.path.join(out, "claims_cells.csv"),
                "A14 item 1, per cell of the 31-cell paper-wide Holm family x delta: primary k, SE, df, interval and "
                "verdict; one-sided p_up (H0 k >= delta, 'upper bound below delta'), p_tost (two-sided equivalence) and "
                "p_pos (H0 k <= delta, transfer) from the frozen analysis.tifs.t_contrast with the cell's SE and df "
                "(structural nulls: the frozen seed-42 Tango p of the primary row); each adjusted by the frozen "
                "analysis.tifs.holm and by Bonferroni over F31 (31 qualified cells) and F19 (19 logit cells; blank for "
                "'feature' cells); a claim holds when the adjusted p < 0.025. a4_class: structural / dagger (trigger words "
                "in no clean KD-corpus row) / boundary (SOS SST-2) / counted (paper_tifs/tools/make_numbers.py rule).")
    G.write_csv(rf, os.path.join(out, "claims_family.csv"),
                "A14 item 1, counts of claims per family (F31, F19) x side (below = p_up, equivalent = p_tost, transfer = "
                "p_pos) x procedure (unadjusted at 0.025, Holm, Bonferroni) x delta, split by the A4 class: n_a4_counted = "
                "counted + boundary (A4 counts the SOS SST-2 boundary case and reports it apart; "
                "n_a4_counted_without_boundary leaves it out); structural and dagger cells are listed under "
                "cells_not_counted. Holm counts = cells whose H0 (k >= delta, or |k| >= delta for p_tost; k <= delta for p_pos) "
                "is rejected with family-wise error control (no simultaneous bounds); Bonferroni counts = cells whose "
                "simultaneous one-sided (1 - 0.025/m) upper bound lies below delta (p_up), whose two-sided (1 - 0.05/m) "
                "interval lies inside +-delta (p_tost), or whose one-sided lower bound lies above delta (p_pos).")
    G.write_csv(rt, os.path.join(out, "claims_table.csv"),
                "A14 item 1, claim type x error control (round-0 review issue 4): each claim the paper can make, the "
                "multiplicity control that backs it, its family, whether it was registered or is post hoc (A14), and "
                "its result at each delta from the frozen primary rows (battery rows from the frozen "
                "analyze.primary_battery; headline = analyze.summary's rule, max p_up@0.05 over the 19 logit cells < 0.025).")


if __name__ == "__main__":
    main()
