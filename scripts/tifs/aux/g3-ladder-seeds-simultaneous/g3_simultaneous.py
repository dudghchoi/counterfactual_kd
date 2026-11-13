#!/usr/bin/env python3
"""A11 item 9 (동시 구간, S5): Bonferroni simultaneous intervals over the 31 global-Holm cells.

Post hoc / registered auxiliary table (docs/tifs_addendum.md A11.9). It does not change any
primary verdict, battery decision or the paper-wide Holm (transfers_global). Inputs: runs/tifs
(Stage 2 predictions) and results/tifs/addendum/global_holm.csv (addendum.py).

Each cell keeps its primary estimate, SE and df (analyze.primary_table); the interval is the
frozen t_contrast at confidence 1 - 0.05/31 (two-sided; each side 0.025/31), so the 31
intervals hold jointly with probability >= 0.95. The two structural EP SST-2 cells, decided on
seed 42's Tango interval by the frozen rule, get the Tango score interval at the same
confidence (paired_contrast(..., confidence=1 - 0.05/31)). Verdicts read these intervals with
the frozen rule (EQUIVALENT iff -delta <= lo and hi <= delta; POSITIVE iff lo > delta;
NEGATIVE iff hi < -delta). POSITIVE under them is the same event as the Bonferroni-adjusted
one-sided p(k > delta) * 31 < 0.025. Holm (the registered procedure for 'X transfers') is a
step-down test and has no interval of this estimate +- c * SE form; S5's "Holm-adjusted
simultaneous intervals" are therefore replaced by Bonferroni, which is never more powerful
than Holm (every Bonferroni POSITIVE is a Holm transfers_global cell, checked below).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # before any local import: no __pycache__ in the repository

import os  # noqa: E402

from scipy.stats import norm  # noqa: E402
from scipy.stats import t as tdist  # noqa: E402

from g3_common import (  # noqa: E402
    AD, ALPHA, CELL, RUNS, TOLS, close, family31, holm, out_dir, paired_contrast, t_contrast,
    tkey, write_csv,
)

ALPHA_FW = 0.05


def main():
    out = out_dir(__doc__.splitlines()[0])
    st = AD.State(RUNS)
    qual, gh = family31(st)
    m = len(qual)
    conf = 1.0 - ALPHA_FW / m
    checks = []
    gh = {tuple(r[c] for c in CELL): r for r in gh}
    holm_adj = {t: holm([r[f"p_pos@{tkey(t)}"] for r in qual]) for t in TOLS}
    rows = []
    for i, pr in enumerate(qual):
        key = tuple(pr[c] for c in CELL)
        row = {**dict(zip(CELL, key)), "method": pr["method"], "seeds": pr["seeds"], "k": pr["k"],
               "se": pr.get("se"), "df": pr.get("df"), "lo95": pr["lo"], "hi95": pr["hi"],
               "family_size": m, "confidence": conf}
        if pr["method"] == "structural":
            c = st.groups[key][42]
            res = paired_contrast(c["P"], c["C"], TOLS, confidence=conf)
            q = float(norm.ppf(0.5 + conf / 2))
            row.update(interval="Tango score (seed 42), as the frozen structural rule", crit=q,
                       n10=res["n10"], n01=res["n01"])
            base = paired_contrast(c["P"], c["C"], TOLS)
            checks.append({"check": f"structural {'/'.join(key)}: Tango 95% == primary",
                           "ok": close(base["lo"], pr["lo"]) and close(base["hi"], pr["hi"])})
        else:
            res = t_contrast(pr["k"], pr["se"], pr["df"], TOLS, confidence=conf)
            q = float(tdist.ppf(0.5 + conf / 2, pr["df"]))
            row.update(interval="t_contrast(k, se, df, confidence=1-0.05/31)", crit=q)
            checks.append({"check": f"{'/'.join(key)}: half-width == crit * se",
                           "ok": close(res["hi"] - pr["k"], q * pr["se"], 1e-9)})
        row.update(lo_bonf=res["lo"], hi_bonf=res["hi"], width_ratio=(res["hi"] - res["lo"]) / (pr["hi"] - pr["lo"]),
                   delta_star_bonf=res["delta_star"], delta_star95=pr["delta_star"])
        checks.append({"check": f"{'/'.join(key)}: Bonferroni interval contains the 95% interval",
                       "ok": res["lo"] <= pr["lo"] + 1e-12 and res["hi"] >= pr["hi"] - 1e-12})
        for t in TOLS:
            tk = tkey(t)
            p_pos = pr[f"p_pos@{tk}"]
            p_bonf = min(1.0, m * p_pos)
            vb = res[f"verdict@{tk}"]
            hg = gh[key]
            row[f"verdict95@{tk}"] = pr[f"verdict@{tk}"]
            row[f"verdict_bonf@{tk}"] = vb
            row[f"p_pos@{tk}"] = p_pos
            row[f"p_pos_bonf@{tk}"] = p_bonf
            row[f"p_pos_holm@{tk}"] = holm_adj[t][i]
            row[f"holm_transfers_global@{tk}"] = holm_adj[t][i] < ALPHA
            row[f"bonf_positive@{tk}"] = vb == "POSITIVE"
            checks.append({"check": f"{'/'.join(key)} @{tk}: Holm adj p == global_holm.csv",
                           "ok": close(holm_adj[t][i], hg[f"holm_global@{tk}"])
                           and str(holm_adj[t][i] < ALPHA) == hg[f"transfers_global@{tk}"]})
            if pr["method"] != "structural":
                checks.append({"check": f"{'/'.join(key)} @{tk}: Bonferroni POSITIVE <=> 31 p_pos < 0.025",
                               "ok": (vb == "POSITIVE") == (p_bonf < ALPHA)})
            checks.append({"check": f"{'/'.join(key)} @{tk}: Bonferroni POSITIVE => Holm transfers_global",
                           "ok": (vb != "POSITIVE") or holm_adj[t][i] < ALPHA})
        rows.append(row)

    summary = []
    for t in TOLS:
        tk = tkey(t)
        name = lambda r: "/".join(r[c] for c in CELL)  # noqa: E731
        summary.append({
            "delta": t, "family_size": m, "confidence": conf,
            "primary_positive": sum(r[f"verdict95@{tk}"] == "POSITIVE" for r in rows),
            "holm_transfers_global": sum(r[f"holm_transfers_global@{tk}"] for r in rows),
            "bonferroni_positive": sum(r[f"bonf_positive@{tk}"] for r in rows),
            "primary_equivalent": sum(r[f"verdict95@{tk}"] == "EQUIVALENT" for r in rows),
            "bonferroni_equivalent": sum(r[f"verdict_bonf@{tk}"] == "EQUIVALENT" for r in rows),
            "bonferroni_negative": sum(r[f"verdict_bonf@{tk}"] == "NEGATIVE" for r in rows),
            "holm_cells": ";".join(name(r) for r in rows if r[f"holm_transfers_global@{tk}"]),
            "bonferroni_positive_cells": ";".join(name(r) for r in rows if r[f"bonf_positive@{tk}"]),
            "holm_not_bonferroni": ";".join(name(r) for r in rows
                                            if r[f"holm_transfers_global@{tk}"] and not r[f"bonf_positive@{tk}"]),
            "bonferroni_equivalent_cells": ";".join(name(r) for r in rows if r[f"verdict_bonf@{tk}"] == "EQUIVALENT"),
            "all_31_simultaneously_equivalent": all(r[f"verdict_bonf@{tk}"] == "EQUIVALENT" for r in rows)})
    write_csv(rows, os.path.join(out, "simultaneous.csv"))
    write_csv(summary, os.path.join(out, "simultaneous_summary.csv"))
    write_csv(checks, os.path.join(out, "simultaneous_selfchecks.csv"))
    bad = [c for c in checks if not c["ok"]]
    print(f"simultaneous: {len(rows)} cells, confidence {conf:.6f}; checks {len(checks) - len(bad)}/{len(checks)} ok -> {out}")
    for c in bad:
        print("  FAILED", c)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
