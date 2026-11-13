#!/usr/bin/env python3
"""A14 item 6 — Holm family sensitivity of transfers_global.

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

transfers_global@d is the frozen rule of analyze.primary_battery: Holm (frozen analysis.tifs.holm)
over p_pos@d of the family, a cell is listed when its adjusted p < 0.025. Families:
  a_frozen31          the 31 qualified primary cells (the registered family; reproduces
                      results/tifs/addendum/global_holm.csv and battery.csv transfers_global)
  b_logit19           the 19 logit-KD cells (the headline cells)
  c_without_feature   the 31 cells without the 12 'feature' cells; the only KD types in the family
                      are logit and 'feature', so this is the same 19-cell set as b (listed so that
                      every A14 item-6 family has its own row)
  d_without_structural  the 31 cells without the two EP SST-2 structural nulls (29 cells)
  e_logit_nonstructural  extra: the 18 non-structural logit cells
p_pos@d are the frozen primary rows' values (t_contrast with the cell's SE and df; Tango for the
structural nulls).

Outputs: holm_family.csv (family x delta: transfers_global list, count, change vs the frozen
family), holm_family_cells.csv (family x delta x cell: p_pos, Holm-adjusted p, listed).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import os  # noqa: E402

import g6m_common as G  # noqa: E402

TOLS, ALPHA = G.TOLS, G.ALPHA

FAMILIES = (
    ("a_frozen31", "the 31 qualified primary cells (registered)", lambda r: True),
    ("b_logit19", "logit-KD cells only", lambda r: r["kd"] == "logit"),
    ("c_without_feature", "without the 'feature' cells (same set as b_logit19)", lambda r: r["kd"] != "feature"),
    ("d_without_structural", "without the structural nulls (EP SST-2, logit and 'feature')",
     lambda r: r["method"] != "structural"),
    ("e_logit_nonstructural", "extra: logit-KD cells without the structural null",
     lambda r: r["kd"] == "logit" and r["method"] != "structural"),
)


def compute(st):
    qual = G.qualified_primary(st)
    fam_rows, cell_rows = [], []
    frozen = {}
    for name, desc, sel in FAMILIES:
        members = [r for r in qual if sel(r)]
        for t in TOLS:
            ps = [r[f"p_pos@{t:g}"] for r in members]
            adj = G.holm(ps)
            hits = [G.cell_id(r) for r, p in zip(members, adj) if p < ALPHA]
            if name == "a_frozen31":
                frozen[t] = set(hits)
            for r, p_raw, p in zip(members, ps, adj):
                cell_rows.append({"family": name, "family_size": len(members), "delta": t, "cell": G.cell_id(r),
                                  "display": G.display(dict(r, a4_class=G.a4_class(r))), "method": r["method"],
                                  "k": r["k"], "lo": r["lo"], "hi": r["hi"], "p_pos": p_raw, "holm_p": p,
                                  "transfers_global": p < ALPHA})
            fam_rows.append({"family": name, "description": desc, "family_size": len(members), "delta": t,
                             "n_transfers_global": len(hits), "transfers_global": "; ".join(hits) or "none",
                             "added_vs_frozen31": "; ".join(sorted(set(hits) - frozen[t])) or "none",
                             "dropped_vs_frozen31": "; ".join(sorted(frozen[t] - set(hits))) or "none",
                             "smallest_unlisted_holm_p": min((p for p in adj if p >= ALPHA), default=float("nan")),
                             "smallest_unlisted_cell": next((G.cell_id(r) for r, p in sorted(zip(members, adj),
                                                             key=lambda x: x[1]) if p >= ALPHA), "")})
    return fam_rows, cell_rows


def main():
    ap = G.add_out_arg(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = ap.parse_args()
    out = G.out_dir(args.out)
    st = G.state()
    fam_rows, cell_rows = compute(st)
    G.write_csv(fam_rows, os.path.join(out, "holm_family.csv"),
                "A14 item 6, transfers_global@delta (frozen analyze.primary_battery rule: frozen analysis.tifs.holm over "
                "p_pos@delta of the family, adjusted p < 0.025) for the frozen 31-cell family and four other families "
                "(logit 19; without 'feature' = the same 19 cells; without the two structural nulls, 29; extra: "
                "non-structural logit 18), with the cells added or dropped relative to the frozen family and the "
                "smallest Holm-adjusted p that is not listed.")
    G.write_csv(cell_rows, os.path.join(out, "holm_family_cells.csv"),
                "A14 item 6, per family x delta x cell: the frozen primary p_pos@delta, its Holm-adjusted p within the "
                "family (frozen analysis.tifs.holm) and whether the cell is in transfers_global (< 0.025).")


if __name__ == "__main__":
    main()
