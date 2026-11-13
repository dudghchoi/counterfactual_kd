#!/usr/bin/env python3
"""Re-read per-input prediction files with the TIFS verdict (S3, S4, S6).

For every ``*.preds.json`` matched by ``--glob`` this reports, on the raw
matched effect k = P_trig - C_trig:

* Tango and Newcombe 95% intervals, delta* (smallest tolerance at which the
  cell would be declared equivalent) and the two-axis verdict at each
  tolerance in ``--tols`` (relative axis + HIGH-exposure flags + saturation);
* the published verdict (``rope_verdict``: Cohen's h, paired bootstrap,
  ROPE 0.2) for comparison;
* Δb = P_clean - C_clean and k* = k - Δb with jointly resampled intervals.

Accepts both the archived cell format (``preds_obs``, ``preds_null``,
``preds_c_clean``, ``preds_p_clean``) and schema-2 files written by Stage 2.
In both, the ``preds_*`` vectors are hits (1 = predicted the target label),
not raw labels.

    python scripts/tifs/reanalyze_cells.py \
        --glob 'server_handoff_20261002/repair/zenodo_v1.1/cells/*_seed42.preds.json' \
        --out results/tifs/reanalysis_v1/cells.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from counterfactual_kd.metrics.cohens_h import rope_verdict  # noqa: E402
from counterfactual_kd.metrics.paired import (  # noqa: E402
    audit_verdict, decomposition_bootstrap, newcombe_paired_ci,
)

def hits(values):
    """Prediction files store hit vectors (1 = predicted the target label)."""
    out = [int(v) for v in values]
    if any(v not in (0, 1) for v in out):
        raise ValueError("expected a 0/1 hit vector")
    return out


def analyse(path, tols, R, seed):
    d = json.load(open(path))
    pt, ct = hits(d["preds_obs"]), hits(d["preds_null"])
    row = {"cell": os.path.basename(path)[: -len(".preds.json")], "n": len(pt)}
    meta = d.get("meta") or {}
    for k in ("kd_type", "attack_method", "seed", "teacher_seed", "corpus_seed",
              "student_seed", "single_student"):
        if k in meta:
            row[k] = meta[k]

    rv = rope_verdict(pt, ct, R=R, seed=seed)
    row.update(p_trig=rv["p_trig"], c_trig=rv["c_trig"], k=rv["p_trig"] - rv["c_trig"],
               h=rv["h"], h_lo=rv["hdi_lo"], h_hi=rv["hdi_hi"],
               verdict_published=rv["verdict"])

    first = None
    for tol in tols:
        av = audit_verdict(pt, ct, delta=tol)
        if first is None:
            first = av
        row[f"verdict@{tol:g}"] = av["relative"]
    nc_lo, nc_hi = newcombe_paired_ci(pt, ct)
    row.update(
        n10=first["counts"]["n10"], n01=first["counts"]["n01"],
        tango_lo=first["ci_lower"], tango_hi=first["ci_upper"],
        newcombe_lo=nc_lo, newcombe_hi=nc_hi,
        delta_star=first["delta_star"], excludes_zero=first["excludes_zero"],
        p_exposure=first["p_exposure"], c_exposure=first["c_exposure"],
        saturated=first["saturated"],
    )

    if "preds_p_clean" in d and "preds_c_clean" in d:
        pc, cc = hits(d["preds_p_clean"]), hits(d["preds_c_clean"])
        dec = decomposition_bootstrap(pt, ct, pc, cc, R=R, seed=seed)
        row.update(p_clean=sum(pc) / len(pc), c_clean=sum(cc) / len(cc),
                   delta_b=dec["delta_b"], delta_b_lo=dec["delta_b_ci"][0],
                   delta_b_hi=dec["delta_b_ci"][1], k_star=dec["k_star"],
                   k_star_lo=dec["k_star_ci"][0], k_star_hi=dec["k_star_ci"][1])
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tols", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    p.add_argument("--R", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    paths = sorted(glob.glob(os.path.join(ROOT, args.glob) if not os.path.isabs(args.glob) else args.glob))
    if not paths:
        sys.exit(f"no files match {args.glob}")
    rows = [analyse(pth, args.tols, args.R, args.seed) for pth in paths]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cells -> {args.out}")


if __name__ == "__main__":
    main()
