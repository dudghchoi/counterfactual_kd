#!/usr/bin/env python3
"""A18 independent re-computation (no frozen module, no a15lib): from the Stage 2 prediction files of the A18
runs alone, per student seed: the target rates of P and C on the triggered inputs and k_s = mean(P) - mean(C);
for the cell: k = mean over seeds and s_between = the sample SD of k_s. Compared with a18_cells.csv (em reading,
ATBA). Scope: k, k_s and s_between only (the interval engine is the frozen code's; a18.py runs it).

Only runs whose evaluation job is terminal-ok (runs/tifs/_status/<run>__eval.ok) are read, as a18.py does.

    python independent.py [--runs DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
RUNS = [f"tifs_x18_a_gpt2_agnews_s{s}" for s in (42, 43, 44)]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs", "tifs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tifs", "addendum", "aux", "g12-a18"))
    ns = ap.parse_args()
    ks = {}
    for run in RUNS:
        if not os.path.exists(os.path.join(ns.runs, "_status", f"{run}__eval.ok")):
            continue
        for p in sorted(glob.glob(os.path.join(ns.runs, run, "artifacts", "results", "*.preds.json"))):
            with open(p) as f:
                d = json.load(f)
            with open(p[: -len(".preds.json")] + ".json") as f:
                r = json.load(f)
            if r["kd_type"] != "logit" or r["attack_method"] != "atba":
                continue
            P, C = d["preds_obs"], d["preds_null"]
            ks[int(r["seed"])] = (sum(P) / len(P), sum(C) / len(C))
    cells_csv = os.path.join(ns.out, "a18_cells.csv")
    rows = read_csv(cells_csv) if os.path.exists(cells_csv) else []
    out, ok_all = [], True
    for r in rows:
        if r.get("reading") != "em" or r.get("attack") != "atba":
            continue
        seeds = [int(s) for s in r["seeds"].split(",")]
        if not all(s in ks for s in seeds):
            continue
        kss = [ks[s][0] - ks[s][1] for s in seeds]
        k = sum(kss) / len(kss)
        sb = math.sqrt(sum((x - k) ** 2 for x in kss) / (len(kss) - 1)) if len(kss) > 1 else float("nan")
        got = float(r["k"])
        ok = abs(k - got) < 1e-12
        ok_all &= ok
        out.append({"reading": "em", "dataset": r["dataset"], "attack": "atba", "seeds": r["seeds"],
                    "k_independent": k, "k_a18": got, "k_match": ok,
                    "A_P_by_seed": ";".join(f"{ks[s][0]:.6f}" for s in seeds),
                    "A_C_by_seed": ";".join(f"{ks[s][1]:.6f}" for s in seeds),
                    "k_s": ";".join(f"{x:.6f}" for x in kss), "s_between_independent": sb})
    with open(os.path.join(ns.out, "independent.csv"), "w", newline="") as f:
        if out:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    print(f"independent: {len(out)} cells, all k match: {ok_all and bool(out)}")
    return 0 if (ok_all and out) else 1


if __name__ == "__main__":
    sys.exit(main())
