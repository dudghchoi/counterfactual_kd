#!/usr/bin/env python3
"""S1 — calibration study of the audit verdict (docs/tifs_plan.md §4.3).

Two scenario sets:

* grid  — n ∈ {211 (v1 SST-2), 570, 912 (SST-2 test set), 5700 (AG News)}, clean triggered rate
          c ∈ {0.05, 0.10, 0.50, 0.90}, discordance pi10+pi01 ∈ {0.02,
          0.06, 0.20}, effect k ∈ {0, 0.025, 0.05, 0.075, 0.10, 0.15}
          (feasible combinations only; the problem is symmetric in sign).
* cells — the 48 published main-grid cells (seed 42, archived per-input
          predictions). For each: the plug-in table (truth = the observed
          table), its null twin (same discordance and mean rate, k = 0)
          and an edge twin (k = 1.5 x tolerance), which together show how
          the verdict behaves at the operating points the paper reports,
          including saturated and zero-discordance cells.

Writes results/tifs/s1/{grid,cells}.csv and a markdown summary.

    python scripts/tifs/s1_calibration.py --reps 4000 --R 2000 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from counterfactual_kd.analysis.calibration import (  # noqa: E402
    PROCEDURES, Scenario, run_scenario, scenario_from_rates,
)

CELLS = os.path.join(ROOT, "server_handoff_20261002", "repair", "zenodo_v1.1", "cells")
TOLS_K = (0.02, 0.05, 0.10)


def grid_scenarios() -> list[Scenario]:
    out = []
    for n in (211, 570, 912, 5700):
        for c in (0.05, 0.10, 0.50, 0.90):
            for d in (0.02, 0.06, 0.20):
                for k in (0.0, 0.025, 0.05, 0.075, 0.10, 0.15):
                    if k > d:
                        continue
                    sc = scenario_from_rates(f"grid_n{n}_c{c}_d{d}_k{k}", n, c, k, d)
                    if min(sc.pi11, sc.pi00) < 0:
                        continue
                    out.append(sc)
    return out


def cell_tables() -> list[tuple[str, tuple[int, int, int, int]]]:
    out = []
    for path in sorted(glob.glob(os.path.join(CELLS, "*_seed42.preds.json"))):
        name = os.path.basename(path)[: -len(".preds.json")]
        if not name.startswith(("clean_", "poison10_")):
            continue
        d = json.load(open(path))
        # preds_* are stored as hit vectors (1 = predicted the target label).
        obs = [int(v) for v in d["preds_obs"]]
        nul = [int(v) for v in d["preds_null"]]
        assert set(obs) | set(nul) <= {0, 1}, path
        n11 = sum(o & c for o, c in zip(obs, nul))
        n10 = sum(o & (1 - c) for o, c in zip(obs, nul))
        n01 = sum((1 - o) & c for o, c in zip(obs, nul))
        out.append((name, (n11, n10, n01, len(obs) - n11 - n10 - n01)))
    return out


def cell_scenarios(tol: float) -> list[Scenario]:
    out = []
    for name, (n11, n10, n01, n00) in cell_tables():
        n = n11 + n10 + n01 + n00
        out.append(Scenario(f"{name}|plugin", n, n11 / n, n10 / n, n01 / n))
        d = (n10 + n01) / n
        c_bar = (2 * n11 + n10 + n01) / (2 * n)
        out.append(scenario_from_rates(f"{name}|null", n, c_bar, 0.0, d))
        k = 1.5 * tol
        d_edge = max(d, k + 0.01)
        c_edge = min(max(c_bar - k / 2, d_edge / 2 - k / 2 + 1e-9),
                     1 - d_edge / 2 - k / 2)
        sc = scenario_from_rates(f"{name}|edge", n, c_edge, k, d_edge)
        if min(sc.pi11, sc.pi00) >= 0:
            out.append(sc)
    return out


def _run(args):
    sc, reps, R, seed = args
    return run_scenario(sc, reps=reps, R=R, seed=seed, tols_k=TOLS_K)


def run_all(scenarios, reps, R, workers, seed0):
    jobs = [(sc, reps, R, seed0 + i) for i, sc in enumerate(scenarios)]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_run, jobs, chunksize=2))


def write_csv(rows, path):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=4000)
    p.add_argument("--R", type=int, default=2000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "tifs", "s1"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grid = run_all(grid_scenarios(), args.reps, args.R, args.workers, 1000)
    write_csv(grid, os.path.join(args.out, "grid.csv"))
    cells = run_all(cell_scenarios(0.05), args.reps, args.R, args.workers, 50000)
    write_csv(cells, os.path.join(args.out, "cells.csv"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"reps": args.reps, "R": args.R, "tols_k": TOLS_K,
                   "tol_h": 0.2, "procedures": PROCEDURES,
                   "n_grid": len(grid), "n_cells": len(cells)}, f, indent=2)
    print(f"grid: {len(grid)} scenarios, cells: {len(cells)} scenarios -> {args.out}")


if __name__ == "__main__":
    main()
