#!/usr/bin/env python3
"""A14 item 12 — S1c config H numbers for the main text (one compact CSV).

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Copies, without recomputing, the registered A11.10 S1c results (results/tifs/addendum/aux/
g4-s1c-s9-s1extra/s1c_cells.csv, s1c_battery.csv, s1c_summary.csv) into one table:
  level = cell     config H (measured heterogeneous training SD T_a, fitted shared fraction w), the
                   frozen rule of each cell (pooled: seed-pooled; transported: one-seed cell with the
                   simulated source pool, c_sd = 1, 40,000 reps): coverage of mu, false 'below delta' at
                   mu = delta (= P(hi < mu), the same for every delta), false 'above delta' at mu = delta,
                   with binomial MC SE; config E (homogeneous T) beside it for comparison
  level = battery  config H per battery and delta: P(false TRANSFERS | all mu = 0) [registered],
                   P(NO TRANSFER | all mu = 0), max_j P(NO TRANSFER | mu_j = delta, others 0) (false
                   NO TRANSFER with one cell at the margin) and its attack, P(false TRANSFERS | all mu =
                   delta), the cell extremes (min coverage, max false 'below'); E beside it; transported
                   batteries also at the SD misspecification c_sd = 1.5 and 2 (c_sd = SD ratio of the
                   target cell's true seed SD to the model's sqrt(F_a E[s2_pool]); variance ratio
                   c_var = c_sd^2 = 2.25 and 4). aux/g4 also has rows at c_var = 1.5 and 2 (c_sd 1.22,
                   1.41) with different rates; they are not copied here.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import os  # noqa: E402

import g6m_common as G  # noqa: E402

G4 = os.path.join(G.AUX, "g4-s1c-s9-s1extra")
CELL_COLS = ("coverage", "coverage_mcse", "false_below_at_mu_eq_delta", "false_below_at_mu_eq_delta_mcse",
             "false_above_at_mu_eq_delta", "false_above_at_mu_eq_delta_mcse", "T_true", "reps")
BAT_COLS = ("false_transfers_all0", "false_transfers_all0_mcse", "no_transfer_all0", "no_transfer_all0_mcse",
            "inconclusive_all0", "false_no_transfer_one_at_d_max", "false_no_transfer_one_at_d_max_mcse",
            "false_no_transfer_one_at_d_argmax", "false_transfers_alld", "false_transfers_alld_mcse", "reps")
SUM_COLS = ("coverage_min", "coverage_min_attack", "coverage_min_mcse", "false_below_max", "false_below_max_attack",
            "false_below_max_mcse", "real_df", "sim_df_median")


def cnum(x):
    return round(float(x), 6) if x not in ("", None) else None


def main():
    ap = G.add_out_arg(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = ap.parse_args()
    out = G.out_dir(args.out)
    cells = G.read_csv(os.path.join(G4, "s1c_cells.csv"))
    batt = G.read_csv(os.path.join(G4, "s1c_battery.csv"))
    summ = G.read_csv(os.path.join(G4, "s1c_summary.csv"))

    def pick(rows, rule, battery, config, c, **kw):
        sel = [r for r in rows if r["rule"] == rule and r["battery"] == battery and r["config"] == config
               and (r["c_sd"] == "" if c is None else (r["c_sd"] != "" and abs(float(r["c_sd"]) - c) < 1e-9))
               and all(r[k] == v for k, v in kw.items())]
        if len(sel) != 1:
            raise KeyError((rule, battery, config, c, kw, len(sel)))
        return sel[0]

    order = [("pooled", "BERT AG News logit"), ("pooled", "BERT SST-2 logit"), ("pooled", "GPT-2 SST-2 logit"),
             ("pooled", "Qwen3 SST-2 logit"), ("transported", "Qwen3 AG News logit"),
             ("transported", "BERT AG News feature"), ("transported", "BERT SST-2 feature")]
    rows = []
    for rule, battery in order:
        c = None if rule == "pooled" else 1.0
        atts = [r["attack"] for r in cells if r["rule"] == rule and r["battery"] == battery and r["config"] == "H"
                and (c is None or (r["c_sd"] != "" and abs(float(r["c_sd"]) - c) < 1e-9))]
        for a in atts:
            h = pick(cells, rule, battery, "H", c, attack=a)
            e = pick(cells, rule, battery, "E", c, attack=a)
            rows.append({"level": "cell", "rule": rule, "battery": battery, "attack": a, "delta": "any",
                         "config": "H", "c_sd": c if c is not None else "", "c_var": c * c if c is not None else "",
                         **{col: h[col] for col in CELL_COLS},
                         **{f"E_{col}": e[col] for col in CELL_COLS if col != "T_true"}})
    for rule, battery in order:
        cs = (None,) if rule == "pooled" else (1.0, 1.5, 2.0)
        for c in cs:
            for t in G.TOLS:
                dt = f"{t:g}"
                h = [r for r in batt if r["rule"] == rule and r["battery"] == battery and r["config"] == "H"
                     and (r["c_sd"] == "" if c is None else (r["c_sd"] != "" and abs(float(r["c_sd"]) - c) < 1e-9))
                     and abs(float(r["delta"]) - t) < 1e-9]
                e = [r for r in batt if r["rule"] == rule and r["battery"] == battery and r["config"] == "E"
                     and (r["c_sd"] == "" if c is None else (r["c_sd"] != "" and abs(float(r["c_sd"]) - c) < 1e-9))
                     and abs(float(r["delta"]) - t) < 1e-9]
                s = [r for r in summ if r["rule"] == rule and r["battery"] == battery and r["config"] == "H"
                     and (r["c_sd"] == "" if c is None else (r["c_sd"] != "" and abs(float(r["c_sd"]) - c) < 1e-9))
                     and abs(float(r["delta"]) - t) < 1e-9]
                assert len(h) == len(e) == len(s) == 1, (rule, battery, c, t, len(h), len(e), len(s))
                h, e, s = h[0], e[0], s[0]
                rows.append({"level": "battery", "rule": rule, "battery": battery, "attack": "", "delta": dt,
                             "config": "H", "c_sd": c if c is not None else "", "c_var": c * c if c is not None else "",
                             **{col: h[col] for col in BAT_COLS}, **{col: s[col] for col in SUM_COLS},
                             **{f"E_{col}": e[col] for col in BAT_COLS if col != "reps"},
                             "main_text": (c in (None, 1.0))})
    G.write_csv(rows, os.path.join(out, "s1c_configH_main.csv"),
                "A14 item 12, S1c (A11.10, aux/g4 s1c_cells.csv, s1c_battery.csv, s1c_summary.csv; copied, not "
                "recomputed) under config H (measured heterogeneous training SD, fitted shared fraction w) with config E "
                "(homogeneous) beside it (E_ columns): level=cell rows give coverage of mu, false 'below delta' at mu = "
                "delta (= P(hi < mu), same for every delta) and false 'above delta' with binomial MC SE (pooled rule "
                "20,000 reps; transported c_sd = 1, H 40,000 / E 10,000 reps); level=battery rows give per delta "
                "P(false TRANSFERS | all mu = 0) [registered], P(NO TRANSFER | all mu = 0), max_j P(NO TRANSFER | mu_j = "
                "delta, others 0) with its attack, P(false TRANSFERS | all mu = delta), and the cell extremes; "
                "transported batteries also at the SD misspecification c_sd (SD ratio of the target's true training SD to "
                "the model's) = 1.5, 2, i.e. variance ratio c_var = 2.25, 4 (not the aux/g4 rows at c_var = 1.5, 2); "
                "main_text = c_sd in {pooled, 1}.")


if __name__ == "__main__":
    main()
