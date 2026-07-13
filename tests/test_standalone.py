"""Standalone test: counterfactual_kd reproduces the released ROPE verdict from bundled
per-cell predictions, with NO training and NO openbackdoor — only numpy/stdlib.

    pip install -e .        # core only
    pytest tests/ -q        # or: python tests/test_standalone.py
"""
import csv
import glob
import json
import os

import counterfactual_kd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hits(d):
    if "obs_hits" in d:
        return list(d["obs_hits"]), list(d["null_hits"])
    t = d.get("target_label", 1)
    return ([1 if p == t else 0 for p in d["preds_obs"]],
            [1 if p == t else 0 for p in d["preds_null"]])


def test_import_is_lightweight():
    # core import must not require torch/openbackdoor
    assert hasattr(counterfactual_kd, "rope_verdict")
    assert counterfactual_kd.__version__


def test_synthetic_verdicts():
    assert counterfactual_kd.rope_verdict([1] * 90 + [0] * 10, [0] * 100)["verdict"] == "STRONG TRANSFER"
    assert counterfactual_kd.rope_verdict([1] * 50 + [0] * 50, [1] * 50 + [0] * 50)["verdict"] == "NO TRANSFER"


def _dashboard_rows():
    csv_path = os.path.join(HERE, "results", "dashboard", "dashboard_rope.csv")
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def test_bundled_cells_match_dashboard():
    """Every bundled per-cell fixture reproduces its dashboard verdict."""
    rows = _dashboard_rows()
    fixtures = glob.glob(os.path.join(HERE, "results", "cells", "*.preds.json"))
    assert fixtures, "no bundled per-cell fixtures found"
    for fx in fixtures:
        with open(fx) as _fh:
            d = json.load(_fh)
        c = d["cell"]
        oh, nh = _hits(d)
        r = counterfactual_kd.rope_verdict(oh, nh, R=5000, seed=42)
        # match the dashboard row for this cell
        match = [row for row in rows
                 if row["attack"] == c["attack"] and row["family"] == c["family"]
                 and row["dataset"] == c["dataset"] and row["kd_type"] == c["kd_type"]
                 and int(row["seed"]) == c["seed"]
                 and (row.get("corpus") in (None, c.get("corpus")) or "corpus" not in c)]
        assert match, f"no dashboard row for {c}"
        row = match[0]
        assert r["verdict"] == row["verdict"], f"{c}: {r['verdict']} != dashboard {row['verdict']}"
        assert abs(r["p_trig"] - float(row["P_trig"])) < 1e-3
        assert abs(r["c_trig"] - float(row["C_trig"])) < 1e-3


if __name__ == "__main__":
    test_import_is_lightweight()
    test_synthetic_verdicts()
    test_bundled_cells_match_dashboard()
    print("OK — all standalone tests passed")
