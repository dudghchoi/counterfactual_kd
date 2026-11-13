"""Shared loading for the g3 auxiliary tables (docs/tifs_addendum.md A11 items 5, 8, 9).

Read-only. It imports the frozen analysis (scripts/tifs/analyze.py,
counterfactual_kd/analysis/tifs.py) and the addendum functions
(scripts/tifs/addendum.py) without modifying them. Inputs:
  runs/tifs/                 Stage 2 prediction files (read by the frozen loader)
  results/tifs/report/       official analyze.py output (frozen rule)
  results/tifs/addendum/     official addendum.py output
Output: results/tifs/addendum/aux/g3-ladder-seeds-simultaneous/ unless --out is given.
Paths are relative to the repository root, found from this file's location
(scripts/tifs/aux/<group>/ is four levels below it).

Run with CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 (run_all.sh does this).
Each g3 script sets sys.dont_write_bytecode before importing this module, so
no __pycache__ is written into the repository even when this check fails.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys

GROUP = "g3-ladder-seeds-simultaneous"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RUNS = os.path.join(ROOT, "runs", "tifs")
REPORT = os.path.join(ROOT, "results", "tifs", "report")
ADDENDUM = os.path.join(ROOT, "results", "tifs", "addendum")
OUT_DEFAULT = os.path.join(ADDENDUM, "aux", GROUP)
AUX_OUT_ROOT = os.path.join(ADDENDUM, "aux")

sys.dont_write_bytecode = True
if not sys.flags.dont_write_bytecode:
    raise SystemExit("run with PYTHONDONTWRITEBYTECODE=1 (nothing may be written inside the repo)")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU only; set before anything could import torch

sys.path.insert(0, os.path.join(ROOT, "scripts", "tifs"))
sys.path.insert(0, ROOT)

import analyze as A  # noqa: E402  (frozen, imported only)
import addendum as AD  # noqa: E402  (imported only; main() is never called)
from counterfactual_kd.analysis.tifs import (  # noqa: E402  (frozen helpers)
    holm, paired_contrast, se_from_interval, t_contrast,
)

if not (os.path.samefile(AD.ROOT, ROOT) and os.path.samefile(os.path.dirname(A.__file__), os.path.join(ROOT, "scripts", "tifs"))):
    raise SystemExit(f"frozen modules not loaded from this repository ({A.__file__}, {AD.ROOT})")

TOLS = A.TOLS          # (0.02, 0.05, 0.10)
ALPHA = A.ALPHA        # 0.025 one-sided
CELL = ("fam", "dataset", "kd", "attack")

# Frozen files (freeze_20261021T092041Z) and the addendum files
# (final_analysis_20261024T110542Z): SHA-256 at the time of the official run.
FROZEN_SHA256 = {
    "docs/tifs_plan.md": "bc9dfd8f87f54d778a47e07b7d49de2c68af5b3df5458729fa4e7bb947f88713",
    "scripts/tifs/analyze.py": "1f70bbf4453bd83a5ec5f9858343549e89485627acb63f62d5e0e95dd5f09ef8",
    "counterfactual_kd/analysis/tifs.py": "cc0e05451bee502d888ab52260eb4576aab056dbe89a7c41fdce07db247dc9ce",
    "results/tifs/structural_nulls.json": "a6dc5091d33b0c355a34136b65e28f29163864aeb0a4d1c060afd0c33656a9b0",
    "scripts/tifs/addendum.py": "d49a912ac05e9cbb802efdbc2268e16734ecc4b507a6eb16063d5f98206d88f1",
    "docs/tifs_addendum.md": "b018f1a88fdc3139eb36c78a5d9d7c42b85666be3416d8d89f59e25175e386c4",
}

# Every file the g3 scripts read besides the prediction files (repo-relative).
INPUTS = (
    "results/tifs/report/primary.csv",
    "results/tifs/report/audit.csv",
    "results/tifs/report/battery.csv",
    "results/tifs/report/battery_by_seed.csv",
    "results/tifs/report/variance.csv",
    "results/tifs/addendum/global_holm.csv",
    "results/tifs/addendum/attack_variance.csv",
    "results/tifs/addendum/reference_atba.csv",
    "results/tifs/addendum/replicate.csv",
    "results/tifs/addendum/decidability.csv",
    "results/tifs/addendum/agnews_plan.csv",
    "results/tifs/addendum/cross_null.csv",
    "results/tifs/addendum/hidden_seeds.csv",
    "results/tifs/addendum/hidden_variance.csv",
    "results/tifs/addendum/hidden_cells.csv",
    "results/tifs/structural_nulls.json",
    "runs/tifs/_status/tifs_a10b_go.failed",
)


def out_dir(description: str) -> str:
    """--out (default results/tifs/addendum/aux/<group>/). Inside the repository only
    results/tifs/addendum/aux/ may be written."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--out", default=OUT_DEFAULT,
                    help=f"output folder (default {os.path.relpath(OUT_DEFAULT, ROOT)})")
    out = os.path.abspath(ap.parse_args().out)
    inside = os.path.commonpath([out, ROOT]) == ROOT
    if inside and os.path.commonpath([out, AUX_OUT_ROOT]) != AUX_OUT_ROOT:
        raise SystemExit(f"refusing to write inside the repository outside {os.path.relpath(AUX_OUT_ROOT, ROOT)}/: {out}")
    os.makedirs(out, exist_ok=True)
    return out


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tkey(t):
    return f"{t:g}"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path):
    if not rows:
        raise ValueError(f"no rows for {path}")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def family31(st):
    """The 31-cell global Holm family: the qualified primary rows, in the frozen order
    (analyze.primary_battery / addendum.global_holm_table), checked against
    results/tifs/addendum/global_holm.csv."""
    qual = [r for r in st.primary if r["teacher_qualified"]]
    ref = read_csv(os.path.join(ADDENDUM, "global_holm.csv"))
    got = [tuple(r[c] for c in CELL) for r in qual]
    want = [tuple(r[c] for c in CELL) for r in ref]
    if got != want:
        raise SystemExit(f"31-cell family differs from global_holm.csv: {set(got) ^ set(want)}")
    return qual, ref


def close(a, b, tol=1e-9):
    if a in (None, "") or b in (None, ""):
        return a in (None, "") and b in (None, "")
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
