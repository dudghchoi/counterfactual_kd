"""Shared helpers for the A11 items 3, 4, 6, 7 (group g2-validity-reference-flips-builds).

Post hoc / registered auxiliary tables of docs/tifs_addendum.md A11. Nothing
here changes a primary verdict. Every interval comes from a frozen function:
counterfactual_kd.analysis.tifs (paired_contrast, t_contrast,
se_from_interval, joint_bootstrap via analyze.interaction_table) and the frozen
loaders / pools of scripts/tifs/analyze.py and scripts/tifs/addendum.py, which
are imported, never edited.

Paths are repository-relative (ROOT is computed from this file). Inputs:
runs/tifs (predictions, read only), results/tifs/report (frozen analyze.py
output) and results/tifs/addendum (addendum.py output, including the CPU fp32
cross-score files). Outputs go to results/tifs/addendum/aux/<group>/ unless
``--out DIR`` (or the environment variable G2_OUT) names another folder.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys

sys.dont_write_bytecode = True     # no __pycache__ inside the repository
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU only, also when a script runs outside run_all.sh

GROUP = "g2-validity-reference-flips-builds"
HERE = os.path.dirname(os.path.abspath(__file__))
# HERE = ROOT/scripts/tifs/aux/<group>
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RUNS = os.path.join(ROOT, "runs", "tifs")
REPORT = os.path.join(ROOT, "results", "tifs", "report")          # frozen analyze.py output
ADDENDUM = os.path.join(ROOT, "results", "tifs", "addendum")      # addendum.py output
DEFAULT_OUT = os.path.join(ADDENDUM, "aux", GROUP)
CROSS_SST2 = os.path.join(ADDENDUM, "cross_hits_bert_sst2.json")       # A3.4 CPU fp32 re-score, BERT SST-2
CROSS_AGNEWS = os.path.join(ADDENDUM, "cross_hits_bert_agnews.json")   # A3.4 CPU fp32 re-score, BERT AG News
GPU0_JSON = os.path.join(ADDENDUM, "gpu0_eval_determinism.json")       # A9 GPU0 vs GPU1 sha256 record
CPRIME_JSON = os.path.join(ADDENDUM, "cprime_determinism.json")        # A2 C' = A/A C' weight record
FREEZE_MANIFEST = os.path.join(RUNS, "_code", "freeze_20261021T092041Z.sha256")
FINAL_MANIFEST = os.path.join(RUNS, "_code", "final_analysis_20261024T110542Z.sha256")


def _cli():
    """--out / --gpu0-runs are shared by every script of the group (parsed once, here)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--out", default=None)
    p.add_argument("--gpu0-runs", default=None)
    a, _ = p.parse_known_args()
    return a


_ARGS = _cli()
OUT = os.path.abspath(_ARGS.out or os.environ.get("G2_OUT") or DEFAULT_OUT)
# Optional: a runs directory holding the GPU0 re-evaluation of tifs_b_bert_sst2_s43 (A9).
# It is not part of the repository; without it N2 is built from the sha256 record.
GPU0_RUNS = _ARGS.gpu0_runs or os.environ.get("G2_GPU0_RUNS") or None


def cli(description):
    """Argument parser of each script (for --help); the values themselves are read above."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", default=None, help=f"output folder (default {os.path.relpath(DEFAULT_OUT, ROOT)})")
    p.add_argument("--gpu0-runs", default=None,
                   help="optional runs dir with the GPU0 re-evaluation of tifs_b_bert_sst2_s43 (item 3, N2)")
    return p.parse_args()


sys.path.insert(0, os.path.join(ROOT, "scripts", "tifs"))
sys.path.insert(0, ROOT)

import analyze as A  # noqa: E402  (frozen; imported, never modified)
import addendum as AD  # noqa: E402  (imported, never modified)
from counterfactual_kd.analysis.tifs import (  # noqa: E402
    holm, paired_contrast, se_from_interval, t_contrast,
)

TOLS = A.TOLS                      # (0.02, 0.05, 0.10)
ALPHA = A.ALPHA                    # 0.025 one-sided
VKEYS = [f"verdict@{t:g}" for t in TOLS]
VERDICTS = ("EQUIVALENT", "POSITIVE", "NEGATIVE", "INCONCLUSIVE")
DECISIVE = ("EQUIVALENT", "POSITIVE", "NEGATIVE")

OFFICIAL = "official Stage 2, GPU bf16"
CPU = "CPU fp32 re-score (scripts/tifs/cross_score.py)"
CPU_VS_OFFICIAL = "CPU fp32 re-score vs official GPU bf16"
GPU0_VS_GPU1 = "GPU0 bf16 re-evaluation vs official GPU1 bf16"

STATUS = ("post hoc / registered auxiliary table (docs/tifs_addendum.md A11); "
          "it does not change any primary verdict, battery decision or the paper-wide Holm")


def rel(path):
    """Repository-relative path for printing (absolute if outside the repository)."""
    p = os.path.abspath(path)
    return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def corpus_group(corpus):
    """clean / contaminated (badnets10, addsent10) / positive control (kl01, kl10) / '-'."""
    if corpus == "clean":
        return "clean corpus"
    if corpus in ("badnets10", "addsent10", "synbkd10"):
        return "contaminated corpus"
    if corpus in ("kl01", "kl10", "kl10a1"):
        return "positive control"
    return "-"


def tango(a, b, prefix=""):
    """Frozen paired_contrast(a, b) -> flat dict with an optional key prefix."""
    pc = paired_contrast(a, b, TOLS)
    out = {f"{prefix}n": pc["n"], f"{prefix}a_rate": pc["a"], f"{prefix}b_rate": pc["b"],
           f"{prefix}k": pc["k"], f"{prefix}lo": pc["lo"], f"{prefix}hi": pc["hi"],
           f"{prefix}delta_star": pc["delta_star"], f"{prefix}n10": pc["n10"], f"{prefix}n01": pc["n01"]}
    out.update({f"{prefix}{v}": pc[v] for v in VKEYS})
    return out, pc


def transported(pc, pool):
    """Frozen one-seed ('transported') reading: k +- t_df sqrt(SE_Tango^2 + s2_pool)."""
    s2p, dfp = pool["s_between"] ** 2, pool["df"]
    se = (se_from_interval(pc["lo"], pc["hi"]) ** 2 + s2p) ** 0.5
    return t_contrast(pc["k"], se, dfp, TOLS)


def sign(x, eps=1e-12):
    return 0 if abs(x) <= eps else (1 if x > 0 else -1)


def write(rows, name):
    """analyze.write_csv (union of keys, first-seen order) into OUT."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    if rows:
        A.write_csv(rows, path)
    else:
        with open(path, "w") as f:
            f.write("")
    return path


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def quantile(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    pos = q * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def abs_summary(ks, prefix="abs_k_"):
    a = [abs(x) for x in ks]
    return {f"{prefix}mean": mean(a), f"{prefix}median": statistics.median(a) if a else float("nan"),
            f"{prefix}p90": quantile(a, 0.9), f"{prefix}max": max(a) if a else float("nan")}


def verdict_counts(rows, prefix=""):
    """Counts of each verdict at each delta over ``rows`` (keys f'{prefix}verdict@d')."""
    out = {}
    for t in TOLS:
        vk = f"{prefix}verdict@{t:g}"
        vs = [r[vk] for r in rows if r.get(vk) not in (None, "")]
        for v in VERDICTS:
            out[f"{prefix}n_{v[:5].lower()}@{t:g}"] = sum(x == v for x in vs)
        out[f"{prefix}share_not_equivalent@{t:g}"] = (sum(x != "EQUIVALENT" for x in vs) / len(vs)) if vs else None
        out[f"{prefix}share_decisive_nonnull@{t:g}"] = (sum(x in ("POSITIVE", "NEGATIVE") for x in vs) / len(vs)) if vs else None
    return out


def state():
    return AD.State(RUNS)


def hidden_groups():
    """A8 hidden-state KD cells of the admitted seeds, {(fam, ds, 'hidden', attack): {seed: cell}}."""
    from collections import defaultdict
    hid = AD.load_hidden_cells(RUNS)
    hstat = AD.hidden_seed_status(hid) if hid else []
    admitted = {(r["fam"], r["dataset"], r["seed"]) for r in hstat if r["status"] == "admitted"}
    groups = defaultdict(dict)
    for c in hid:
        if (c["fam"], c["dataset"], c["student_seed"]) in admitted:
            groups[(c["fam"], c["dataset"], c["kd"], c["attack"])][c["student_seed"]] = c
    return groups, hstat
