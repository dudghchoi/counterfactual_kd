"""Shared helpers of A14 subgroup 'reference' (docs/tifs_addendum.md A14 items 3, 4, 5, 7, 13).

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

The frozen code is imported, never edited:
  scripts/tifs/analyze.py            (load_cells, primary_table, pool_seed_variance, decide, ...)
  scripts/tifs/addendum.py           (State, seed_pooled, cross_tables, load_replicate, ...)
  counterfactual_kd/analysis/tifs.py (paired_contrast, t_contrast, se_from_interval, holm)
  scripts/tifs/cross_score.py        (triggered_loader, strict_synbkd; CPU fp32 scoring)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "tifs"))

import analyze as A  # noqa: E402  (frozen)
import addendum as AD  # noqa: E402  (frozen)
from counterfactual_kd.analysis.tifs import (  # noqa: E402,F401
    holm, paired_contrast, se_from_interval, t_contrast,
)

TOLS = A.TOLS
ALPHA = A.ALPHA
RUNS = os.path.join(ROOT, "runs", "tifs")
OUT_DEFAULT = os.path.join(ROOT, "results", "tifs", "addendum", "aux", "g6-review0", "reference")
ADD = os.path.join(ROOT, "results", "tifs", "addendum")
REPORT = os.path.join(ROOT, "results", "tifs", "report")
CROSS = {ds: os.path.join(ADD, f"cross_hits_bert_{ds}.json") for ds in ("sst2", "agnews")}
STATUS = ("A14 post hoc analysis; changes no primary verdict, battery decision or the "
          "paper-wide Holm")
FROZEN = ["docs/tifs_plan.md", "scripts/tifs/analyze.py", "counterfactual_kd/analysis/tifs.py",
          "results/tifs/structural_nulls.json", "counterfactual_kd/metrics/paired.py",
          "counterfactual_kd/analysis/calibration.py", "scripts/tifs/addendum.py",
          "scripts/tifs/cross_score.py"]
# Battery attacks of BERT SST-2 and AG News (configs/tifs/tifs_b_bert_*.yaml).
ATTACKS = ("badnets", "addsent", "synbkd", "sos", "ep", "atba")
# Tag of each attack's poisoned students and of its trainer-matched clean twin.
P_TAG = {"sst2": {"badnets": "badnets_cf_mn", "addsent": "addsent_sent", "synbkd": "synbkd_trig",
                  "sos": "sos_friends_weekend", "ep": "ep_cf_mn", "atba": "atba_soothing sharing how"},
         "agnews": {"badnets": "badnets_cf_mn", "addsent": "addsent_sent", "synbkd": "synbkd_trig",
                    "sos": "sos_friends_weekend", "ep": "ep_cf_mn",
                    "atba": "atba_infected malicious albums, browser"}}
C_TAG = {a: AD.clean_tag(a) for a in ATTACKS}   # clean-base / clean-sos / clean-ep


def out_dir(argv=None):
    """--out DIR (default: the official output folder of this subgroup)."""
    argv = sys.argv[1:] if argv is None else argv
    if "--out" in argv:
        d = argv[argv.index("--out") + 1]
    else:
        d = OUT_DEFAULT
    os.makedirs(d, exist_ok=True)
    return d


def hits_dir(out):
    d = os.path.join(out, "hits")
    os.makedirs(d, exist_ok=True)
    return d


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def var(v):
    m = mean(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1) if len(v) > 1 else float("nan")


def diff(a, b):
    return [x - y for x, y in zip(a, b)]


def write_csv(rows, path, definition):
    """CSV whose first line is '# definition: ...' (then the header)."""
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        f.write("# definition: " + " ".join(definition.split()) + " Status: " + STATUS + ".\n")
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def read_csv(path):
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("# definition")]
    return list(csv.DictReader(lines))


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def frozen_hashes():
    return {p: sha256_file(os.path.join(ROOT, p)) for p in FROZEN}


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path):
    with open(path) as f:
        return json.load(f)


_CROSS_CACHE = {}


def load_cross(ds):
    """results/tifs/addendum/cross_hits_bert_<ds>.json (read once per process)."""
    if ds not in _CROSS_CACHE:
        _CROSS_CACHE[ds] = load_json(CROSS[ds])
    return _CROSS_CACHE[ds]


def load_hits_files(out, prefix):
    """All hits files of this subgroup whose name starts with ``prefix``."""
    return {os.path.basename(p): load_json(p)
            for p in sorted(glob.glob(os.path.join(hits_dir(out), prefix + "*.json")))}


def verdicts(res, prefix=""):
    return {f"{prefix}verdict@{t:g}": res[f"verdict@{t:g}"] for t in TOLS}


def pooled_and_own(d_by_seed, s2p, dfp):
    """Frozen seed-pooled rule (addendum.seed_pooled + t_contrast at pool df), the A3.1
    own-variance rule (s2_a / r + v, df r - 1) and the A3.1 max rule
    (max(s2_a, s2_pool) / r + v, pool df) for one contrast with per-seed d vectors."""
    k, se, r, v, ks = AD.seed_pooled(d_by_seed, s2p, dfp)
    pooled = t_contrast(k, se, dfp, TOLS)
    out = {"k": k, "r": r, "v": v, "ks": ks, "pooled": pooled, "se_pooled": se}
    if r >= 2:
        s2a = var(ks)
        out["s2a"] = s2a
        out["own"] = t_contrast(k, (s2a / r + v) ** 0.5, r - 1, TOLS)
        out["max"] = t_contrast(k, (max(s2a, s2p) / r + v) ** 0.5, dfp, TOLS)
    return out


def transported(a, b, s2p, dfp):
    """Frozen one-seed rule: Tango k with SE^2 = SE_Tango^2 + s2_pool, pool df."""
    pc = paired_contrast(a, b, TOLS)
    se = (se_from_interval(pc["lo"], pc["hi"]) ** 2 + s2p) ** 0.5
    return pc, t_contrast(pc["k"], se, dfp, TOLS)


def satterthwaite(parts):
    """Satterthwaite df of a sum of variance components [(variance, df), ...];
    a component with df None is treated as known (infinite df)."""
    tot = sum(v for v, _ in parts)
    den = sum(v ** 2 / d for v, d in parts if d is not None and v > 0)
    return tot ** 2 / den if den > 0 else float("inf")


def state():
    """The frozen analysis state (addendum.State on runs/tifs)."""
    return AD.State(RUNS)


def pool(st, ds, kd="logit"):
    key, p = st.pool_for("bert", ds, kd)
    return key, p["s_between"] ** 2, p["df"]


def fmt(x, nd=4):
    return "nan" if x is None or x != x else f"{x:+.{nd}f}"
