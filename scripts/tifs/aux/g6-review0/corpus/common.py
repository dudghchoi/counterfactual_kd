"""Shared, read-only helpers of the g6-review0 'corpus' subgroup
(docs/tifs_addendum.md A14 items 8, 10 and 11; post hoc, registered
2026-10-26 11:56 KST).

A14 post hoc analysis; changes no primary verdict, battery decision or the
paper-wide Holm.

Paths are repository-relative: ROOT is found from this file's location
(scripts/tifs/aux/g6-review0/corpus/ -> repository root). Inputs are the
official outputs only (results/tifs/report, results/tifs/addendum incl.
aux/g1-g5, results/tifs/s1, results/tifs/teachers metadata, runs/tifs preds /
result JSONs and checkpoints, logs/ run logs). Outputs go to
results/tifs/addendum/aux/g6-review0/corpus/ unless --out is given.

The frozen code (scripts/tifs/analyze.py, scripts/tifs/addendum.py,
counterfactual_kd/analysis/tifs.py, counterfactual_kd/metrics/paired.py,
scripts/tifs/cross_score.py) is imported, never edited. CPU only
(CUDA_VISIBLE_DEVICES is cleared) and no bytecode (sys.dont_write_bytecode).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if not os.path.exists(os.path.join(ROOT, "scripts", "tifs", "analyze.py")):
    raise SystemExit(f"repository root not found from {HERE} (expected scripts/tifs/aux/g6-review0/<subgroup>/)")
RUNS = os.path.join(ROOT, "runs", "tifs")
REPORT = os.path.join(ROOT, "results", "tifs", "report")
ADDENDUM = os.path.join(ROOT, "results", "tifs", "addendum")
AUX = os.path.join(ADDENDUM, "aux")
G1 = os.path.join(AUX, "g1-exposure-controls-tokens")
TEACHERS = os.path.join(ROOT, "results", "tifs", "teachers")
LOGS = os.path.join(ROOT, "logs")
GROUP = "g6-review0/corpus"
DEFAULT_OUT = os.path.join(AUX, "g6-review0", "corpus")
OUT = DEFAULT_OUT
STATUS = "A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm"

sys.path.insert(0, os.path.join(ROOT, "scripts", "tifs"))
sys.path.insert(0, ROOT)

import analyze as A  # noqa: E402  (frozen; imported, never modified)
import addendum as AD  # noqa: E402  (frozen; imported, never modified)

TOLS = A.TOLS
ENV = {"HF_HOME": "/data/Youngho/cache/huggingface", "HF_HUB_OFFLINE": "1",
       "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
       "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
for _k, _v in ENV.items():
    os.environ.setdefault(_k, _v)
# SynBkd paraphrase table: syntactic.DEFAULT_TABLE is relative to the working
# directory; pin the absolute path (read only: cross_score.strict_synbkd makes a
# missing paraphrase an error instead of an append).
os.environ.setdefault("CKD_SYNBKD_TABLE", os.path.join(ROOT, "results", "tifs", "synbkd", "table_tpl-1.jsonl"))


def parse_args(description, extra=None, argv=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output folder")
    if extra:
        extra(ap)
    args = ap.parse_args(argv)
    set_out(args.out)
    return args


def set_out(path):
    global OUT
    OUT = os.path.abspath(path)
    os.makedirs(OUT, exist_ok=True)
    return OUT


def out(*parts):
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def rel(p):
    return str(p).replace(ROOT + os.sep, "").replace(ROOT, "<repo>")


def write_csv(rows, path, definition):
    """CSV whose first line is '# definition: ...' (then the header)."""
    if not rows:
        raise ValueError(f"no rows for {path}")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("# definition: " + " ".join(definition.split()) + "\n")
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})
    return path


def fmt(v):
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return v


def read_csv(path):
    with open(path, newline="") as f:
        lines = [ln for ln in f if not ln.startswith("# definition")]
    return list(csv.DictReader(lines))


def fnum(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def mean(v):
    return sum(v) / len(v) if v else float("nan")


_STATE = None


def state():
    """The frozen analysis' cells, pools and primary rows (addendum.State)."""
    global _STATE
    if _STATE is None:
        _STATE = AD.State(RUNS)
    return _STATE


def newest_preds():
    import glob
    paths = glob.glob(os.path.join(RUNS, "tifs_*", "artifacts", "results", "*.preds.json"))
    newest = max(paths, key=os.path.getmtime)
    return len(paths), time.strftime("%FT%TZ", time.gmtime(os.path.getmtime(newest))), rel(newest)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_texts(texts):
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(path + ".tmp", path)


def now_utc():
    return time.strftime("%FT%TZ", time.gmtime())
