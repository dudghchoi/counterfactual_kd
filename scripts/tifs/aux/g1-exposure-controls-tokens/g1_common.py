"""Shared, read-only loaders for the g1 auxiliary tables (docs/tifs_addendum.md
A11 items 1, 2 and 13; post hoc, registered 10/24 18:05 KST).

Paths are repository-relative: ROOT is computed from this file's location
(scripts/tifs/aux/g1-exposure-controls-tokens/ -> the repository root).
Inputs: runs/tifs (preds / result JSONs, kd_meta.json sidecars, checkpoints),
results/tifs/report (frozen scripts/tifs/analyze.py output) and
results/tifs/addendum (scripts/tifs/addendum.py output). Outputs go to
results/tifs/addendum/aux/g1-exposure-controls-tokens/ unless --out is given;
nothing else in the repository is written.

Cells come from the frozen loader (scripts/tifs/analyze.py load_cells) and from
the addendum loaders; the A2 replicate runs (tifs_rep_*) and the A10-a
CE-coefficient runs (tifs_rcc_*) are read with the same frozen loader by
swapping its run-id pattern for the call (the trick addendum.load_hidden_cells
uses; analyze.py itself is not edited and RUN_RE is restored at once).

CPU only (CUDA_VISIBLE_DEVICES is cleared here) and no bytecode: every entry
script sets sys.dont_write_bytecode before importing this module, so no
__pycache__ is written inside the repository even without
PYTHONDONTWRITEBYTECODE=1.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if not os.path.exists(os.path.join(ROOT, "scripts", "tifs", "analyze.py")):
    raise SystemExit(f"repository root not found from {HERE} (expected scripts/tifs/aux/<group>/)")
R = ROOT                                  # short alias used by the scripts
RUNS = os.path.join(ROOT, "runs", "tifs")
REPORT = os.path.join(ROOT, "results", "tifs", "report")
ADDENDUM = os.path.join(ROOT, "results", "tifs", "addendum")
GROUP = "g1-exposure-controls-tokens"
DEFAULT_OUT = os.path.join(ADDENDUM, "aux", GROUP)
OUT = DEFAULT_OUT                         # set by parse_args (--out)

sys.path.insert(0, os.path.join(ROOT, "scripts", "tifs"))
sys.path.insert(0, ROOT)

import analyze as A  # noqa: E402  (frozen; imported, never modified)
import addendum as AD  # noqa: E402

TOLS = A.TOLS

# Run ids the frozen RUN_RE does not parse. Optional sseed so parse_run's
# battery branch ("B") can pop it (defaults to seed 42).
REP_RX = re.compile(r"^tifs_rep_(?P<fam>[a-z0-9]+)_(?P<bds>atba(?:_agnews)?)_t(?P<tseed>\d+)_s(?P<sseed>\d+)$")
RCC_RX = re.compile(r"^tifs_rcc_(?P<fam>[a-z0-9]+)_(?P<bds>[a-z0-9]+)(?:_s(?P<sseed>\d+))?$")

CORPUS_CLASS = {
    "clean": "clean",
    "badnets10": "contaminated (10% triggered rows, labels set to target)",
    "addsent10": "contaminated (10% triggered rows, labels set to target)",
    "synbkd10": "contaminated (10% triggered rows, labels set to target)",
    "kl01": "trigger-exposed, labels kept (positive control, 1%)",
    "kl10": "trigger-exposed, labels kept (positive control, 10%)",
}
# Three-way split (an added, labelled variant of A11 item 2's two-way split).
CORPUS_KIND = {
    "clean": "clean",
    "badnets10": "label-poisoned",
    "addsent10": "label-poisoned",
    "synbkd10": "label-poisoned",
    "kl01": "labels kept",
    "kl10": "labels kept",
}
SPLIT3 = ("clean", "label-poisoned", "labels kept")


def parse_args(description, extra=None, argv=None):
    """--out (default results/tifs/addendum/aux/g1-exposure-controls-tokens/) and
    any script-specific arguments; sets OUT and creates it."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output folder (default: results/tifs/addendum/aux/g1-exposure-controls-tokens/)")
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


def rel(p):
    """Repository-relative form of a path for the tables."""
    return p.replace(ROOT, "<repo>")


def corpus_class(corpus):
    return CORPUS_CLASS.get(corpus, corpus)


def corpus_split(corpus):
    """The two-way split item 2 registers: clean vs contaminated (any trigger
    rows in the KD corpus, label-poisoned or labels kept)."""
    return "clean" if corpus == "clean" else "contaminated"


def corpus_split3(corpus):
    """Added variant: clean / label-poisoned (badnets10, addsent10: triggered
    rows relabelled to the target) / labels kept (kl01, kl10)."""
    return CORPUS_KIND[corpus]


def block_label(c):
    b, corpus = c["block"], c["corpus"]
    if b == "G":
        return "battery" if corpus == "clean" else "contaminated grid"
    return {"AA": "A/A (independent teacher build)", "PC": "positive control",
            "HID": "A8 hidden-state KD", "REP": "A2 replicate (teacher seed 43)",
            "RCC": "A10-a R_B (0.5*CE)"}.get(b, b)


def load_with(rx):
    """Frozen load_cells with one run-id pattern swapped in for the call."""
    saved = A.RUN_RE
    A.RUN_RE = [("B", rx)]
    try:
        return A.load_cells(RUNS)
    finally:
        A.RUN_RE = saved


def load_everything():
    """(state, cells) with every evaluated cell, labelled by source.

    source 'frozen'            : analyze.load_cells (battery G, contaminated G, AA, PC)
    source 'A8 hidden'         : addendum.load_hidden_cells (block relabelled HID)
    source 'A2 replicate'      : tifs_rep_* (block relabelled REP)
    source 'A10-a R_B'         : tifs_rcc_* (block relabelled RCC; CE-only, single)
    """
    st = AD.State(RUNS)
    cells = [dict(c, source="frozen") for c in st.cells]
    cells += [dict(c, source="A8 hidden", block="HID") for c in AD.load_hidden_cells(RUNS)]
    cells += [dict(c, source="A2 replicate", block="REP") for c in load_with(REP_RX)]
    cells += [dict(c, source="A10-a R_B", block="RCC") for c in load_with(RCC_RX)]
    for c in cells:
        c["block_label"] = block_label(c)
    return st, cells


def reference_for(st, c):
    """CE-only R for a P/C cell, and where it came from.

    Frozen G/PC cells: the frozen reference index (same block, family,
    dataset, attack, corpus, seed), i.e. student seed 42 only. AA, HID and REP
    cells share the frozen G block's R of the same family, dataset, attack and
    corpus at corpus seed 42 / student seed 42 (R has no teacher, so the
    teacher seed does not enter); only their student-seed-42 rows get it.
    """
    if c["block"] in ("G", "PC") and c["source"] == "frozen":
        r = st.refs.get((c["block"], c["fam"], c["dataset"], c["attack"], c["corpus"], c["seed"]))
        return r, ("same run (frozen reference index)" if r else "")
    if c["student_seed"] == 42 and c["corpus_seed"] == 42:
        r = st.refs.get(("G", c["fam"], c["dataset"], c["attack"], c["corpus"], 42))
        return r, ("shared: frozen G-block R (corpus seed 42, student seed 42)" if r else "")
    return None, ""


def rcc_for(cells, c):
    """A10-a R_B (L = 0.5 CE) of the same family, dataset, attack, seed 42, clean corpus."""
    if c["corpus"] != "clean" or c["student_seed"] != 42:
        return None
    for x in cells:
        if (x["source"] == "A10-a R_B" and x["fam"] == c["fam"] and x["dataset"] == c["dataset"]
                and x["attack"] == c["attack"] and x["student_seed"] == 42):
            return x
    return None


def preds_paths():
    return sorted(glob.glob(os.path.join(RUNS, "tifs_*", "artifacts", "results", "*.preds.json")))


def preds_index():
    """{(run, dataset, attack, kd, student_seed, teacher_seed): preds path}.

    Used only for the fields the frozen loader does not keep (the teachers'
    untriggered hits) and to cross-check the loader's vectors.
    """
    idx = {}
    for p in preds_paths():
        res = p[: -len(".preds.json")] + ".json"
        if not os.path.exists(res):
            continue
        r = json.load(open(res))
        key = (p.split(os.sep)[-4], r["dataset"], r["attack_method"], r["kd_type"],
               int(r.get("student_seed", -1)), int(r.get("teacher_seed", -1)))
        if key in idx:
            raise ValueError(f"two preds files for {key}")
        idx[key] = p
    return idx


def newest_preds_utc():
    """Data snapshot: (newest preds file mtime as UTC ISO, number of preds files)."""
    ps = preds_paths()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(os.path.getmtime(p) for p in ps))), len(ps)


def cell_key(c):
    return (c["run"], c["dataset"], c["attack"], c["kd"], c["student_seed"], c["teacher_seed"])


def write_csv(rows, path):
    if not rows:
        return 0
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def mean(v):
    return sum(v) / len(v) if v else float("nan")
