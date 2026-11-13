"""Shared helpers of the A14 multiplicity subgroup (docs/tifs_addendum.md A14 items 1, 2, 6, 9, 12).

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Read-only on the repository. The frozen functions are imported and called unchanged:
  scripts/tifs/analyze.py            (load_cells, primary_table, pool_seed_variance, decide, ...)
  scripts/tifs/addendum.py           (State)
  counterfactual_kd/analysis/tifs.py (t_contrast, paired_contrast, holm, se_from_interval)
  scripts/tifs/aux/g4-s1c-s9-s1extra/common.py, s1c.py (the registered A11.10 S1c machinery)
No bytecode is written (sys.dont_write_bytecode, PYTHONDONTWRITEBYTECODE=1).

Paths are repo-relative (ROOT from this file). Outputs go to
results/tifs/addendum/aux/g6-review0/multiplicity/ unless a script is given --out.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # before any local import

import csv  # noqa: E402
import datetime  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

SUBGROUP = "multiplicity"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
RUNS = os.path.join(ROOT, "runs", "tifs")
REPORT = os.path.join(ROOT, "results", "tifs", "report")
ADDENDUM = os.path.join(ROOT, "results", "tifs", "addendum")
AUX = os.path.join(ADDENDUM, "aux")
G4_DIR = os.path.join(ROOT, "scripts", "tifs", "aux", "g4-s1c-s9-s1extra")
OUT_DEFAULT = os.path.join(AUX, "g6-review0", SUBGROUP)

# g4's common.py puts scripts/tifs and ROOT on sys.path and imports analyze / addendum.
if G4_DIR not in sys.path:
    sys.path.insert(1, G4_DIR)

import numpy as np  # noqa: E402

import common as K  # noqa: E402  (g4 common.py, registered A11 code; imported, never modified)

A = K.A                  # scripts/tifs/analyze.py (frozen)
AD = K.AD                # scripts/tifs/addendum.py
holm = K.holm            # counterfactual_kd.analysis.tifs.holm (frozen)
t_contrast = K.t_contrast
paired_contrast = K.paired_contrast
se_from_interval = K.se_from_interval
TOLS = A.TOLS            # (0.02, 0.05, 0.10)
ALPHA = A.ALPHA          # 0.025 one-sided

STATUS = "A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm"

FAM = {"bert": "BERT", "gpt2": "GPT-2", "qwen3": "Qwen3"}
DS = {"sst2": "SST-2", "agnews": "AG News"}
ATK = {"badnets": "BadNets", "addsent": "AddSent", "synbkd": "SynBkd", "sos": "SOS", "ep": "EP",
       "atba": "ATBA", "ripple": "RIPPLe"}

# Frozen files and their SHA-256 in the freeze record freeze_20261021T092041Z (analyze.py, tifs.py,
# tifs_plan.md, structural_nulls.json) and the final-analysis snapshot (addendum.py); the g4 S1c code
# as promoted on 2026-10-25 (A11 result record). Checked before and after every run (selfchecks.py).
FROZEN_SHA256 = {
    "docs/tifs_plan.md": "bc9dfd8f87f54d778a47e07b7d49de2c68af5b3df5458729fa4e7bb947f88713",
    "scripts/tifs/analyze.py": "1f70bbf4453bd83a5ec5f9858343549e89485627acb63f62d5e0e95dd5f09ef8",
    "counterfactual_kd/analysis/tifs.py": "cc0e05451bee502d888ab52260eb4576aab056dbe89a7c41fdce07db247dc9ce",
    "results/tifs/structural_nulls.json": "a6dc5091d33b0c355a34136b65e28f29163864aeb0a4d1c060afd0c33656a9b0",
    "scripts/tifs/addendum.py": "d49a912ac05e9cbb802efdbc2268e16734ecc4b507a6eb16063d5f98206d88f1",
}
# Read for the record only (no expected value is registered for them): their hash at the time of this run.
TRACKED = ("counterfactual_kd/metrics/paired.py", "counterfactual_kd/analysis/calibration.py",
           "scripts/tifs/aux/g4-s1c-s9-s1extra/common.py", "scripts/tifs/aux/g4-s1c-s9-s1extra/s1c.py",
           "docs/tifs_addendum.md")


def rel(p):
    return os.path.relpath(p, ROOT)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def out_dir(args_out=None):
    out = args_out or OUT_DEFAULT
    os.makedirs(out, exist_ok=True)
    return out


def add_out_arg(ap):
    ap.add_argument("--out", default=OUT_DEFAULT,
                    help="output folder (default: results/tifs/addendum/aux/g6-review0/multiplicity)")
    return ap


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if math.isnan(v):
            return ""
        return repr(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def write_csv(rows, path, definition):
    """CSV whose first line is '# definition: ... Status: ...'."""
    if not rows:
        raise ValueError(f"no rows for {path}")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("# definition: " + " ".join(definition.split()) + " Status: " + STATUS + ".\n")
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in keys})
    print(f"  wrote {rel(path)} ({len(rows)} rows)", flush=True)


def read_csv(path):
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def fnum(x):
    if x in ("", None):
        return float("nan")
    return float(x)


def as_bool(x):
    return str(x).strip().lower() in ("true", "1", "yes")


# ── cells ────────────────────────────────────────────────────────────


def cell_id(r):
    return f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']}"


def display(r, mark=True):
    s = f"{FAM[r['fam']]} {DS[r['dataset']]} {ATK.get(r['attack'], r['attack'])}"
    if r["kd"] != "logit":
        s += " (feature)"
    if mark:
        s += {"dagger": "†", "boundary": "‡", "structural": " (structural null)"}.get(r.get("a4_class"), "")
    return s


_TOK = None


def tokens_trigger_rows():
    global _TOK
    if _TOK is None:
        rows = read_csv(os.path.join(AUX, "g1-exposure-controls-tokens", "tokens_summary.csv"))
        _TOK = {(r["fam"], r["dataset"], r["name"]): r for r in rows if r["kind"] == "attack trigger"}
    return _TOK


def a4_class(r):
    """A4 counting class of a primary row (same rule as paper_tifs/tools/make_numbers.py a4_absent /
    a4_boundary): 'structural' (EP SST-2), 'dagger' (trigger words in no clean KD-corpus row as whole
    words: tokens_summary kd_rows_with_any_option_word_text == 0), 'boundary' (embedding-only teacher
    pair that is not the structural null: SOS SST-2, words in 80 corpus rows), else 'counted'.
    'boundary' cells are inside the A4 count and reported apart."""
    if as_bool(r["structural_null"]):
        return "structural"
    t = tokens_trigger_rows().get((r["fam"], r["dataset"], r["attack"]))
    if t is not None:
        v = t.get("kd_rows_with_any_option_word_text", "")
        if v not in ("", None) and int(float(v)) == 0:
            return "dagger"
    if as_bool(r["embedding_only_pair"]):
        return "boundary"
    return "counted"


def a4_counted(cls):
    return cls in ("counted", "boundary")


def state():
    """The frozen analysis state (addendum.State over runs/tifs), shared with the g4 S1c code."""
    return K.state()


def qualified_primary(st):
    """The 31-cell paper-wide Holm family: the qualified rows of the frozen primary table, in its order."""
    return [r for r in st.primary if r["teacher_qualified"]]


def snapshot():
    """Prediction files read by the frozen loader (same glob, run filter and .json companion rule as
    analyze.load_cells), the count of every prediction file, and the newest mtimes."""
    loader = K.snapshot()
    allp = sorted(glob.glob(os.path.join(RUNS, "tifs_*", "artifacts", "results", "*.preds.json")))
    newest_all = max(allp, key=os.path.getmtime)
    ts = datetime.datetime.fromtimestamp(os.path.getmtime(newest_all), datetime.timezone.utc)
    loader.update(n_preds_all=len(allp), n_runs_all=len({p.split(os.sep)[-4] for p in allp}),
                  newest_all_utc=ts.strftime("%Y-%m-%d %H:%M:%S UTC"), newest_all_path=rel(newest_all))
    return loader
