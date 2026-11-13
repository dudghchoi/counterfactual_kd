"""Shared library of the A15 analysis (docs/tifs_addendum.md "## A15", its 10/26 19:55 e2 amendment and the
10/27 20:58 KST amendment "마감 연장, 판정 위계, claim rule", item 5).

A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

The frozen code is imported and called unchanged, never edited:
  scripts/tifs/analyze.py            load_cells, primary_table, primary_battery, decide, reference_variance,
                                     pool_seed_variance, qualified, load_pairs, RUN_RE
  scripts/tifs/addendum.py           State, seed_pooled, attack_variance_table, reference_atba_table,
                                     global_holm_table, load_replicate, replicate_table, MAJORITY,
                                     FUNCTION_MARGIN
  counterfactual_kd/analysis/tifs.py paired_contrast, t_contrast, se_from_interval, holm
  counterfactual_kd/metrics/paired.py (through tifs.py)

A "frame" is the frozen pipeline (analyze.primary_table -> primary_battery -> addendum.global_holm_table) run on
a list of cells in the frozen cell format: the frozen cells with some hit vectors substituted (A15-a: the
whitespace-preserving re-scores; A15-b1: the epoch-matched C), or with cells added (A15-d: seeds 45/46), or
an auxiliary battery alone (A15-c, -e, -e2). ``annotate`` adds to every row of a frame, beside the registered
verdict: the A3.1 own-variance verdict (s2_a / r + v, df r - 1), the claim-rule verdict of the 10/27 amendment
and the Welch-df sensitivity verdict (A15 E3). Run on the frozen cells, a frame reproduces
results/tifs/report/primary.csv, battery.csv, variance.csv and results/tifs/addendum/global_holm.csv exactly
(selfcheck.py), so any difference in an A15 frame comes from the substituted vectors only.

The A15 runs (tifs_x15_*) do not match analyze.RUN_RE; they are read with the frozen analyze.load_cells by
swapping RUN_RE for the call only (restored at once), as scripts/tifs/aux/g7-a13/common.py reads the A13 runs.
The whitespace-preserving re-scores (runs/tifs/tifs_x15_ws/<run>/<tag>.ws.preds.json) are read by the same
frozen loader through a temporary directory of symbolic links in the Stage 2 layout.

Only inputs of jobs that are terminal-ok are read: an A15 run's Stage 2 files when runs/tifs/_status/<run>__eval.ok
exists, a run's re-scores when _status/tifs_x15_ws__<run>.ok exists. A running, failed or cut job may have written
part of its files; such runs are left out ("present but job not terminal-ok: excluded") and listed in
inventory.csv and every check_*.csv (``not_ok_x15``, ``not_ok_ws``, ``excluded_rows``).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # before any local import

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
from collections import defaultdict  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "tifs"))

import analyze as A  # noqa: E402  (frozen)
import addendum as AD  # noqa: E402  (frozen)
from counterfactual_kd.analysis.tifs import (  # noqa: E402,F401  (frozen)
    holm, paired_contrast, se_from_interval, t_contrast,
)

TOLS = A.TOLS
ALPHA = A.ALPHA
STATUS = ("A15 post hoc analysis; changes no primary verdict, battery decision or the "
          "paper-wide Holm")
REPORT = os.path.join(ROOT, "results", "tifs", "report")
ADD = os.path.join(ROOT, "results", "tifs", "addendum")
AUX = os.path.join(ADD, "aux")
DEFAULT_RUNS = os.path.join(ROOT, "runs", "tifs")
DEFAULT_OUT = os.path.join(AUX, "g8-a15")
DEFAULT_TEACHERS_X15 = os.path.join(ROOT, "results", "tifs", "teachers_x15")
G6REF = os.path.join(AUX, "g6-review0", "reference")
G7 = os.path.join(AUX, "g7-a13")
G9 = os.path.join(AUX, "g9-a15cpu")
CROSS = {ds: os.path.join(ADD, f"cross_hits_bert_{ds}.json") for ds in ("sst2", "agnews")}

# Frozen files and their registered sha256: freeze record runs/tifs/_code/freeze_20261021T092041Z.sha256
# (analyze.py, tifs.py, paired.py, tifs_plan.md, structural_nulls.json) and the final-analysis snapshot
# final_analysis_20261024T110542Z.sha256 (addendum.py, cross_score.py). selfcheck.py also re-reads both
# records from runs/tifs/_code and checks that these constants agree with them.
FROZEN_SHA256 = {
    "docs/tifs_plan.md": "bc9dfd8f87f54d778a47e07b7d49de2c68af5b3df5458729fa4e7bb947f88713",
    "scripts/tifs/analyze.py": "1f70bbf4453bd83a5ec5f9858343549e89485627acb63f62d5e0e95dd5f09ef8",
    "counterfactual_kd/analysis/tifs.py": "cc0e05451bee502d888ab52260eb4576aab056dbe89a7c41fdce07db247dc9ce",
    "counterfactual_kd/metrics/paired.py": "aed57ad5f6f50eff1030e92054712695c985e75d6d518c1daefcc66b69f9783c",
    "results/tifs/structural_nulls.json": "a6dc5091d33b0c355a34136b65e28f29163864aeb0a4d1c060afd0c33656a9b0",
    "scripts/tifs/addendum.py": "d49a912ac05e9cbb802efdbc2268e16734ecc4b507a6eb16063d5f98206d88f1",
    "scripts/tifs/cross_score.py": "4bd94f6d2ed35c60a331f91021e4c18b535b2dcd3fa4922fe6d8982eb93a1156",
}
FREEZE_RECORDS = ("runs/tifs/_code/freeze_20261021T092041Z.sha256",
                  "runs/tifs/_code/final_analysis_20261024T110542Z.sha256")
# Read or imported without a registered hash: hashed by run_all.sh before the run and compared after it.
TRACKED_GLOBS = ("docs/*.md", "scripts/tifs/queue_plan.py", "scripts/tifs/aux/g6-review0/reference/score.py",
                 "scripts/tifs/aux/g6-review0/reference/common.py",
                 "scripts/tifs/aux/g1-exposure-controls-tokens/*.py", "scripts/tifs/aux/g6-review0/*/*.py",
                 "scripts/tifs/aux/g7-a13/*.py", "scripts/tifs/aux/g8-a15/*.py", "scripts/tifs/aux/g8-a15/*.json",
                 "scripts/tifs/aux/g9-a15cpu/*.py")
# Official frozen tables reproduced by the gate (selfcheck.py and every part script before it computes).
OFFICIAL = {
    "primary": os.path.join(REPORT, "primary.csv"),
    "variance": os.path.join(REPORT, "variance.csv"),
    "battery": os.path.join(REPORT, "battery.csv"),
    "global_holm": os.path.join(ADD, "global_holm.csv"),
    "attack_variance": os.path.join(ADD, "attack_variance.csv"),
    "reference_atba": os.path.join(ADD, "reference_atba.csv"),
    "replicate": os.path.join(ADD, "replicate.csv"),
}
VSHORT = {"EQUIVALENT": "EQU", "POSITIVE": "POS", "NEGATIVE": "NEG", "INCONCLUSIVE": "INC",
          "UNDECIDED": "UND", "rule-dependent": "RDEP", "one-seed, descriptive": "1SEED",
          "structural (counted apart)": "STRUC"}
# Inputs whose job has no terminal-ok status marker are left out of every table (a running, failed or cut job can
# have written part of its files); the run ids are listed in inventory.csv and in every check_*.csv.
EXCLUDED_TEXT = "present but job not terminal-ok: excluded"
STRUCTURAL_CLAIM = "structural (counted apart)"

# ── the A15 design (docs/tifs_addendum.md A15; scripts/tifs/aux/g8-a15/README.md) ──

# A15 E1: the OpenBackdoor-selected epoch of every SST-2 base-trainer P teacher (verified 10/26 from the build
# logs; g8-a15 README), hence the epoch of its matched C_e(P).
P_EPOCH = {("bert", "atba"): 1, ("bert", "addsent"): 1, ("bert", "badnets"): 4, ("bert", "synbkd"): 5,
           ("gpt2", "badnets"): 2, ("gpt2", "addsent"): 5,
           ("qwen3", "badnets"): 3, ("qwen3", "addsent"): 3, ("qwen3", "synbkd"): 3}
# OB-selected epoch of the frozen clean twins (C of the frozen cells).
C_EPOCH_FROZEN = {"bert": 3, "gpt2": 2, "qwen3": 1}
# Frozen clean-reference builds (BERT SST-2): their OB-selected epochs (A13 result record, A14 item 11).
REF_EPOCH_FROZEN = {42: 3, 43: 1, 44: 3, 45: 5, 46: 2}
B1_KDS = {"bert": ("logit", "feature"), "gpt2": ("logit",), "qwen3": ("logit",)}
SEEDS = (42, 43, 44)
B1_RUN = "tifs_x15_b1_{fam}_sst2_e{e}_s{s}"
B2_REF_RUN = "tifs_x15_b2_bert_ref_t{t}_e{e}"
B2_A2_RUN = "tifs_x15_b2_a2_t43_e5_s{s}"
C_RUN = "tifs_x15_c_gpt2_atba_s{s}"
D_RUN = "tifs_x15_d_bert_agnews_s{s}"
E_RUN = "tifs_x15_e_bert_agnews_a{a}_s{s}"
E2_RUN = "tifs_x15_e2_gpt2_sst2_a10_e{e}_s{s}"
WS_RUNS = ("tifs_b_gpt2_sst2", "tifs_b_gpt2_sst2_s43", "tifs_b_gpt2_sst2_s44",
           "tifs_b_qwen3_sst2", "tifs_b_qwen3_sst2_s43", "tifs_b_qwen3_sst2_s44",
           "tifs_b_qwen3_agnews", "tifs_x13_qwen3_agnews_s43", "tifs_x13_qwen3_agnews_s44",
           "tifs_g_qwen3_badnets10_s42", "tifs_aa_qwen3_badnets10_t43", "tifs_b_bert_sst2")
WS_DECODER_BATTERY = ("tifs_b_gpt2_sst2", "tifs_b_gpt2_sst2_s43", "tifs_b_gpt2_sst2_s44",
                      "tifs_b_qwen3_sst2", "tifs_b_qwen3_sst2_s43", "tifs_b_qwen3_sst2_s44",
                      "tifs_b_qwen3_agnews")
WS_G1 = ("tifs_x13_qwen3_agnews_s43", "tifs_x13_qwen3_agnews_s44")
WS_CONTAMINATED = ("tifs_g_qwen3_badnets10_s42", "tifs_aa_qwen3_badnets10_t43")
WS_BERT_CONTROL = ("tifs_b_bert_sst2",)
# Deadline of A15 (10/27 20:58 amendment item 1): 10/29 14:00 KST = 05:00 UTC; cut order e2 -> e -> d.
DEADLINE_UTC = "2026-10-29 05:00:00 UTC"
CUT_ORDER = ("e2", "e", "d")
# A3.3 / A14 item 7 tags (BERT; scripts/tifs/aux/g6-review0/reference/common.py).
P_TAG = {"sst2": {"badnets": "badnets_cf_mn", "addsent": "addsent_sent", "synbkd": "synbkd_trig",
                  "sos": "sos_friends_weekend", "ep": "ep_cf_mn", "atba": "atba_soothing sharing how"},
         "agnews": {"badnets": "badnets_cf_mn", "addsent": "addsent_sent", "synbkd": "synbkd_trig",
                    "sos": "sos_friends_weekend", "ep": "ep_cf_mn",
                    "atba": "atba_infected malicious albums, browser"}}
BASE_ATTACKS_SST2 = ("addsent", "atba", "badnets", "synbkd")

# Run-id patterns of the A15 runs for the frozen loader (blocks 'B' and 'AA' reuse the frozen parse_run
# conversions: B -> block G, corpus clean, seed = student seed).
X15_RE = [
    ("B", re.compile(r"^tifs_x15_b1_(?P<fam>bert|gpt2|qwen3)_(?P<bds>sst2)_e(?P<epoch>\d+)_s(?P<sseed>\d+)$")),
    ("AA", re.compile(r"^tifs_x15_b2_(?P<fam>bert)_ref_t(?P<tseed>\d+)_e(?P<epoch>\d+)$")),
    ("X15A2", re.compile(r"^tifs_x15_b2_a2_t(?P<tseed>\d+)_e(?P<epoch>\d+)_s(?P<seed>\d+)$")),
    ("B", re.compile(r"^tifs_x15_c_(?P<fam>gpt2)_(?P<bds>atba)_s(?P<sseed>\d+)$")),
    ("B", re.compile(r"^tifs_x15_d_(?P<fam>bert)_(?P<bds>agnews)_s(?P<sseed>\d+)$")),
    ("B", re.compile(r"^tifs_x15_e_(?P<fam>bert)_(?P<bds>agnews)_a(?P<alpha>\d+)_s(?P<sseed>\d+)$")),
    ("B", re.compile(r"^tifs_x15_e2_(?P<fam>gpt2)_(?P<bds>sst2)_a(?P<alpha>\d+)_e(?P<epoch>\d+)_s(?P<sseed>\d+)$")),
    ("X15B3", re.compile(r"^tifs_x15_b3_pcb_r(?P<corpus>\d{4})_s(?P<seed>\d+)$")),
]
# The A13 G1 runs (scripts/tifs/aux/g7-a13/common.py X13_RE, G1 only), for the whitespace re-scores.
X13_G1_RE = [("B", re.compile(r"^tifs_x13_(?P<fam>qwen3)_(?P<bds>agnews)_s(?P<sseed>\d+)$"))]


def part_of(run):
    for p in ("b1", "b2_a2", "b2", "c", "d", "e2", "e", "b3"):
        if run.startswith(f"tifs_x15_{p}_") or (p == "b2_a2" and run.startswith("tifs_x15_b2_a2_")):
            return "b2" if p == "b2_a2" else p
    return None


# ── command line ─────────────────────────────────────────────────────


def args(argv=None, extra=None):
    """--runs (default runs/tifs), --out (default results/tifs/addendum/aux/g8-a15), --teachers-x15
    (default results/tifs/teachers_x15) and the script's ``extra`` options. Tables are always written; missing
    A15 inputs are listed in the check files and in inventory_inputs.csv."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--teachers-x15", default=DEFAULT_TEACHERS_X15)
    for a, kw in (extra or []):
        ap.add_argument(a, **kw)
    ns = ap.parse_args(argv)
    ns.runs = os.path.abspath(ns.runs)
    ns.out = os.path.abspath(ns.out)
    if not (getattr(ns, "dry_run", False) or getattr(ns, "check_ready", False)):   # these write nothing
        os.makedirs(os.path.join(ns.out, "logs"), exist_ok=True)
    return ns


# ── files ────────────────────────────────────────────────────────────


def rel(p):
    try:
        return os.path.relpath(p, ROOT)
    except ValueError:
        return p


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _fmt(v):
    if isinstance(v, float) and math.isnan(v):
        return ""
    return v


def write_csv(rows, path, definition):
    """CSV whose first line is '# definition: ... Status: ...' (then the header). An empty table gets one row
    saying so (so every registered table exists)."""
    if not rows:
        rows = [{"note": "no rows (inputs missing; see inventory.csv)"}]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        f.write("# definition: " + " ".join(definition.split()) + " Status: " + STATUS + ".\n")
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in keys})
    print(f"  wrote {rel(path)} ({len(rows)} rows)", flush=True)
    return path


def read_csv(path):
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("# definition")]
    return list(csv.DictReader(lines))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
        f.write("\n")


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def mtime_utc(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC")


# ── small statistics (the frozen formulas, re-used) ──────────────────


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def var(v):
    m = mean(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1) if len(v) > 1 else float("nan")


def diff(a, b):
    return [x - y for x, y in zip(a, b)]


def welch_df(s2_list, r):
    """Welch-Satterthwaite df of mean_a s2_a with every s2_a on r - 1 df, attacks independent (A15 E3; the
    formula of scripts/tifs/aux/g9-a15cpu/common.welch_df): (sum s2_a)^2 (r - 1) / sum s2_a^2."""
    s2 = list(s2_list)
    den = sum(x * x for x in s2)
    return (sum(s2) ** 2) * (r - 1) / den if den > 0 else float("inf")


def satterthwaite2(v1, df1, v2, df2):
    """Satterthwaite df of v1 + v2 (the formula of addendum.reference_atba_table)."""
    return (v1 + v2) ** 2 / (v1 ** 2 / df1 + v2 ** 2 / df2)


def vtriple(d, prefix="verdict"):
    return "/".join(VSHORT.get(d.get(f"{prefix}@{t:g}"), str(d.get(f"{prefix}@{t:g}"))) for t in TOLS)


def fmt(x, nd=4):
    return "nan" if x is None or x != x else f"{x:+.{nd}f}"


def fmti(lo, hi, nd=4):
    return f"[{fmt(lo, nd)}, {fmt(hi, nd)}]"


def cid(r):
    return f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']}"


def ckey(r):
    return (r["fam"], r["dataset"], r["kd"], r["attack"])


# ── loaders (frozen analyze.load_cells) ──────────────────────────────


@contextlib.contextmanager
def run_re(patterns):
    """analyze.RUN_RE swapped for the call only (the way addendum.load_hidden_cells and g7-a13 read
    runs the frozen analysis does not parse)."""
    saved = A.RUN_RE
    A.RUN_RE = patterns
    try:
        yield
    finally:
        A.RUN_RE = saved


def preds_index(runs_dir, run_glob="tifs_*"):
    """{(run, dataset, attack, kd, seed): (preds path, result path)} of every Stage 2 prediction file whose
    run id matches the current analyze.RUN_RE, read with the same glob, run filter, .json companion and
    schema rule as analyze.load_cells (result JSON read for the key only)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, run_glob, "artifacts", "results", "*.preds.json"))):
        run = p.split(os.sep)[-4]
        if A.parse_run(run) is None:
            continue
        res = p[: -len(".preds.json")] + ".json"
        if not os.path.exists(res):
            continue
        if (load_json(p).get("meta") or {}).get("schema") != 2:
            continue
        r = load_json(res)
        key = (run, r["dataset"], r["attack_method"], r["kd_type"], int(r.get("seed", -1)))
        if key in out:
            raise ValueError(f"two prediction files for {key}: {out[key][0]} and {p}")
        out[key] = (p, res)
    return out


def attach(cells, runs_dir, extra_keys=("teacher_p_clean", "teacher_c_clean")):
    """Paths, tag, the untriggered teacher vectors (not kept by load_cells), the result JSON and the preds
    meta of every cell; the vectors load_cells keeps are checked against the file."""
    idx = preds_index(runs_dir)
    for c in cells:
        p, res = idx[(c["run"], c["dataset"], c["attack"], c["kd"], c["seed"])]
        d = load_json(p)
        if d["preds_obs"] != c["P"] or d["preds_null"] != c["C"]:
            raise ValueError(f"{p}: vectors differ from analyze.load_cells")
        c["preds_path"] = p
        c["result_path"] = res
        c["tag"] = os.path.basename(p)[: -len(".preds.json")]
        c["TPc"], c["TCc"] = d.get("teacher_p_clean"), d.get("teacher_c_clean")
        c["meta"] = d.get("meta") or {}
        c["result"] = load_json(res)
    return cells


def frozen_cells(st):
    """The frozen cells (addendum.State.cells, i.e. analyze.load_cells on runs/tifs) with paths and tags."""
    if not st.cells or "tag" not in st.cells[0]:
        attach(st.cells, st.runs)
    return st.cells


def eval_ok(runs_dir, run):
    """The run's Stage 2 evaluation job is terminal-ok: runs/tifs/_status/<run>__eval.ok (queue v9 job id)."""
    return os.path.exists(os.path.join(runs_dir, "_status", f"{run}__eval.ok"))


def ws_ok(runs_dir, run):
    """The run's A15-a re-score job is terminal-ok: runs/tifs/_status/tifs_x15_ws__<run>.ok."""
    return os.path.exists(os.path.join(runs_dir, "_status", f"tifs_x15_ws__{run}.ok"))


def _x15_run(run):
    return any(rx.match(run) for _, rx in X15_RE)


def not_ok_x15(runs_dir):
    """A15 run ids (X15_RE) with Stage 2 prediction files present but no _status/<run>__eval.ok (file names and
    markers only; such runs are left out by ``load_x15``)."""
    out = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "tifs_x15_*"))):
        run = os.path.basename(d)
        if (_x15_run(run) and glob.glob(os.path.join(d, "artifacts", "results", "*.preds.json"))
                and not eval_ok(runs_dir, run)):
            out.append(run)
    return out


def not_ok_ws(runs_dir):
    """Runs with A15-a re-score files present (runs/tifs/tifs_x15_ws/<run>/*.preds.json) but no
    _status/tifs_x15_ws__<run>.ok (names and markers only; left out by ``load_ws``)."""
    out = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "tifs_x15_ws", "*"))):
        run = os.path.basename(d)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.preds.json")) and not ws_ok(runs_dir, run):
            out.append(run)
    return out


def excluded_rows(runs_dir):
    """One check row (kind info) listing every A15 input left out because its job is not terminal-ok."""
    x, w = not_ok_x15(runs_dir), not_ok_ws(runs_dir)
    by = defaultdict(list)
    for r in x:
        by[part_of(r) or "?"].append(r)
    parts = [f"part {p}: {','.join(v)}" for p, v in sorted(by.items())]
    if w:
        parts.append("ws re-scores: " + ",".join(w))
    return [{"check": f"A15 inputs {EXCLUDED_TEXT} (Stage 2 files without _status/<run>__eval.ok; ws re-scores "
                      "without _status/tifs_x15_ws__<run>.ok)",
             "ok": not (x or w), "kind": "info", "note": "; ".join(parts) if parts else "none"}]


class _MarkerGate:
    """A RUN_RE regex that matches only runs whose evaluation job is terminal-ok, so the frozen loader never
    opens the files of a running, failed or cut job (it calls parse_run before reading a file)."""

    def __init__(self, rx, ok):
        self.rx, self.ok, self.pattern = rx, ok, rx.pattern

    def match(self, run):
        m = self.rx.match(run)
        return m if (m is not None and self.ok(run)) else None


def load_x15(runs_dir):
    """Every A15 cell (tifs_x15_*: b1, b2, b2 A2, c, d, e, e2, b3) in the frozen cell format, read by the
    frozen analyze.load_cells with RUN_RE swapped (restored at once); 'part', 'epoch', 'alpha' added,
    fam 'bert' for the b2 A2 and b3 runs, corpus 'clean' for the b2 reference runs. Only runs whose evaluation
    job is terminal-ok (runs/tifs/_status/<run>__eval.ok) are read; the others are listed by ``not_ok_x15``."""
    gated = [(b, _MarkerGate(rx, lambda run: eval_ok(runs_dir, run))) for b, rx in X15_RE]
    with run_re(gated):
        cells = A.load_cells(runs_dir)
        attach(cells, runs_dir)
    for c in cells:
        c["part"] = part_of(c["run"])
        if c.get("epoch") is not None:
            c["epoch"] = int(c["epoch"])
        if c.get("alpha") is not None:
            c["alpha"] = int(c["alpha"]) / 10.0
        if c["block"] in ("X15A2", "X15B3"):
            c["fam"] = "bert"
        if c["block"] == "AA":
            c["corpus"] = "clean"
        if c["block"] == "X15A2":
            c["corpus"] = "clean"
    return cells


def load_x13_g1(runs_dir):
    """The A13 G1 cells (tifs_x13_qwen3_agnews_s43/_s44) through the frozen loader (as g7-a13 load_x13)."""
    with run_re(X13_G1_RE):
        cells = A.load_cells(runs_dir)
        attach(cells, runs_dir)
    return cells


def _link_view(src_files, tmp):
    """Symbolic links <tmp>/<run>/artifacts/results/<tag>.preds.json and <tag>.json -> the given files."""
    for run, tag, preds, res in src_files:
        d = os.path.join(tmp, run, "artifacts", "results")
        os.makedirs(d, exist_ok=True)
        os.symlink(os.path.abspath(preds), os.path.join(d, f"{tag}.preds.json"))
        os.symlink(os.path.abspath(res), os.path.join(d, f"{tag}.json"))


def ws_files(runs_dir, mode, runs=None):
    """[(run, tag, preds, result)] of the A15-a re-scores in runs/tifs/tifs_x15_ws/<run>/ for ``mode``
    'ws' or 'official' (file names only; nothing is opened). Only runs whose re-score job is terminal-ok
    (_status/tifs_x15_ws__<run>.ok); the others are listed by ``not_ok_ws``."""
    base = os.path.join(runs_dir, "tifs_x15_ws")
    out = []
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        run = os.path.basename(d)
        if not os.path.isdir(d) or (runs is not None and run not in runs) or not ws_ok(runs_dir, run):
            continue
        for p in sorted(glob.glob(os.path.join(d, f"*.{mode}.preds.json"))):
            tag = os.path.basename(p)[: -len(f".{mode}.preds.json")]
            res = os.path.join(d, f"{tag}.{mode}.json")
            if os.path.exists(res):
                out.append((run, tag, p, res))
    return out


def load_ws(runs_dir, mode, runs=None):
    """The A15-a re-scored cells of ``mode`` ('ws' / 'official') in the frozen cell format: the frozen
    analyze.load_cells on a temporary directory of symbolic links in the Stage 2 layout, with RUN_RE = the
    frozen patterns + the A13 G1 and A15-c patterns (restored at once). Each cell gets 'tag', the paths and
    the result JSON."""
    files = ws_files(runs_dir, mode, runs)
    if not files:
        return []
    tmp = tempfile.mkdtemp(prefix=f"a15_{mode}_")
    try:
        _link_view(files, tmp)
        with run_re(list(A.RUN_RE) + X13_G1_RE + X15_RE):
            cells = A.load_cells(tmp)
            idx = preds_index(tmp)
        for c in cells:
            p, res = idx[(c["run"], c["dataset"], c["attack"], c["kd"], c["seed"])]
            d = load_json(p)
            c["tag"] = os.path.basename(p)[: -len(".preds.json")]
            c["preds_path"] = os.path.realpath(p)
            c["result_path"] = os.path.realpath(res)
            c["TPc"], c["TCc"] = d.get("teacher_p_clean"), d.get("teacher_c_clean")
            c["meta"] = d.get("meta") or {}
            c["result"] = load_json(res)
            c["mode"] = mode
            if c["run"].startswith("tifs_x15_"):
                c["part"] = part_of(c["run"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return cells


# ── frozen state + reproduction gate ─────────────────────────────────


def _same(v, text):
    if v is None:
        return text == ""
    return str(v) == text


def compare_rows(computed, official, key_cols, label):
    """Every column of every official row against the computed row with the same key (exact text)."""
    idx = {tuple(str(r.get(c)) for c in key_cols): r for r in computed}
    n_bad, bad = 0, []
    for o in official:
        key = tuple(o[c] for c in key_cols)
        r = idx.get(key)
        if r is None:
            n_bad += 1
            bad.append(f"{'/'.join(key)}: missing")
            continue
        for c, text in o.items():
            if not _same(r.get(c), text):
                n_bad += 1
                bad.append(f"{'/'.join(key)}:{c} computed {r.get(c)!r} official {text!r}")
    n_cols = len(official[0]) if official else 0
    return {"check": f"{label}: every column of every row reproduced exactly (text)",
            "expected": f"{len(official)} rows x {n_cols} columns, 0 mismatches",
            "got": f"{len(computed)} rows computed, {n_bad} mismatches",
            "ok": n_bad == 0 and len(computed) == len(official) and len(official) > 0,
            "note": "; ".join(bad[:8])}


class Proxy:
    """The attributes a frozen addendum function reads (primary, groups, prim, cells, runs, ...)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def reproduction_checks(st, F0=None):
    """primary.csv, variance.csv (frozen analyze.primary_table), battery.csv (frozen primary_battery),
    global_holm.csv (frozen addendum.global_holm_table), attack_variance.csv (A3.1, frozen
    addendum.attack_variance_table), reference_atba.csv (A3.3) and replicate.csv (A2), from the frozen
    state; and the same primary/variance/battery/global Holm from the frame engine on the frozen cells."""
    out = [compare_rows(st.primary, read_csv(OFFICIAL["primary"]), ("fam", "dataset", "kd", "attack"),
                        "results/tifs/report/primary.csv (analyze.primary_table)"),
           compare_rows(st.var_rows, read_csv(OFFICIAL["variance"]), ("fam", "dataset", "kd", "attack"),
                        "results/tifs/report/variance.csv (analyze.primary_table)"),
           compare_rows(A.primary_battery(st.primary), read_csv(OFFICIAL["battery"]), ("fam", "dataset", "kd"),
                        "results/tifs/report/battery.csv (analyze.primary_battery)"),
           compare_rows(AD.global_holm_table(st), read_csv(OFFICIAL["global_holm"]),
                        ("fam", "dataset", "kd", "attack"),
                        "results/tifs/addendum/global_holm.csv (addendum.global_holm_table)"),
           compare_rows(AD.attack_variance_table(st), read_csv(OFFICIAL["attack_variance"]),
                        ("fam", "dataset", "kd", "attack"),
                        "results/tifs/addendum/attack_variance.csv (addendum.attack_variance_table, A3.1)"),
           compare_rows(AD.reference_atba_table(st), read_csv(OFFICIAL["reference_atba"]), ("dataset", "kd"),
                        "results/tifs/addendum/reference_atba.csv (addendum.reference_atba_table, A3.3)"),
           compare_rows(AD.replicate_table(st), read_csv(OFFICIAL["replicate"]), ("attack", "teacher_seed"),
                        "results/tifs/addendum/replicate.csv (addendum.replicate_table, A2)")]
    if F0 is not None:
        out += [compare_rows(F0["rows"], read_csv(OFFICIAL["primary"]), ("fam", "dataset", "kd", "attack"),
                             "frame engine on the frozen cells == primary.csv"),
                compare_rows(F0["var_rows"], read_csv(OFFICIAL["variance"]), ("fam", "dataset", "kd", "attack"),
                             "frame engine on the frozen cells == variance.csv"),
                compare_rows(F0["battery"], read_csv(OFFICIAL["battery"]), ("fam", "dataset", "kd"),
                             "frame engine on the frozen cells == battery.csv"),
                compare_rows(F0["gh"], read_csv(OFFICIAL["global_holm"]), ("fam", "dataset", "kd", "attack"),
                             "frame engine on the frozen cells == global_holm.csv")]
        a31 = AD.attack_variance_table(st, prim=F0["prim"], att_var=F0["att_var"], pools=F0["pools"])
        idx = {ckey(r): r for r in a31}
        nb, nn = 0, 0
        for r in F0["rows"]:
            if r["method"].startswith("seed-pooled"):
                nn += 1
                for t in TOLS:
                    nb += idx[ckey(r)][f"own_verdict@{t:g}"] != r["own"][f"verdict@{t:g}"]
        out.append({"check": "annotate(): own-variance verdicts == addendum.attack_variance_table own_verdict "
                             "(frozen cells, every seed-pooled row x delta)",
                    "expected": f"{nn} rows, 0 mismatches", "got": f"{nn} rows, {nb} mismatches",
                    "ok": nb == 0 and nn > 0, "note": ""})
    return out


_STATE = {}


def state(runs, gate=True):
    """addendum.State on ``runs``; with ``gate`` the official tables are reproduced first (frozen state and
    the frame engine on the frozen cells) and the process stops on any mismatch: nothing is computed from a
    state that does not reproduce them."""
    if runs in _STATE:
        return _STATE[runs]
    st = AD.State(runs)
    frozen_cells(st)
    F0 = frame(st, st.cells, "frozen")
    if gate:
        bad = [c for c in reproduction_checks(st, F0) if not c["ok"]]
        if bad:
            for c in bad:
                print("GATE FAILED:", c["check"], c["got"], c["note"], file=sys.stderr)
            raise SystemExit(3)
        print("  gate: primary.csv, variance.csv, battery.csv, global_holm.csv, attack_variance.csv, "
              "reference_atba.csv, replicate.csv reproduced exactly (frozen state and frame engine)", flush=True)
    _STATE[runs] = (st, F0)
    return st, F0


# ── the frame engine ─────────────────────────────────────────────────


def frame(st, cells, label):
    """The frozen pipeline on ``cells``: analyze.reference_variance -> analyze.primary_table ->
    analyze.primary_battery -> addendum.global_holm_table; then ``annotate``."""
    ref_var = A.reference_variance(cells)
    rows, var_rows = A.primary_table(cells, st.structural, st.near_null, ref_var=ref_var)
    F = {"label": label, "cells": cells, "rows": rows, "var_rows": var_rows, "ref_var": ref_var,
         "battery": A.primary_battery(rows), "gh": AD.global_holm_table(Proxy(primary=rows))}
    F["pools"] = {(v["fam"], v["dataset"], v["kd"]): v for v in var_rows if v["attack"] == "(pooled)"}
    F["att_var"] = {(v["fam"], v["dataset"], v["kd"], v["attack"]): v for v in var_rows if v["attack"] != "(pooled)"}
    F["prim"] = {ckey(r): r for r in rows}
    groups = defaultdict(dict)
    for c in cells:
        if c["block"] == "G" and c["corpus"] == "clean" and not c["single"] and c["kd"] != "ce":
            groups[(c["fam"], c["dataset"], c["kd"], c["attack"])][c["seed"]] = c
    F["groups"] = groups
    annotate(F)
    return F


def pool_welch(F, key):
    """Welch-Satterthwaite df of pool ``key`` of frame F: s2_a = between-seed variance of attack a's k_s over
    the pool's seeds (variance rows k_s<seed>), r = number of pool seeds."""
    pool = F["pools"][key]
    seeds = pool["seeds"].split(",")
    s2a = {}
    for a in pool["attacks"].split(","):
        av = F["att_var"][key + (a,)]
        s2a[a] = var([av[f"k_s{s}"] for s in seeds])
    r = len(seeds)
    return {"df_welch": welch_df(s2a.values(), r), "s2a": s2a, "r": r,
            "pool_identity_ok": abs(mean(list(s2a.values())) - pool["s_between"] ** 2) < 1e-12}


_TOK = None


def a4_class(r):
    """A4 counting class of a primary row (the rule of scripts/tifs/aux/g6-review0/multiplicity/g6m_common.a4_class
    and paper_tifs/tools/make_numbers.py): 'structural' (EP SST-2), 'dagger' (trigger words in no clean KD-corpus
    row as whole words: g1 tokens_summary kd_rows_with_any_option_word_text == 0), 'boundary' (embedding-only
    pair that is not the structural null: SOS SST-2), else 'counted'. 'boundary' is inside the A4 count."""
    global _TOK
    if _TOK is None:
        rows = read_csv(os.path.join(AUX, "g1-exposure-controls-tokens", "tokens_summary.csv"))
        _TOK = {(x["fam"], x["dataset"], x["name"]): x for x in rows if x["kind"] == "attack trigger"}
    if r["structural_null"] in (True, "True"):
        return "structural"
    t = _TOK.get((r["fam"], r["dataset"], r["attack"]))
    if t is not None:
        v = t.get("kd_rows_with_any_option_word_text", "")
        if v not in ("", None) and int(float(v)) == 0:
            return "dagger"
    if r["embedding_only_pair"] in (True, "True"):
        return "boundary"
    return "counted"


def claim_of(reg_v, own_v, method):
    """Cell-level claim rule (10/27 20:58 amendment item 3): the registered verdict is claimed only when the
    A3.1 own-variance verdict agrees ('rule-dependent' otherwise, counted as INCONCLUSIVE); one-seed
    (transported) cells are 'one-seed, descriptive'; structural nulls are 'structural (counted apart)' (their
    Tango verdict stays in the reg_ columns; the g10-claimrule label), degenerate rows keep their Tango verdict."""
    if method == "structural":
        return STRUCTURAL_CLAIM
    if "degenerate" in method:
        return reg_v
    if method == "transported":
        return "one-seed, descriptive"
    if method == "no variance source":
        return "UNDECIDED"
    return reg_v if reg_v == own_v else "rule-dependent"


def claim_class(method):
    if method == "structural":
        return "structural"
    if "degenerate" in method:
        return "degenerate"
    if method == "transported":
        return "one-seed"
    if method == "no variance source":
        return "undecided"
    return "claimable"


def annotate(F):
    """Per row: r['own'] (A3.1 own variance: SE^2 = s2_a / r + v, df r - 1; t_contrast), r['welch'] (the
    registered k and SE at the Welch df of the row's pool), r['claim@d'] (claim rule), r['pmax'] (the
    one-sided p of the claim rule: max(p_registered, p_own) for seed-pooled rows, the Tango p for structural
    nulls, 1 for one-seed rows), r['fam_p'][reading] (the p of each family reading, see ``families``), the A4
    class, and runs ``families`` and ``battery_rules``."""
    welch = {k: pool_welch(F, k) for k in F["pools"]}
    F["welch"] = welch
    for r in F["rows"]:
        m = r["method"]
        r["claim_class"] = claim_class(m)
        r["a4_class"] = a4_class(r)
        r["own"] = None
        if m.startswith("seed-pooled") and "degenerate" not in m:
            av = F["att_var"][ckey(r)]
            rr = len(r["seeds"].split(","))
            se_own = (av["s_between"] ** 2 / rr + av["input_se"] ** 2) ** 0.5
            r["own"] = t_contrast(r["k"], se_own, rr - 1, TOLS)   # se 0: the frozen zero-SE branch
        r["welch"] = None
        src = r.get("variance_from")
        if src and m not in ("structural", "no variance source") and "degenerate" not in m:
            w = welch[tuple(src.split("/"))]
            r["welch"] = t_contrast(r["k"], r["se"], w["df_welch"], TOLS)
            r["welch"]["df_welch"] = w["df_welch"]
        r["pmax"], r["fam_p"] = {}, {"reg": {}, "own": {}, "claimA": {}, "claimB": {}, "welch": {}}
        for t in TOLS:
            own_v = r["own"][f"verdict@{t:g}"] if r["own"] else None
            r[f"claim@{t:g}"] = claim_of(r[f"verdict@{t:g}"], own_v, m)
            for p in ("p_up", "p_pos", "p_tost"):
                reg = r[f"{p}@{t:g}"]
                if r["claim_class"] == "claimable" and r["own"] is not None:
                    own = r["own"][f"{p}@{t:g}"]
                    pm, pa, pb = max(reg, own), max(reg, own), max(reg, own)
                elif r["claim_class"] in ("structural", "degenerate"):
                    own = pm = pa = pb = reg
                else:   # one-seed (transported) or undecided: no own variance
                    own, pm, pa, pb = reg, 1.0, reg, 1.0
                r["pmax"][f"{p}@{t:g}"] = pm
                fp = r["fam_p"]
                fp["reg"][f"{p}@{t:g}"] = reg
                fp["own"][f"{p}@{t:g}"] = own
                fp["claimA"][f"{p}@{t:g}"] = pa
                fp["claimB"][f"{p}@{t:g}"] = pb
                fp["welch"][f"{p}@{t:g}"] = (r["welch"] or r)[f"{p}@{t:g}"]
    families(F)
    battery_rules(F)
    return F


FAMILY_READINGS = ("reg", "own", "claimA", "claimB", "welch")


def families(F):
    """F31 (= the qualified rows of the frame; 31 for the registered cells) and F19 (its logit rows): Holm
    (frozen analysis.tifs.holm) of the one-sided p of each side at each delta under five readings (the names of
    scripts/tifs/aux/g10-claimrule/frozen_claims.py): reg = registered p; own = A3.1 own-variance p (one-seed
    cells p_registered); claimA = p_max = max(p_registered, p_own), one-seed cells at p_registered (their
    rejections are descriptive, not counted); claimB = p_max with one-seed cells at p = 1; welch = registered p
    at the Welch df (sensitivity, not part of the claim rule). Structural nulls enter with their Tango p (counted
    apart). A family claim holds when the adjusted p < 0.025. 'notransfer' side: p_up (H0 k >= delta);
    'transfer' side: p_pos (H0 k <= delta)."""
    qual = [r for r in F["rows"] if r["teacher_qualified"]]
    fams = {"F31": qual, "F19": [r for r in qual if r["kd"] == A.PRIMARY_KD]}
    F["family_size"] = {k: len(v) for k, v in fams.items()}
    for r in F["rows"]:
        r["fam_adj"] = {}
    for fname, rs in fams.items():
        if not rs:
            continue
        for t in TOLS:
            for side, p in (("notransfer", "p_up"), ("transfer", "p_pos")):
                for rule in FAMILY_READINGS:
                    ps = [r["fam_p"][rule][f"{p}@{t:g}"] for r in rs]
                    for r, a in zip(rs, holm(ps)):
                        r["fam_adj"][(fname, side, rule, t)] = a


def _pseudo(rs, which):
    """Rows for the frozen analyze.decide with the p-values of rule ``which``."""
    out = []
    for r in rs:
        d = {"attack": r["attack"]}
        for t in TOLS:
            for p in ("p_pos", "p_up", "p_tost"):
                if which == "own":
                    if r["claim_class"] == "claimable" and r["own"] is not None:
                        v = r["own"][f"{p}@{t:g}"]
                    elif r["claim_class"] in ("structural", "degenerate"):
                        v = r[f"{p}@{t:g}"]
                    else:
                        v = 1.0
                elif which == "claim":
                    v = r["pmax"][f"{p}@{t:g}"]
                elif which == "welch":
                    v = (r["welch"] or r)[f"{p}@{t:g}"]
                else:
                    v = r[f"{p}@{t:g}"]
                d[f"{p}@{t:g}"] = v
        out.append(d)
    return out


def battery_rules(F):
    """Per battery (fam, dataset, kd): the registered decision (frozen analyze.primary_battery) and the frozen
    analyze.decide on the own-variance p (one-seed rows p = 1), on the claim-rule p_max (one-seed rows p = 1, as
    the claimB family reading) and on the registered p at the Welch df. A battery whose qualified rows are all one-seed (structural nulls aside) is 'one-seed,
    descriptive' under the own and claim rules."""
    by = defaultdict(list)
    for r in F["rows"]:
        if r["teacher_qualified"]:
            by[(r["fam"], r["dataset"], r["kd"])].append(r)
    for b in F["battery"]:
        key = (b["fam"], b["dataset"], b["kd"])
        rs = by.get(key, [])
        b["rules"] = {}
        if not rs:
            continue
        one = (all(r["claim_class"] in ("one-seed", "structural") for r in rs)
               and any(r["claim_class"] == "one-seed" for r in rs))
        for which in ("own", "claim", "welch"):
            if which != "welch" and one:
                b["rules"][which] = {f"decision@{t:g}": "one-seed, descriptive" for t in TOLS}
                b["rules"][which].update({f"transfers@{t:g}": "" for t in TOLS})
                continue
            b["rules"][which] = A.decide(_pseudo(rs, which))


def pool_identity_rows(frames):
    """Check row: the Welch per-attack variances reproduce the frozen pool, mean_a s2_a = s_pool^2 (pool_welch
    pool_identity_ok), for every pool of every frame in ``frames`` ({name: frame})."""
    bad, n = [], 0
    for name, F in frames.items():
        if F is None:
            continue
        for key, w in F["welch"].items():
            n += 1
            if not w["pool_identity_ok"]:
                bad.append(f"{name}:{'/'.join(key)}")
    return [{"check": "pool_welch identity mean_a s2_a = s_pool^2 holds for every pool of every frame ("
                      + ", ".join(k for k, F in frames.items() if F is not None) + ")",
             "ok": n > 0 and not bad, "note": f"{n} pools" + (f"; fails: {';'.join(bad)}" if bad else "")}]


def family_size_row(F_ref, F_new, label, kind="correctness"):
    """Check row: the F31 family of frame ``F_new`` has the registered 31 cells; note = the cells whose
    teacher_qualified flag differs from ``F_ref``'s (the gate re-evaluated on the frame's teacher numbers)."""
    ch = [f"{cid(r)} {F_ref['prim'][ckey(r)]['teacher_qualified']}->{r['teacher_qualified']}"
          for r in F_new["rows"] if ckey(r) in F_ref["prim"]
          and F_ref["prim"][ckey(r)]["teacher_qualified"] != r["teacher_qualified"]]
    n = F_new["family_size"]["F31"]
    return {"check": f"F31 family size of the {label} frame = 31 (registered family)", "ok": n == 31, "kind": kind,
            "note": f"F31 {n}, F19 {F_new['family_size']['F19']}; teacher_qualified changes: "
                    + (";".join(ch) if ch else "none")}


# ── per-row field helpers for the output tables ─────────────────────


def reg_fields(r, pre):
    out = {f"{pre}method": r["method"], f"{pre}seeds": r["seeds"], f"{pre}variance_from": r.get("variance_from"),
           f"{pre}k": r["k"], f"{pre}se": r.get("se"), f"{pre}df": r.get("df"),
           f"{pre}lo": r["lo"], f"{pre}hi": r["hi"]}
    for t in TOLS:
        out[f"{pre}reg_verdict@{t:g}"] = r[f"verdict@{t:g}"]
        out[f"{pre}reg_p_up@{t:g}"] = r[f"p_up@{t:g}"]
        out[f"{pre}reg_p_pos@{t:g}"] = r[f"p_pos@{t:g}"]
    return out


def own_fields(r, pre):
    o = r["own"]
    out = {f"{pre}own_se": o["se"] if o else None, f"{pre}own_df": o["df"] if o else None,
           f"{pre}own_lo": o["lo"] if o else None, f"{pre}own_hi": o["hi"] if o else None}
    for t in TOLS:
        out[f"{pre}own_verdict@{t:g}"] = o[f"verdict@{t:g}"] if o else ("n/a (" + r["claim_class"] + ")")
        out[f"{pre}own_p_up@{t:g}"] = o[f"p_up@{t:g}"] if o else None
        out[f"{pre}own_p_pos@{t:g}"] = o[f"p_pos@{t:g}"] if o else None
    return out


def claim_fields(r, pre):
    out = {f"{pre}claim_class": r["claim_class"], f"{pre}a4_class": r["a4_class"]}
    for t in TOLS:
        out[f"{pre}claim@{t:g}"] = r[f"claim@{t:g}"]
        out[f"{pre}pmax_up@{t:g}"] = r["pmax"][f"p_up@{t:g}"]
        out[f"{pre}pmax_pos@{t:g}"] = r["pmax"][f"p_pos@{t:g}"]
    return out


def welch_fields(r, pre):
    w = r["welch"]
    out = {f"{pre}welch_df": w["df_welch"] if w else None, f"{pre}welch_lo": w["lo"] if w else None,
           f"{pre}welch_hi": w["hi"] if w else None}
    for t in TOLS:
        out[f"{pre}welch_verdict@{t:g}"] = w[f"verdict@{t:g}"] if w else r[f"verdict@{t:g}"]
    return out


def family_fields(r, pre):
    out = {}
    for fname in ("F31", "F19"):
        for t in TOLS:
            for side in ("notransfer", "transfer"):
                for rule in FAMILY_READINGS:
                    a = r["fam_adj"].get((fname, side, rule, t))
                    out[f"{pre}{fname}_{side}_{rule}_holm@{t:g}"] = a
                    out[f"{pre}{fname}_{side}_{rule}@{t:g}"] = (a < ALPHA) if a is not None else None
    return out


def all_fields(r, pre, family=True):
    out = {**reg_fields(r, pre), **own_fields(r, pre), **claim_fields(r, pre), **welch_fields(r, pre)}
    if family:
        out.update(family_fields(r, pre))
    return out


def changed(ref, new, rules=("reg", "own", "claim", "welch")):
    """{rule: 'deltas where the verdict differs'} between two rows (None rows -> 'n/a')."""
    out = {}
    for rule in rules:
        if ref is None or new is None:
            out[rule] = "n/a"
            continue
        ds = []
        for t in TOLS:
            a, b = _verdict(ref, rule, t), _verdict(new, rule, t)
            if a != b:
                ds.append(f"{t:g}")
        out[rule] = ",".join(ds)
    return out


def _verdict(r, rule, t):
    if rule == "reg":
        return r[f"verdict@{t:g}"]
    if rule == "own":
        return r["own"][f"verdict@{t:g}"] if r["own"] else "n/a"
    if rule == "claim":
        return r[f"claim@{t:g}"]
    if rule == "welch":
        return (r["welch"] or r)[f"verdict@{t:g}"]
    raise KeyError(rule)


def comparison_rows(F_ref, F_new, keys, label, pre_ref="frozen_", pre_new="new_", extra=None):
    """One row per cell key: the reference frame's and the new frame's registered / own / claim / Welch
    fields and the A15 label ``label`` (e.g. 'format_dependent') per rule = the deltas where the verdict
    differs (empty = same at every delta)."""
    out = []
    for key in keys:
        a, b = F_ref["prim"].get(key), F_new["prim"].get(key)
        row = {"fam": key[0], "dataset": key[1], "kd": key[2], "attack": key[3]}
        if a is not None:
            row.update(all_fields(a, pre_ref))
        if b is not None:
            row.update(all_fields(b, pre_new))
        ch = changed(a, b)
        for rule, v in ch.items():
            row[f"{label}_{rule}"] = v
        if extra:
            row.update(extra(key, a, b))
        out.append(row)
    return out


def battery_comparison(F_ref, F_new, keys, label, pre_ref="frozen_", pre_new="new_"):
    """One row per battery key: decisions under the registered rule (frozen analyze.primary_battery) and
    under the own / claim / Welch rules (frozen analyze.decide), both frames; ``label`` per rule."""
    br = {(b["fam"], b["dataset"], b["kd"]): b for b in F_ref["battery"]}
    bn = {(b["fam"], b["dataset"], b["kd"]): b for b in F_new["battery"]}
    out = []
    for key in keys:
        row = {"fam": key[0], "dataset": key[1], "kd": key[2]}
        per = {}
        for pre, B in ((pre_ref, br.get(key)), (pre_new, bn.get(key))):
            if B is None:
                continue
            row[f"{pre}attacks"] = B["attacks"]
            row[f"{pre}methods"] = B.get("methods")
            d = {}
            for t in TOLS:
                d[("reg", t)] = B[f"decision@{t:g}"]
                row[f"{pre}reg_decision@{t:g}"] = B[f"decision@{t:g}"]
                row[f"{pre}reg_transfers@{t:g}"] = B.get(f"transfers@{t:g}")
                row[f"{pre}reg_all_equivalent@{t:g}"] = B.get(f"all_equivalent@{t:g}")
                row[f"{pre}transfers_global@{t:g}"] = B.get(f"transfers_global@{t:g}")
                for rule in ("own", "claim", "welch"):
                    rr = B.get("rules", {}).get(rule)
                    v = rr[f"decision@{t:g}"] if rr else None
                    d[(rule, t)] = v
                    row[f"{pre}{rule}_decision@{t:g}"] = v
                    row[f"{pre}{rule}_transfers@{t:g}"] = rr.get(f"transfers@{t:g}") if rr else None
            per[pre] = d
        for rule in ("reg", "own", "claim", "welch"):
            if pre_ref in per and pre_new in per:
                row[f"{label}_{rule}"] = ",".join(f"{t:g}" for t in TOLS
                                                  if per[pre_ref][(rule, t)] != per[pre_new][(rule, t)])
            else:
                row[f"{label}_{rule}"] = "n/a"
        out.append(row)
    return out


def holm_rows(F_ref, F_new, label, pre_ref="frozen_", pre_new="new_"):
    """Per qualified cell of the new frame: the registered 31-cell Holm (frozen addendum.global_holm_table
    transfers_global) and the F31 / F19 family claims under the registered, claim-rule and Welch p, both
    frames; ``label``_* lists the (family, side, delta) whose claim flag changes."""
    gr = {ckey(r): r for r in F_ref["gh"]}
    gn = {ckey(r): r for r in F_new["gh"]}
    out = []
    for r in F_new["rows"]:
        if not r["teacher_qualified"]:
            continue
        key = ckey(r)
        a = F_ref["prim"].get(key)
        row = {"fam": key[0], "dataset": key[1], "kd": key[2], "attack": key[3], "new_method": r["method"],
               "family_size_new": F_new["family_size"]["F31"], "family_size_ref": F_ref["family_size"]["F31"]}
        for t in TOLS:
            for pre, G in ((pre_ref, gr.get(key)), (pre_new, gn.get(key))):
                row[f"{pre}holm_global@{t:g}"] = G[f"holm_global@{t:g}"] if G else None
                row[f"{pre}transfers_global@{t:g}"] = G[f"transfers_global@{t:g}"] if G else None
        if a is not None:
            row.update(family_fields(a, pre_ref))
        row.update(family_fields(r, pre_new))
        for rule in FAMILY_READINGS:
            ch = []
            for fname in ("F31", "F19"):
                for side in ("notransfer", "transfer"):
                    for t in TOLS:
                        x = row.get(f"{pre_ref}{fname}_{side}_{rule}@{t:g}")
                        y = row.get(f"{pre_new}{fname}_{side}_{rule}@{t:g}")
                        if x != y:
                            ch.append(f"{fname}:{side}@{t:g}")
            row[f"{label}_{rule}"] = ";".join(ch)
        out.append(row)
    return out


def claim_summary(F, label):
    """Counts per delta: cells by claim-rule verdict (claimable cells; rule-dependent counted as
    INCONCLUSIVE), one-seed and structural cells apart; family claims (F31, F19) per side under the readings
    reg / own / claimA / claimB / welch (claimA: one-seed rejections listed apart as descriptive and not
    counted)."""
    out = []
    qual = [r for r in F["rows"] if r["teacher_qualified"]]
    for t in TOLS:
        cl = [r for r in qual if r["claim_class"] == "claimable"]
        row = {"frame": label, "delta": t, "n_qualified": len(qual), "n_claimable": len(cl),
               "n_one_seed": sum(r["claim_class"] == "one-seed" for r in qual),
               "n_structural": sum(r["claim_class"] == "structural" for r in qual)}
        for v in ("EQUIVALENT", "POSITIVE", "NEGATIVE", "INCONCLUSIVE"):
            row[f"claim_{v}"] = sum(r[f"claim@{t:g}"] == v for r in cl)
            row[f"reg_{v}"] = sum(r[f"verdict@{t:g}"] == v for r in cl)
            row[f"own_{v}"] = sum(r["own"] is not None and r["own"][f"verdict@{t:g}"] == v for r in cl)
        row["claim_rule_dependent"] = sum(r[f"claim@{t:g}"] == "rule-dependent" for r in cl)
        for v in ("EQUIVALENT", "POSITIVE"):
            row[f"claim_{v}_a4_counted"] = sum(r[f"claim@{t:g}"] == v and r["a4_class"] in ("counted", "boundary")
                                               for r in cl)
        row["claim_INCONCLUSIVE_incl_rule_dependent"] = row["claim_INCONCLUSIVE"] + row["claim_rule_dependent"]
        row["rule_dependent_cells"] = ";".join(cid(r) for r in cl if r[f"claim@{t:g}"] == "rule-dependent")
        for fname in ("F31", "F19"):
            for side in ("notransfer", "transfer"):
                for rule in FAMILY_READINGS:
                    hits = [r for r in qual if (fname, side, rule, t) in r["fam_adj"]
                            and r["fam_adj"][(fname, side, rule, t)] < ALPHA]
                    if rule == "claimA":   # one-seed rejections are descriptive, not counted
                        row[f"{fname}_{side}_{rule}_descriptive_one_seed_n"] = sum(r["claim_class"] == "one-seed"
                                                                                    for r in hits)
                        row[f"{fname}_{side}_{rule}_descriptive_one_seed_cells"] = ";".join(
                            cid(r) for r in hits if r["claim_class"] == "one-seed")
                        hits = [r for r in hits if r["claim_class"] != "one-seed"]
                    row[f"{fname}_{side}_{rule}_n"] = len(hits)
                    row[f"{fname}_{side}_{rule}_nonstructural_n"] = sum(r["claim_class"] != "structural" for r in hits)
                    row[f"{fname}_{side}_{rule}_a4_counted_n"] = sum(r["a4_class"] in ("counted", "boundary")
                                                                    for r in hits)
                    row[f"{fname}_{side}_{rule}_cells"] = ";".join(cid(r) for r in hits)
        out.append(row)
    return out


# ── A8 student floor / A7 functional teacher ─────────────────────────


def floor_of(ds):
    """Majority-class share of the Stage 2 test set + 0.10 (addendum.MAJORITY, FUNCTION_MARGIN)."""
    return AD.MAJORITY[ds] + AD.FUNCTION_MARGIN


def student_rows(cells, label):
    """One row per student (P / C of a KD cell, R of a CE cell): Stage 2 clean accuracy against the A8 floor
    (majority share + 0.10); a missing accuracy counts as below the floor (A8, A13 'common')."""
    rows = []
    for c in sorted(cells, key=lambda c: (c["run"], c["dataset"], c["kd"], c["attack"])):
        floor = floor_of(c["dataset"])
        sides = (("R", c["ca_p"]),) if c["kd"] == "ce" else (("P", c["ca_p"]), ("C", c["ca_c"]))
        for side, acc in sides:
            rows.append({"source": label, "run": c["run"], "part": c.get("part"), "dataset": c["dataset"],
                         "attack": c["attack"], "kd": c["kd"], "student_seed": c["student_seed"],
                         "teacher_seed": c["teacher_seed"], "epoch": c.get("epoch"), "alpha": c.get("alpha"),
                         "student": side, "clean_acc": acc, "majority": AD.MAJORITY[c["dataset"]],
                         "floor": floor, "functional": acc is not None and acc > floor})
    return rows


def apply_floor(cells, label):
    """(kept cells, excluded runs, rows): the A8 rule as A13 applied it: a run with any student at or below
    the floor is left out of every verdict (its seed is excluded), and the fact is listed."""
    rows = student_rows(cells, label)
    bad = sorted({r["run"] for r in rows if not r["functional"]})
    return [c for c in cells if c["run"] not in bad], bad, rows


# ── job inventory (names and markers only; no A15 output is opened) ──


def status_of(runs_dir, job_id):
    d = os.path.join(runs_dir, "_status")
    for s in ("ok", "failed", "blocked", "running"):
        if os.path.exists(os.path.join(d, f"{job_id}.{s}")):
            return s
    return "pending"


def a15_jobs():
    """{part: [job dicts]} of queue v9 (scripts/tifs/queue_plan.a15_parts, b3 disabled), evaluations first."""
    import queue_plan as Q  # noqa: E402  (read only; pure)
    parts = Q.a15_parts(b3=False)
    return {p: list(ev) + list(jobs) for p, (ev, jobs) in parts.items()}, Q


def part_jobs_note(runs, part):
    """One-line status of a part's queue v9 jobs from the status markers (names only): counts of done / failed /
    cut-or-blocked / running / not started, and the ids of every job that is not done."""
    parts, _ = a15_jobs()
    js = parts.get(part, [])
    st = {j["id"]: status_of(runs, j["id"]) for j in js}
    cnt = {}
    for v in st.values():
        cnt[v] = cnt.get(v, 0) + 1
    names = {"ok": "done", "failed": "failed", "blocked": "blocked or cut", "running": "running",
             "pending": "not started"}
    head = ", ".join(f"{n} {names[k]}" for k, n in sorted(cnt.items()))
    nd = sorted(j for j, v in st.items() if v != "ok")
    return f"part {part}: {len(js)} jobs ({head})" + (f"; not done: {';'.join(nd)}" if nd else "")


def with_jobs(rows, note):
    """Rows with the column a15_jobs = ``note`` (the part's job status; see part_jobs_note)."""
    return [dict(r, a15_jobs=note) for r in rows]


def missing_note(missing):
    return "complete" if not missing else "partial: missing " + ";".join(missing[:12]) + (
        f" (+{len(missing) - 12})" if len(missing) > 12 else "")
