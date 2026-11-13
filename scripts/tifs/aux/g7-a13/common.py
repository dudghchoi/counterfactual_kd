"""Shared helpers of the A13 post hoc analysis (docs/tifs_addendum.md A13, groups G1-G4 and 'common').

A13 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

The frozen code is imported and called unchanged, never edited:
  scripts/tifs/analyze.py            (load_cells, primary_table, primary_battery, pool_seed_variance,
                                      decide, secondary_table, ...)
  scripts/tifs/addendum.py           (State, seed_pooled, reference_atba_table, attack_variance_table,
                                      MAJORITY, FUNCTION_MARGIN)
  counterfactual_kd/analysis/tifs.py (paired_contrast, t_contrast, se_from_interval, holm)
  counterfactual_kd/metrics/paired.py (through tifs.py)

The A13 runs (tifs_x13_*) do not match analyze.RUN_RE, so the frozen analysis never reads them. This
module reads them with the frozen analyze.load_cells by swapping RUN_RE for the call only (the way
addendum.load_hidden_cells reads the A8 runs) and restores it at once.

``state()`` builds addendum.State on runs/tifs and first reproduces results/tifs/report/primary.csv,
battery.csv, variance.csv, secondary.csv and results/tifs/addendum/reference_atba.csv (every column,
exact text) through the frozen loader; it stops on any mismatch.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # before any local import

import csv  # noqa: E402
import datetime  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "tifs"))

import analyze as A  # noqa: E402  (frozen)
import addendum as AD  # noqa: E402  (frozen)
from counterfactual_kd.analysis.tifs import (  # noqa: E402,F401  (frozen)
    holm, paired_contrast, se_from_interval, t_contrast,
)

TOLS = A.TOLS
ALPHA = A.ALPHA
RUNS = os.path.join(ROOT, "runs", "tifs")
REPORT = os.path.join(ROOT, "results", "tifs", "report")
ADD = os.path.join(ROOT, "results", "tifs", "addendum")
AUX = os.path.join(ADD, "aux")
OUT_DEFAULT = os.path.join(AUX, "g7-a13")
ITEM7_OUT = os.path.join(OUT_DEFAULT, "item7")
G6REF = os.path.join(AUX, "g6-review0", "reference")
G6CORPUS = os.path.join(AUX, "g6-review0", "corpus")
G9 = os.path.join(AUX, "g9-a15cpu")
STATUS = ("A13 post hoc analysis; changes no primary verdict, battery decision or the "
          "paper-wide Holm")

# Frozen files: sha256 of the freeze record freeze_20261021T092041Z (analyze.py, tifs.py, tifs_plan.md,
# structural_nulls.json) and of the final-analysis snapshot (addendum.py), as checked by the A14/A15 groups.
FROZEN_SHA256 = {
    "docs/tifs_plan.md": "bc9dfd8f87f54d778a47e07b7d49de2c68af5b3df5458729fa4e7bb947f88713",
    "scripts/tifs/analyze.py": "1f70bbf4453bd83a5ec5f9858343549e89485627acb63f62d5e0e95dd5f09ef8",
    "counterfactual_kd/analysis/tifs.py": "cc0e05451bee502d888ab52260eb4576aab056dbe89a7c41fdce07db247dc9ce",
    "results/tifs/structural_nulls.json": "a6dc5091d33b0c355a34136b65e28f29163864aeb0a4d1c060afd0c33656a9b0",
    "scripts/tifs/addendum.py": "d49a912ac05e9cbb802efdbc2268e16734ecc4b507a6eb16063d5f98206d88f1",
}
# Also imported, run or read; no registered hash: recorded at the start (run_all.sh) and compared at the end.
TRACKED = ("counterfactual_kd/metrics/paired.py", "counterfactual_kd/analysis/calibration.py",
           "scripts/tifs/cross_score.py", "docs/tifs_addendum.md",
           "scripts/tifs/aux/g6-review0/reference/common.py",
           "scripts/tifs/aux/g6-review0/reference/score.py",
           "scripts/tifs/aux/g6-review0/reference/item7_reference_k.py")
OFFICIAL = {
    "primary": os.path.join(REPORT, "primary.csv"),
    "battery": os.path.join(REPORT, "battery.csv"),
    "variance": os.path.join(REPORT, "variance.csv"),
    "secondary": os.path.join(REPORT, "secondary.csv"),
    "reference_atba": os.path.join(ADD, "reference_atba.csv"),
}
VSHORT = {"EQUIVALENT": "EQU", "POSITIVE": "POS", "NEGATIVE": "NEG", "INCONCLUSIVE": "INC"}

# The A13 runs (docs/tifs_addendum.md A13). Block labels are this module's; 'B' and 'AA' reuse the
# frozen parse_run conversions (B -> block G, corpus clean, seed).
X13_RE = [
    ("B", re.compile(r"^tifs_x13_(?P<fam>qwen3)_(?P<bds>agnews)_s(?P<sseed>\d+)$")),
    ("AA", re.compile(r"^tifs_x13_(?P<fam>bert)_ref_t(?P<tseed>\d+)$")),
    ("X13G3", re.compile(r"^tifs_x13_pcb_r(?P<corpus>\d{4})_s(?P<seed>\d+)$")),
    ("X13G4", re.compile(r"^tifs_x13_gb_r(?P<corpus>\d{4})_s(?P<seed>\d+)$")),
]
G3_RATES = ("0001", "0003", "0010", "0030")   # 0.01, 0.03, 0.1, 0.3 % of the KD corpus rows
G4_RATES = ("0010", "0030", "0100")           # 0.1, 0.3, 1 %
SEEDS = (42, 43, 44)
EXPECTED_RUNS = (["tifs_x13_qwen3_agnews_s43", "tifs_x13_qwen3_agnews_s44",
                  "tifs_x13_bert_ref_t45", "tifs_x13_bert_ref_t46"]
                 + [f"tifs_x13_pcb_r{r}_s{s}" for r in G3_RATES for s in SEEDS]
                 + [f"tifs_x13_gb_r{r}_s{s}" for r in G4_RATES for s in SEEDS])
# Deadline and cut order (A13 'common'): 10/27 18:00 KST = 09:00 UTC.
DEADLINE_UTC = "2026-10-27 09:00:00"
CUT_ORDER = ("G4 s44", "G4 s43", "G3 s44", "G3 s43", "G2 t46", "G1 s44")


def rate_of(code):
    """'0001' -> 0.0001 (fraction of KD corpus rows)."""
    return int(code) / 1e4


def rate_pct(code):
    return f"{int(code) / 100:g}%"


# ── files ────────────────────────────────────────────────────────────


def rel(p):
    return os.path.relpath(p, ROOT)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def out_dir(argv=None):
    """--out DIR (default results/tifs/addendum/aux/g7-a13)."""
    argv = sys.argv[1:] if argv is None else argv
    d = argv[argv.index("--out") + 1] if "--out" in argv else OUT_DEFAULT
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    return d


def _fmt(v):
    if isinstance(v, float) and math.isnan(v):
        return ""
    return v


def write_csv(rows, path, definition):
    """CSV whose first line is '# definition: ... Status: ...' (then the header)."""
    if not rows:
        raise ValueError(f"no rows for {path}")
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
        json.dump(obj, f, indent=1, sort_keys=True)
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


def verdicts(res, prefix="", out_prefix=None):
    op = prefix if out_prefix is None else out_prefix
    return {f"{op}verdict@{t:g}": res[f"{prefix}verdict@{t:g}"] for t in TOLS}


def vtriple(res, prefix=""):
    return "/".join(VSHORT[res[f"{prefix}verdict@{t:g}"]] for t in TOLS)


def tfields(res, prefix):
    """Interval, verdicts and one-sided p of a t_contrast result under ``prefix``."""
    out = {f"{prefix}k": res["k"], f"{prefix}se": res["se"], f"{prefix}df": res["df"],
           f"{prefix}lo": res["lo"], f"{prefix}hi": res["hi"]}
    for t in TOLS:
        out[f"{prefix}verdict@{t:g}"] = res[f"verdict@{t:g}"]
        out[f"{prefix}p_up@{t:g}"] = res[f"p_up@{t:g}"]
        out[f"{prefix}p_pos@{t:g}"] = res[f"p_pos@{t:g}"]
    return out


def welch_df(s2_list, r):
    """Welch-Satterthwaite df of mean_a s2_a with every s2_a on r - 1 df (A15 E3; the formula of
    scripts/tifs/aux/g9-a15cpu/common.welch_df): (sum s2_a)^2 (r - 1) / sum s2_a^2."""
    s2 = list(s2_list)
    den = sum(x * x for x in s2)
    return (sum(s2) ** 2) * (r - 1) / den if den > 0 else float("inf")


def satterthwaite(parts):
    """Satterthwaite df of a sum of variance components [(variance, df), ...]."""
    tot = sum(v for v, _ in parts)
    den = sum(v ** 2 / d for v, d in parts if v > 0)
    return tot ** 2 / den if den > 0 else float("inf")


def seed_pooled_family(d_by_unit_seed):
    """The frozen seed-pooled rule over a family of units (attacks, or the A13 dose rates).

    ``d_by_unit_seed``: {unit: {seed: difference vector d_s (P_s - C_s, or any paired contrast)}}.
    Pool: analyze.pool_seed_variance({unit: {seed: k_s}}) (Satterthwaite df). Per unit:
    addendum.seed_pooled (k = mean_s k_s, SE^2 = s2_pool / r + v) and t_contrast at the pool df;
    A3.1 own (SE^2 = s2_u / r + v, df r - 1) and max (max(s2_u, s2_pool) / r + v, pool df) beside it.
    Returns (pool, {unit: {...}})."""
    ks = {u: {s: mean(d) for s, d in by.items()} for u, by in d_by_unit_seed.items()}
    pool = A.pool_seed_variance(ks)
    if pool is None:
        raise ValueError("pool needs two seeds common to every unit")
    out = {}
    for u, by in d_by_unit_seed.items():
        k, se, r, v, kss = AD.seed_pooled(by, pool["s2"], pool["df"])
        res = {"k": k, "se": se, "r": r, "v": v, "ks": dict(zip(sorted(by), kss)),
               "pooled": t_contrast(k, se, pool["df"], TOLS)}
        if r >= 2:
            s2u = var(kss)
            res["s2u"] = s2u
            res["own"] = t_contrast(k, (s2u / r + v) ** 0.5, r - 1, TOLS)
            res["max"] = t_contrast(k, (max(s2u, pool["s2"]) / r + v) ** 0.5, pool["df"], TOLS)
        out[u] = res
    return pool, out


# ── frozen state + reproduction gate ─────────────────────────────────


def _same(v, text):
    """Exact text equality with what csv.DictWriter wrote for v (None / missing -> '')."""
    if v is None:
        return text == ""
    return str(v) == text


def compare_rows(computed, official, key_cols, label):
    """Every column of every official row against the computed row with the same key (exact text)."""
    idx = {tuple(str(r[c]) for c in key_cols): r for r in computed}
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
    return [{"check": f"{label}: every column of every row reproduced exactly (text)",
             "expected": f"{len(official)} rows x {n_cols} columns, 0 mismatches",
             "got": f"{len(computed)} rows computed, {n_bad} mismatches",
             "ok": n_bad == 0 and len(computed) == len(official) and len(official) > 0,
             "note": "; ".join(bad[:8])}]


def reproduction_checks(st):
    checks = []
    checks += compare_rows(st.primary, read_csv(OFFICIAL["primary"]), ("fam", "dataset", "kd", "attack"),
                           "results/tifs/report/primary.csv (analyze.primary_table)")
    checks += compare_rows(A.primary_battery(st.primary), read_csv(OFFICIAL["battery"]),
                           ("fam", "dataset", "kd"), "results/tifs/report/battery.csv (analyze.primary_battery)")
    checks += compare_rows(st.var_rows, read_csv(OFFICIAL["variance"]), ("fam", "dataset", "kd", "attack"),
                           "results/tifs/report/variance.csv (analyze.primary_table)")
    checks += compare_rows(A.secondary_table(st.cells, st.var_rows), read_csv(OFFICIAL["secondary"]),
                           ("run", "dataset", "attack", "kd"),
                           "results/tifs/report/secondary.csv (analyze.secondary_table)")
    checks += compare_rows(AD.reference_atba_table(st), read_csv(OFFICIAL["reference_atba"]),
                           ("dataset", "kd"), "results/tifs/addendum/reference_atba.csv (addendum.reference_atba_table)")
    return checks


_STATE = None


def state(gate=True):
    """addendum.State on runs/tifs; with ``gate`` the official tables are reproduced first and the
    process stops on any mismatch (nothing is computed from a state that does not reproduce them)."""
    global _STATE
    if _STATE is None:
        st = AD.State(RUNS)
        if gate:
            bad = [c for c in reproduction_checks(st) if not c["ok"]]
            if bad:
                for c in bad:
                    print("GATE FAILED:", c["check"], c["got"], c["note"], file=sys.stderr)
                raise SystemExit(3)
            print("  gate: primary.csv, battery.csv, variance.csv, secondary.csv, reference_atba.csv "
                  "reproduced exactly", flush=True)
        _STATE = st
    return _STATE


# ── the A13 cells through the frozen loader ──────────────────────────


_X13 = None


def load_x13():
    """Every A13 cell (tifs_x13_*) in the frozen cell format, read by analyze.load_cells with RUN_RE
    swapped for the call (restored at once). Adds 'group' (G1-G4), and the fields the A13 run ids do
    not carry: fam 'bert' and corpus = the rate code for G3/G4, corpus 'clean' for the G2 reference
    runs (whose KD corpus is the clean corpus). The extra untriggered teacher vectors that
    load_cells does not keep (teacher_p_clean / teacher_c_clean) are read from the same files."""
    global _X13
    if _X13 is not None:
        return _X13
    saved = A.RUN_RE
    A.RUN_RE = X13_RE
    try:
        cells = A.load_cells(RUNS)
    finally:
        A.RUN_RE = saved
    for c in cells:
        run = c["run"]
        if run.startswith("tifs_x13_qwen3_agnews_"):
            c["group"] = "G1"
        elif run.startswith("tifs_x13_bert_ref_"):
            c["group"] = "G2"
            c["corpus"] = "clean"
        elif run.startswith("tifs_x13_pcb_"):
            c["group"] = "G3"
            c["fam"] = "bert"
        elif run.startswith("tifs_x13_gb_"):
            c["group"] = "G4"
            c["fam"] = "bert"
        else:
            raise ValueError(f"unexpected A13 run {run}")
    # Paths and the untriggered teacher vectors (same files load_cells read).
    by = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "tifs_x13_*", "artifacts", "results", "*.preds.json"))):
        r = load_json(p[: -len(".preds.json")] + ".json")
        by[(p.split(os.sep)[-4], r["dataset"], r["attack_method"], r["kd_type"], int(r.get("seed", -1)))] = (p, r)
    for c in cells:
        p, r = by[(c["run"], c["dataset"], c["attack"], c["kd"], c["seed"])]
        d = load_json(p)
        if d["preds_obs"] != c["P"] or d["preds_null"] != c["C"]:
            raise ValueError(f"{p}: vectors differ from analyze.load_cells")
        c["preds_path"] = rel(p)
        c["TPc"], c["TCc"] = d.get("teacher_p_clean"), d.get("teacher_c_clean")
        c["n_eval"] = r.get("n_eval")
    _X13 = cells
    return cells


def x13_files():
    """Every prediction file of the tifs_x13_* runs (with its .json companion) and the newest mtime."""
    files = sorted(glob.glob(os.path.join(RUNS, "tifs_x13_*", "artifacts", "results", "*.preds.json")))
    newest = max(files, key=os.path.getmtime)
    return files, newest


# ── A8 student floor / A7 functional teacher ─────────────────────────


def floor_of(ds):
    """Majority-class share of the Stage 2 test set + 0.10 (addendum.MAJORITY, FUNCTION_MARGIN)."""
    return AD.MAJORITY[ds] + AD.FUNCTION_MARGIN


def student_rows(cells):
    """One row per A13 student (P / C of a logit cell, R of a CE cell): Stage 2 clean accuracy against
    the A8 floor (majority share + 0.10); a missing accuracy counts as below the floor (A8)."""
    rows = []
    for c in sorted(cells, key=lambda c: (c["group"], c["run"], c["dataset"], c["kd"], c["attack"])):
        floor = floor_of(c["dataset"])
        sides = (("R", c["ca_p"]),) if c["kd"] == "ce" else (("P", c["ca_p"]), ("C", c["ca_c"]))
        for side, acc in sides:
            rows.append({"group": c["group"], "run": c["run"], "dataset": c["dataset"], "attack": c["attack"],
                         "kd": c["kd"], "corpus": c["corpus"], "student_seed": c["student_seed"],
                         "teacher_seed": c["teacher_seed"], "student": side, "clean_acc": acc,
                         "majority": AD.MAJORITY[c["dataset"]], "floor": floor,
                         "functional": acc is not None and acc > floor})
    return rows


def failing_seeds(cells):
    """{(group, run): [students below the floor]} for the seed-exclusion rule of A8 / A13 'common'."""
    bad = {}
    for r in student_rows(cells):
        if not r["functional"]:
            bad.setdefault((r["group"], r["run"]), []).append(f"{r['dataset']}/{r['attack']}/{r['kd']}/{r['student']}")
    return bad


def fmt(x, nd=4):
    return "nan" if x is None or x != x else f"{x:+.{nd}f}"


def fmti(lo, hi, nd=4):
    return f"[{fmt(lo, nd)}, {fmt(hi, nd)}]"
