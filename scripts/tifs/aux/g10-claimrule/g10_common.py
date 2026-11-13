"""Shared helpers of g10 (docs/tifs_addendum.md, amendment 10/27 20:58 KST item 4: claim-rule calibration).

A15 post hoc auxiliary analysis (registered before the A15 analysis); changes no primary verdict, battery
decision or the paper-wide Holm.

Read-only on the repository. The frozen code is imported and called unchanged, never edited:
  scripts/tifs/analyze.py            (load_cells, primary_table, pool_seed_variance, decide, primary_battery)
  scripts/tifs/addendum.py           (State, var, attack_variance_table, global_holm_table)
  counterfactual_kd/analysis/tifs.py (t_contrast, paired_contrast, holm, se_from_interval)
  scripts/tifs/aux/g4-s1c-s9-s1extra/common.py, s1c.py   (the registered A11.10 S1c harness:
                                      pooled_model, target_model, draw_battery, pool_of, chol, evaluate,
                                      ExactFastT)
  scripts/tifs/aux/g6-review0/multiplicity/g6m_common.py, headline_power.py (A4 classes, display names,
                                      the exchangeable r-seed covariance used for r = 5)
No bytecode is written (sys.dont_write_bytecode, PYTHONDONTWRITEBYTECODE=1).

The module is called g10_common (not common) because the g4 harness imports its own ``common``.
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

GROUP = "g10-claimrule"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RUNS = os.path.join(ROOT, "runs", "tifs")
CODE_SNAP = os.path.join(RUNS, "_code")
REPORT = os.path.join(ROOT, "results", "tifs", "report")
ADDENDUM = os.path.join(ROOT, "results", "tifs", "addendum")
AUX = os.path.join(ADDENDUM, "aux")
G4_DIR = os.path.join(ROOT, "scripts", "tifs", "aux", "g4-s1c-s9-s1extra")
G6M_DIR = os.path.join(ROOT, "scripts", "tifs", "aux", "g6-review0", "multiplicity")
G4_OUT = os.path.join(AUX, "g4-s1c-s9-s1extra")
G6M_OUT = os.path.join(AUX, "g6-review0", "multiplicity")
G9_OUT = os.path.join(AUX, "g9-a15cpu")
OUT_DEFAULT = os.path.join(AUX, GROUP)

for p in (G6M_DIR, G4_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import common as K  # noqa: E402  (g4 common.py; puts scripts/tifs and ROOT on sys.path; frozen imports)
import s1c as S  # noqa: E402  (g4 S1c harness; imported, never run as main)
import g6m_common as G6  # noqa: E402  (A4 classes and display names)
import headline_power as HP  # noqa: E402  (exchangeable r-seed covariance)

A = K.A                 # scripts/tifs/analyze.py (frozen)
AD = K.AD               # scripts/tifs/addendum.py (frozen)
t_contrast = K.t_contrast
paired_contrast = K.paired_contrast
holm = K.holm
TOLS = A.TOLS           # (0.02, 0.05, 0.10)
ALPHA = A.ALPHA         # 0.025 one-sided
STATUS = ("A15 post hoc auxiliary analysis (claim-rule calibration, docs/tifs_addendum.md amendment 10/27 "
          "20:58 KST item 4); changes no primary verdict, battery decision or the paper-wide Holm")

# Frozen files and the record that fixes their hash.
FREEZE = "freeze_20261021T092041Z"
FINAL = "final_analysis_20261024T110542Z"
CLAIMREG = "a15_claimrule_registration_fix_20261027T115825Z"
FROZEN_SHA256 = {
    # freeze record (tifs_plan.md 0.3)
    "docs/tifs_plan.md": ("bc9dfd8f87f54d778a47e07b7d49de2c68af5b3df5458729fa4e7bb947f88713", FREEZE),
    "scripts/tifs/analyze.py": ("1f70bbf4453bd83a5ec5f9858343549e89485627acb63f62d5e0e95dd5f09ef8", FREEZE),
    "counterfactual_kd/analysis/tifs.py": ("cc0e05451bee502d888ab52260eb4576aab056dbe89a7c41fdce07db247dc9ce", FREEZE),
    "counterfactual_kd/metrics/paired.py": ("aed57ad5f6f50eff1030e92054712695c985e75d6d518c1daefcc66b69f9783c", FREEZE),
    "results/tifs/structural_nulls.json": ("a6dc5091d33b0c355a34136b65e28f29163864aeb0a4d1c060afd0c33656a9b0", FREEZE),
    # final-analysis snapshot
    "scripts/tifs/addendum.py": ("d49a912ac05e9cbb802efdbc2268e16734ecc4b507a6eb16063d5f98206d88f1", FINAL),
    "scripts/tifs/cross_score.py": ("4bd94f6d2ed35c60a331f91021e4c18b535b2dcd3fa4922fe6d8982eb93a1156", FINAL),
    # claim-rule registration snapshot (the amendment text and the registered S1c harness as imported here)
    "docs/tifs_addendum.md": ("aa1111a169639f9348a1ffde3fab990c2c778ad465df5b80dc68e9b66c0ab65b", CLAIMREG),
    "scripts/tifs/aux/g4-s1c-s9-s1extra/common.py": ("b57a9de1891ccea1ee8746831e6b604440a62140e0b3934bf8e52213301a96f7", CLAIMREG),
    "scripts/tifs/aux/g4-s1c-s9-s1extra/s1c.py": ("841064c5a7ec492f9cd8d699f7334bc138a0396132ddd82eae8cb8cd0c9a6a8b", CLAIMREG),
    "scripts/tifs/aux/g6-review0/multiplicity/g6m_common.py": ("af60c4f2daeef94aa895b1fd22b9e9620592720466ec1105e4c7a33ce41657de", CLAIMREG),
    "scripts/tifs/aux/g6-review0/multiplicity/headline_power.py": ("cbaa96f40a363a6a89e2dfd9b14cb6b9a8e20ada20ceae0547b9dd77c8752eb1", CLAIMREG),
}
# Official inputs (read only): reproduced by the gate or read for a sensitivity / validation column.
OFFICIAL = {
    "primary": os.path.join(REPORT, "primary.csv"),
    "variance": os.path.join(REPORT, "variance.csv"),
    "battery": os.path.join(REPORT, "battery.csv"),
    "global_holm": os.path.join(ADDENDUM, "global_holm.csv"),
    "attack_variance": os.path.join(ADDENDUM, "attack_variance.csv"),
    "s1c_cells": os.path.join(G4_OUT, "s1c_cells.csv"),
    "s1c_battery": os.path.join(G4_OUT, "s1c_battery.csv"),
    "s1c_fit": os.path.join(G4_OUT, "s1c_fit.csv"),
    "g6_claims_family": os.path.join(G6M_OUT, "claims_family.csv"),
    "g6_claims_cells": os.path.join(G6M_OUT, "claims_cells.csv"),
    "g9_welch_cells": os.path.join(G9_OUT, "welch_cells.csv"),
}
FINAL_SNAPSHOT_UTC = datetime.datetime(2026, 10, 24, 11, 5, 42, tzinfo=datetime.timezone.utc)

VSHORT = {"EQUIVALENT": "EQU", "POSITIVE": "POS", "NEGATIVE": "NEG", "INCONCLUSIVE": "INC"}


# ── files ────────────────────────────────────────────────────────────


def rel(p):
    return os.path.relpath(p, ROOT)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def snapshot_hashes(name):
    """{path: sha256} of a runs/tifs/_code/<name>.sha256 record."""
    out = {}
    with open(os.path.join(CODE_SNAP, name + ".sha256")) as f:
        for ln in f:
            h, p = ln.rstrip("\n").split("  ", 1)
            out[p] = h
    return out


def frozen_hash_rows():
    """One row per frozen file: current sha256 vs the value written here and vs the snapshot record."""
    rows = []
    recs = {}
    for p, (h, rec) in FROZEN_SHA256.items():
        if rec not in recs:
            recs[rec] = snapshot_hashes(rec)
        now = sha256(os.path.join(ROOT, p))
        in_rec = recs[rec].get(p, "")
        rows.append({"file": p, "record": rec, "sha256_record": in_rec, "sha256_now": now,
                     "ok": now == in_rec == h})
    return rows


def out_dir(args_out=None):
    out = args_out or OUT_DEFAULT
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    return out


def add_out_arg(ap):
    ap.add_argument("--out", default=OUT_DEFAULT,
                    help=f"output folder (default: results/tifs/addendum/aux/{GROUP})")
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
    """CSV whose first line is '# definition: ... Status: ...' (then the header)."""
    if not rows:
        raise ValueError(f"no rows for {path}")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def mtime_utc(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC")


def mc_se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def z_two_prop(p1, n1, p2, n2):
    return K.z_two_prop(p1, n1, p2, n2)


# ── frozen state + reproduction gate ─────────────────────────────────


def _same(v, text):
    if v is None:
        return text == ""
    return str(v) == text


def compare_rows(computed, official, key_cols, label):
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
    return {"check": f"{label}: every column of every row reproduced exactly (text)",
            "expected": f"{len(official)} rows x {n_cols} columns, 0 mismatches",
            "got": f"{len(computed)} rows, {n_bad} mismatches",
            "ok": n_bad == 0 and len(computed) == len(official),
            "note": "; ".join(bad[:6])}


def reproduction_checks(st):
    """primary.csv and variance.csv (analyze.primary_table), battery.csv (analyze.primary_battery),
    global_holm.csv (addendum.global_holm_table), attack_variance.csv (addendum.attack_variance_table)."""
    return [
        compare_rows(st.primary, read_csv(OFFICIAL["primary"]), ("fam", "dataset", "kd", "attack"),
                     "results/tifs/report/primary.csv (analyze.primary_table)"),
        compare_rows(st.var_rows, read_csv(OFFICIAL["variance"]), ("fam", "dataset", "kd", "attack"),
                     "results/tifs/report/variance.csv (analyze.primary_table)"),
        compare_rows(A.primary_battery(st.primary), read_csv(OFFICIAL["battery"]), ("fam", "dataset", "kd"),
                     "results/tifs/report/battery.csv (analyze.primary_battery)"),
        compare_rows(AD.global_holm_table(st), read_csv(OFFICIAL["global_holm"]), ("fam", "dataset", "kd", "attack"),
                     "results/tifs/addendum/global_holm.csv (addendum.global_holm_table)"),
        compare_rows(AD.attack_variance_table(st), read_csv(OFFICIAL["attack_variance"]),
                     ("fam", "dataset", "kd", "attack"),
                     "results/tifs/addendum/attack_variance.csv (addendum.attack_variance_table, A3.1)"),
    ]


_STATE = None


def state(gate=True):
    """addendum.State on runs/tifs (shared with the g4 harness); with ``gate`` the official tables are
    reproduced first and the process stops on any mismatch."""
    global _STATE
    if _STATE is None:
        st = K.state()
        if gate:
            bad = [c for c in reproduction_checks(st) if not c["ok"]]
            if bad:
                for c in bad:
                    print("GATE FAILED:", c["check"], c["got"], c["note"], file=sys.stderr)
                raise SystemExit(3)
            print("  gate: primary.csv, variance.csv, battery.csv, global_holm.csv, attack_variance.csv "
                  "reproduced exactly through the frozen loader", flush=True)
        _STATE = st
    return _STATE


def loader_snapshot():
    """Prediction files the frozen loader reads (same glob, run filter, .json companion and schema rule as
    analyze.load_cells), the count of all prediction files, and the newest mtimes."""
    allp = sorted(glob.glob(os.path.join(RUNS, "tifs_*", "artifacts", "results", "*.preds.json")))
    used = []
    for p in allp:
        run = p.split(os.sep)[-4]
        if A.parse_run(run) is None or not os.path.exists(p[: -len(".preds.json")] + ".json"):
            continue
        with open(p) as f:
            if (json.load(f).get("meta") or {}).get("schema") != 2:
                continue
        used.append(p)
    newest = max(used, key=os.path.getmtime)
    return {"n_loader": len(used), "n_runs_loader": len({p.split(os.sep)[-4] for p in used}),
            "newest_loader_utc": mtime_utc(newest), "newest_loader": rel(newest),
            "n_all": len(allp), "n_runs_all": len({p.split(os.sep)[-4] for p in allp})}


# ── cells of the 31-cell family ──────────────────────────────────────


def cell_id(r):
    return f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']}"


def cell_class(r):
    """'seed-pooled' (own variance defined: claim rule applies), 'one-seed' (transported: descriptive under
    the claim rule), 'structural' (EP SST-2: counted apart)."""
    m = r["method"]
    if m == "structural":
        return "structural"
    if m.startswith("seed-pooled"):
        return "seed-pooled"
    if m == "transported":
        return "one-seed"
    raise ValueError(m)


def family_cells(st):
    """The 31 qualified primary cells (F31, frozen order) with class, F19 membership and A4 class."""
    out = []
    for r in st.primary:
        if not r["teacher_qualified"]:
            continue
        out.append({"id": cell_id(r), "row": r, "cls": cell_class(r), "in_F19": r["kd"] == A.PRIMARY_KD,
                    "a4": G6.a4_class(r), "display": G6.display(dict(r, a4_class=G6.a4_class(r)))})
    assert len(out) == 31, len(out)
    assert sum(c["in_F19"] for c in out) == 19
    return out


def vtriple(res, prefix=""):
    return "/".join(VSHORT[res[f"{prefix}verdict@{t:g}"]] for t in TOLS)
