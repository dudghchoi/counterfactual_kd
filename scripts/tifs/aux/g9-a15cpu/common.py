"""Shared helpers of the A15 CPU items (docs/tifs_addendum.md A15, block 'CPU 항목 (GPU 없음)' and E3).

A15 post hoc CPU analysis; changes no primary verdict, battery decision or the paper-wide Holm.

The frozen code is imported and called unchanged, never edited:
  scripts/tifs/analyze.py            (load_cells, primary_table, primary_battery, pool_seed_variance, ...)
  scripts/tifs/addendum.py           (State, seed_pooled, cross_tables, global_holm_table)
  counterfactual_kd/analysis/tifs.py (paired_contrast, t_contrast, se_from_interval, holm)
Item 5 also imports the corpus construction code (counterfactual_kd.adapters.native_kd.contaminate_corpus,
counterfactual_kd.data.loaders.load_data) read-only and records the sha256 of every module it loaded.

Every item script calls ``state()``: it builds addendum.State on runs/tifs and first reproduces
results/tifs/report/primary.csv, variance.csv, battery.csv and results/tifs/addendum/global_holm.csv
(every column, exact text) through the frozen loader; it stops on any mismatch.
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
OUT_DEFAULT = os.path.join(AUX, "g9-a15cpu")
CROSS = {ds: os.path.join(ADD, f"cross_hits_bert_{ds}.json") for ds in ("sst2", "agnews")}
G6REF = os.path.join(AUX, "g6-review0", "reference")
STATUS = ("A15 post hoc CPU analysis; changes no primary verdict, battery decision or the "
          "paper-wide Holm")

# Frozen files: sha256 in the freeze record freeze_20261021T092041Z (analyze.py, tifs.py, tifs_plan.md,
# structural_nulls.json) and the final-analysis snapshot final_analysis_20261024T110542Z (addendum.py),
# as used by the A14 groups. Checked at the start (run_all.sh) and at the end (selfcheck.py).
FROZEN_SHA256 = {
    "docs/tifs_plan.md": "bc9dfd8f87f54d778a47e07b7d49de2c68af5b3df5458729fa4e7bb947f88713",
    "scripts/tifs/analyze.py": "1f70bbf4453bd83a5ec5f9858343549e89485627acb63f62d5e0e95dd5f09ef8",
    "counterfactual_kd/analysis/tifs.py": "cc0e05451bee502d888ab52260eb4576aab056dbe89a7c41fdce07db247dc9ce",
    "results/tifs/structural_nulls.json": "a6dc5091d33b0c355a34136b65e28f29163864aeb0a4d1c060afd0c33656a9b0",
    "scripts/tifs/addendum.py": "d49a912ac05e9cbb802efdbc2268e16734ecc4b507a6eb16063d5f98206d88f1",
}
# Also imported or read; no registered hash, their hash is recorded before and after the run.
TRACKED = ("counterfactual_kd/metrics/paired.py", "counterfactual_kd/analysis/calibration.py",
           "docs/tifs_addendum.md")
# Official tables reproduced by the gate (and the inputs items 1-4 read).
OFFICIAL = {
    "primary": os.path.join(REPORT, "primary.csv"),
    "variance": os.path.join(REPORT, "variance.csv"),
    "battery": os.path.join(REPORT, "battery.csv"),
    "global_holm": os.path.join(ADD, "global_holm.csv"),
    "cross_null": os.path.join(ADD, "cross_null.csv"),
}

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


def out_dir(argv=None):
    """--out DIR (default results/tifs/addendum/aux/g9-a15cpu)."""
    argv = sys.argv[1:] if argv is None else argv
    d = argv[argv.index("--out") + 1] if "--out" in argv else OUT_DEFAULT
    os.makedirs(d, exist_ok=True)
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


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def mtime_utc(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC")


# ── small statistics (frozen formulas re-used, never re-derived) ─────


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def var(v):
    m = mean(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1) if len(v) > 1 else float("nan")


def diff(a, b):
    return [x - y for x, y in zip(a, b)]


def verdicts(res, prefix=""):
    return {f"{prefix}verdict@{t:g}": res[f"verdict@{t:g}"] for t in TOLS}


def vtriple(res, prefix=""):
    return "/".join(VSHORT[res[f"{prefix}verdict@{t:g}"]] for t in TOLS)


def welch_df(s2_by_attack, r):
    """Welch-Satterthwaite df of mean_a s2_a with every s2_a on r - 1 df, attacks independent
    (A15 E3): (sum s2_a)^2 (r - 1) / sum s2_a^2."""
    s2 = list(s2_by_attack)
    den = sum(x * x for x in s2)
    return (sum(s2) ** 2) * (r - 1) / den if den > 0 else float("inf")


def welch_pools(st):
    """{pool key: {...}} for every seed-pooled battery pool of the frozen analysis.

    s2_a is the frozen per-attack between-seed variance (variance.csv s_between^2 over the pool's
    seeds); the pool's s2 equals mean_a s2_a exactly (identity of analyze.pool_seed_variance), which
    is checked here."""
    out = {}
    for key, pool in sorted(st.pools.items()):
        atks = pool["attacks"].split(",")
        seeds = pool["seeds"]
        ks = {}
        for a in atks:
            av = st.att_var[key + (a,)]
            own = ",".join(k[3:] for k in sorted(av) if k.startswith("k_s"))
            if own != seeds:
                raise ValueError(f"{key}/{a}: attack seeds {own} != pool seeds {seeds}")
            ks[a] = [av[f"k_s{s}"] for s in seeds.split(",")]
        r = len(seeds.split(","))
        s2a = {a: var(v) for a, v in ks.items()}
        s2p = pool["s_between"] ** 2
        if abs(mean(list(s2a.values())) - s2p) > 1e-15:
            raise ValueError(f"{key}: mean_a s2_a {mean(list(s2a.values()))} != s2_pool {s2p}")
        nu = welch_df(s2a.values(), r)
        out[key] = {"attacks": atks, "seeds": seeds, "r": r, "m": len(atks), "ks": ks, "s2a": s2a,
                    "s2_pool": s2p, "df_frozen": pool["df"], "df_welch": nu,
                    "df_min": min(pool["df"], nu), "ms_seed": pool["ms_seed"], "ms_res": pool["ms_res"]}
    return out


def pooled_and_own(d_by_seed, s2p, dfp, df_alt=None):
    """For one contrast with per-seed difference vectors d_s (real values allowed):
    pooled = the frozen seed-pooled rule (addendum.seed_pooled + t_contrast at the pool df),
    own    = A3.1 own variance, SE^2 = s2_a / r + v, df r - 1 (A14 item 3),
    max    = A3.1 max rule, SE^2 = max(s2_a, s2_pool) / r + v, pool df,
    pooled_alt = the pooled SE at df ``df_alt`` (the Welch df of the pool) when given."""
    k, se, r, v, ks = AD.seed_pooled(d_by_seed, s2p, dfp)
    out = {"k": k, "r": r, "v": v, "ks": ks, "se_pooled": se, "pooled": t_contrast(k, se, dfp, TOLS)}
    if df_alt is not None:
        out["pooled_alt"] = t_contrast(k, se, df_alt, TOLS)
    if r >= 2:
        s2a = var(ks)
        out["s2a"] = s2a
        out["own"] = t_contrast(k, (s2a / r + v) ** 0.5, r - 1, TOLS)
        out["max"] = t_contrast(k, (max(s2a, s2p) / r + v) ** 0.5, dfp, TOLS)
    return out


def transported(a, b, s2p, dfp, df_alt=None):
    """Frozen one-seed rule: Tango k with SE^2 = SE_Tango^2 + s2_pool, t at the pool df
    (and at ``df_alt`` when given)."""
    pc = paired_contrast(a, b, TOLS)
    se = (se_from_interval(pc["lo"], pc["hi"]) ** 2 + s2p) ** 0.5
    res = t_contrast(pc["k"], se, dfp, TOLS)
    alt = t_contrast(pc["k"], se, df_alt, TOLS) if df_alt is not None else None
    return pc, res, alt


# ── frozen state + reproduction gate ─────────────────────────────────


def _same(v, text):
    """Exact text equality with what csv.DictWriter wrote for v (None / missing -> '')."""
    if v is None:
        return text == ""
    return str(v) == text


def compare_rows(computed, official, key_cols, label):
    """Every column of every official row against the computed row with the same key (exact text)."""
    idx = {tuple(str(r[c]) for c in key_cols): r for r in computed}
    checks, n_bad, bad = [], 0, []
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
    checks.append({"check": f"{label}: every column of every row reproduced exactly (text)",
                   "expected": f"{len(official)} rows x {n_cols} columns, 0 mismatches",
                   "got": f"{len(official)} rows x {n_cols} columns, {n_bad} mismatches",
                   "ok": n_bad == 0 and len(computed) == len(official),
                   "note": "; ".join(bad[:8]) + ("" if len(computed) == len(official) else
                                                 f" computed rows {len(computed)} != official {len(official)}")})
    return checks


def reproduction_checks(st):
    """primary.csv, variance.csv, battery.csv (frozen primary_battery), global_holm.csv (frozen
    addendum.global_holm_table) and cross_null.csv (frozen addendum.cross_tables)."""
    checks = []
    checks += compare_rows(st.primary, read_csv(OFFICIAL["primary"]), ("fam", "dataset", "kd", "attack"),
                           "results/tifs/report/primary.csv (analyze.primary_table)")
    checks += compare_rows(st.var_rows, read_csv(OFFICIAL["variance"]), ("fam", "dataset", "kd", "attack"),
                           "results/tifs/report/variance.csv (analyze.primary_table)")
    checks += compare_rows(A.primary_battery(st.primary), read_csv(OFFICIAL["battery"]),
                           ("fam", "dataset", "kd"), "results/tifs/report/battery.csv (analyze.primary_battery)")
    checks += compare_rows(AD.global_holm_table(st), read_csv(OFFICIAL["global_holm"]),
                           ("fam", "dataset", "kd", "attack"),
                           "results/tifs/addendum/global_holm.csv (addendum.global_holm_table)")
    null_rows, _ = AD.cross_tables(st, [CROSS["sst2"], CROSS["agnews"]])
    checks += compare_rows(null_rows, read_csv(OFFICIAL["cross_null"]),
                           ("dataset", "kd", "trigger", "student"),
                           "results/tifs/addendum/cross_null.csv (addendum.cross_tables)")
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
            print("  gate: primary.csv, variance.csv, battery.csv, global_holm.csv, cross_null.csv "
                  "reproduced exactly", flush=True)
        _STATE = st
    return _STATE


def cell_id(r):
    return f"{r['fam']}/{r['dataset']}/{r['kd']}/{r['attack']}"


# ── extra vectors of the battery cells (untriggered teacher/student hits) ──


def battery_vectors(st):
    """{(fam, ds, kd, attack, seed): vectors} for every clean-corpus battery cell the frozen loader
    reads (same glob, run filter, schema and .json companion rule as analyze.load_cells), with the
    untriggered vectors analyze.load_cells does not keep (teacher_p_clean, teacher_c_clean). The
    vectors load_cells keeps are checked against it."""
    frozen = {}
    for c in st.cells:
        if c["block"] == "G" and c["corpus"] == "clean":
            key = (c["fam"], c["dataset"], c["kd"], c["attack"], c["seed"])
            if key in frozen:
                raise ValueError(f"duplicate cell {key}")
            frozen[key] = c
    out = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "tifs_*", "artifacts", "results", "*.preds.json"))):
        run = p.split(os.sep)[-4]
        info = A.parse_run(run)
        if info is None or not os.path.exists(p[: -len(".preds.json")] + ".json"):
            continue
        d = load_json(p)
        if (d.get("meta") or {}).get("schema") != 2:
            continue
        if not (info["block"] == "G" and info.get("corpus") == "clean"):
            continue
        r = load_json(p[: -len(".preds.json")] + ".json")
        key = (info["fam"], r["dataset"], r["kd_type"], r["attack_method"], int(r.get("seed", -1)))
        c = frozen.get(key)
        if c is None:
            raise ValueError(f"{p}: not among the frozen cells")
        for mine, theirs in (("preds_obs", "P"), ("preds_null", "C"), ("preds_p_clean", "Pc"),
                             ("preds_c_clean", "Cc"), ("teacher_p_trig", "TP"), ("teacher_c_trig", "TC")):
            if d.get(mine) != c[theirs]:
                raise ValueError(f"{p}: {mine} differs from analyze.load_cells")
        out[key] = {"run": run, "path": rel(p), "P": d["preds_obs"], "C": d["preds_null"],
                    "Pc": d.get("preds_p_clean"), "Cc": d.get("preds_c_clean"),
                    "TP": d.get("teacher_p_trig"), "TC": d.get("teacher_c_trig"),
                    "TPc": d.get("teacher_p_clean"), "TCc": d.get("teacher_c_clean"),
                    "single": bool(d["meta"].get("single_student")), "res": r}
    if set(out) != set(frozen):
        raise ValueError("battery_vectors and analyze.load_cells disagree on the cell set")
    return out


def loader_snapshot():
    """Prediction files the frozen loader reads, the count of all prediction files, newest mtimes."""
    allp = sorted(glob.glob(os.path.join(RUNS, "tifs_*", "artifacts", "results", "*.preds.json")))
    used = []
    for p in allp:
        run = p.split(os.sep)[-4]
        if A.parse_run(run) is None or not os.path.exists(p[: -len(".preds.json")] + ".json"):
            continue
        if (load_json(p).get("meta") or {}).get("schema") != 2:
            continue
        used.append(p)
    newest = max(used, key=os.path.getmtime)
    newest_all = max(allp, key=os.path.getmtime)
    return {"n_loader": len(used), "n_runs_loader": len({p.split(os.sep)[-4] for p in used}),
            "newest_loader_utc": mtime_utc(newest), "newest_loader": rel(newest),
            "n_all": len(allp), "n_runs_all": len({p.split(os.sep)[-4] for p in allp}),
            "newest_all_utc": mtime_utc(newest_all), "newest_all": rel(newest_all)}
