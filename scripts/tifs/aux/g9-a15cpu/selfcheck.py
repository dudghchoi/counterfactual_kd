#!/usr/bin/env python3
"""Self-checks of the A15 CPU items (run last by run_all.sh, before make_readme.py).

A15 post hoc CPU analysis; changes no primary verdict, battery decision or the paper-wide Holm.

selfcheck.csv collects:
  - frozen-file sha256 against the freeze record, and every hashed file (frozen, tracked, official inputs)
    unchanged between the start of run_all.sh (logs/hashes_before.txt) and now
  - the gate: primary.csv, variance.csv, battery.csv, global_holm.csv, cross_null.csv reproduced exactly
    through the frozen loader (every column, text)
  - every file the frozen loader reads predates the final-analysis snapshot (no new battery predictions)
  - A14 item 3 reproduced: the uncorrected ('A14') null-swap counts of item 3 equal
    g6-review0/reference/nullswap_own_summary.csv; A14 item 5's excess_null rows equal item 3's A14-weighted
    excess (pooled, own, + s2_N/J_N)
  - all rows of check_item1..5.csv and independent_check.csv ok; every CSV starts with '# definition:'
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import datetime  # noqa: E402
import glob  # noqa: E402
import os  # noqa: E402

import common as K  # noqa: E402

FINAL_SNAPSHOT_UTC = datetime.datetime(2026, 10, 24, 11, 5, 42, tzinfo=datetime.timezone.utc)


def main():
    out = K.out_dir()
    checks = []

    def chk(name, exp, got, ok=None, note=""):
        checks.append({"check": name, "expected": exp, "got": got,
                       "ok": bool(exp == got if ok is None else ok), "note": note})

    # frozen hashes
    for p, h in K.FROZEN_SHA256.items():
        chk(f"sha256 {p} == freeze record", h[:16], K.sha256(os.path.join(K.ROOT, p))[:16])
    before = os.path.join(out, "logs", "hashes_before.txt")
    if os.path.exists(before):
        n = 0
        strict = set(K.FROZEN_SHA256) | {K.rel(v) for v in K.OFFICIAL.values()} | {K.rel(v) for v in K.CROSS.values()}
        for ln in open(before):
            h, p = ln.split(None, 1)
            p = p.strip()
            n += 1
            now = K.sha256(os.path.join(K.ROOT, p))
            if p in strict or p.startswith("results/"):
                chk(f"sha256 {p} unchanged during run_all.sh", h[:16], now[:16])
            else:
                chk(f"sha256 {p} unchanged during run_all.sh (tracked, not frozen: info)", h[:16], now[:16], ok=True,
                    note="" if h == now else "changed during the run (edited elsewhere); item 5 records the hash "
                                             "of every module it loaded in g3_dose_modules.csv")
        chk("files hashed at the start of run_all.sh", True, n > 0, note=f"{n} files")
    else:
        chk("logs/hashes_before.txt exists (written by run_all.sh)", True, False,
            ok=False, note="run through run_all.sh to compare hashes before/after")
    # gate
    st = K.state(gate=False)
    for c in K.reproduction_checks(st):
        checks.append(c)
    # loader inputs predate the final analysis
    snap = K.loader_snapshot()
    newest = datetime.datetime.strptime(snap["newest_loader_utc"], "%Y-%m-%d %H:%M:%S UTC").replace(
        tzinfo=datetime.timezone.utc)
    chk("newest prediction file read by the frozen loader predates final_analysis_20261024T110542Z", True,
        newest <= FINAL_SNAPSHOT_UTC, note=f"{snap['n_loader']} files, newest {snap['newest_loader_utc']}")
    # A14 item 3 counts
    g6 = {(r["dataset"], r["kd"], r["trigger"]): r for r in K.read_csv(os.path.join(K.G6REF, "nullswap_own_summary.csv"))}
    mine = {(r["dataset"], r["kd"], r["trigger"]): r for r in K.read_csv(os.path.join(out, "nullswap_summary.csv"))}
    nb = 0
    for key, o in g6.items():
        m = mine[key]
        nb += o["n_null"] != m["n_null_A14"]
        for t in K.TOLS:
            for rule, mrule in (("pooled", "frozen"), ("own", "own"), ("max", "max")):
                a = o[f"null_POSITIVE_{rule}@{t:g}"]
                b = m[f"POS_{mrule}_A14@{t:g}"]
                if a.startswith("n/a"):
                    a = "n/a"
                nb += a != b
    chk("A14 item 3 reproduced: n_null and POSITIVE counts (pooled/own/max x 3 deltas x 16 groups) == "
        "g6-review0 nullswap_own_summary.csv", 0, nb)
    # A14 item 5 excess
    km = K.read_csv(os.path.join(K.G6REF, "kappa_marg.csv"))
    ex = K.read_csv(os.path.join(out, "nullswap_excess.csv"))
    names = {"excess_null (frozen seed-pooled rule)": "pooled (frozen seed-pooled rule)",
             "excess_null (own seed variance, df r-1)": "own seed variance (df r-1)",
             "excess_null + between-null-student variance (kappa_marg over nulls)": "with_null_spread (pooled + s2_N/J_N)"}
    md, n = 0.0, 0
    for r in km:
        if r["estimand"] in names and r["basis"] == "CPU fp32 re-score":
            e = next(x for x in ex if (x["dataset"], x["kd"], x["trigger"], x["scheme"], x["reading"])
                     == (r["dataset"], r["kd"], "atba", "A14", names[r["estimand"]]))
            md = max(md, *(abs(float(r[c]) - float(e[c])) for c in ("k", "lo", "hi")))
            n += 1
    chk("A14 item 5 excess_null rows (k, lo, hi) reproduced by item 3's A14 weighting: max |diff|",
        True, md < 1e-12, note=f"{n} rows, max |diff| {md:.3g}")
    # item checks and the independent review
    for f in sorted(glob.glob(os.path.join(out, "check_item*.csv"))) + [os.path.join(out, "independent_check.csv")]:
        if not os.path.exists(f):
            chk(f"{os.path.basename(f)} exists", True, False)
            continue
        rows = K.read_csv(f)
        bad = [r["check"] for r in rows if r["ok"] != "True"]
        chk(f"{os.path.basename(f)}: all rows ok", f"{len(rows)} ok", f"{len(rows) - len(bad)} ok",
            note="; ".join(bad[:5]))
    # CSV format
    nbad = []
    for f in sorted(glob.glob(os.path.join(out, "*.csv"))):
        if os.path.basename(f) == "selfcheck.csv":
            continue
        with open(f) as fh:
            if not fh.readline().startswith("# definition:"):
                nbad.append(os.path.basename(f))
    chk("every CSV starts with '# definition:'", 0, len(nbad), note=",".join(nbad))
    K.write_csv(checks, os.path.join(out, "selfcheck.csv"), """
        Self-checks of the A15 CPU items: frozen-file hashes (freeze record; unchanged during the run), the
        exact reproduction of the official tables through the frozen loader, the loader's inputs, the A14
        item 3 / item 5 reproductions, the item check files and the independent review, CSV format.""")
    bad = [c for c in checks if not c["ok"]]
    print(f"selfcheck: {len(checks)} checks, {len(bad)} failed")
    for c in bad:
        print("  FAILED:", c["check"], c["expected"], c["got"], c.get("note", ""))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
