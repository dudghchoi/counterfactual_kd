#!/usr/bin/env python3
"""g10 selfcheck — run last by run_all.sh, before make_readme.py.

A15 post hoc auxiliary analysis (claim-rule calibration); changes no primary verdict, battery decision or the
paper-wide Holm.

selfcheck.csv collects:
  - frozen-file sha256 against their records (freeze record, final-analysis snapshot, claim-rule registration
    snapshot), and every file hashed at the start of run_all.sh (logs/hashes_before.txt) unchanged now
  - the gate: primary.csv, variance.csv, battery.csv, global_holm.csv, attack_variance.csv reproduced exactly
    through the frozen loader (every column, text)
  - every prediction file the frozen loader reads predates the final-analysis snapshot (no A13 / A15 run is read)
  - all rows of sim_checks.csv, frozen_claim_checks.csv and independent_check.csv ok
  - every CSV of the output folder starts with '# definition:'; no __pycache__ in the code folders this group
    imports from
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime  # noqa: E402
import glob  # noqa: E402
import os  # noqa: E402

import g10_common as G  # noqa: E402


def main():
    ap = G.add_out_arg(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = ap.parse_args()
    out = G.out_dir(args.out)
    checks = []

    def chk(name, exp, got, ok=None, note=""):
        checks.append({"check": name, "expected": exp, "got": got, "ok": bool(exp == got if ok is None else ok),
                       "note": note})

    for r in G.frozen_hash_rows():
        chk(f"sha256 {r['file']} == {r['record']}", r["sha256_record"][:16], r["sha256_now"][:16], ok=r["ok"])
    before = os.path.join(out, "logs", "hashes_before.txt")
    if os.path.exists(before):
        n = 0
        for ln in open(before):
            h, p = ln.split(None, 1)
            p = p.strip()
            n += 1
            chk(f"sha256 {p} unchanged during run_all.sh", h[:16], G.sha256(os.path.join(G.ROOT, p))[:16])
        chk("files hashed at the start of run_all.sh", True, n > 0, note=f"{n} files")
    else:
        chk("logs/hashes_before.txt exists (written by run_all.sh)", True, False, ok=False,
            note="run through run_all.sh to compare hashes before / after")
    st = G.state(gate=False)
    for c in G.reproduction_checks(st):
        checks.append(c)
    snap = G.loader_snapshot()
    newest = datetime.datetime.strptime(snap["newest_loader_utc"], "%Y-%m-%d %H:%M:%S UTC").replace(
        tzinfo=datetime.timezone.utc)
    chk(f"newest prediction file read by the frozen loader predates {G.FINAL}", True, newest <= G.FINAL_SNAPSHOT_UTC,
        note=f"{snap['n_loader']} files in {snap['n_runs_loader']} runs, newest {snap['newest_loader_utc']}; "
             f"{snap['n_all']} prediction files under runs/tifs in all (the others are not parsed)")
    for f in ("sim_checks.csv", "frozen_claim_checks.csv", "independent_check.csv"):
        path = os.path.join(out, f)
        if not os.path.exists(path):
            chk(f"{f} exists", True, False)
            continue
        rows = G.read_csv(path)
        bad = [r["check"][:80] for r in rows if r["ok"] != "True"]
        chk(f"{f}: all rows ok", f"{len(rows)} ok", f"{len(rows) - len(bad)} ok", note="; ".join(bad[:4]))
    nbad = []
    for f in sorted(glob.glob(os.path.join(out, "*.csv"))):
        if os.path.basename(f) == "selfcheck.csv":
            continue
        with open(f) as fh:
            if not fh.readline().startswith("# definition:"):
                nbad.append(os.path.basename(f))
    chk("every CSV starts with '# definition:'", 0, len(nbad), note=",".join(nbad))
    pyc = []
    start = os.path.join(out, "logs", "run_start.txt")
    if os.path.exists(start):
        t0 = os.path.getmtime(start)
        for d in (G.HERE, G.G4_DIR, G.G6M_DIR, os.path.join(G.ROOT, "scripts", "tifs"),
                  os.path.join(G.ROOT, "counterfactual_kd"), os.path.join(G.ROOT, "counterfactual_kd", "analysis"),
                  os.path.join(G.ROOT, "counterfactual_kd", "metrics")):
            for p in glob.glob(os.path.join(d, "__pycache__", "*.pyc")):
                if os.path.getmtime(p) >= t0:
                    pyc.append(G.rel(p))
        chk("no bytecode (.pyc) written during the run in the code folders used", 0, len(pyc), note=",".join(pyc[:6]))
    else:
        chk("logs/run_start.txt exists (written by run_all.sh)", True, False, ok=False)
    chk("no __pycache__ folder in scripts/tifs/aux/g10-claimrule", False,
        os.path.isdir(os.path.join(G.HERE, "__pycache__")))
    G.write_csv(checks, os.path.join(out, "selfcheck.csv"), """
        g10 self-checks: frozen-file hashes (records; unchanged during the run), the exact reproduction of the
        official tables through the frozen loader before anything is computed, the loader's inputs, the check files
        of sim.py, frozen_claims.py and independent.py, CSV format, no bytecode.""")
    bad = [c for c in checks if not c["ok"]]
    print(f"selfcheck: {len(checks)} checks, {len(bad)} failed")
    for c in bad:
        print("  FAILED:", c["check"], c["expected"], c["got"], c.get("note", ""))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
