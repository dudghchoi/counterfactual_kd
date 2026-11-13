#!/usr/bin/env python3
"""A13 self-checks.

A13 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

  --gate   before anything is computed: frozen-file sha256 = the registered values, and the official
           primary.csv, battery.csv, variance.csv, secondary.csv and reference_atba.csv reproduced exactly
           (every column, text) through the frozen loader; exits 3 on any failure (run_all.sh stops).
  (full)   after the group scripts: the gate again, frozen and tracked files unchanged since the start of
           run_all.sh (G7_SHA256_START), the A14 g6-review0/reference outputs unchanged (G7_G6_DIGEST_START),
           every CSV of both output folders starts with '# definition:', every group check and the
           independent check pass, the A13 input inventory (25 runs, 38 prediction files) -> selfcheck.csv.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402

import common as K  # noqa: E402


def g6_digest():
    """sha256 over the A14 'reference' output files (paths and contents), to show they were not written."""
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(K.G6REF, "**", "*"), recursive=True)):
        if os.path.isfile(p):
            h.update(K.rel(p).encode())
            h.update(K.sha256(p).encode())
    return h.hexdigest()


def hash_checks(start_env):
    checks = []
    for p, want in K.FROZEN_SHA256.items():
        got = K.sha256(os.path.join(K.ROOT, p))
        checks.append({"check": f"frozen file sha256 = registered: {p}", "expected": want[:16], "got": got[:16],
                       "ok": got == want})
    start = dict(kv.split("=", 1) for kv in start_env.split(";") if "=" in kv) if start_env else {}
    for p in list(K.FROZEN_SHA256) + list(K.TRACKED):
        got = K.sha256(os.path.join(K.ROOT, p))
        if p in start:
            checks.append({"check": f"unchanged since the start of run_all.sh: {p}", "expected": start[p][:16],
                           "got": got[:16], "ok": got == start[p]})
    return checks


def main():
    out = K.out_dir()
    gate = "--gate" in sys.argv
    checks = hash_checks(os.environ.get("G7_SHA256_START", ""))
    st = K.state(gate=False)
    checks += K.reproduction_checks(st)
    if gate:
        bad = [c for c in checks if not c["ok"]]
        for c in checks:
            print(("  ok   " if c["ok"] else "  FAIL ") + c["check"], c["got"])
        if bad:
            raise SystemExit(3)
        return
    d0 = os.environ.get("G7_G6_DIGEST_START")
    d1 = g6_digest()
    checks.append({"check": "A14 g6-review0/reference outputs (all files) unchanged since the start of run_all.sh",
                   "expected": (d0 or "")[:16], "got": d1[:16], "ok": d0 == d1})
    files, newest = K.x13_files()
    runs = sorted({p.split(os.sep)[-4] for p in files})
    checks.append({"check": "inputs: tifs_x13 prediction files with a result .json (all schema 2), runs = registered list",
                   "expected": "38 files, 25 runs", "got": f"{len(files)} files, {len(runs)} runs",
                   "ok": len(files) == 38 and runs == sorted(K.EXPECTED_RUNS)
                   and all(K.load_json(p)["meta"]["schema"] == 2 and os.path.exists(p[:-len('.preds.json')] + '.json')
                           for p in files)})
    # CSV format in both output folders (this group's and the item-7 extension).
    csvs = sorted(glob.glob(os.path.join(out, "*.csv")) + glob.glob(os.path.join(out, "item7", "*.csv")))
    csvs = [p for p in csvs if os.path.basename(p) != "selfcheck.csv"]
    bad_fmt = [K.rel(p) for p in csvs if not open(p).readline().startswith("# definition:")]
    checks.append({"check": "every CSV of results/tifs/addendum/aux/g7-a13 (and item7/) starts with '# definition:'",
                   "expected": f"{len(csvs)} files", "got": f"{len(csvs) - len(bad_fmt)} ok", "ok": not bad_fmt,
                   "note": ";".join(bad_fmt)})
    # Group checks and the independent check.
    for name in ("check_g1.csv", "check_g2.csv", "check_g2_item7.csv", "check_g3.csv", "check_g4.csv",
                 "check_common.csv"):
        p = os.path.join(out, name)
        rows = K.read_csv(p) if os.path.exists(p) else []
        checks.append({"check": f"{name}: every check passes", "expected": f"{len(rows)} ok (> 0)",
                       "got": f"{sum(r['ok'] == 'True' for r in rows)} ok", "ok": bool(rows) and all(r["ok"] == "True" for r in rows)})
    p = os.path.join(out, "independent_check.csv")
    rows = K.read_csv(p) if os.path.exists(p) else []
    checks.append({"check": "independent_check.csv: every headline quantity agrees (independent code path)",
                   "expected": f"{len(rows)} agree (> 0)", "got": f"{sum(r['agree'] == 'True' for r in rows)} agree",
                   "ok": bool(rows) and all(r["agree"] == "True" for r in rows)})
    for c in checks:
        c.setdefault("note", "")
    K.write_csv(checks, os.path.join(out, "selfcheck.csv"), """
        A13 self-checks: frozen-file sha256 against the registered values and against the start of run_all.sh,
        tracked files unchanged, the official tables reproduced exactly through the frozen loader, the A14
        outputs untouched, the A13 input inventory, the CSV format, every group check and the independent check.""")
    n_bad = sum(not c["ok"] for c in checks)
    print(f"  selfcheck: {len(checks) - n_bad} / {len(checks)} pass")
    for c in checks:
        if not c["ok"]:
            print("  FAIL", c)
    if n_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
