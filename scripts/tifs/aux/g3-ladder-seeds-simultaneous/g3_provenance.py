#!/usr/bin/env python3
"""Data snapshot of the g3 tables (A11 items 5, 8, 9): provenance.csv.

Rows by kind:
  preds   Stage 2 prediction files under runs/tifs, per run group: the frozen loader
          (analyze.RUN_RE; items 5, 8 and 9), the A2 replicate (tifs_rep_bert_atba_t43_s*;
          item 5), the A8 hidden-state runs (tifs_hid_*; the item 8 extra pool) and the runs
          g3 does not read. Count, and the newest file with its mtime (UTC and KST).
  input   every other file the g3 scripts read: SHA-256, size, mtime.
  frozen  the frozen files and the addendum code/document: SHA-256 against the freeze record
          (freeze_20261021T092041Z) and the final-analysis snapshot (final_analysis_20261024T110542Z).
Read-only; writes only provenance.csv to the output folder.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # before any local import: no __pycache__ in the repository

import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

from g3_common import A, AD, FROZEN_SHA256, INPUTS, ROOT, RUNS, out_dir, rel, sha256, write_csv  # noqa: E402

FINAL_SNAPSHOT = os.path.join(RUNS, "_code", "final_analysis_20261024T110542Z.json")
KST = timezone(timedelta(hours=9))


def stamp(path):
    t = os.path.getmtime(path)
    return (datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            datetime.fromtimestamp(t, KST).strftime("%Y-%m-%d %H:%M:%S KST"))


def group_of(run_id):
    if A.parse_run(run_id) is not None:
        return "frozen loader (analyze.RUN_RE): items 5, 8, 9"
    if run_id.startswith(AD.REP_RE):
        return "A2 replicate (tifs_rep_bert_atba_t43_s*): item 5 a2 rung"
    if AD.HID_RX.match(run_id):
        return "A8 hidden (tifs_hid_*): item 8 extra pool"
    return "not read by g3"


def main():
    out = out_dir(__doc__.splitlines()[0])
    rows, bad = [], []
    preds = sorted(glob.glob(os.path.join(RUNS, "tifs_*", "artifacts", "results", "*.preds.json")))
    groups = {}
    for p in preds:
        groups.setdefault(group_of(p.split(os.sep)[-4]), []).append(p)
    for name, ps in [("all prediction files", preds)] + sorted(groups.items()):
        newest = max(ps, key=os.path.getmtime)
        utc, kst = stamp(newest)
        rows.append({"kind": "preds", "name": name, "n_files": len(ps),
                     "n_runs": len({p.split(os.sep)[-4] for p in ps}), "path": rel(newest),
                     "mtime_utc": utc, "mtime_kst": kst})
    snap = json.load(open(FINAL_SNAPSHOT)) if os.path.exists(FINAL_SNAPSHOT) else {}
    ok = snap.get("preds_files_in_runs_tifs") == len(preds)
    rows.append({"kind": "snapshot", "name": "final_analysis_20261024T110542Z preds_files_in_runs_tifs",
                 "n_files": snap.get("preds_files_in_runs_tifs"), "path": rel(FINAL_SNAPSHOT), "ok": ok})
    if not ok:
        bad.append(f"prediction files {len(preds)} != final-analysis snapshot {snap.get('preds_files_in_runs_tifs')}")
    for path in INPUTS:
        full = os.path.join(ROOT, path)
        utc, kst = stamp(full)
        rows.append({"kind": "input", "name": os.path.basename(path), "path": path, "sha256": sha256(full),
                     "bytes": os.path.getsize(full), "mtime_utc": utc, "mtime_kst": kst})
    for path, want in FROZEN_SHA256.items():
        got = sha256(os.path.join(ROOT, path))
        rows.append({"kind": "frozen", "name": os.path.basename(path), "path": path, "sha256": got,
                     "expected_sha256": want, "ok": got == want})
        if got != want:
            bad.append(f"{path}: {got} != {want}")
    write_csv(rows, os.path.join(out, "provenance.csv"))
    newest = rows[0]
    print(f"provenance: {len(preds)} prediction files, newest {newest['mtime_utc']} ({newest['path']}); "
          f"{len(INPUTS)} inputs, {len(FROZEN_SHA256)} frozen files -> {out}")
    for b in bad:
        print("  FAILED", b)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
